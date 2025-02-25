# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import traceback
from math import log
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pypdfium2 as libpdfium
import nv_ingest.util.nim.yolox as yolox_utils

from nv_ingest.schemas.metadata_schema import AccessLevelEnum
from nv_ingest.schemas.metadata_schema import TableFormatEnum
from nv_ingest.schemas.metadata_schema import TextTypeEnum
from nv_ingest.schemas.pdf_extractor_schema import PDFiumConfigSchema
from nv_ingest.util.image_processing.transforms import crop_image
from nv_ingest.util.image_processing.transforms import numpy_to_base64
from nv_ingest.util.nim.helpers import create_inference_client
from nv_ingest.util.pdf.metadata_aggregators import Base64Image
from nv_ingest.util.pdf.metadata_aggregators import CroppedImageWithContent
from nv_ingest.util.pdf.metadata_aggregators import construct_image_metadata_from_pdf_image
from nv_ingest.util.pdf.metadata_aggregators import construct_table_and_chart_metadata
from nv_ingest.util.pdf.metadata_aggregators import construct_text_metadata
from nv_ingest.util.pdf.metadata_aggregators import extract_pdf_metadata
from nv_ingest.util.pdf.pdfium import PDFIUM_PAGEOBJ_MAPPING
from nv_ingest.util.pdf.pdfium import pdfium_pages_to_numpy
from nv_ingest.util.pdf.pdfium import pdfium_try_get_bitmap_as_numpy

YOLOX_MAX_BATCH_SIZE = 8
YOLOX_MAX_WIDTH = 1536
YOLOX_MAX_HEIGHT = 1536
YOLOX_NUM_CLASSES = 3
YOLOX_CONF_THRESHOLD = 0.01
YOLOX_IOU_THRESHOLD = 0.5
YOLOX_MIN_SCORE = 0.1
YOLOX_FINAL_SCORE = 0.48

logger = logging.getLogger(__name__)


def extract_tables_and_charts_using_image_ensemble(
    pages: List,  # List[libpdfium.PdfPage]
    config: PDFiumConfigSchema,
    trace_info: Optional[List] = None,
) -> List[Tuple[int, object]]:  # List[Tuple[int, CroppedImageWithContent]]
    tables_and_charts = []

    try:
        model_interface = yolox_utils.YoloxPageElementsModelInterface()
        yolox_client = create_inference_client(
            config.yolox_endpoints, model_interface, config.auth_token, config.yolox_infer_protocol
        )

        batches = []
        i = 0
        max_batch_size = YOLOX_MAX_BATCH_SIZE
        while i < len(pages):
            batch_size = min(2 ** int(log(len(pages) - i, 2)), max_batch_size)
            batches.append(pages[i : i + batch_size])  # noqa: E203
            i += batch_size

        page_index = 0
        for batch in batches:
            original_images, _ = pdfium_pages_to_numpy(
                batch, scale_tuple=(YOLOX_MAX_WIDTH, YOLOX_MAX_HEIGHT), trace_info=trace_info
            )

            # Prepare data
            data = {"images": original_images}

            # Perform inference using NimClient
            inference_results = yolox_client.infer(
                data,
                model_name="yolox",
                num_classes=YOLOX_NUM_CLASSES,
                conf_thresh=YOLOX_CONF_THRESHOLD,
                iou_thresh=YOLOX_IOU_THRESHOLD,
                min_score=YOLOX_MIN_SCORE,
                final_thresh=YOLOX_FINAL_SCORE,
                trace_info=trace_info,  # traceable_func arg
                stage_name="pdf_content_extractor",  # traceable_func arg
            )

            # Process results
            for annotation_dict, original_image in zip(inference_results, original_images):
                extract_table_and_chart_images(
                    annotation_dict,
                    original_image,
                    page_index,
                    tables_and_charts,
                )
                page_index += 1

    except TimeoutError:
        logger.error("Timeout error during table/chart extraction.")
        raise

    except Exception as e:
        logger.error(f"Unhandled error during table/chart extraction: {str(e)}")
        traceback.print_exc()
        raise e

    finally:
        if yolox_client:
            yolox_client.close()

    logger.debug(f"Extracted {len(tables_and_charts)} tables and charts.")

    return tables_and_charts


# Handle individual table/chart extraction and model inference
def extract_table_and_chart_images(
    annotation_dict,
    original_image,
    page_idx,
    tables_and_charts,
):
    """
    Handle the extraction of tables and charts from the inference results and run additional model inference.

    Parameters
    ----------
    annotation_dict : dict/
        A dictionary containing detected objects and their bounding boxes.
    original_image : np.ndarray
        The original image from which objects were detected.
    page_idx : int
        The index of the current page being processed.
    tables_and_charts : List[Tuple[int, ImageTable]]
        A list to which extracted tables and charts will be appended.

    Notes
    -----
    This function iterates over detected objects, crops the original image to the bounding boxes,
    and runs additional inference on the cropped images to extract detailed information about tables
    and charts.

    Examples
    --------
    >>> annotation_dict = {"table": [], "chart": []}
    >>> original_image = np.random.rand(1536, 1536, 3)
    >>> tables_and_charts = []
    >>> extract_table_and_chart_images(annotation_dict, original_image, 0, tables_and_charts)
    """

    width, height, *_ = original_image.shape
    for label in ["table", "chart"]:
        if not annotation_dict:
            continue

        objects = annotation_dict[label]
        for idx, bboxes in enumerate(objects):
            *bbox, _ = bboxes
            h1, w1, h2, w2 = bbox * np.array([height, width, height, width])

            cropped = crop_image(original_image, (h1, w1, h2, w2))
            base64_img = numpy_to_base64(cropped)

            table_data = CroppedImageWithContent(
                content="",
                image=base64_img,
                bbox=(w1, h1, w2, h2),
                max_width=width,
                max_height=height,
                type_string=label,
            )
            tables_and_charts.append((page_idx, table_data))


# Define a helper function to use unstructured-io to extract text from a base64
# encoded bytestream PDF
def pdfium_extractor(
    pdf_stream,
    extract_text: bool,
    extract_images: bool,
    extract_tables: bool,
    extract_charts: bool,
    trace_info=None,
    **kwargs,
):
    """
    Helper function to use pdfium to extract text from a bytestream PDF.

    Parameters
    ----------
    pdf_stream : io.BytesIO
        A bytestream PDF.
    extract_text : bool
        Specifies whether to extract text.
    extract_images : bool
        Specifies whether to extract images.
    extract_tables : bool
        Specifies whether to extract tables.
    extract_charts : bool
        Specifies whether to extract tables.
    **kwargs
        The keyword arguments are used for additional extraction parameters.

        kwargs.pdfium_config : dict, optional[PDFiumConfigSchema]

    Returns
    -------
    str
        A string of extracted text.
    """
    logger.debug("Extracting PDF with pdfium backend.")

    row_data = kwargs.get("row_data")
    source_id = row_data["source_id"]
    text_depth = kwargs.get("text_depth", "page")
    text_depth = TextTypeEnum[text_depth.upper()]
    paddle_output_format = kwargs.get("paddle_output_format", "pseudo_markdown")
    paddle_output_format = TableFormatEnum[paddle_output_format.upper()]

    # get base metadata
    metadata_col = kwargs.get("metadata_column", "metadata")

    pdfium_config = kwargs.get("pdfium_config", {})
    pdfium_config = pdfium_config if pdfium_config is not None else {}

    base_unified_metadata = row_data[metadata_col] if metadata_col in row_data.index else {}

    base_source_metadata = base_unified_metadata.get("source_metadata", {})
    source_location = base_source_metadata.get("source_location", "")
    collection_id = base_source_metadata.get("collection_id", "")
    partition_id = base_source_metadata.get("partition_id", -1)
    access_level = base_source_metadata.get("access_level", AccessLevelEnum.LEVEL_1)

    pages = []
    extracted_data = []
    doc = libpdfium.PdfDocument(pdf_stream)
    pdf_metadata = extract_pdf_metadata(doc, source_id)

    source_metadata = {
        "source_name": pdf_metadata.filename,
        "source_id": source_id,
        "source_location": source_location,
        "source_type": pdf_metadata.source_type,
        "collection_id": collection_id,
        "date_created": pdf_metadata.date_created,
        "last_modified": pdf_metadata.last_modified,
        "summary": "",
        "partition_id": partition_id,
        "access_level": access_level,
    }

    logger.debug(f"Extracting text from PDF with {pdf_metadata.page_count} pages.")
    logger.debug(f"Extract text: {extract_text}")
    logger.debug(f"extract images: {extract_images}")
    logger.debug(f"extract tables: {extract_tables}")
    logger.debug(f"extract tables: {extract_charts}")

    # Pdfium does not support text extraction at the document level
    accumulated_text = []
    text_depth = text_depth if text_depth == TextTypeEnum.PAGE else TextTypeEnum.DOCUMENT
    for page_idx in range(pdf_metadata.page_count):
        page = doc.get_page(page_idx)
        page_width, page_height = doc.get_page_size(page_idx)

        # https://pypdfium2.readthedocs.io/en/stable/python_api.html#module-pypdfium2._helpers.textpage
        if extract_text:
            textpage = page.get_textpage()
            page_text = textpage.get_text_bounded()
            accumulated_text.append(page_text)

            if text_depth == TextTypeEnum.PAGE and len(accumulated_text) > 0:
                text_extraction = construct_text_metadata(
                    accumulated_text,
                    pdf_metadata.keywords,
                    page_idx,
                    -1,
                    -1,
                    -1,
                    pdf_metadata.page_count,
                    text_depth,
                    source_metadata,
                    base_unified_metadata,
                )

                extracted_data.append(text_extraction)
                accumulated_text = []

        # Image extraction
        if extract_images:
            for obj in page.get_objects():
                obj_type = PDFIUM_PAGEOBJ_MAPPING.get(obj.type, "UNKNOWN")
                if obj_type == "IMAGE":
                    try:
                        # Attempt to retrieve the image bitmap
                        image_numpy: np.ndarray = pdfium_try_get_bitmap_as_numpy(obj)  # noqa
                        image_base64: str = numpy_to_base64(image_numpy)
                        image_bbox = obj.get_pos()
                        image_size = obj.get_size()
                        image_data = Base64Image(
                            image=image_base64,
                            bbox=image_bbox,
                            width=image_size[0],
                            height=image_size[1],
                            max_width=page_width,
                            max_height=page_height,
                        )

                        extracted_image_data = construct_image_metadata_from_pdf_image(
                            image_data,
                            page_idx,
                            pdf_metadata.page_count,
                            source_metadata,
                            base_unified_metadata,
                        )

                        extracted_data.append(extracted_image_data)
                    except Exception as e:
                        logger.error(f"Unhandled error extracting image: {e}")
                        pass  # Pdfium failed to extract the image associated with this object - corrupt or missing.

        # Table and chart collection
        if extract_tables or extract_charts:
            pages.append(page)

    if extract_text and text_depth == TextTypeEnum.DOCUMENT and len(accumulated_text) > 0:
        text_extraction = construct_text_metadata(
            accumulated_text,
            pdf_metadata.keywords,
            -1,
            -1,
            -1,
            -1,
            pdf_metadata.page_count,
            text_depth,
            source_metadata,
            base_unified_metadata,
        )

        extracted_data.append(text_extraction)

    if extract_tables or extract_charts:
        for page_idx, table_and_charts in extract_tables_and_charts_using_image_ensemble(
            pages,
            pdfium_config,
            trace_info=trace_info,
        ):
            if (extract_tables and (table_and_charts.type_string == "table")) or (
                extract_charts and (table_and_charts.type_string == "chart")
            ):
                if table_and_charts.type_string == "table":
                    table_and_charts.content_format = paddle_output_format

                extracted_data.append(
                    construct_table_and_chart_metadata(
                        table_and_charts,
                        page_idx,
                        pdf_metadata.page_count,
                        source_metadata,
                        base_unified_metadata,
                    )
                )

    logger.debug(f"Extracted {len(extracted_data)} items from PDF.")

    return extracted_data
