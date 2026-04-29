# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-friendly content row transforms used by example pipelines."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.io.image_store import resolve_image_b64
from nemo_retriever.ocr.ocr import _crop_b64_image_by_norm_bbox
from nemo_retriever.params.models import IMAGE_MODALITIES

logger = logging.getLogger(__name__)

_CONTENT_COLUMNS = ("table", "chart", "infographic")


def _combine_text_with_content(row: Any, text_column: str, content_columns: Sequence[str]) -> str:
    """Combine page text with OCR content text for embedding."""
    parts = []
    base = row.get(text_column)
    if isinstance(base, str) and base.strip():
        parts.append(base.strip())
    for col in content_columns:
        content_list = row.get(col)
        if isinstance(content_list, list):
            for item in content_list:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
                    caption = item.get("caption", "")
                    if isinstance(caption, str) and caption.strip():
                        parts.append(caption.strip())
    return "\n\n".join(parts) if parts else ""


def _deep_copy_row(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Copy row dicts without sharing nested mutable values across exploded rows."""
    import copy

    out: Dict[str, Any] = {}
    for key, value in row_dict.items():
        if isinstance(value, (dict, list)):
            out[key] = copy.deepcopy(value)
        else:
            out[key] = value
    return out


def explode_content_to_rows(
    batch_df: Any,
    *,
    text_column: str = "text",
    content_columns: Sequence[str] = _CONTENT_COLUMNS,
    modality: str = "text",
    text_elements_modality: Optional[str] = None,
    structured_elements_modality: Optional[str] = None,
) -> Any:
    """Expand each page row into multiple rows for per-element embedding."""
    text_mod = text_elements_modality or modality
    struct_mod = structured_elements_modality or modality

    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df

    any_images = text_mod in IMAGE_MODALITIES or struct_mod in IMAGE_MODALITIES

    if not any(column in batch_df.columns for column in content_columns):
        batch_df = batch_df.copy()
        if text_mod in IMAGE_MODALITIES and "page_image" in batch_df.columns:
            batch_df["_image_b64"] = batch_df["page_image"].apply(
                lambda page_image: resolve_image_b64(page_image) if isinstance(page_image, dict) else None
            )
        if "page_image" in batch_df.columns:
            batch_df["_stored_image_uri"] = batch_df["page_image"].apply(
                lambda page_image: page_image.get("stored_image_uri") if isinstance(page_image, dict) else None
            )
        batch_df["_embed_modality"] = text_mod
        return batch_df

    new_rows: List[Dict[str, Any]] = []
    for _, row in batch_df.iterrows():
        row_dict = row.to_dict()
        exploded_any = False

        page_image = row_dict.get("page_image")
        page_image_b64: Optional[str] = None
        page_stored_uri: Optional[str] = None
        if isinstance(page_image, dict):
            page_stored_uri = page_image.get("stored_image_uri")
            if any_images:
                page_image_b64 = resolve_image_b64(page_image)

        page_text = row_dict.get(text_column)
        if isinstance(page_text, str) and page_text.strip():
            page_row = _deep_copy_row(row_dict)
            page_row["_embed_modality"] = text_mod
            page_row["_content_type"] = "text"
            if text_mod in IMAGE_MODALITIES:
                page_row["_image_b64"] = page_image_b64
            page_row["_stored_image_uri"] = page_stored_uri
            page_row["_bbox_xyxy_norm"] = None
            new_rows.append(page_row)
            exploded_any = True

        for column in content_columns:
            content_list = row_dict.get(column)
            if not isinstance(content_list, list):
                continue
            for item in content_list:
                if not isinstance(item, dict):
                    continue
                item_b64 = resolve_image_b64(item) if struct_mod in IMAGE_MODALITIES else None
                # Emit rows for text and (optionally) caption fields.
                for field, content_type in [("text", column), ("caption", f"{column}_caption")]:
                    value = item.get(field, "")
                    if not isinstance(value, str) or not value.strip():
                        continue
                    content_row = _deep_copy_row(row_dict)
                    content_row[text_column] = value.strip()
                    content_row["_embed_modality"] = struct_mod
                    content_row["_content_type"] = content_type
                    if struct_mod in IMAGE_MODALITIES:
                        if item_b64:
                            content_row["_image_b64"] = item_b64
                        elif page_image_b64:
                            bbox = item.get("bbox_xyxy_norm")
                            if bbox and len(bbox) == 4:
                                cropped_b64, _ = _crop_b64_image_by_norm_bbox(page_image_b64, bbox_xyxy_norm=bbox)
                                content_row["_image_b64"] = cropped_b64
                            else:
                                content_row["_image_b64"] = page_image_b64
                        else:
                            content_row["_image_b64"] = None
                    content_row["_stored_image_uri"] = item.get("stored_image_uri") or page_stored_uri
                    content_row["_bbox_xyxy_norm"] = item.get("bbox_xyxy_norm")
                    new_rows.append(content_row)
                    exploded_any = True

        if not exploded_any:
            preserved = _deep_copy_row(row_dict)
            preserved["_embed_modality"] = text_mod
            preserved["_content_type"] = "text"
            if text_mod in IMAGE_MODALITIES:
                preserved["_image_b64"] = page_image_b64
            preserved["_stored_image_uri"] = page_stored_uri
            preserved["_bbox_xyxy_norm"] = None
            new_rows.append(preserved)

    return pd.DataFrame(new_rows).reset_index(drop=True)


def collapse_content_to_page_rows(
    batch_df: Any,
    *,
    text_column: str = "text",
    content_columns: Sequence[str] = _CONTENT_COLUMNS,
    modality: str = "text",
) -> Any:
    """Collapse each page into a single row for page-level embedding."""
    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df

    batch_df = batch_df.copy()
    batch_df[text_column] = batch_df.apply(
        lambda row: _combine_text_with_content(row, text_column, content_columns),
        axis=1,
    )

    if modality in IMAGE_MODALITIES:
        if "page_image" in batch_df.columns:
            batch_df["_image_b64"] = batch_df["page_image"].apply(
                lambda page_image: resolve_image_b64(page_image) if isinstance(page_image, dict) else None
            )
        else:
            batch_df["_image_b64"] = None

    if "page_image" in batch_df.columns:
        batch_df["_stored_image_uri"] = batch_df["page_image"].apply(
            lambda page_image: page_image.get("stored_image_uri") if isinstance(page_image, dict) else None
        )

    batch_df["_embed_modality"] = modality
    return batch_df


def _row_segment_window(row: Any) -> tuple[str | None, float | None, float | None, str]:
    """Pull ``(source, segment_start, segment_end, text)`` from a row's metadata."""
    meta = row.get("metadata")
    if not isinstance(meta, dict):
        return None, None, None, row.get("text") or ""
    source = meta.get("source_path") or meta.get("source_video")
    start = meta.get("segment_start_seconds")
    end = meta.get("segment_end_seconds")
    try:
        start_f = float(start) if start is not None else None
        end_f = float(end) if end is not None else None
    except (TypeError, ValueError):
        start_f = end_f = None
    return (str(source) if source else None, start_f, end_f, row.get("text") or "")


# Cap merged OCR per audio row so a chatty UI screenshot doesn't dominate the
# embedding for a 10-word transcript utterance. The 95th percentile of observed
# OCR is ~134 words; capping at 60 trims the long tail without losing typical
# slide content.
_OCR_MAX_WORDS_PER_ROW = 60


def _clean_ocr_text(text: str) -> str:
    """Drop low-signal OCR strings; collapse runs of identical adjacent words.

    - empty / whitespace-only → ""
    - <10 chars after stripping → ""  (filters "TED", "NVIDIA", etc.)
    - no token longer than 3 chars → ""  (filters "+ + + +" symbol noise)
    - consecutive duplicate words ("Mosaic AI Mosaic AI" → "Mosaic AI")
    """
    if not isinstance(text, str):
        return ""
    s = " ".join(text.split())  # normalize whitespace
    if len(s) < 10:
        return ""
    tokens = s.split()
    if not any(len(tok) > 3 for tok in tokens):
        return ""
    # Collapse adjacent duplicate words (case-insensitive).
    deduped: List[str] = []
    last_lower: str | None = None
    for tok in tokens:
        if tok.lower() == last_lower:
            continue
        deduped.append(tok)
        last_lower = tok.lower()
    return " ".join(deduped)


@designer_component(
    name="Merge Video Frame Text Into Audio",
    category="Video",
    compute="cpu",
    description="Folds frame-OCR text into the audio row whose time window contains the frame.",
    category_color="#ff6bbb",
)
class MergeVideoFrameTextIntoAudioActor(AbstractOperator, CPUOperator):
    """Stateful: buffer frame OCR per ``source_video`` across batches, fold each
    frame's text into the audio row whose midpoint falls inside the frame window,
    and drop standalone frame rows.

    Frame rows (with ``page_image``) carry per-segment time windows (e.g. 60-75s);
    audio rows carry per-utterance windows from ASR fan-out (e.g. 62.3-64.1s).
    For ``recall_match_mode=audio_segment`` the matcher tests midpoint-in-window
    with a small tolerance, so frame rows seldom score on their own. Folding their
    OCR text into the temporally overlapping audio row preserves the OCR signal
    at the audio's fine-grained timestamps.

    Stateful design: state lives per actor instance, so this operator must be run
    with ``concurrency=1`` (the executor's default). The buffer holds one entry
    per (source, frame_window), bounded by total frame count across the dataset.

    Order independence: frames buffered as they arrive; audio rows look up the
    buffer immediately on receipt. As long as a video's frame rows precede its
    audio rows in upstream output (which the ASR concat order guarantees), every
    audio row sees the relevant frames.
    """

    def __init__(self) -> None:
        super().__init__()
        # source -> list[(start, end, text, page_image)]
        # page_image is the frame's image dict; carried alongside OCR text so
        # that a downstream embed stage can use it when embed_modality is
        # "text_image" (the VL embedder picks both signals up via the standard
        # explode/collapse path that derives _image_b64 from page_image).
        self._frames_by_source: Dict[str, List[tuple[float, float, str, Any]]] = {}

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, batch_df: Any, **kwargs: Any) -> Any:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return batch_df
        if "page_image" not in batch_df.columns:
            return batch_df  # no frames possible — pass through

        is_frame = batch_df["page_image"].apply(lambda v: isinstance(v, dict))
        frames = batch_df[is_frame]
        audio = batch_df[~is_frame]

        # Buffer this batch's frames keyed by source video. Skip frames whose
        # OCR is pure UI/symbol noise so the long tail doesn't dominate the
        # merged embedding.
        for _, row in frames.iterrows():
            source, start, end, text = _row_segment_window(row)
            if source is None or start is None or end is None:
                continue
            cleaned = _clean_ocr_text(text)
            if not cleaned:
                continue
            page_image = row.get("page_image")
            self._frames_by_source.setdefault(source, []).append((start, end, cleaned, page_image))

        if audio.empty:
            # Frames-only batch — they're now buffered, drop them from output.
            return batch_df.iloc[:0]

        # Look up buffered frames for each audio row. Pick the single closest
        # frame (by midpoint distance) when multiple windows contain the audio
        # midpoint — concatenating multiple frames' OCR amplifies UI chrome.
        new_texts: List[str] = []
        new_page_images: List[Any] = []
        for _, row in audio.iterrows():
            source, a_start, a_end, existing = _row_segment_window(row)
            if source is None or a_start is None or a_end is None:
                new_texts.append(existing)
                new_page_images.append(None)
                continue
            a_mid = (a_start + a_end) / 2.0
            # (midpoint_distance, text, page_image)
            best: tuple[float, str, Any] | None = None
            for f_start, f_end, text, page_image in self._frames_by_source.get(source, ()):
                if not (f_start <= a_mid <= f_end):
                    continue
                dist = abs(((f_start + f_end) / 2.0) - a_mid)
                if best is None or dist < best[0]:
                    best = (dist, text, page_image)
            if best is not None:
                ocr_blob = " ".join(best[1].split()[:_OCR_MAX_WORDS_PER_ROW])
                new_texts.append(f"{existing}\n\n{ocr_blob}" if existing else ocr_blob)
                new_page_images.append(best[2])
            else:
                new_texts.append(existing)
                new_page_images.append(None)

        merged = audio.copy()
        merged["text"] = new_texts
        # Always copy the matched frame's page_image. With embed_modality="text"
        # this column is unused; with embed_modality="text_image" the standard
        # explode/collapse path turns it into _image_b64 for the VL embedder.
        merged["page_image"] = new_page_images
        return merged.reset_index(drop=True)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, batch_df: Any) -> Any:
        return self.run(batch_df)
