# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VideoFrameExtractActor: Ray Data map_batches callable for video → image-page rows.

For each input video it uses MediaInterface to probe duration, computes segment
boundaries for the requested split strategy, and calls ffmpeg once per segment
to grab a single frame (demuxer seek, no re-segmentation to disk). Frames are
emitted in the same schema as image_bytes_to_pages_df so downstream detection
stages (PageElementDetection / OCR / TableStructure / GraphicElements) work
unchanged.
"""

from __future__ import annotations

import base64
import logging
import math
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from PIL import Image

from nemo_retriever.audio.media_interface import MediaInterface, is_media_available
from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.graph.operator_archetype import ArchetypeOperator
from nemo_retriever.params import VideoExtractParams

logger = logging.getLogger(__name__)

_PAGE_COLUMNS = [
    "path",
    "page_number",
    "source_id",
    "text",
    "page_image",
    "images",
    "tables",
    "charts",
    "infographics",
    "metadata",
]

_DEFAULT_DPI = 200


@designer_component(
    name="Video Frame Extractor",
    category="Video",
    compute="cpu",
    description="Splits video into N-second segments and extracts one frame per segment.",
    category_color="#ff6bbb",
)
class VideoFrameExtractCPUActor(AbstractOperator, CPUOperator):
    """Ray Data map_batches callable: DataFrame with ``path`` -> image-page rows.

    Each output row matches the image / PDF-extraction schema so downstream
    detection stages can consume it unchanged.
    """

    def __init__(self, params: VideoExtractParams | None = None) -> None:
        super().__init__(params=params)
        if not is_media_available():
            raise RuntimeError(
                "VideoFrameExtractActor requires ffmpeg. Install with: pip install ffmpeg-python and system ffmpeg."
            )
        self._params = params or VideoExtractParams()
        self._interface = MediaInterface()

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        if not isinstance(data, pd.DataFrame) or data.empty:
            return pd.DataFrame(columns=_PAGE_COLUMNS)
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        if not isinstance(data, pd.DataFrame) or data.empty:
            return pd.DataFrame(columns=_PAGE_COLUMNS)

        out_rows: List[Dict[str, Any]] = []
        for _, row in data.iterrows():
            path = row.get("path")
            if path is None:
                continue
            path_str = str(path)
            if not path_str.strip():
                continue
            try:
                out_rows.extend(_extract_frames_one(path_str, self._params, self._interface))
            except Exception as exc:
                logger.exception("Frame extraction failed for %s: %s", path_str, exc)
                continue

        if not out_rows:
            return pd.DataFrame(columns=_PAGE_COLUMNS)
        return pd.DataFrame(out_rows)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        return self.run(batch_df)


class VideoFrameExtractActor(ArchetypeOperator):
    """Graph-facing archetype resolving to the CPU frame-extraction actor."""

    _cpu_variant_class = VideoFrameExtractCPUActor

    def __init__(self, params: VideoExtractParams | None = None) -> None:
        resolved_params = params or VideoExtractParams()
        super().__init__(params=resolved_params)
        self._params = resolved_params


def _extract_frames_one(
    source_path: str,
    params: VideoExtractParams,
    interface: MediaInterface,
) -> List[Dict[str, Any]]:
    """Probe video, compute segment boundaries, extract one frame per segment."""
    path_file = Path(source_path)
    probe, num_splits, duration = interface.probe_media(path_file, params.split_interval, params.split_type)
    if not num_splits or not duration or num_splits <= 0:
        return []

    num_splits_int = int(num_splits)
    segment_time = math.ceil(duration / num_splits_int)
    rows: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="retriever_video_frames_") as tmpdir:
        for idx in range(num_splits_int):
            seg_start = float(idx * segment_time)
            seg_end = float(min((idx + 1) * segment_time, duration))
            seek_s = _frame_seek_seconds(params.frame_position, seg_start, seg_end)

            frame_path = Path(tmpdir) / f"{path_file.stem}_segment_{idx:04d}.{params.frame_format}"
            try:
                _ffmpeg_extract_single_frame(source_path, frame_path, seek_s, params.frame_format)
            except Exception as exc:
                logger.warning("ffmpeg failed to extract frame for %s segment %d: %s", source_path, idx, exc)
                continue

            try:
                frame_bytes = frame_path.read_bytes()
            except OSError as exc:
                logger.warning("Could not read frame file %s: %s", frame_path, exc)
                continue

            row = _frame_bytes_to_page_row(
                frame_bytes=frame_bytes,
                source_path=source_path,
                segment_index=idx,
                segment_start=seg_start,
                segment_end=seg_end,
                frame_format=params.frame_format,
                frame_position_sec=seek_s,
            )
            if row is not None:
                rows.append(row)

    return rows


def _frame_seek_seconds(position: str, seg_start: float, seg_end: float) -> float:
    if position == "first":
        return seg_start
    if position == "last":
        return max(seg_start, seg_end - 0.1)
    return (seg_start + seg_end) / 2.0


def _ffmpeg_extract_single_frame(source: str, out_path: Path, seek_s: float, fmt: str) -> None:
    """Run ffmpeg with demuxer-level seek and emit exactly one frame."""
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{seek_s:.3f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
    ]
    if fmt == "jpeg":
        args += ["-q:v", "2"]
    args += ["-f", "image2", str(out_path)]
    proc = subprocess.run(args, capture_output=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', errors='replace').strip()}")


def _frame_bytes_to_page_row(
    *,
    frame_bytes: bytes,
    source_path: str,
    segment_index: int,
    segment_start: float,
    segment_end: float,
    frame_format: str,
    frame_position_sec: float,
) -> Optional[Dict[str, Any]]:
    """Build a single image-page-schema row from one extracted frame."""
    try:
        img = Image.open(BytesIO(frame_bytes)).convert("RGB")
    except Exception as exc:
        logger.warning("PIL failed to decode frame from %s segment %d: %s", source_path, segment_index, exc)
        return None

    buf = BytesIO()
    img.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    h, w = img.height, img.width

    page_image: Dict[str, Any] = {
        "image_b64": image_b64,
        "encoding": "png",
        "orig_shape_hw": (h, w),
    }
    image_entry: Dict[str, Any] = {
        "image_b64": image_b64,
        "text": "",
        "bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0],
    }

    return {
        "path": source_path,
        "page_number": segment_index + 1,
        "source_id": None,
        "text": "",
        "page_image": page_image,
        "images": [image_entry],
        "tables": [],
        "charts": [],
        "infographics": [],
        "metadata": {
            "has_text": False,
            "needs_ocr_for_text": True,
            "dpi": _DEFAULT_DPI,
            "source_path": source_path,
            "source_video": source_path,
            "segment_index": segment_index,
            "segment_start_seconds": segment_start,
            "segment_end_seconds": segment_end,
            "frame_position_seconds": frame_position_sec,
            "frame_format": frame_format,
            "modality": "video_frame",
            "error": None,
        },
    }


def video_frames_path_to_pages_df(
    path: str,
    params: VideoExtractParams | None = None,
) -> pd.DataFrame:
    """Synchronous loader: one video path -> DataFrame of image-page rows (one per segment)."""
    if not is_media_available():
        raise RuntimeError("video_frames_path_to_pages_df requires ffmpeg.")
    params = params or VideoExtractParams()
    rows = _extract_frames_one(path, params, MediaInterface())
    if not rows:
        return pd.DataFrame(columns=_PAGE_COLUMNS)
    return pd.DataFrame(rows)
