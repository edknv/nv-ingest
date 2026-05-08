# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VideoFrameExtractActor: per-time-chunk frame extraction.

Consumes ``video_time_chunk`` rows (descriptors carrying chunk start/end
timestamps) and emits ``video_frame`` rows.  This is where the actual
ffmpeg decode + per-chunk scene detection + SSIM key-frame select +
advanced dedup happen.  The split between :class:`VideoSplitActor`
(emits chunk descriptors) and this actor lets Ray Data parallelise long
videos across multiple actors.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from nemo_retriever.audio.media_interface import MediaInterface, is_media_available
from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.ocr.shared import concat_with_passthrough
from nemo_retriever.params import VideoFrameParams
from nemo_retriever.video import _content_types as _CT
from nemo_retriever.video.frame_actor import _extract_one

logger = logging.getLogger(__name__)


def _rewrite_scene_id_for_chunk(rows: list, chunk_idx: int) -> list:
    """Make ``metadata.scene_id`` globally unique by encoding the chunk index.

    With per-video time chunking, each chunk's frames carry chunk-local
    scene_ids (0..N).  AudioVisualFuser's per_scene mode groups by
    ``(source_path, scene_id)``, which would erroneously merge scenes
    from different chunks of the same source.  Encoding the chunk index
    in the scene_id makes the groupby key globally unique.
    """
    if chunk_idx <= 0:
        return rows
    for row in rows:
        md = row.get("metadata")
        if not isinstance(md, dict):
            continue
        local_id = md.get("scene_id")
        if local_id is None:
            continue
        try:
            md["scene_id"] = chunk_idx * 10000 + int(local_id)
        except (TypeError, ValueError):
            pass
    return rows


@designer_component(
    name="Video Frame Extract",
    category="Video",
    compute="cpu",
    description="Per-time-chunk frame extraction; consumes video_time_chunk descriptors.",
    category_color="#ff6b6b",
)
class VideoFrameExtractActor(AbstractOperator, CPUOperator):
    """Extract frames per ``video_time_chunk`` row.

    Audio rows and any other content type pass through unchanged so this
    actor is safe to wire after :class:`~nemo_retriever.video.VideoSplitActor`
    on a graph that emits both audio chunks and time-chunk descriptors.
    """

    def __init__(self, params: VideoFrameParams | None = None) -> None:
        super().__init__(params=params)
        if not is_media_available():
            raise RuntimeError("VideoFrameExtractActor requires ffmpeg.")
        self._params = params or VideoFrameParams()
        self._interface = MediaInterface()

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, batch_df: Any, **kwargs: Any) -> Any:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return batch_df
        if "_content_type" not in batch_df.columns:
            return batch_df

        is_chunk = batch_df["_content_type"].astype(str) == _CT.VIDEO_TIME_CHUNK
        chunks = batch_df[is_chunk].reset_index(drop=True)
        passthrough = batch_df[~is_chunk].reset_index(drop=True)

        if chunks.empty:
            return passthrough

        out_rows: List[Dict[str, Any]] = []
        for _, row in chunks.iterrows():
            md = row.get("metadata") or {}
            source_path = str(
                md.get("source_path") or row.get("source_path") or row.get("path") or ""
            )
            if not source_path:
                continue
            chunk_idx = int(md.get("chunk_index") or 0)
            start = float(md.get("chunk_start_seconds") or 0.0)
            end = float(md.get("chunk_end_seconds") or 0.0)
            try:
                # ``_extract_one`` forwards the time bounds to both the
                # legacy and scene-aware code paths and shifts the
                # extracted frames' timestamps back into the source
                # video's absolute timeline.
                frame_rows = _extract_one(
                    source_path,
                    self._params,
                    self._interface,
                    start_seconds=start,
                    end_seconds=end if end > 0.0 else None,
                )
            except Exception:
                logger.exception(
                    "Per-chunk frame extraction failed for %s [%.1f, %.1f]",
                    source_path,
                    start,
                    end,
                )
                continue
            # Rewrite scene_ids so per_scene fusion doesn't merge across chunks.
            _rewrite_scene_id_for_chunk(frame_rows, chunk_idx)
            out_rows.extend(frame_rows)

        if not out_rows:
            return passthrough
        frames_df = pd.DataFrame(out_rows)
        return concat_with_passthrough(frames_df, passthrough)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
