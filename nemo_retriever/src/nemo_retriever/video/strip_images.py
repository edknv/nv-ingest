# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VideoFrameStripImagesActor: drop image_b64/bytes from frame rows.

Runs after the captioner/OCR has produced ``text``.  By that point the
image bytes are dead weight — VideoFrameTextDedup, AudioVisualFuser, and
the text-modality embedder all consume only ``text`` + metadata.  Stripping
the byte-heavy columns here is the streaming-mode memory-pressure relief
that ``StoreOperator`` used to provide as a side effect of writing images
to disk; with single-run pipelines there is no reason to pay the disk
write.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.video import _content_types as _CT

logger = logging.getLogger(__name__)


_BYTE_HEAVY_COLUMNS = ("image_b64", "bytes")


@designer_component(
    name="Video Frame Strip Images",
    category="Video",
    compute="cpu",
    description="Drops image_b64/bytes columns from video_frame rows after the captioner/OCR has produced text.",
)
class VideoFrameStripImagesActor(AbstractOperator, CPUOperator):
    """Null out image_b64 and bytes columns to release object-store memory.

    Audio rows (and any other content type) pass through unchanged.
    The strip happens by setting columns to ``None`` rather than dropping
    them, so the DataFrame schema stays consistent across batches and
    downstream stages don't need to special-case missing columns.
    """

    def __init__(self) -> None:
        super().__init__()

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, batch_df: Any, **kwargs: Any) -> Any:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return batch_df
        cols_to_strip = [c for c in _BYTE_HEAVY_COLUMNS if c in batch_df.columns]
        if not cols_to_strip:
            return batch_df
        # Only strip frame rows. Audio chunk rows carry their MP4 bytes in
        # the `bytes` column for downstream ASR (their `path` points at a
        # tmpdir that was torn down on context-manager exit). Without
        # `_content_type` set, treat the whole batch as legacy frame rows
        # for back-compat with callers that never tagged content type.
        if "_content_type" not in batch_df.columns:
            out = batch_df.copy()
            for col in cols_to_strip:
                out[col] = None
            return out
        is_frame = batch_df["_content_type"].astype(str) == _CT.VIDEO_FRAME
        if not bool(is_frame.any()):
            return batch_df
        out = batch_df.copy()
        for col in cols_to_strip:
            out.loc[is_frame, col] = None
        return out

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
