# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video pipeline: per-segment frame extraction feeding the image detection pipeline."""

from __future__ import annotations

from nemo_retriever.params import VideoExtractParams
from nemo_retriever.video.frame_actor import (
    VideoFrameExtractActor,
    VideoFrameExtractCPUActor,
    video_frames_path_to_pages_df,
)
from nemo_retriever.video.frame_ocr import (
    VideoFrameOCRActor,
    VideoFrameOCRCPUActor,
)
from nemo_retriever.video.split import (
    VideoSplitActor,
    VideoSplitCPUActor,
)

__all__ = [
    "VideoExtractParams",
    "VideoFrameExtractActor",
    "VideoFrameExtractCPUActor",
    "VideoFrameOCRActor",
    "VideoFrameOCRCPUActor",
    "VideoSplitActor",
    "VideoSplitCPUActor",
    "video_frames_path_to_pages_df",
]
