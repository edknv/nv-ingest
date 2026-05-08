# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo_retriever.video.strip_images."""

from __future__ import annotations

import pandas as pd
import pytest

from nemo_retriever.video.strip_images import VideoFrameStripImagesActor


def test_strips_image_columns_on_frame_rows() -> None:
    actor = VideoFrameStripImagesActor()
    df = pd.DataFrame(
        [
            {
                "_content_type": "video_frame",
                "image_b64": "AAAA",
                "bytes": b"raw",
                "text": "caption",
                "metadata": {"frame_timestamp_seconds": 1.0},
            }
        ]
    )
    out = actor.process(df)
    assert out.iloc[0]["image_b64"] is None
    assert out.iloc[0]["bytes"] is None
    assert out.iloc[0]["text"] == "caption"
    assert out.iloc[0]["_content_type"] == "video_frame"


def test_passes_through_when_byte_columns_absent() -> None:
    actor = VideoFrameStripImagesActor()
    df = pd.DataFrame(
        [
            {
                "_content_type": "audio",
                "text": "spoken",
                "metadata": {"segment_start_seconds": 0.0},
            }
        ]
    )
    out = actor.process(df)
    assert out.iloc[0]["text"] == "spoken"
    # No new None columns introduced.
    assert "image_b64" not in out.columns
    assert "bytes" not in out.columns


def test_empty_dataframe_passthrough() -> None:
    actor = VideoFrameStripImagesActor()
    df = pd.DataFrame()
    out = actor.process(df)
    assert out.empty


def test_non_dataframe_input_returned_as_is() -> None:
    actor = VideoFrameStripImagesActor()
    out = actor.process(None)
    assert out is None


def test_strips_multiple_rows_uniformly() -> None:
    actor = VideoFrameStripImagesActor()
    df = pd.DataFrame(
        [
            {"_content_type": "video_frame", "image_b64": "A", "bytes": b"a", "text": "c1"},
            {"_content_type": "video_frame", "image_b64": "B", "bytes": b"b", "text": "c2"},
            {"_content_type": "audio", "image_b64": None, "bytes": None, "text": "spoken"},
        ]
    )
    out = actor.process(df)
    assert out["image_b64"].tolist() == [None, None, None]
    assert out["bytes"].tolist() == [None, None, None]
    assert out["text"].tolist() == ["c1", "c2", "spoken"]
