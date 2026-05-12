# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_retriever.video.caption_actor."""

from __future__ import annotations

import base64
import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from PIL import Image

from nemo_retriever.params import CaptionParams
from nemo_retriever.video.caption_actor import (
    VideoFrameCaptionActor,
    VideoFrameCaptionCPUActor,
    VideoFrameCaptionGPUActor,
)


def _png_b64(color: tuple[int, int, int]) -> str:
    """Return a base64-encoded 32x32 solid-color PNG (real image for caption helpers)."""
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_frame_df(image_b64s: list[str]) -> pd.DataFrame:
    rows = []
    for i, b64 in enumerate(image_b64s):
        rows.append(
            {
                "path": f"/tmp/frame_{i}.png",
                "source_path": "/tmp/v.mp4",
                "image_b64": b64,
                "page_number": i,
                "metadata": {
                    "source_path": "/tmp/v.mp4",
                    "frame_index": i,
                    "fps": 1.0,
                    "frame_timestamp_seconds": float(i) + 0.5,
                    "segment_start_seconds": float(i),
                    "segment_end_seconds": float(i) + 1.0,
                    "_content_type": "video_frame",
                    "modality": "video_frame",
                },
                "bytes": b"fake",
            }
        )
    return pd.DataFrame(rows)


def test_archetype_prefers_cpu_when_endpoint_url_set() -> None:
    remote = CaptionParams(endpoint_url="https://example/vlm")
    local = CaptionParams()
    assert VideoFrameCaptionActor.prefers_cpu_variant({"params": remote}) is True
    assert VideoFrameCaptionActor.prefers_cpu_variant({"params": local}) is False
    # Missing params dict / kwargs falls back to GPU.
    assert VideoFrameCaptionActor.prefers_cpu_variant({}) is False
    assert VideoFrameCaptionActor.prefers_cpu_variant(None) is False


def test_cpu_actor_calls_remote_batched_with_b64_list() -> None:
    red = _png_b64((255, 0, 0))
    blue = _png_b64((0, 0, 255))
    df = _make_frame_df([red, blue, ""])  # empty b64 row should be dropped (empty caption)

    nim_client = MagicMock()
    nim_client.infer = MagicMock(return_value=["hello world", "frame two"])

    params = CaptionParams(endpoint_url="https://example/vlm", api_key="k", batch_size=4)
    with patch("nemo_retriever.video.caption_actor._create_remote_client", return_value=nim_client):
        actor = VideoFrameCaptionCPUActor(params)
    out = actor.run(df)

    assert isinstance(out, pd.DataFrame)
    assert len(out) == 2  # empty-b64 row dropped
    assert out["text"].tolist() == ["hello world", "frame two"]
    nim_client.infer.assert_called_once()
    sent_payload = nim_client.infer.call_args.args[0]
    # Two valid frames were forwarded to the VLM; the empty-b64 row was dropped.
    assert len(sent_payload["base64_images"]) == 2


def test_cpu_actor_requires_endpoint_url() -> None:
    with pytest.raises(ValueError, match="endpoint_url"):
        VideoFrameCaptionCPUActor(CaptionParams())


def test_gpu_actor_rejects_endpoint_url() -> None:
    with pytest.raises(ValueError, match="remote endpoint"):
        VideoFrameCaptionGPUActor(CaptionParams(endpoint_url="https://example/vlm"))


def test_gpu_actor_invokes_local_model() -> None:
    red = _png_b64((255, 0, 0))
    blue = _png_b64((0, 0, 255))
    df = _make_frame_df([red, blue])

    fake_model = MagicMock()
    fake_model.caption_batch = MagicMock(return_value=["alpha", "beta"])

    actor = VideoFrameCaptionGPUActor(CaptionParams())
    actor._model = fake_model

    out = actor.run(df)
    assert isinstance(out, pd.DataFrame)
    assert out["text"].tolist() == ["alpha", "beta"]
    assert fake_model.caption_batch.call_count == 1
    sent_b64s = fake_model.caption_batch.call_args.args[0]
    assert sent_b64s == [red, blue]


def test_gpu_actor_drops_empty_caption_rows() -> None:
    red = _png_b64((255, 0, 0))
    blue = _png_b64((0, 0, 255))
    df = _make_frame_df([red, blue])

    fake_model = MagicMock()
    fake_model.caption_batch = MagicMock(return_value=["alpha", ""])

    actor = VideoFrameCaptionGPUActor(CaptionParams())
    actor._model = fake_model

    out = actor.run(df)
    assert len(out) == 1
    assert out["text"].iloc[0] == "alpha"
    assert out["page_number"].iloc[0] == 0
