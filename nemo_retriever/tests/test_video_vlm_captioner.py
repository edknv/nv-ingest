# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo_retriever.video.vlm_captioner."""

from __future__ import annotations

import base64
import io
from typing import Any, List
from unittest import mock

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PIL")

from PIL import Image

from nemo_retriever.params import VideoFrameVLMParams


def _b64_png(value: int = 100, size: int = 32) -> str:
    arr = np.full((size, size, 3), value, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _frame_row(idx: int) -> dict:
    return {
        "path": f"/tmp/v.mp4",
        "source_path": "/tmp/v.mp4",
        "image_b64": _b64_png(idx * 10 + 100),
        "page_number": idx,
        "_content_type": "video_frame",
        "metadata": {
            "_content_type": "video_frame",
            "modality": "video_frame",
            "frame_timestamp_seconds": float(idx),
            "segment_start_seconds": float(idx) - 0.25,
            "segment_end_seconds": float(idx) + 0.25,
            "fps": 2.0,
        },
        "text": "",
    }


def _audio_row() -> dict:
    return {
        "path": "/tmp/v.mp4",
        "source_path": "/tmp/v.mp4",
        "_content_type": "audio",
        "metadata": {
            "_content_type": "audio",
            "modality": "audio_segment",
            "segment_start_seconds": 0.0,
            "segment_end_seconds": 1.0,
        },
        "text": "the lecturer is explaining slides",
    }


def test_video_frame_vlm_captioner_writes_captions_to_text(monkeypatch) -> None:
    from nemo_retriever.video.vlm_captioner import VideoFrameVLMCaptionerCPUActor

    fake_responses = ["caption A", "caption B"]
    captured: List[List[str]] = []

    def fake_caption_images(batch_df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        captured.append(list(batch_df["image_b64"]))
        out = batch_df.copy()
        out["text"] = fake_responses[: len(batch_df)]
        out["metadata"] = [
            {**(m or {}), "frame_text_method": "vlm"} for m in batch_df["metadata"]
        ]
        return out

    params = VideoFrameVLMParams(
        endpoint_url="https://fake.example/vlm",
        api_key="dummy",
    )
    with mock.patch(
        "nemo_retriever.video.vlm_captioner._caption_video_frames", side_effect=fake_caption_images
    ):
        actor = VideoFrameVLMCaptionerCPUActor(params=params)
        df = pd.DataFrame([_audio_row(), _frame_row(0), _frame_row(1)])
        out = actor.process(df)

    assert captured  # captioner was invoked
    frame_out = out[out["_content_type"] == "video_frame"].reset_index(drop=True)
    assert list(frame_out["text"]) == ["caption A", "caption B"]
    for _, row in frame_out.iterrows():
        assert row["metadata"]["frame_text_method"] == "vlm"

    audio_out = out[out["_content_type"] == "audio"].reset_index(drop=True)
    assert audio_out.iloc[0]["text"] == "the lecturer is explaining slides"


def test_video_frame_vlm_captioner_passthrough_when_no_frames(monkeypatch) -> None:
    from nemo_retriever.video.vlm_captioner import VideoFrameVLMCaptionerCPUActor

    params = VideoFrameVLMParams(endpoint_url="https://fake.example/vlm", api_key="x")
    with mock.patch("nemo_retriever.video.vlm_captioner._caption_video_frames"):
        actor = VideoFrameVLMCaptionerCPUActor(params=params)
        df = pd.DataFrame([_audio_row()])
        out = actor.process(df)
    assert (out["_content_type"] == "audio").all()


def test_archetype_resolves_to_cpu_when_endpoint_url_set() -> None:
    from nemo_retriever.video.vlm_captioner import VideoFrameVLMCaptioner

    params_remote = VideoFrameVLMParams(endpoint_url="https://fake.example/vlm")
    assert VideoFrameVLMCaptioner.prefers_cpu_variant({"params": params_remote}) is True

    params_local = VideoFrameVLMParams(endpoint_url=None)
    assert VideoFrameVLMCaptioner.prefers_cpu_variant({"params": params_local}) is False


def test_cpu_actor_requires_endpoint() -> None:
    from nemo_retriever.video.vlm_captioner import VideoFrameVLMCaptionerCPUActor

    with pytest.raises(ValueError):
        VideoFrameVLMCaptionerCPUActor(params=VideoFrameVLMParams(endpoint_url=None))


def test_caption_video_frames_local_path_uses_caption_batch_with_correct_kwargs() -> None:
    """Regression: local-model path must call model.caption_batch with the right kwargs.

    The previous implementation used model.caption(image_b64=..., max_new_tokens=...)
    which silently raised TypeError and emitted empty captions for every frame.
    """
    from nemo_retriever.video.vlm_captioner import _caption_video_frames

    captured = {}

    class FakeModel:
        def caption_batch(self, base64_images, *, prompt, system_prompt, temperature, top_p=None, max_tokens=None):
            captured["images"] = list(base64_images)
            captured["prompt"] = prompt
            captured["temperature"] = temperature
            captured["max_tokens"] = max_tokens
            return [f"local caption {i}" for i in range(len(base64_images))]

    params = VideoFrameVLMParams(
        endpoint_url=None,
        prompt="Describe.",
        temperature=0.0,
        max_tokens=128,
    )
    df = pd.DataFrame([_frame_row(0), _frame_row(1)])
    out = _caption_video_frames(df, params=params, local_model=FakeModel())

    assert list(out["text"]) == ["local caption 0", "local caption 1"]
    assert captured["prompt"] == "Describe."
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 128
    for _, row in out.iterrows():
        assert row["metadata"]["frame_text_method"] == "vlm"
