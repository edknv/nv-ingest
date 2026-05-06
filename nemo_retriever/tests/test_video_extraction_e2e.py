# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test wiring all 5 reference-faithful video gaps.

Builds a synthetic 3-color-block video, runs:
    VideoSplitActor (scene_detection + key_frame_select + advanced_dedup)
    -> VideoFrameVLMCaptioner (mocked)
    -> VideoFrameTextDedup
    -> AudioVisualFuser (per_scene)
and asserts the resulting DataFrame has fused rows with scene_id metadata
and VLM-caption text.

Skips if ffmpeg / scenedetect / etc are unavailable.
"""

from __future__ import annotations

import base64
import io
import pathlib
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scenedetect")
pytest.importorskip("skimage.metrics")
pytest.importorskip("imagehash")
pytest.importorskip("PIL")
pytest.importorskip("ffmpeg")

from nemo_retriever.audio.media_interface import is_media_available

if not is_media_available():
    pytest.skip("ffmpeg not available", allow_module_level=True)

from PIL import Image

from nemo_retriever.params import (
    AudioChunkParams,
    AudioVisualFuseParams,
    VideoAdvancedDedupParams,
    VideoFrameParams,
    VideoFrameTextDedupParams,
    VideoFrameVLMParams,
    VideoKeyFrameSelectParams,
    VideoSceneDetectParams,
)
from nemo_retriever.video import (
    AudioVisualFuser,
    VideoFrameTextDedup,
    VideoFrameVLMCaptionerCPUActor,
    VideoSplitActor,
)


def _write_three_block_video(path: pathlib.Path, fps: int = 24, secs_per_block: int = 1) -> None:
    """Three-second 320x240 mp4: red/green/blue blocks with low-variance noise overlay."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))
    rng = np.random.default_rng(0)
    try:
        for color_bgr in [(0, 0, 200), (0, 200, 0), (200, 0, 0)]:
            base = np.full((240, 320, 3), color_bgr, dtype=np.int32)
            for _ in range(fps * secs_per_block):
                noise = rng.integers(0, 30, size=(240, 320, 3), dtype=np.int32)
                frame = np.clip(base + noise, 0, 255).astype(np.uint8)
                writer.write(frame)
    finally:
        writer.release()


def test_full_pipeline_with_all_five_gaps_enabled(tmp_path: pathlib.Path) -> None:
    """All 5 gaps wired into one run: scene + SSIM + advanced-dedup + VLM + per_scene fusion."""
    video = tmp_path / "three_blocks.mp4"
    _write_three_block_video(video)

    # Stage 1: VideoSplitActor — scene-aware extraction (gaps 1, 2, 3).
    frame_params = VideoFrameParams(
        enabled=True,
        fps=4.0,
        dedup=False,
        scene_detection=VideoSceneDetectParams(enabled=True, threshold=15.0),
        key_frame_selection=VideoKeyFrameSelectParams(enabled=True, z_threshold=2.0),
        advanced_dedup=VideoAdvancedDedupParams(enabled=True),
        frame_text_method="vlm",
        vlm=VideoFrameVLMParams(endpoint_url="https://fake.example/vlm", api_key="x"),
    )
    split_actor = VideoSplitActor(
        audio_chunk_params=AudioChunkParams(enabled=False),
        video_frame_params=frame_params,
    )
    rows = split_actor.process(pd.DataFrame([{"path": str(video)}]))
    assert (rows["_content_type"] == "video_frame").any()
    assert {row["metadata"]["scene_id"] for _, row in rows.iterrows()} == {0, 1, 2}

    # Stage 2: VideoFrameVLMCaptioner — gap 4. Mock the actual VLM call.
    def fake_caption_video_frames(batch_df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        out = batch_df.copy()
        out["text"] = [f"caption {i}" for i in range(len(out))]
        out["metadata"] = [
            {**(m or {}), "frame_text_method": "vlm"} for m in batch_df["metadata"]
        ]
        return out

    with mock.patch(
        "nemo_retriever.video.vlm_captioner._caption_video_frames",
        side_effect=fake_caption_video_frames,
    ):
        captioner = VideoFrameVLMCaptionerCPUActor(params=frame_params.vlm)
        captioned = captioner.process(rows)

    frame_rows = captioned[captioned["_content_type"] == "video_frame"].reset_index(drop=True)
    assert len(frame_rows) >= 3
    for _, row in frame_rows.iterrows():
        assert row["text"].startswith("caption")
        assert row["metadata"]["frame_text_method"] == "vlm"

    # Stage 3: VideoFrameTextDedup — collapses runs of identical OCR/VLM text.
    dedup_actor = VideoFrameTextDedup(VideoFrameTextDedupParams(enabled=True))
    deduped = dedup_actor.process(captioned)
    assert (deduped["_content_type"] == "video_frame").any()

    # Stage 4: synthesise an ASR row covering one of the scenes so per_scene
    # has both audio and visual to fuse.
    audio_row = {
        "path": str(video),
        "source_path": str(video),
        "_content_type": "audio",
        "metadata": {
            "_content_type": "audio",
            "modality": "audio_segment",
            "segment_start_seconds": 0.0,
            "segment_end_seconds": 3.0,
            "segment_count": 1,
            "segment_index": 0,
        },
        "text": "spoken commentary",
    }
    with_audio = pd.concat([deduped, pd.DataFrame([audio_row])], ignore_index=True)

    # Stage 5: AudioVisualFuser per_scene — gap 5.
    fuser = AudioVisualFuser(AudioVisualFuseParams(mode="per_scene"))
    fused = fuser.process(with_audio)
    av_rows = fused[fused["_content_type"] == "audio_visual"].reset_index(drop=True)
    assert len(av_rows) >= 1
    text = av_rows.iloc[0]["text"]
    assert "[AUDIO]" in text and "[VISUAL]" in text
    assert "spoken commentary" in text
    assert "caption" in text  # at least one frame caption made it in
    assert av_rows.iloc[0]["metadata"]["fusion_mode"] == "per_scene"
    assert "scene_id" in av_rows.iloc[0]["metadata"]
