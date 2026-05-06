# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo_retriever.video.scene_detection."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scenedetect")

from nemo_retriever.video.scene_detection import (
    assign_scene_ids,
    detect_scenes,
)


def _write_three_block_video(path: pathlib.Path, fps: int = 24, secs_per_block: int = 1) -> None:
    """Write a 3-second 320x240 mp4 with hard cuts between red/green/blue blocks.

    A small amount of additive uniform noise is overlaid on each frame so the
    decoded grayscale frames carry non-zero Laplacian variance. Without this
    the blur filter in :mod:`advanced_dedup` (Laplacian variance < threshold
    = "blurry") rejects every frame in the synthetic video. Noise is bounded
    (max +29 / channel) so per-scene color shifts dominate and PySceneDetect
    still finds three shots at threshold 15.0.
    """
    rng = np.random.default_rng(0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))
    try:
        for color_bgr in [(0, 0, 200), (0, 200, 0), (200, 0, 0)]:
            base = np.full((240, 320, 3), color_bgr, dtype=np.int16)
            for _ in range(fps * secs_per_block):
                noise = rng.integers(0, 30, size=(240, 320, 3), dtype=np.int16)
                frame = np.clip(base + noise, 0, 255).astype(np.uint8)
                writer.write(frame)
    finally:
        writer.release()


def test_detect_scenes_finds_three_shots(tmp_path: pathlib.Path) -> None:
    video = tmp_path / "three_blocks.mp4"
    _write_three_block_video(video)
    scenes = detect_scenes(str(video), threshold=15.0)  # lower than the public default; synthetic solid-colour cuts are weaker than real shot changes
    assert len(scenes) == 3
    assert scenes[0][0] == 0.0
    for (a_start, a_end), (b_start, _) in zip(scenes, scenes[1:]):
        assert a_end == pytest.approx(b_start, abs=0.05)


def test_detect_scenes_zero_scenes_falls_back_to_one(tmp_path: pathlib.Path) -> None:
    """A static video produces zero ContentDetector scenes; helper returns one."""
    video = tmp_path / "static.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video), fourcc, 10, (64, 64))
    try:
        frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        for _ in range(20):
            writer.write(frame)
    finally:
        writer.release()
    scenes = detect_scenes(str(video), threshold=30.0)
    assert len(scenes) == 1
    assert scenes[0][0] == 0.0
    assert scenes[0][1] > 0.0


def test_assign_scene_ids_labels_each_frame() -> None:
    scenes = [(0.0, 1.0), (1.0, 2.5), (2.5, 4.0)]
    timestamps = [0.1, 0.5, 1.0, 1.7, 2.6, 3.9]
    assigned = assign_scene_ids(timestamps, scenes)
    assert assigned == [
        (0, 0.0, 1.0),
        (0, 0.0, 1.0),
        (1, 1.0, 2.5),
        (1, 1.0, 2.5),
        (2, 2.5, 4.0),
        (2, 2.5, 4.0),
    ]


def test_assign_scene_ids_clamps_overrun_to_last_scene() -> None:
    """Frames past the last scene's end (rounding noise) clamp to the last scene."""
    scenes = [(0.0, 1.0), (1.0, 2.0)]
    assigned = assign_scene_ids([0.0, 1.5, 2.05], scenes)
    assert [row[0] for row in assigned] == [0, 1, 1]


def test_assign_scene_ids_empty_scenes_raises() -> None:
    with pytest.raises(ValueError):
        assign_scene_ids([0.0], [])


def test_assign_scene_ids_clamps_negative_to_first_scene() -> None:
    scenes = [(0.0, 1.0), (1.0, 2.0)]
    assigned = assign_scene_ids([-0.5, -0.01], scenes)
    assert [row[0] for row in assigned] == [0, 0]


def test_is_scenedetect_available_returns_true_when_installed() -> None:
    from nemo_retriever.video.scene_detection import is_scenedetect_available

    # The test fixture imports scenedetect at module load (importorskip).
    assert is_scenedetect_available() is True


def test_detect_scenes_raises_runtimeerror_when_scenedetect_missing(monkeypatch) -> None:
    """When scenedetect can't be imported the public detect_scenes raises a clear RuntimeError."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "scenedetect":
            raise ImportError("simulated missing scenedetect")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from nemo_retriever.video import scene_detection as sd

    assert sd.is_scenedetect_available() is False

    with pytest.raises(RuntimeError, match="multimedia"):
        sd.detect_scenes("/tmp/whatever.mp4")


def test_video_split_actor_scene_aware_paths(tmp_path) -> None:
    """End-to-end: scene_detection + key_frame_selection + advanced_dedup
    keep at least one frame per scene with scene metadata stamped."""
    pytest.importorskip("ffmpeg")
    from nemo_retriever.audio.media_interface import is_media_available

    if not is_media_available():
        pytest.skip("ffmpeg not available")

    import pandas as pd

    from nemo_retriever.params import (
        AudioChunkParams,
        VideoAdvancedDedupParams,
        VideoFrameParams,
        VideoKeyFrameSelectParams,
        VideoSceneDetectParams,
    )
    from nemo_retriever.video import VideoSplitActor

    video = tmp_path / "three_blocks.mp4"
    _write_three_block_video(video, fps=24, secs_per_block=1)

    params = VideoFrameParams(
        enabled=True,
        fps=4.0,
        scene_detection=VideoSceneDetectParams(enabled=True, threshold=15.0),
        key_frame_selection=VideoKeyFrameSelectParams(enabled=True, z_threshold=2.0),
        advanced_dedup=VideoAdvancedDedupParams(enabled=True),
        dedup=False,  # legacy dhash off
    )
    audio = AudioChunkParams(enabled=False)
    actor = VideoSplitActor(audio_chunk_params=audio, video_frame_params=params)

    out = actor.process(pd.DataFrame([{"path": str(video)}]))

    frame_rows = out[out["_content_type"] == "video_frame"].reset_index(drop=True)
    assert len(frame_rows) >= 3  # at least one keeper per of three scenes

    scene_ids = {row["metadata"]["scene_id"] for _, row in frame_rows.iterrows()}
    assert scene_ids == {0, 1, 2}
    for _, row in frame_rows.iterrows():
        md = row["metadata"]
        assert md["scene_start_seconds"] >= 0.0
        assert md["scene_end_seconds"] > md["scene_start_seconds"]
