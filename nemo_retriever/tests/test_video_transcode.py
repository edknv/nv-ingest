# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo_retriever.video.transcode."""
from __future__ import annotations

from unittest import mock

import pandas as pd

from nemo_retriever.params import VideoTranscodeParams


def test_disabled_passthrough() -> None:
    from nemo_retriever.video.transcode import VideoTranscodeActor

    with mock.patch("nemo_retriever.video.transcode._encoder_available", return_value=True):
        actor = VideoTranscodeActor(VideoTranscodeParams(enabled=False))
    df = pd.DataFrame([{"path": "/tmp/foo.mp4", "bytes": b"abc"}])
    out = actor.process(df)
    assert out.iloc[0]["path"] == "/tmp/foo.mp4"
    assert out.iloc[0]["bytes"] == b"abc"


def test_skips_h264_codec(monkeypatch) -> None:
    from nemo_retriever.video.transcode import VideoTranscodeActor

    with mock.patch("nemo_retriever.video.transcode._encoder_available", return_value=True):
        actor = VideoTranscodeActor(VideoTranscodeParams(enabled=True))
    monkeypatch.setattr("nemo_retriever.video.transcode._ffprobe_codec", lambda _: "h264")
    transcode_called = []
    monkeypatch.setattr(
        "nemo_retriever.video.transcode._transcode_one",
        lambda *a, **kw: transcode_called.append(True),
    )
    df = pd.DataFrame([{"path": "/tmp/already_h264.mp4"}])
    out = actor.process(df)
    assert out.iloc[0]["path"] == "/tmp/already_h264.mp4"
    assert not transcode_called


def test_av1_triggers_transcode_then_caches(tmp_path, monkeypatch) -> None:
    from nemo_retriever.video.transcode import VideoTranscodeActor

    with mock.patch("nemo_retriever.video.transcode._encoder_available", return_value=True):
        actor = VideoTranscodeActor(
            VideoTranscodeParams(enabled=True, cache_dir=str(tmp_path), encoder="h264_nvenc"),
        )
    monkeypatch.setattr("nemo_retriever.video.transcode._ffprobe_codec", lambda _: "av1")

    transcode_calls = []
    def fake_transcode(src, dest, *, encoder, preset, crf):
        transcode_calls.append((src, dest, encoder))
        # Simulate writing the cached file.
        from pathlib import Path
        Path(dest).touch()
    monkeypatch.setattr("nemo_retriever.video.transcode._transcode_one", fake_transcode)

    df = pd.DataFrame([{"path": "/some/place/foo.mp4"}])
    out = actor.process(df)
    assert len(transcode_calls) == 1
    assert out.iloc[0]["path"].endswith(".transcoded.h264.mp4")

    # Run again — should NOT call _transcode_one (cached).
    transcode_calls.clear()
    out2 = actor.process(df)
    assert len(transcode_calls) == 0
    assert out2.iloc[0]["path"] == out.iloc[0]["path"]
