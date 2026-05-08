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

    with mock.patch("nemo_retriever.video.transcode._encoder_works", return_value=True):
        actor = VideoTranscodeActor(VideoTranscodeParams(enabled=False))
    df = pd.DataFrame([{"path": "/tmp/foo.mp4", "bytes": b"abc"}])
    out = actor.process(df)
    assert out.iloc[0]["path"] == "/tmp/foo.mp4"
    assert out.iloc[0]["bytes"] == b"abc"


def test_skips_h264_codec(monkeypatch) -> None:
    from nemo_retriever.video.transcode import VideoTranscodeActor

    with mock.patch("nemo_retriever.video.transcode._encoder_works", return_value=True):
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

    with mock.patch("nemo_retriever.video.transcode._encoder_works", return_value=True):
        actor = VideoTranscodeActor(
            VideoTranscodeParams(enabled=True, cache_dir=str(tmp_path), encoder="h264_nvenc"),
        )
    monkeypatch.setattr("nemo_retriever.video.transcode._ffprobe_codec", lambda _: "av1")

    transcode_calls = []
    def fake_transcode(src, dest, *, encoder, preset, crf, threads=4, max_height=0):
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


def test_resolve_encoder_falls_back_when_primary_probe_fails(monkeypatch) -> None:
    from nemo_retriever.video.transcode import _resolve_encoder

    probed = []
    def fake_works(enc: str) -> bool:
        probed.append(enc)
        return enc == "libx264"

    monkeypatch.setattr("nemo_retriever.video.transcode._encoder_works", fake_works)
    chosen = _resolve_encoder(
        VideoTranscodeParams(encoder="h264_nvenc", encoder_fallback="libx264"),
    )
    assert chosen == "libx264"
    assert probed == ["h264_nvenc", "libx264"]


def test_normalize_preset_translates_nvenc_to_x264() -> None:
    from nemo_retriever.video.transcode import _normalize_preset

    # NVENC encoder: pass through.
    assert _normalize_preset("h264_nvenc", "p4") == "p4"
    # libx264 encoder: translate.
    assert _normalize_preset("libx264", "p4") == "medium"
    assert _normalize_preset("libx264", "p1") == "ultrafast"
    assert _normalize_preset("libx264", "p7") == "veryslow"
    # Non-mapped preset: pass through.
    assert _normalize_preset("libx264", "medium") == "medium"


def test_max_height_adds_scale_filter(monkeypatch) -> None:
    """When max_height>0 the ffmpeg cmd includes a scale=...:min(ih,N) filter."""
    captured = []
    def fake_run(cmd, **kw):
        captured.append(list(cmd))
        from unittest.mock import Mock
        return Mock(returncode=0, stderr="")
    monkeypatch.setattr("nemo_retriever.video.transcode.subprocess.run", fake_run)

    from nemo_retriever.video.transcode import _transcode_one
    _transcode_one(
        "/tmp/in.mp4", "/tmp/out.mp4",
        encoder="libx264", preset="ultrafast", crf=23, threads=4, max_height=720,
    )
    assert any(arg.startswith("scale=") for arg in captured[0])
    # Should mention min(ih,720)
    vf_idx = captured[0].index("-vf")
    assert "720" in captured[0][vf_idx + 1]


def test_max_height_zero_skips_scale_filter(monkeypatch) -> None:
    """When max_height=0 there's no -vf filter."""
    captured = []
    def fake_run(cmd, **kw):
        captured.append(list(cmd))
        from unittest.mock import Mock
        return Mock(returncode=0, stderr="")
    monkeypatch.setattr("nemo_retriever.video.transcode.subprocess.run", fake_run)

    from nemo_retriever.video.transcode import _transcode_one
    _transcode_one(
        "/tmp/in.mp4", "/tmp/out.mp4",
        encoder="libx264", preset="medium", crf=23, threads=4, max_height=0,
    )
    assert "-vf" not in captured[0]
