# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the new per_scene / per_sentence modes on AudioVisualFuser."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import pytest

from nemo_retriever.params import AudioVisualFuseParams
from nemo_retriever.video import AudioVisualFuser


def _audio(start: float, end: float, text: str, *, segment_count: Optional[int] = None) -> dict:
    md = {
        "_content_type": "audio",
        "modality": "audio_segment",
        "segment_start_seconds": start,
        "segment_end_seconds": end,
    }
    if segment_count is not None:
        md["segment_count"] = segment_count
        md["segment_index"] = 0
    return {
        "path": "/v.mp4",
        "source_path": "/v.mp4",
        "_content_type": "audio",
        "metadata": md,
        "text": text,
    }


def _frame(start: float, end: float, text: str, scene_id: Optional[int] = None) -> dict:
    md = {
        "_content_type": "video_frame",
        "modality": "video_frame",
        "segment_start_seconds": start,
        "segment_end_seconds": end,
        "fps": 1.0,
    }
    if scene_id is not None:
        md["scene_id"] = scene_id
        md["scene_start_seconds"] = float(int(start))
        md["scene_end_seconds"] = float(int(start) + 5)
    return {
        "path": "/v.mp4",
        "source_path": "/v.mp4",
        "_content_type": "video_frame",
        "metadata": md,
        "text": text,
    }


def test_per_scene_mode_emits_one_fused_row_per_scene_with_widened_window() -> None:
    fuser = AudioVisualFuser(AudioVisualFuseParams(mode="per_scene"))
    df = pd.DataFrame(
        [
            _audio(0.5, 1.5, "intro", segment_count=2),
            _audio(2.0, 3.0, "main point", segment_count=2),
            _frame(0.4, 1.4, "Title slide", scene_id=0),
            _frame(2.2, 2.7, "Bullet points", scene_id=0),
        ]
    )
    out = fuser.process(df)
    fused = out[out["_content_type"] == "audio_visual"].reset_index(drop=True)
    assert len(fused) == 1
    row = fused.iloc[0]
    assert "[AUDIO]" in row["text"] and "intro" in row["text"] and "main point" in row["text"]
    assert "Title slide" in row["text"] and "Bullet points" in row["text"]
    assert row["metadata"]["fusion_mode"] == "per_scene"
    assert row["metadata"]["scene_id"] == 0


def test_per_scene_mode_requires_scene_id_metadata() -> None:
    fuser = AudioVisualFuser(AudioVisualFuseParams(mode="per_scene"))
    df = pd.DataFrame(
        [
            _audio(0.0, 1.0, "no scenes here", segment_count=1),
            _frame(0.1, 0.9, "frame text"),  # no scene_id
        ]
    )
    with pytest.raises(ValueError, match="per_scene"):
        fuser.process(df)


def test_per_utterance_mode_unchanged_behaviour() -> None:
    """Regression: default mode emits one fused row per utterance, single best frame."""
    fuser = AudioVisualFuser(AudioVisualFuseParams(mode="per_utterance"))
    df = pd.DataFrame(
        [
            _audio(0.0, 2.0, "presenter speaking"),
            _frame(0.5, 1.0, "Slide: introduction"),
            _frame(1.2, 1.8, "Slide: agenda"),
        ]
    )
    out = fuser.process(df)
    fused = out[out["_content_type"] == "audio_visual"].reset_index(drop=True)
    assert len(fused) == 1
    text = fused.iloc[0]["text"]
    assert text.startswith("[AUDIO]") and "[VISUAL]" in text


def test_per_sentence_mode_includes_all_overlapping_frames() -> None:
    fuser = AudioVisualFuser(
        AudioVisualFuseParams(
            mode="per_sentence",
            per_sentence_per_frame_max_chars=20,
            per_sentence_total_visual_max_chars=200,
        )
    )
    df = pd.DataFrame(
        [
            _audio(0.0, 3.0, "sentence one", segment_count=1),
            _frame(0.2, 1.0, "Slide A text"),
            _frame(1.0, 2.0, "Slide B text"),
            _frame(2.0, 3.0, "Slide C text"),
        ]
    )
    out = fuser.process(df)
    fused = out[out["_content_type"] == "audio_visual"].reset_index(drop=True)
    assert len(fused) == 1
    text = fused.iloc[0]["text"]
    assert "Slide A" in text and "Slide B" in text and "Slide C" in text
    assert fused.iloc[0]["metadata"]["fusion_mode"] == "per_sentence"


def test_per_sentence_mode_requires_segment_audio_metadata() -> None:
    fuser = AudioVisualFuser(AudioVisualFuseParams(mode="per_sentence"))
    df = pd.DataFrame(
        [
            _audio(0.0, 30.0, "long unsegmented chunk"),  # no segment_count
            _frame(1.0, 2.0, "Slide"),
        ]
    )
    with pytest.raises(ValueError, match="segment_audio"):
        fuser.process(df)


def test_per_sentence_per_frame_cap_truncates_long_visual_text() -> None:
    """Each frame's text is truncated to per_sentence_per_frame_max_chars before joining."""
    fuser = AudioVisualFuser(
        AudioVisualFuseParams(
            mode="per_sentence",
            per_sentence_per_frame_max_chars=10,
            per_sentence_total_visual_max_chars=200,
        )
    )
    long_text = "a" * 50  # 50 chars
    df = pd.DataFrame(
        [
            _audio(0.0, 3.0, "speaks", segment_count=1),
            _frame(0.5, 2.5, long_text),
        ]
    )
    out = fuser.process(df)
    fused = out[out["_content_type"] == "audio_visual"].reset_index(drop=True)
    assert len(fused) == 1
    visual_part = fused.iloc[0]["text"].split("[VISUAL]", 1)[1].strip()
    # The 50-char "aaaa..." input should have been clipped to 10 chars.
    assert len(visual_part) == 10
    assert visual_part == "a" * 10


def test_per_sentence_mode_validates_every_audio_row_not_just_first() -> None:
    """Mixed batch where some sources are segmented and some are not must raise.

    Prior implementation only checked .iloc[0] which silently accepted
    later rows that lacked segment_count.
    """
    fuser = AudioVisualFuser(AudioVisualFuseParams(mode="per_sentence"))
    df = pd.DataFrame(
        [
            _audio(0.0, 1.0, "first sentence", segment_count=1),
            _audio(2.0, 30.0, "unsegmented chunk"),  # no segment_count
            _frame(0.5, 1.0, "Slide A"),
            _frame(5.0, 6.0, "Slide B"),
        ]
    )
    with pytest.raises(ValueError, match="segment_audio"):
        fuser.process(df)


def test_per_scene_mode_keeps_scene_zero_start_at_zero() -> None:
    """Regression for scene_start_seconds=0.0 falsy bug: an audio row at
    [0.0, 0.3] inside scene 0 (whose window starts at 0.0) must fuse, not
    be silently dropped by the truthy-evaluation of 0.0."""
    fuser = AudioVisualFuser(AudioVisualFuseParams(mode="per_scene"))
    df = pd.DataFrame(
        [
            _audio(0.0, 0.3, "first words", segment_count=2),
            _audio(0.5, 1.0, "rest", segment_count=2),
            _frame(0.4, 1.0, "Title slide", scene_id=0),
        ]
    )
    # Manually set scene_start_seconds=0.0 to exercise the falsy edge case.
    df.iloc[2]["metadata"]["scene_start_seconds"] = 0.0
    df.iloc[2]["metadata"]["scene_end_seconds"] = 5.0
    out = fuser.process(df)
    fused = out[out["_content_type"] == "audio_visual"].reset_index(drop=True)
    assert len(fused) == 1
    text = fused.iloc[0]["text"]
    assert "first words" in text  # the audio at [0.0, 0.3] was NOT dropped
    assert "rest" in text
    assert "Title slide" in text
