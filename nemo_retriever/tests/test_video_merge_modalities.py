# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd

from nemo_retriever.video.merge_modalities import merge_video_frame_audio_rows


_FAKE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def _frame_row(
    start: float,
    end: float,
    ocr: str,
    *,
    idx: int = 0,
    source: str = "/v.mp4",
    image_b64: str = _FAKE_B64,
) -> dict:
    return {
        "path": source,
        "page_number": idx + 1,
        "text": ocr,
        "page_image": {"image_b64": image_b64, "encoding": "png", "orig_shape_hw": (1, 1)},
        "images": [{"image_b64": image_b64, "text": "", "bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0]}],
        "metadata": {
            "modality": "video_frame",
            "_content_type": "image",
            "source_path": source,
            "segment_index": idx,
            "segment_start_seconds": start,
            "segment_end_seconds": end,
            "frame_position_seconds": (start + end) / 2.0,
        },
        "_content_type": "image",
    }


def _audio_row(start: float, end: float, transcript: str, *, idx: int = 0, source: str = "/v.mp4") -> dict:
    return {
        "path": f"/chunk_{idx}.wav",
        "page_number": idx,
        "text": transcript,
        "metadata": {
            "modality": "audio_segment",
            "_content_type": "audio",
            "source_path": source,
            "chunk_index": idx,
            "duration": end - start,
            "segment_start_seconds": start,
            "segment_end_seconds": end,
        },
        "_content_type": "audio",
    }


def test_audio_aligned_with_single_frame_produces_one_merged_row() -> None:
    df = pd.DataFrame(
        [
            _frame_row(0.0, 14.0, "Title slide", idx=0),
            _audio_row(0.0, 14.0, "welcome to the talk", idx=0),
        ]
    )
    out = merge_video_frame_audio_rows(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["metadata"]["modality"] == "video_segment"
    assert row["metadata"]["_content_type"] == "video_segment"
    assert row["text"] == "welcome to the talk\nTitle slide"
    assert row["metadata"]["audio_text"] == "welcome to the talk"
    assert row["metadata"]["ocr_text"] == "Title slide"
    assert row["metadata"]["segment_start_seconds"] == 0.0
    assert row["metadata"]["segment_end_seconds"] == 14.0
    # Embedding stays text-only so the doc encoder matches the query encoder
    # the harness uses; visual signal is preserved as OCR text in `text`.
    assert "_embed_modality" not in row or pd.isna(row.get("_embed_modality"))
    assert "_image_b64" not in row or pd.isna(row.get("_image_b64"))


def test_many_short_audio_rows_share_one_long_frame() -> None:
    """Per-utterance audio + a 120s frame: each utterance keeps its span."""
    df = pd.DataFrame(
        [
            _frame_row(0.0, 120.0, "Frame OCR", idx=0),
            _audio_row(2.0, 5.0, "first utterance", idx=0),
            _audio_row(30.0, 35.0, "second utterance", idx=1),
            _audio_row(80.0, 90.0, "third utterance", idx=2),
        ]
    )
    out = merge_video_frame_audio_rows(df)
    assert len(out) == 3
    spans = [(r["metadata"]["segment_start_seconds"], r["metadata"]["segment_end_seconds"]) for _, r in out.iterrows()]
    assert spans == [(2.0, 5.0), (30.0, 35.0), (80.0, 90.0)]
    for _, row in out.iterrows():
        assert row["metadata"]["modality"] == "video_segment"
        assert row["metadata"]["ocr_text"] == "Frame OCR"
        assert "Frame OCR" in row["text"]


def test_audio_with_no_overlapping_frame_stays_audio() -> None:
    df = pd.DataFrame(
        [
            _frame_row(0.0, 14.0, "frame zero", idx=0),
            _audio_row(20.0, 34.0, "later utterance", idx=1),
        ]
    )
    out = merge_video_frame_audio_rows(df)
    modalities = sorted(out["metadata"].apply(lambda m: m["modality"]).tolist())
    # Frame survives as standalone (no audio overlap), audio survives as standalone.
    assert modalities == ["audio_segment", "video_frame"]


def test_unclaimed_frame_emitted_as_standalone() -> None:
    df = pd.DataFrame(
        [
            _frame_row(0.0, 14.0, "claimed frame", idx=0),
            _frame_row(14.0, 28.0, "silent frame", idx=1),
            _audio_row(0.0, 14.0, "speech", idx=0),
        ]
    )
    out = merge_video_frame_audio_rows(df)
    by_modality = out["metadata"].apply(lambda m: m["modality"]).value_counts().to_dict()
    assert by_modality == {"video_segment": 1, "video_frame": 1}
    silent = out[out["metadata"].apply(lambda m: m["modality"] == "video_frame")].iloc[0]
    assert silent["text"] == "silent frame"


def test_audio_midpoint_inside_first_frame_picks_first_frame() -> None:
    """Audio [10..20], midpoint 15. Frame A covers [0..16], frame B [16..30]."""
    df = pd.DataFrame(
        [
            _frame_row(0.0, 16.0, "FRAME-A", idx=0),
            _frame_row(16.0, 30.0, "FRAME-B", idx=1),
            _audio_row(10.0, 20.0, "voice", idx=0),
        ]
    )
    out = merge_video_frame_audio_rows(df)
    # Audio midpoint = 15 is inside frame A's [0..16] span.
    merged = out[out["metadata"].apply(lambda m: m["modality"] == "video_segment")].iloc[0]
    assert merged["metadata"]["ocr_text"] == "FRAME-A"
    # Frame B never matched any audio; it's emitted standalone.
    standalone_frames = out[out["metadata"].apply(lambda m: m["modality"] == "video_frame")]
    assert len(standalone_frames) == 1
    assert standalone_frames.iloc[0]["text"] == "FRAME-B"


def test_merge_is_per_source() -> None:
    df = pd.DataFrame(
        [
            _frame_row(0.0, 14.0, "A frame", idx=0, source="/a.mp4"),
            _frame_row(0.0, 14.0, "B frame", idx=0, source="/b.mp4"),
            _audio_row(0.0, 14.0, "A audio", idx=0, source="/a.mp4"),
            _audio_row(0.0, 14.0, "B audio", idx=0, source="/b.mp4"),
        ]
    )
    out = merge_video_frame_audio_rows(df)
    assert len(out) == 2
    by_source = {row["metadata"]["source_path"]: row for _, row in out.iterrows()}
    assert by_source["/a.mp4"]["text"] == "A audio\nA frame"
    assert by_source["/b.mp4"]["text"] == "B audio\nB frame"


def test_no_audio_passes_through_unchanged() -> None:
    df = pd.DataFrame(
        [
            _frame_row(0.0, 14.0, "frame only", idx=0),
        ]
    )
    out = merge_video_frame_audio_rows(df)
    assert len(out) == 1
    assert out.iloc[0]["metadata"]["modality"] == "video_frame"
    assert out.iloc[0]["text"] == "frame only"


def test_empty_dataframe_returns_empty() -> None:
    df = pd.DataFrame(columns=["path", "text", "metadata"])
    out = merge_video_frame_audio_rows(df)
    assert len(out) == 0
