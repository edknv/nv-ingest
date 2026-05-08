# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo_retriever.video.frame_extract."""

from __future__ import annotations

import pandas as pd
import pytest


def test_passthrough_when_no_time_chunk_rows() -> None:
    from nemo_retriever.video.frame_extract import VideoFrameExtractActor
    from nemo_retriever.params import VideoFrameParams

    actor = VideoFrameExtractActor(params=VideoFrameParams())
    df = pd.DataFrame([{"_content_type": "audio", "text": "spoken", "metadata": {}}])
    out = actor.process(df)
    assert (out["_content_type"] == "audio").all()


def test_processes_video_time_chunk_rows(monkeypatch) -> None:
    from nemo_retriever.video.frame_extract import VideoFrameExtractActor
    from nemo_retriever.params import VideoFrameParams

    def fake_extract_one(source_path, params, interface, *, start_seconds=0.0, end_seconds=None):
        return [
            {
                "path": source_path,
                "source_path": source_path,
                "_content_type": "video_frame",
                "page_number": 0,
                "image_b64": "",
                "metadata": {
                    "_content_type": "video_frame",
                    "modality": "video_frame",
                    "frame_timestamp_seconds": start_seconds + 0.5,
                    "segment_start_seconds": start_seconds,
                    "segment_end_seconds": start_seconds + 1.0,
                    "fps": 1.0,
                    "source_path": source_path,
                },
            }
        ]

    monkeypatch.setattr("nemo_retriever.video.frame_extract._extract_one", fake_extract_one)
    monkeypatch.setattr("nemo_retriever.video.frame_extract.is_media_available", lambda: True)

    actor = VideoFrameExtractActor(params=VideoFrameParams())
    df = pd.DataFrame(
        [
            {
                "path": "/tmp/v.mp4",
                "source_path": "/tmp/v.mp4",
                "_content_type": "video_time_chunk",
                "metadata": {
                    "source_path": "/tmp/v.mp4",
                    "chunk_start_seconds": 0.0,
                    "chunk_end_seconds": 600.0,
                    "fps": 1.0,
                },
            },
            {
                "path": "/tmp/v.mp4",
                "source_path": "/tmp/v.mp4",
                "_content_type": "video_time_chunk",
                "metadata": {
                    "source_path": "/tmp/v.mp4",
                    "chunk_start_seconds": 600.0,
                    "chunk_end_seconds": 1200.0,
                    "fps": 1.0,
                },
            },
            {
                "path": "/tmp/v.mp4",
                "source_path": "/tmp/v.mp4",
                "_content_type": "audio",
                "text": "spoken",
                "metadata": {"_content_type": "audio"},
            },
        ]
    )
    out = actor.process(df)
    frame_rows = out[out["_content_type"] == "video_frame"].reset_index(drop=True)
    assert len(frame_rows) == 2
    # Per-chunk extracted frame timestamps should reflect their chunk's start.
    assert frame_rows.iloc[0]["metadata"]["frame_timestamp_seconds"] == 0.5
    assert frame_rows.iloc[1]["metadata"]["frame_timestamp_seconds"] == 600.5
    # Audio passes through.
    audio_rows = out[out["_content_type"] == "audio"].reset_index(drop=True)
    assert len(audio_rows) == 1
    assert audio_rows.iloc[0]["text"] == "spoken"


def test_emit_video_time_chunks_with_known_duration(monkeypatch) -> None:
    from nemo_retriever.params import VideoFrameParams, VideoFrameTimeChunkParams
    from nemo_retriever.video.frame_actor import _emit_video_time_chunks

    monkeypatch.setattr(
        "nemo_retriever.video.scene_detection._probe_duration_seconds",
        lambda _path: 1500.0,  # 25 minutes
    )

    params = VideoFrameParams(
        fps=1.0,
        time_chunking=VideoFrameTimeChunkParams(enabled=True, chunk_seconds=600),
    )
    chunks = _emit_video_time_chunks("/tmp/long.mp4", params)
    assert len(chunks) == 3  # 0-600, 600-1200, 1200-1500
    starts = [c["metadata"]["chunk_start_seconds"] for c in chunks]
    ends = [c["metadata"]["chunk_end_seconds"] for c in chunks]
    assert starts == [0.0, 600.0, 1200.0]
    assert ends == [600.0, 1200.0, 1500.0]
    for c in chunks:
        assert c["_content_type"] == "video_time_chunk"
        assert c["metadata"]["fps"] == 1.0


def test_emit_video_time_chunks_unknown_duration_single_chunk(monkeypatch) -> None:
    from nemo_retriever.params import VideoFrameParams, VideoFrameTimeChunkParams
    from nemo_retriever.video.frame_actor import _emit_video_time_chunks

    monkeypatch.setattr(
        "nemo_retriever.video.scene_detection._probe_duration_seconds",
        lambda _path: 0.0,  # unknown
    )

    params = VideoFrameParams(
        fps=1.0,
        time_chunking=VideoFrameTimeChunkParams(enabled=True, chunk_seconds=600),
    )
    chunks = _emit_video_time_chunks("/tmp/v.mp4", params)
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["chunk_end_seconds"] == 0.0  # 0 = "until end"


def test_chunk_index_makes_scene_ids_globally_unique(monkeypatch) -> None:
    """Two chunks of the same source must produce non-overlapping scene_ids
    so AudioVisualFuser's per_scene groupby doesn't merge across chunks."""
    from nemo_retriever.video.frame_extract import VideoFrameExtractActor
    from nemo_retriever.params import VideoFrameParams

    def fake_extract_one(source_path, params, interface, *, start_seconds=0.0, end_seconds=None):
        # Each chunk produces 3 local scenes 0, 1, 2.
        return [
            {
                "path": source_path,
                "source_path": source_path,
                "_content_type": "video_frame",
                "page_number": idx,
                "image_b64": "",
                "metadata": {
                    "_content_type": "video_frame",
                    "modality": "video_frame",
                    "frame_timestamp_seconds": start_seconds + idx + 0.5,
                    "segment_start_seconds": start_seconds + idx,
                    "segment_end_seconds": start_seconds + idx + 1.0,
                    "fps": 1.0,
                    "scene_id": idx,  # local scene_id 0, 1, 2
                    "scene_start_seconds": start_seconds + idx,
                    "scene_end_seconds": start_seconds + idx + 1.0,
                    "source_path": source_path,
                },
            }
            for idx in range(3)
        ]

    monkeypatch.setattr("nemo_retriever.video.frame_extract._extract_one", fake_extract_one)
    monkeypatch.setattr("nemo_retriever.video.frame_extract.is_media_available", lambda: True)

    actor = VideoFrameExtractActor(params=VideoFrameParams())
    df = pd.DataFrame(
        [
            {
                "path": "/tmp/v.mp4",
                "source_path": "/tmp/v.mp4",
                "_content_type": "video_time_chunk",
                "metadata": {
                    "source_path": "/tmp/v.mp4",
                    "chunk_index": 0,
                    "chunk_start_seconds": 0.0,
                    "chunk_end_seconds": 600.0,
                    "fps": 1.0,
                },
            },
            {
                "path": "/tmp/v.mp4",
                "source_path": "/tmp/v.mp4",
                "_content_type": "video_time_chunk",
                "metadata": {
                    "source_path": "/tmp/v.mp4",
                    "chunk_index": 1,
                    "chunk_start_seconds": 600.0,
                    "chunk_end_seconds": 1200.0,
                    "fps": 1.0,
                },
            },
        ]
    )
    out = actor.process(df)
    frame_rows = out[out["_content_type"] == "video_frame"].reset_index(drop=True)
    assert len(frame_rows) == 6
    scene_ids = sorted({int(row["metadata"]["scene_id"]) for _, row in frame_rows.iterrows()})
    # Chunk 0: scene_ids 0, 1, 2  (chunk_idx=0 → no rewrite)
    # Chunk 1: scene_ids 10000, 10001, 10002  (chunk_idx=1 * 10000 + local)
    assert scene_ids == [0, 1, 2, 10000, 10001, 10002]


def test_emit_video_time_chunks_stamps_chunk_index(monkeypatch) -> None:
    from nemo_retriever.params import VideoFrameParams, VideoFrameTimeChunkParams
    from nemo_retriever.video.frame_actor import _emit_video_time_chunks

    monkeypatch.setattr(
        "nemo_retriever.video.scene_detection._probe_duration_seconds",
        lambda _path: 1500.0,
    )
    params = VideoFrameParams(
        fps=1.0,
        time_chunking=VideoFrameTimeChunkParams(enabled=True, chunk_seconds=600),
    )
    chunks = _emit_video_time_chunks("/tmp/long.mp4", params)
    indices = [c["metadata"]["chunk_index"] for c in chunks]
    assert indices == [0, 1, 2]
