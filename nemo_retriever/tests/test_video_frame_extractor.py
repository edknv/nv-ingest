# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for VideoFrameExtractActor and video_frames_path_to_pages_df."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from nemo_retriever.audio.media_interface import is_media_available
from nemo_retriever.params import VideoExtractParams
from nemo_retriever.video.frame_actor import (
    VideoFrameExtractActor,
    video_frames_path_to_pages_df,
)

_PAGE_COLUMNS = {
    "path",
    "page_number",
    "source_id",
    "text",
    "page_image",
    "images",
    "tables",
    "charts",
    "infographics",
    "metadata",
}


def _ffmpeg_available() -> bool:
    return is_media_available() and shutil.which("ffmpeg") is not None


def _make_tiny_mp4(out_path: Path, duration_sec: int = 4, color: str = "red") -> None:
    """Generate a short MP4 via ffmpeg's lavfi color source."""
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=320x240:d={duration_sec}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not available")
def test_video_frames_path_to_pages_df_produces_image_schema(tmp_path: Path):
    video = tmp_path / "tiny.mp4"
    _make_tiny_mp4(video, duration_sec=4)

    params = VideoExtractParams(split_type="time", split_interval=2, frame_format="jpeg")
    df = video_frames_path_to_pages_df(str(video), params=params)

    assert len(df) == 2
    assert _PAGE_COLUMNS.issubset(set(df.columns))

    first = df.iloc[0]
    assert first["page_image"]["encoding"] == "png"
    assert isinstance(first["page_image"]["image_b64"], str) and len(first["page_image"]["image_b64"]) > 100
    assert first["page_image"]["orig_shape_hw"] == (240, 320)
    assert first["page_number"] == 1
    assert first["path"] == str(video)

    meta = first["metadata"]
    assert meta["source_video"] == str(video)
    assert meta["source_path"] == str(video)
    assert meta["segment_index"] == 0
    # recall.core._hit_to_audio_segment_key reads these exact keys.
    assert meta["segment_start"] == pytest.approx(0.0)
    assert meta["segment_end"] == pytest.approx(2.0)
    assert meta["modality"] == "video_frame"
    assert meta["frame_format"] == "jpeg"
    assert meta["needs_ocr_for_text"] is True
    assert meta["error"] is None

    # Frames must NOT carry page_elements_v3 — video OCR skips that stage.
    assert "page_elements_v3" not in df.columns

    assert df.iloc[1]["metadata"]["segment_index"] == 1
    assert df.iloc[1]["page_number"] == 2
    assert df.iloc[1]["metadata"]["segment_start"] == pytest.approx(2.0)


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not available")
def test_video_frame_extract_actor_batch(tmp_path: Path):
    video = tmp_path / "batch.mp4"
    _make_tiny_mp4(video, duration_sec=4)

    batch = pd.DataFrame([{"path": str(video)}])
    params = VideoExtractParams(split_type="time", split_interval=2)
    actor = VideoFrameExtractActor(params=params)
    out = actor.run(batch)

    assert isinstance(out, pd.DataFrame)
    assert len(out) == 2
    assert all(out["metadata"].apply(lambda m: m["source_video"] == str(video)))
    assert set(out["metadata"].apply(lambda m: m["segment_index"]).tolist()) == {0, 1}


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not available")
def test_video_frame_extract_actor_empty_batch():
    actor = VideoFrameExtractActor(params=VideoExtractParams())
    out = actor.run(pd.DataFrame(columns=["path"]))
    assert isinstance(out, pd.DataFrame)
    assert out.empty
    assert _PAGE_COLUMNS.issubset(set(out.columns))


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not available")
def test_video_frame_extract_unreadable_file_yields_empty(tmp_path: Path):
    bad = tmp_path / "corrupt.mp4"
    bad.write_bytes(b"this is not a video file")

    df = video_frames_path_to_pages_df(str(bad), params=VideoExtractParams(split_type="time", split_interval=2))
    assert isinstance(df, pd.DataFrame)
    assert df.empty


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not available")
def test_video_branch_runs_ocr_only_pipeline(tmp_path: Path, monkeypatch):
    """Video frames go directly to VideoFrameOCRActor; PageElementDetection and
    the generic OCRActor are both skipped."""
    from nemo_retriever.graph.multi_type_extract_operator import MultiTypeExtractCPUActor

    video = tmp_path / "route.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=320x240:d=3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        capture_output=True,
    )

    op = MultiTypeExtractCPUActor(
        extraction_mode="auto",
        video_params=VideoExtractParams(split_interval=1, extract_audio=False),
    )

    forbidden: list[str] = []
    video_ocr_called: list[int] = []

    def _fake_instantiate(cls, **kwargs):
        name = cls.__name__
        if name in {"PageElementDetectionActor", "OCRActor", "TableStructureActor", "GraphicElementsActor"}:
            forbidden.append(name)
            raise AssertionError(f"{name} must be skipped for video frames")

        if name == "VideoFrameOCRActor":

            class _FakeVideoOCR:
                def run(self, df):
                    video_ocr_called.append(len(df))
                    df = df.copy()
                    df["text"] = [f"ocr_segment_{idx}" for idx in df["metadata"].apply(lambda m: m["segment_index"])]
                    df["ocr"] = [{"timing": {"seconds": 0.0}, "error": None}] * len(df)
                    return df

            return _FakeVideoOCR()

        raise AssertionError(f"Unexpected operator in video frame pipeline: {name}")

    monkeypatch.setattr(op, "_instantiate_resolved", _fake_instantiate)

    batch = pd.DataFrame([{"path": str(video), "bytes": video.read_bytes()}])
    result = op.process(batch)

    assert forbidden == []
    assert video_ocr_called == [3]
    assert result["text"].tolist() == ["ocr_segment_0", "ocr_segment_1", "ocr_segment_2"]
    assert result["metadata"].apply(lambda m: m["modality"]).unique().tolist() == ["video_frame"]


def test_video_frame_ocr_cpu_actor_writes_text(monkeypatch):
    """VideoFrameOCRCPUActor invokes the OCR NIM once per frame and writes text."""
    from nemo_retriever.video.frame_ocr import VideoFrameOCRCPUActor

    actor = VideoFrameOCRCPUActor(ocr_invoke_url="http://fake/ocr", api_key="secret")

    captured_invocations: list[dict] = []

    def _fake_invoke(self, **kwargs):
        captured_invocations.append(kwargs)
        # NIM response shape: one item per input image with top-level text_detections.
        return [
            {
                "text_detections": [
                    {
                        "text_prediction": {"text": f"page{i}"},
                        "bounding_box": {"points": [{"x": 0, "y": 0}]},
                    }
                ]
            }
            for i, _ in enumerate(kwargs["image_b64_list"])
        ]

    monkeypatch.setattr(
        "nemo_retriever.nim.nim.NIMClient.invoke_image_inference_batches",
        _fake_invoke,
    )

    df = pd.DataFrame(
        [
            {
                "path": "/v.mp4",
                "page_number": 1,
                "text": "",
                "page_image": {"image_b64": "AAAA", "encoding": "png", "orig_shape_hw": (2, 2)},
                "metadata": {"modality": "video_frame"},
            },
            {
                "path": "/v.mp4",
                "page_number": 2,
                "text": "",
                "page_image": {"image_b64": "BBBB", "encoding": "png", "orig_shape_hw": (2, 2)},
                "metadata": {"modality": "video_frame"},
            },
        ]
    )

    out = actor.run(df)

    assert len(captured_invocations) == 1
    assert captured_invocations[0]["invoke_url"] == "http://fake/ocr"
    assert captured_invocations[0]["api_key"] == "secret"
    assert captured_invocations[0]["image_b64_list"] == ["AAAA", "BBBB"]
    assert out["text"].tolist() == ["page0", "page1"]
    assert all(meta["error"] is None for meta in out["ocr"])


def test_video_frame_ocr_cpu_actor_requires_invoke_url():
    """The CPU (remote-NIM) variant must refuse to instantiate without a URL."""
    from nemo_retriever.video.frame_ocr import VideoFrameOCRCPUActor

    with pytest.raises(ValueError, match="ocr_invoke_url"):
        VideoFrameOCRCPUActor()


def test_video_frame_ocr_cpu_actor_missing_page_image(monkeypatch):
    """Rows without page_image are skipped; no NIM call happens when batch is empty."""
    from nemo_retriever.video.frame_ocr import VideoFrameOCRCPUActor

    actor = VideoFrameOCRCPUActor(ocr_invoke_url="http://fake/ocr")

    call_count = {"n": 0}

    def _fake_invoke(self, **kwargs):
        call_count["n"] += 1
        return []

    monkeypatch.setattr(
        "nemo_retriever.nim.nim.NIMClient.invoke_image_inference_batches",
        _fake_invoke,
    )

    df = pd.DataFrame([{"path": "/v.mp4", "page_number": 1, "text": "", "page_image": None, "metadata": {}}])
    out = actor.run(df)
    assert call_count["n"] == 0
    assert out["text"].tolist() == [""]


def test_video_frame_ocr_archetype_prefers_gpu_by_default():
    """Without ocr_invoke_url, the archetype routes to the local GPU model."""
    from nemo_retriever.video.frame_ocr import VideoFrameOCRActor

    assert VideoFrameOCRActor.prefers_cpu_variant({}) is False
    assert VideoFrameOCRActor.prefers_cpu_variant({"ocr_invoke_url": ""}) is False
    # Remote is only chosen when a URL is explicitly provided.
    assert VideoFrameOCRActor.prefers_cpu_variant({"ocr_invoke_url": "http://x/ocr"}) is True


def test_multi_type_extractor_needs_gpu_for_video_by_default():
    """Outer MultiTypeExtractOperator must pick its GPU variant for video
    unless a remote OCR URL is set; otherwise Ray places it on a CPU-only
    worker and loading the local Nemotron OCR model blows up with a CUDA
    deserialization error."""
    from nemo_retriever.graph.multi_type_extract_operator import MultiTypeExtractOperator
    from nemo_retriever.params import ExtractParams

    # Default video mode (no extract_params, no extract_params.ocr_invoke_url) -> prefers GPU.
    assert MultiTypeExtractOperator.prefers_cpu_variant({"extraction_mode": "video", "extract_params": None}) is False
    assert (
        MultiTypeExtractOperator.prefers_cpu_variant({"extraction_mode": "video", "extract_params": ExtractParams()})
        is False
    )

    # With a remote OCR endpoint, CPU variant is fine.
    assert (
        MultiTypeExtractOperator.prefers_cpu_variant(
            {"extraction_mode": "video", "extract_params": ExtractParams(ocr_invoke_url="http://x/ocr")}
        )
        is True
    )

    # Audio mode is unaffected.
    assert MultiTypeExtractOperator.prefers_cpu_variant({"extraction_mode": "audio", "extract_params": None}) is True


def test_video_frame_ocr_gpu_actor_uses_local_model(monkeypatch):
    """GPU variant drives the local NemotronOCRV1.invoke once per frame."""
    from nemo_retriever.video import frame_ocr as fo

    class _FakeModel:
        def __init__(self):
            self.calls = []

        def invoke(self, b64, merge_level="paragraph"):
            self.calls.append((b64, merge_level))
            # Nemotron OCR v1 returns a list of text blocks (any of several schemas).
            return [{"text": f"frame-{b64}", "left": 0.0, "right": 1.0, "upper": 0.0, "lower": 1.0}]

    fake_model = _FakeModel()

    def _fake_init(self, **kwargs):
        from nemo_retriever.graph.abstract_operator import AbstractOperator

        AbstractOperator.__init__(self, **kwargs)
        self._model = fake_model
        self._merge_level = "paragraph"

    monkeypatch.setattr(fo.VideoFrameOCRGPUActor, "__init__", _fake_init)

    actor = fo.VideoFrameOCRGPUActor()
    df = pd.DataFrame(
        [
            {
                "path": "/v.mp4",
                "page_number": 1,
                "text": "",
                "page_image": {"image_b64": "AAAA", "encoding": "png", "orig_shape_hw": (2, 2)},
                "metadata": {"modality": "video_frame"},
            },
            {
                "path": "/v.mp4",
                "page_number": 2,
                "text": "",
                "page_image": {"image_b64": "BBBB", "encoding": "png", "orig_shape_hw": (2, 2)},
                "metadata": {"modality": "video_frame"},
            },
        ]
    )
    out = actor.run(df)

    assert [b for b, _ in fake_model.calls] == ["AAAA", "BBBB"]
    assert out["text"].tolist() == ["frame-AAAA", "frame-BBBB"]
    assert all(meta["error"] is None for meta in out["ocr"])


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not available")
def test_video_frame_position_first_vs_middle(tmp_path: Path):
    """'first' seek lands at segment start; 'middle' lands at midpoint."""
    video = tmp_path / "pos.mp4"
    _make_tiny_mp4(video, duration_sec=4)

    df_first = video_frames_path_to_pages_df(
        str(video),
        params=VideoExtractParams(split_type="time", split_interval=2, frame_position="first"),
    )
    df_mid = video_frames_path_to_pages_df(
        str(video),
        params=VideoExtractParams(split_type="time", split_interval=2, frame_position="middle"),
    )
    assert df_first.iloc[0]["metadata"]["frame_position_seconds"] == pytest.approx(0.0)
    assert df_mid.iloc[0]["metadata"]["frame_position_seconds"] == pytest.approx(1.0)


def test_graph_pipeline_infers_input_type_from_extension(tmp_path: Path):
    """_infer_input_type maps common extensions to their input-type label."""
    from nemo_retriever.examples.graph_pipeline import _infer_input_type

    cases = {
        "clip.mp4": "video",
        "clip.mov": "video",
        "clip.mkv": "video",
        "sample.wav": "audio",
        "doc.pdf": "pdf",
        "slide.pptx": "doc",
        "note.txt": "txt",
        "page.html": "html",
        "img.png": "image",
    }
    for name, expected in cases.items():
        p = tmp_path / name
        p.write_bytes(b"")
        assert _infer_input_type(p) == expected, f"{name} → {expected}"


def test_graph_pipeline_infers_input_type_from_directory(tmp_path: Path):
    """A directory of one type resolves; a mix raises typer.BadParameter."""
    import typer
    from nemo_retriever.examples.graph_pipeline import _infer_input_type

    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "a.mp4").write_bytes(b"")
    (videos / "b.mkv").write_bytes(b"")
    assert _infer_input_type(videos) == "video"

    mixed = tmp_path / "mixed"
    mixed.mkdir()
    (mixed / "a.mp4").write_bytes(b"")
    (mixed / "b.pdf").write_bytes(b"")
    with pytest.raises(typer.BadParameter, match="mixed input types"):
        _infer_input_type(mixed)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(typer.BadParameter, match="No supported files"):
        _infer_input_type(empty)
