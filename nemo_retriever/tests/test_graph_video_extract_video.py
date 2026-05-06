# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for OCR-vs-VLM selection in the video graph builder."""

from __future__ import annotations

from typing import Any

import pytest

from nemo_retriever.params import (
    AudioChunkParams,
    AudioVisualFuseParams,
    VideoFrameParams,
    VideoFrameTextDedupParams,
    VideoFrameVLMParams,
)


def _build_video_graph(frame_params: VideoFrameParams) -> Any:
    """Test helper: invoke the same internal builder used by the runtime."""
    from nemo_retriever.graph.ingestor_runtime import build_graph

    return build_graph(
        extraction_mode="auto",
        extract_params=None,
        text_params=None,
        html_params=None,
        audio_chunk_params=AudioChunkParams(enabled=False),
        asr_params=None,
        caption_params=None,
        video_frame_params=frame_params,
        video_text_dedup_params=VideoFrameTextDedupParams(enabled=False),
        av_fuse_params=AudioVisualFuseParams(enabled=False),
    )


def _operator_class_names(graph: Any) -> set[str]:
    """Pull operator class names from a Graph, using the codebase's standard walker."""
    from nemo_retriever.graph.graph_pipeline_registry import collect_nodes

    return {node.operator_class.__name__ for node in collect_nodes(graph)}


def test_extract_video_default_wires_ocr_actor() -> None:
    graph = _build_video_graph(VideoFrameParams())
    names = _operator_class_names(graph)
    assert any("OCR" in n for n in names), f"expected OCR actor in {names}"


def test_extract_video_method_vlm_wires_vlm_captioner() -> None:
    params = VideoFrameParams(
        frame_text_method="vlm",
        vlm=VideoFrameVLMParams(
            endpoint_url="https://fake.example/vlm",
            api_key="x",
        ),
    )
    graph = _build_video_graph(params)
    names = _operator_class_names(graph)
    assert any(n.startswith("VideoFrameVLMCaptioner") for n in names), f"expected VLM captioner in {names}"
    assert not any("OCR" in n for n in names), f"OCR should not be wired: {names}"


def test_extract_video_method_none_wires_neither() -> None:
    params = VideoFrameParams(frame_text_method="none")
    graph = _build_video_graph(params)
    names = _operator_class_names(graph)
    assert not any(n.endswith("OCRActor") for n in names), f"OCR present: {names}"
    assert not any(n.startswith("VideoFrameVLMCaptioner") for n in names), f"VLM present: {names}"


def test_extract_video_method_vlm_without_endpoint_falls_back_to_local() -> None:
    """vlm path with no endpoint resolves to GPU/local archetype variant.

    This test only asserts the wiring; it doesn't construct the GPU model.
    """
    params = VideoFrameParams(
        frame_text_method="vlm",
        vlm=VideoFrameVLMParams(endpoint_url=None),
    )
    # Building the graph should not raise -- the archetype can defer
    # actor instantiation until the operator runs.
    graph = _build_video_graph(params)
    names = _operator_class_names(graph)
    assert any(n.startswith("VideoFrameVLMCaptioner") for n in names), f"expected VLM captioner in {names}"
