# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _ParamsModel._resolve_api_keys model validator."""

import pytest
from pydantic import ValidationError

from nemo_retriever.params.models import EmbedParams, ExtractParams, NO_API_KEY, StoreParams, VideoFrameParams


class TestVideoFrameParams:
    def test_fps_zero_rejected(self) -> None:
        """``fps=0`` would div-by-zero in ``_extract_one``; reject at the model boundary."""
        with pytest.raises(ValidationError):
            VideoFrameParams(fps=0)


class TestStoreParams:
    def test_storage_options_redacted_from_repr(self) -> None:
        params = StoreParams(storage_options={"key": "AKIA_TEST", "secret": "SECRET_TEST"})

        rendered = repr(params)

        assert "AKIA_TEST" not in rendered
        assert "SECRET_TEST" not in rendered
        assert "storage_options=***" in rendered
        assert params.storage_options == {"key": "AKIA_TEST", "secret": "SECRET_TEST"}


class TestResolveApiKeys:
    def test_nvidia_api_key_env_var(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        monkeypatch.delenv("NGC_API_KEY", raising=False)
        assert EmbedParams().api_key == "nvapi-test"

    def test_ngc_api_key_fallback(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.setenv("NGC_API_KEY", "ngc-test")
        assert EmbedParams().api_key == "ngc-test"

    def test_explicit_value_not_overwritten(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        assert EmbedParams(api_key="explicit-key").api_key == "explicit-key"

    def test_no_env_var_remains_none(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.delenv("NGC_API_KEY", raising=False)
        assert EmbedParams().api_key is None

    def test_no_api_key_sentinel_suppresses_resolution(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        assert EmbedParams(api_key=NO_API_KEY).api_key is None

    def test_all_api_key_fields_resolved_on_extract_params(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        monkeypatch.delenv("NGC_API_KEY", raising=False)
        params = ExtractParams()
        assert params.api_key == "nvapi-test"
        assert params.page_elements_api_key == "nvapi-test"
        assert params.ocr_api_key == "nvapi-test"


def test_video_scene_detect_params_defaults_and_roundtrip():
    from nemo_retriever.params import VideoSceneDetectParams

    p = VideoSceneDetectParams()
    assert p.enabled is False
    assert p.threshold == 30.0
    p2 = VideoSceneDetectParams.model_validate(p.model_dump())
    assert p2 == p


def test_video_key_frame_select_params_defaults():
    from nemo_retriever.params import VideoKeyFrameSelectParams

    p = VideoKeyFrameSelectParams()
    assert p.enabled is False
    assert p.z_threshold == 2.0


def test_video_advanced_dedup_params_defaults():
    from nemo_retriever.params import VideoAdvancedDedupParams

    p = VideoAdvancedDedupParams()
    assert p.enabled is False
    assert p.blur_threshold == 100.0
    assert p.similarity_threshold == 5
    assert p.entropy_gain_threshold == 0.1


def test_video_frame_vlm_params_defaults():
    from nemo_retriever.params import VideoFrameVLMParams

    p = VideoFrameVLMParams()
    assert p.endpoint_url is None
    assert p.api_key is None
    assert p.model_name == "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
    assert p.prompt == "Transcribe this image, word to word."
    assert p.temperature == 0.0
    assert p.max_tokens == 1024


def test_video_frame_params_extended_defaults_and_nesting():
    from nemo_retriever.params import (
        VideoAdvancedDedupParams,
        VideoFrameParams,
        VideoFrameVLMParams,
        VideoKeyFrameSelectParams,
        VideoSceneDetectParams,
    )

    p = VideoFrameParams()
    assert p.frame_text_method == "ocr"
    assert isinstance(p.scene_detection, VideoSceneDetectParams)
    assert isinstance(p.key_frame_selection, VideoKeyFrameSelectParams)
    assert isinstance(p.advanced_dedup, VideoAdvancedDedupParams)
    assert isinstance(p.vlm, VideoFrameVLMParams)
    p2 = VideoFrameParams.model_validate(p.model_dump())
    assert p2 == p


def test_audio_visual_fuse_params_modes():
    from nemo_retriever.params import AudioVisualFuseParams

    p = AudioVisualFuseParams()
    assert p.enabled is True
    assert p.mode == "per_utterance"
    assert p.scene_visual_max_chars == 800
    assert p.per_sentence_per_frame_max_chars == 100
    assert p.per_sentence_total_visual_max_chars == 400


def test_video_frame_params_rejects_unknown_method():
    import pydantic

    from nemo_retriever.params import VideoFrameParams

    with pytest.raises(pydantic.ValidationError):
        VideoFrameParams(frame_text_method="vqa")  # not in literal set


def test_audio_visual_fuse_params_rejects_unknown_mode():
    import pydantic

    from nemo_retriever.params import AudioVisualFuseParams

    with pytest.raises(pydantic.ValidationError):
        AudioVisualFuseParams(mode="bogus")
