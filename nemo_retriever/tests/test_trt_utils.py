# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_retriever.utils.trt_utils helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_retriever.utils.trt_utils import is_trt_enabled, try_compile_model, try_compile_trt


# ---------------------------------------------------------------------------
# is_trt_enabled()
# ---------------------------------------------------------------------------


class TestIsTrtEnabled:
    @pytest.mark.parametrize("val", ["1", "true", "True", "TRUE", "yes", "YES", "on", "ON", " true "])
    def test_truthy_values(self, monkeypatch, val):
        monkeypatch.setenv("RETRIEVER_ENABLE_TORCH_TRT", val)
        assert is_trt_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", "", "random"])
    def test_falsy_values(self, monkeypatch, val):
        monkeypatch.setenv("RETRIEVER_ENABLE_TORCH_TRT", val)
        assert is_trt_enabled() is False

    def test_unset(self, monkeypatch):
        monkeypatch.delenv("RETRIEVER_ENABLE_TORCH_TRT", raising=False)
        assert is_trt_enabled() is False


# ---------------------------------------------------------------------------
# try_compile_model() — torch.compile wrapper
# ---------------------------------------------------------------------------


class TestTryCompileModel:
    def test_uses_torch_tensorrt_backend_when_available(self):
        model = MagicMock(spec=torch.nn.Module)
        compiled = MagicMock(spec=torch.nn.Module)

        with patch("nemo_retriever.utils.trt_utils._has_torch_tensorrt", return_value=True), \
             patch("torch.compile", return_value=compiled) as mock_compile:
            result = try_compile_model(model)

        mock_compile.assert_called_once_with(model, backend="torch_tensorrt")
        assert result is compiled

    def test_uses_inductor_backend_when_torch_tensorrt_missing(self):
        model = MagicMock(spec=torch.nn.Module)
        compiled = MagicMock(spec=torch.nn.Module)

        with patch("nemo_retriever.utils.trt_utils._has_torch_tensorrt", return_value=False), \
             patch("torch.compile", return_value=compiled) as mock_compile:
            result = try_compile_model(model)

        mock_compile.assert_called_once_with(model, backend="inductor")
        assert result is compiled

    def test_returns_original_model_on_compile_failure(self):
        model = MagicMock(spec=torch.nn.Module)

        with patch("nemo_retriever.utils.trt_utils._has_torch_tensorrt", return_value=False), \
             patch("torch.compile", side_effect=RuntimeError("compilation failed")):
            result = try_compile_model(model)

        assert result is model


# ---------------------------------------------------------------------------
# try_compile_trt() — explicit torch_tensorrt.compile for vision submodules
# ---------------------------------------------------------------------------


class TestTryCompileTrt:
    def test_returns_compiled_model_on_success(self):
        model = MagicMock(spec=torch.nn.Module)
        compiled = MagicMock(spec=torch.nn.Module)
        trt_inputs = [MagicMock()]

        with patch.dict("sys.modules", {"torch_tensorrt": MagicMock(compile=MagicMock(return_value=compiled))}):
            result = try_compile_trt(model, trt_inputs, {torch.float16})

        assert result is compiled

    def test_returns_original_model_when_torch_tensorrt_missing(self):
        model = MagicMock(spec=torch.nn.Module)
        trt_inputs = [MagicMock()]

        with patch.dict("sys.modules", {"torch_tensorrt": None}):
            result = try_compile_trt(model, trt_inputs, {torch.float16})

        assert result is model

    def test_returns_original_model_on_compile_failure(self):
        model = MagicMock(spec=torch.nn.Module)
        trt_inputs = [MagicMock()]

        mock_trt = MagicMock()
        mock_trt.compile.side_effect = RuntimeError("compilation failed")

        with patch.dict("sys.modules", {"torch_tensorrt": mock_trt}):
            result = try_compile_trt(model, trt_inputs, {torch.float16})

        assert result is model

    def test_forwards_kwargs_to_compile(self):
        model = MagicMock(spec=torch.nn.Module)
        compiled = MagicMock(spec=torch.nn.Module)
        trt_inputs = [MagicMock()]

        mock_trt = MagicMock()
        mock_trt.compile.return_value = compiled

        with patch.dict("sys.modules", {"torch_tensorrt": mock_trt}):
            try_compile_trt(
                model,
                trt_inputs,
                {torch.float16},
                torch_executed_ops={"torchvision::nms"},
            )

        mock_trt.compile.assert_called_once_with(
            model,
            inputs=trt_inputs,
            enabled_precisions={torch.float16},
            torch_executed_ops={"torchvision::nms"},
        )
