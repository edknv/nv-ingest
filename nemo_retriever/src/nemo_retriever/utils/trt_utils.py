# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for optional model compilation (torch.compile / torch_tensorrt)."""

from __future__ import annotations

import logging
import os
from typing import Any, Set

import torch

logger = logging.getLogger(__name__)

_TRT_ENV_VAR = "RETRIEVER_ENABLE_TORCH_TRT"


def is_trt_enabled() -> bool:
    """Return True when the user has opted into model compilation."""
    return os.getenv(_TRT_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# torch.compile — for text models (embedder, reranker)
# ---------------------------------------------------------------------------


def _has_torch_tensorrt() -> bool:
    """Check if torch_tensorrt is importable."""
    try:
        import torch_tensorrt  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def try_compile_model(model: torch.nn.Module) -> torch.nn.Module:
    """Best-effort compilation via ``torch.compile``.

    * If ``torch_tensorrt`` is installed, uses it as the backend for
      maximum inference throughput.
    * Otherwise, falls back to the built-in ``inductor`` backend which
      still provides significant speedups with zero extra dependencies.

    Returns the original *model* unchanged on any failure.
    """
    if _has_torch_tensorrt():
        backend = "torch_tensorrt"
    else:
        backend = "inductor"

    try:
        compiled = torch.compile(model, backend=backend)
        logger.info("torch.compile succeeded (backend=%s)", backend)
        return compiled
    except Exception:
        logger.warning(
            "torch.compile (backend=%s) failed — falling back to eager PyTorch",
            backend,
            exc_info=True,
        )
        return model


# ---------------------------------------------------------------------------
# torch_tensorrt.compile — for vision models (OCR, page-elements)
# that need explicit submodule compilation with fixed input shapes
# ---------------------------------------------------------------------------


def try_compile_trt(
    model: torch.nn.Module,
    trt_inputs: list[Any],
    enabled_precisions: Set[torch.dtype],
    **kwargs: Any,
) -> torch.nn.Module:
    """Best-effort TensorRT compilation for vision submodules.

    This uses the explicit ``torch_tensorrt.compile()`` API which is
    needed when compiling individual submodules with fixed input shapes
    and excluded ops (e.g. NMS).  For text models, prefer
    :func:`try_compile_model` which uses ``torch.compile``.

    Returns the original *model* unchanged on any failure.
    """
    try:
        import torch_tensorrt  # type: ignore
    except Exception:
        logger.info("torch_tensorrt not available — skipping TRT compilation")
        return model

    try:
        compiled = torch_tensorrt.compile(
            model,
            inputs=trt_inputs,
            enabled_precisions=enabled_precisions,
            **kwargs,
        )
        logger.info("TensorRT compilation succeeded")
        return compiled
    except Exception:
        logger.warning("TensorRT compilation failed — falling back to eager PyTorch", exc_info=True)
        return model
