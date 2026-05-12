# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VideoFrameCaptionActor: full-frame captioning for video frames.

Thin wrapper around the shared :func:`caption_full_image_df` helper in
:mod:`nemo_retriever.caption.caption`. The CPU/GPU variants own the model
lifecycle (NIM client init, lazy local VLM load); the batch-shape handling
and caption call are reused from the page-elements caption pipeline so a
single code path covers both image and video frame captioning.

Configuration is read from :class:`CaptionParams`, the same params object
that drives :class:`CaptionActor` for the image / PDF pipelines.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from nemo_retriever.caption.caption import (
    _create_local_model,
    _create_remote_client,
    caption_full_image_df,
)
from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.graph.gpu_operator import GPUOperator
from nemo_retriever.graph.operator_archetype import ArchetypeOperator
from nemo_retriever.ocr.shared import concat_with_passthrough, split_ocrable_rows
from nemo_retriever.params import CaptionParams
from nemo_retriever.video import _content_types as _CT

_CAPTIONABLE_CONTENT_TYPES = ("", _CT.VIDEO_FRAME)

logger = logging.getLogger(__name__)


def _has_remote_endpoint(params: CaptionParams) -> bool:
    return bool(str(getattr(params, "endpoint_url", None) or "").strip())


class VideoFrameCaptionGPUActor(AbstractOperator, GPUOperator):
    """Local VLM captioner on full video frames."""

    def __init__(self, params: CaptionParams) -> None:
        super().__init__(params=params)
        self._params = params
        if _has_remote_endpoint(params):
            raise ValueError(
                "VideoFrameCaptionGPUActor does not support remote endpoint execution. "
                "Use VideoFrameCaptionCPUActor instead."
            )
        self._kwargs = params.model_dump(mode="python")
        self._model = None  # lazily loaded on first call

    def _ensure_model(self) -> None:
        if self._model is None:
            self._model = _create_local_model(self._kwargs)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, batch_df: Any, **kwargs: Any) -> Any:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return pd.DataFrame()
        cap_df, passthrough = split_ocrable_rows(batch_df, _CAPTIONABLE_CONTENT_TYPES)
        if cap_df.empty:
            return passthrough
        self._ensure_model()
        out = caption_full_image_df(
            cap_df,
            model=self._model,
            model_name=self._kwargs.get("model_name"),
            prompt=self._kwargs.get("prompt"),
            system_prompt=self._kwargs.get("system_prompt"),
            temperature=self._kwargs.get("temperature", 1.0),
            top_p=self._kwargs.get("top_p"),
            max_tokens=int(self._kwargs.get("max_tokens", 1024)),
            batch_size=int(self._kwargs.get("batch_size", 8)),
        )
        return concat_with_passthrough(out, passthrough)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class VideoFrameCaptionCPUActor(AbstractOperator, CPUOperator):
    """Remote VLM captioner (NIM) on full video frames, batched per call."""

    def __init__(self, params: CaptionParams) -> None:
        super().__init__(params=params)
        self._params = params
        if not _has_remote_endpoint(params):
            raise ValueError("VideoFrameCaptionCPUActor requires params.endpoint_url to be set.")
        self._kwargs = params.model_dump(mode="python")
        self._nim_client = _create_remote_client(
            self._kwargs.get("endpoint_url"),
            self._kwargs.get("api_key"),
        )

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, batch_df: Any, **kwargs: Any) -> Any:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return pd.DataFrame()
        cap_df, passthrough = split_ocrable_rows(batch_df, _CAPTIONABLE_CONTENT_TYPES)
        if cap_df.empty:
            return passthrough

        from nemo_retriever.caption.caption import caption_b64_to_text

        out = cap_df.copy()
        b64s = [b if isinstance(b, str) else "" for b in out.get("image_b64", [])]
        captions = caption_b64_to_text(
            b64s,
            nim_client=self._nim_client,
            model_name=self._kwargs.get("model_name"),
            prompt=self._kwargs.get("prompt"),
            system_prompt=self._kwargs.get("system_prompt"),
            temperature=self._kwargs.get("temperature", 1.0),
            top_p=self._kwargs.get("top_p"),
            max_tokens=int(self._kwargs.get("max_tokens", 1024)),
            batch_size=int(self._kwargs.get("batch_size", 8)),
        )
        out["text"] = captions
        out = out[out["text"].astype(bool)].reset_index(drop=True)
        return concat_with_passthrough(out, passthrough)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


@designer_component(
    name="Video Frame Caption",
    category="Video",
    compute="gpu",
    description="Generates captions for full video frames using a vision-language model",
)
class VideoFrameCaptionActor(ArchetypeOperator):
    """Graph-facing archetype that resolves to GPU or CPU variant.

    Routes to the CPU (NIM) variant when ``CaptionParams.endpoint_url`` is
    set; otherwise loads a local Nemotron VLM captioner on GPU.
    """

    @classmethod
    def prefers_cpu_variant(cls, operator_kwargs: dict[str, Any] | None = None) -> bool:
        params = (operator_kwargs or {}).get("params")
        return bool(str(getattr(params, "endpoint_url", None) or "").strip())

    @classmethod
    def cpu_variant_class(cls):
        return VideoFrameCaptionCPUActor

    @classmethod
    def gpu_variant_class(cls):
        return VideoFrameCaptionGPUActor

    def __init__(self, params: CaptionParams) -> None:
        super().__init__(params=params)
        self._params = params
