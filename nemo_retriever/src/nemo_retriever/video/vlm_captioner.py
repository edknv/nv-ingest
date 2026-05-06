# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VideoFrameVLMCaptioner: per-frame VLM captioning.

Mutually exclusive with :class:`~nemo_retriever.video.VideoFrameOCRActor`.
The graph wires either one based on
``VideoFrameParams.frame_text_method``.  Reuses the local Nemotron VLM
and remote NIM clients used by
:class:`~nemo_retriever.caption.CaptionActor`.
"""

from __future__ import annotations

import logging
from typing import Any, List

import pandas as pd

from nemo_retriever.caption.caption import (
    _caption_batch_local,
    _caption_batch_remote,
    _create_remote_client,
    _get_cached_local_model,
)
from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.graph.gpu_operator import GPUOperator
from nemo_retriever.graph.operator_archetype import ArchetypeOperator
from nemo_retriever.ocr.shared import concat_with_passthrough
from nemo_retriever.params import VideoFrameVLMParams
from nemo_retriever.video import _content_types as _CT

logger = logging.getLogger(__name__)


def _split_frame_rows(batch_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "_content_type" not in batch_df.columns:
        return pd.DataFrame(), batch_df
    is_frame = batch_df["_content_type"].astype(str) == _CT.VIDEO_FRAME
    return batch_df[is_frame].reset_index(drop=True), batch_df[~is_frame].reset_index(drop=True)


def _stamp_method(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["metadata"] = [
        {**(m or {}), "frame_text_method": "vlm"} for m in df["metadata"]
    ]
    return df


def _caption_video_frames(
    frame_df: pd.DataFrame,
    *,
    params: VideoFrameVLMParams,
    nim_client: Any | None = None,
    local_model: Any | None = None,
) -> pd.DataFrame:
    """Apply VLM captioning to a frame-only DataFrame; write the caption to ``text``.

    Either ``nim_client`` (remote) or ``local_model`` (GPU) must be provided.
    Frames whose VLM call fails get an empty ``text``; downstream
    ``VideoFrameTextDedup`` drops empty-text rows.
    """
    if frame_df.empty:
        return frame_df

    images: List[str] = list(frame_df["image_b64"].astype(str))
    captions: List[str]
    if nim_client is not None:
        try:
            captions = _caption_batch_remote(
                images,
                nim_client=nim_client,
                model_name=params.model_name,
                prompt=params.prompt,
                system_prompt=None,
                temperature=params.temperature,
                max_tokens=params.max_tokens,
            )
        except Exception:
            logger.warning("Remote VLM call failed; emitting empty captions", exc_info=True)
            captions = ["" for _ in images]
    elif local_model is not None:
        try:
            captions = _caption_batch_local(
                images,
                model=local_model,
                prompt=params.prompt,
                system_prompt=None,
                temperature=params.temperature,
                max_tokens=params.max_tokens,
            )
        except Exception:
            logger.warning("Local VLM call failed; emitting empty captions", exc_info=True)
            captions = ["" for _ in images]
    else:
        raise RuntimeError("_caption_video_frames requires nim_client or local_model")

    out = frame_df.copy()
    out["text"] = captions
    return _stamp_method(out)


class VideoFrameVLMCaptionerCPUActor(AbstractOperator, CPUOperator):
    """Remote VLM captioning over a NIM endpoint."""

    def __init__(self, params: VideoFrameVLMParams) -> None:
        super().__init__(params=params)
        self._params = params
        endpoint = (params.endpoint_url or "").strip()
        if not endpoint:
            raise ValueError("VideoFrameVLMCaptionerCPUActor requires params.endpoint_url to be set.")
        self._nim_client = _create_remote_client(endpoint, params.api_key)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, batch_df: Any, **kwargs: Any) -> Any:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return batch_df
        frames, passthrough = _split_frame_rows(batch_df)
        if frames.empty:
            return passthrough
        captioned = _caption_video_frames(frames, params=self._params, nim_client=self._nim_client)
        return concat_with_passthrough(captioned, passthrough)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class VideoFrameVLMCaptionerGPUActor(AbstractOperator, GPUOperator):
    """Local Nemotron VLM captioning."""

    def __init__(self, params: VideoFrameVLMParams) -> None:
        super().__init__(params=params)
        self._params = params
        self._local_model = _get_cached_local_model(params.model_dump(mode="python"))

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, batch_df: Any, **kwargs: Any) -> Any:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return batch_df
        frames, passthrough = _split_frame_rows(batch_df)
        if frames.empty:
            return passthrough
        captioned = _caption_video_frames(frames, params=self._params, local_model=self._local_model)
        return concat_with_passthrough(captioned, passthrough)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


@designer_component(
    name="Video Frame VLM Captioner",
    category="Video",
    compute="gpu",
    description="Captions each video frame with a VLM (resolves to local Nemotron VLM or remote NIM)",
)
class VideoFrameVLMCaptioner(ArchetypeOperator):
    """Graph-facing archetype that resolves to GPU or CPU variant.

    Routes to the CPU (NIM) variant when ``params.endpoint_url`` is set;
    otherwise loads the local Nemotron VLM on GPU.
    """

    @classmethod
    def prefers_cpu_variant(cls, operator_kwargs: dict[str, Any] | None = None) -> bool:
        params = (operator_kwargs or {}).get("params")
        endpoint = getattr(params, "endpoint_url", None)
        return bool(str(endpoint or "").strip())

    @classmethod
    def cpu_variant_class(cls):
        return VideoFrameVLMCaptionerCPUActor

    @classmethod
    def gpu_variant_class(cls):
        return VideoFrameVLMCaptionerGPUActor

    def __init__(self, params: VideoFrameVLMParams) -> None:
        super().__init__(params=params)
        self._params = params
