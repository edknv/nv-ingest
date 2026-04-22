# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VideoFrameOCRActor: OCR a video frame directly as a full page image.

Video frames don't benefit from per-element detection (tables / charts /
infographics are rare mid-frame), so instead of running PageElementDetection +
the crop-based ``ocr_page_elements`` stage we invoke the OCR model once per
frame on the whole image and write the result into ``text``.

By default the archetype resolves to the **GPU** variant which loads the
local Nemotron OCR v1 model.  The CPU variant (remote NIM HTTP) is only
selected when ``ocr_invoke_url`` is explicitly provided.
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Any, List, Optional

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.graph.gpu_operator import GPUOperator
from nemo_retriever.graph.operator_archetype import ArchetypeOperator
from nemo_retriever.nim.nim import NIMClient
from nemo_retriever.ocr.shared import (
    _blocks_to_text,
    _extract_remote_ocr_item,
    _parse_ocr_result,
)
from nemo_retriever.params import RemoteRetryParams

logger = logging.getLogger(__name__)


class VideoFrameOCRCPUActor(AbstractOperator, CPUOperator):
    """Remote-NIM variant: one OCR HTTP call per frame, batched."""

    def __init__(
        self,
        *,
        ocr_invoke_url: Optional[str] = None,
        api_key: Optional[str] = None,
        inference_batch_size: int = 8,
        request_timeout_s: float = 120.0,
        remote_max_pool_workers: int = 16,
        remote_max_retries: int = 10,
        remote_max_429_retries: int = 5,
    ) -> None:
        super().__init__(
            ocr_invoke_url=ocr_invoke_url,
            api_key=api_key,
            inference_batch_size=inference_batch_size,
            request_timeout_s=request_timeout_s,
            remote_max_pool_workers=remote_max_pool_workers,
            remote_max_retries=remote_max_retries,
            remote_max_429_retries=remote_max_429_retries,
        )
        invoke_url = str(ocr_invoke_url or "").strip()
        if not invoke_url:
            raise ValueError(
                "VideoFrameOCRCPUActor requires ocr_invoke_url. "
                "Instantiate VideoFrameOCRActor instead — it resolves to the GPU variant (local model) by default."
            )
        self._invoke_url = invoke_url
        self._api_key = api_key
        self._inference_batch_size = max(1, int(inference_batch_size))
        self._timeout_s = float(request_timeout_s)
        self._retry = RemoteRetryParams(
            remote_max_pool_workers=int(remote_max_pool_workers),
            remote_max_retries=int(remote_max_retries),
            remote_max_429_retries=int(remote_max_429_retries),
        )
        self._nim_client = NIMClient(max_pool_workers=int(remote_max_pool_workers))

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        if not isinstance(data, pd.DataFrame) or data.empty:
            return data

        b64s, positions = _collect_frame_b64s(data)
        out = data.copy()
        if "text" not in out.columns:
            out["text"] = [""] * len(out)

        if not b64s:
            out["ocr"] = [{"timing": None, "error": None}] * len(out)
            return out

        t0 = time.perf_counter()
        try:
            response_items = self._nim_client.invoke_image_inference_batches(
                invoke_url=self._invoke_url,
                image_b64_list=b64s,
                api_key=self._api_key,
                timeout_s=self._timeout_s,
                max_batch_size=self._inference_batch_size,
                max_retries=int(self._retry.remote_max_retries),
                max_429_retries=int(self._retry.remote_max_429_retries),
            )
        except BaseException as exc:
            elapsed = time.perf_counter() - t0
            logger.warning("Video frame OCR failed: %s: %s", type(exc).__name__, exc)
            out["ocr"] = [{"timing": {"seconds": float(elapsed)}, "error": _exc_payload(exc)}] * len(out)
            return out

        elapsed = time.perf_counter() - t0
        ocr_meta = [{"timing": {"seconds": float(elapsed)}, "error": None} for _ in range(len(out))]
        if len(response_items) != len(b64s):
            logger.warning("Video frame OCR returned %d responses for %d inputs", len(response_items), len(b64s))
        for local_i, pos in enumerate(positions):
            if local_i >= len(response_items):
                break
            preds = _extract_remote_ocr_item(response_items[local_i])
            text = _blocks_to_text(_parse_ocr_result(preds))
            out.at[out.index[pos], "text"] = text
        out["ocr"] = ocr_meta
        return out

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, batch_df: Any, **kwargs: Any) -> Any:
        return self.run(batch_df, **kwargs)


class VideoFrameOCRGPUActor(AbstractOperator, GPUOperator):
    """Local-model variant: loads Nemotron OCR v1 once per actor and invokes it per frame."""

    def __init__(
        self,
        *,
        ocr_invoke_url: Optional[str] = None,
        api_key: Optional[str] = None,
        inference_batch_size: int = 8,
        merge_level: str = "paragraph",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            ocr_invoke_url=ocr_invoke_url,
            api_key=api_key,
            inference_batch_size=inference_batch_size,
            merge_level=merge_level,
        )
        # ocr_invoke_url/api_key are accepted for signature parity with the CPU
        # variant; they are ignored here (local model). The archetype routes to
        # the CPU variant when the caller actually wants remote inference.
        from nemo_retriever.model.local import NemotronOCRV1

        self._model = NemotronOCRV1()
        self._merge_level = str(merge_level or "paragraph")

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        if not isinstance(data, pd.DataFrame) or data.empty:
            return data

        b64s, positions = _collect_frame_b64s(data)
        out = data.copy()
        if "text" not in out.columns:
            out["text"] = [""] * len(out)

        if not b64s:
            out["ocr"] = [{"timing": None, "error": None}] * len(out)
            return out

        t0 = time.perf_counter()
        ocr_meta: List[dict] = [{"timing": None, "error": None} for _ in range(len(out))]
        for local_i, pos in enumerate(positions):
            b64 = b64s[local_i]
            try:
                preds = self._model.invoke(b64, merge_level=self._merge_level)
            except BaseException as exc:
                logger.warning("Local video frame OCR failed for row %d: %s", pos, exc)
                ocr_meta[pos] = {"timing": None, "error": _exc_payload(exc)}
                continue
            text = _blocks_to_text(_parse_ocr_result(preds))
            out.at[out.index[pos], "text"] = text
        elapsed = time.perf_counter() - t0
        for meta in ocr_meta:
            if meta.get("error") is None:
                meta["timing"] = {"seconds": float(elapsed)}
        out["ocr"] = ocr_meta
        return out

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, batch_df: Any, **kwargs: Any) -> Any:
        return self.run(batch_df, **kwargs)


@designer_component(
    name="Video Frame OCR",
    category="Video",
    compute="gpu",
    description="Runs OCR on each video frame as a full page image (local GPU by default; remote NIM if URL set).",
    category_color="#ff6bbb",
)
class VideoFrameOCRActor(ArchetypeOperator):
    """Graph-facing archetype: GPU (local model) by default, CPU (remote NIM) when URL set."""

    _cpu_variant_class = VideoFrameOCRCPUActor
    _gpu_variant_class = VideoFrameOCRGPUActor

    @classmethod
    def prefers_cpu_variant(cls, operator_kwargs: dict[str, Any] | None = None) -> bool:
        kwargs = operator_kwargs or {}
        return bool(str(kwargs.get("ocr_invoke_url") or kwargs.get("invoke_url") or "").strip())

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


def _collect_frame_b64s(data: pd.DataFrame) -> tuple[List[str], List[int]]:
    """Extract page_image.image_b64 from each row, skipping rows without one."""
    b64s: List[str] = []
    positions: List[int] = []
    for pos, (_, row) in enumerate(data.iterrows()):
        page_image = row.get("page_image")
        if not isinstance(page_image, dict):
            continue
        b64 = page_image.get("image_b64")
        if isinstance(b64, str) and b64:
            b64s.append(b64)
            positions.append(pos)
    return b64s, positions


def _exc_payload(exc: BaseException) -> dict:
    return {
        "stage": "video_frame_ocr",
        "type": exc.__class__.__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }
