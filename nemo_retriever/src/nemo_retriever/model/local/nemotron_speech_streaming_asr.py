# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Local ASR using ``nvidia/nemotron-speech-streaming-en-0.6b`` via the NVIDIA
NeMo toolkit. Model is a cache-aware RNN-T; ``ASRModel.transcribe()`` handles
chunking internally for offline batch use. Input: 16 kHz mono.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from nemo_retriever.utils.audio_io import load_audio_16k, write_wav_16k
from nemo_retriever.utils.nvtx import gpu_inference_range

logger = logging.getLogger(__name__)

MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"


try:
    import nemo.collections.asr as _nemo_asr

    _NEMO_AVAILABLE = True
except ImportError:
    _nemo_asr = None  # type: ignore[assignment]
    _NEMO_AVAILABLE = False


def _require_nemo() -> None:
    if not _NEMO_AVAILABLE:
        raise RuntimeError(
            "NemotronSpeechStreamingASR requires the NVIDIA NeMo toolkit. "
            'Install with: uv pip install -e "./nemo_retriever[audio]" '
            "(adds nemo_toolkit[asr])."
        )


def _hypothesis_text(item: Any) -> str:
    """Extract the transcript string from a NeMo output element.

    NeMo's ``ASRModel.transcribe`` return shape is decoder- and version-
    dependent: RNN-T may return ``Hypothesis`` objects; CTC or newer releases
    return plain ``str``; some paths wrap per-utterance output in a one-item
    list. Handle all three by unwrapping and falling back to ``.text``.
    """
    if item is None:
        return ""
    if isinstance(item, (list, tuple)):
        return _hypothesis_text(item[0]) if item else ""
    text = getattr(item, "text", None)
    if text is None:
        text = item
    return str(text).strip()


class NemotronSpeechStreamingASR:
    """Local ASR using ``nvidia/nemotron-speech-streaming-en-0.6b`` via NeMo."""

    def __init__(self, device: Optional[str] = None, model_name: Optional[str] = None) -> None:
        self._device = device
        self._model_name = model_name or MODEL_ID
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        _require_nemo()

        import torch
        from huggingface_hub import try_to_load_from_cache

        assert _nemo_asr is not None  # for type checkers

        # NeMo's ASRModel.from_pretrained() calls HfApi.file_exists() unconditionally
        # before consulting the cache, which raises OfflineModeIsEnabled under
        # HF_HUB_OFFLINE=1 even when the .nemo checkpoint is fully cached. When the
        # file is already local, load it via restore_from() to bypass the HF API
        # probe; fall back to from_pretrained() only when a first-time download
        # is actually needed.
        filename = f"{self._model_name.split('/')[-1]}.nemo"
        cached = try_to_load_from_cache(repo_id=self._model_name, filename=filename)
        if isinstance(cached, str):
            model = _nemo_asr.models.ASRModel.restore_from(cached)
        else:
            model = _nemo_asr.models.ASRModel.from_pretrained(model_name=self._model_name)
        model.eval()
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device.startswith("cuda") and torch.cuda.is_available():
            model = model.to(device)
        self._model = model
        logger.info("NemotronSpeechStreamingASR: loaded %s on %s", self._model_name, device)

    def transcribe(self, paths: List[str]) -> List[str]:
        """Transcribe audio files (any soundfile/ffmpeg-readable format) to text."""
        self._ensure_loaded()
        audios: List[Optional[np.ndarray]] = [load_audio_16k(p) for p in paths]
        return self.transcribe_audios(audios)

    def transcribe_audios(self, audios: List[Optional[np.ndarray]]) -> List[str]:
        """Transcribe a batch of 16 kHz mono float32 numpy arrays."""
        self._ensure_loaded()
        valid_idx: List[int] = []
        valid_arrays: List[np.ndarray] = []
        for i, audio in enumerate(audios):
            if audio is not None and audio.size > 0:
                valid_idx.append(i)
                valid_arrays.append(audio)

        result = [""] * len(audios)
        if not valid_arrays:
            return result

        tmp_paths: List[str] = []
        try:
            for audio in valid_arrays:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp_paths.append(f.name)
                write_wav_16k(audio, tmp_paths[-1])

            # RNN-T joint-network memory is O(B * T * U * V); with long
            # per-chunk audio (hours, in streaming configs) a Ray batch of even
            # a few utterances blows past 80 GiB. Transcribe one at a time —
            # NeMo's streaming decoder is already serialised internally and
            # there's no real throughput win from batching here.
            with gpu_inference_range("NemotronSpeechStreaming06B", batch_size=len(valid_arrays)):
                outputs = self._model.transcribe(tmp_paths, batch_size=1)
        except Exception as e:
            logger.warning("ASR (nemo) batch failed: %s", e)
            return result
        finally:
            for p in tmp_paths:
                Path(p).unlink(missing_ok=True)

        # Newer NeMo releases return ``(best_hypotheses, all_hypotheses)`` as a
        # 2-tuple; older releases return the ``best_hypotheses`` list directly.
        # Unwrap the tuple form so we always iterate the per-utterance list.
        if isinstance(outputs, tuple) and outputs and isinstance(outputs[0], (list, tuple)):
            outputs = outputs[0]

        if not isinstance(outputs, (list, tuple)) or len(outputs) < len(valid_idx):
            logger.warning(
                "ASR (nemo) returned %d items for %d inputs; dropping batch",
                len(outputs) if hasattr(outputs, "__len__") else -1,
                len(valid_idx),
            )
            return result

        for idx, item in zip(valid_idx, outputs):
            result[idx] = _hypothesis_text(item)
        return result
