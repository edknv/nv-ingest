# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared audio I/O helpers for local ASR and related callers."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000


def load_audio_16k(path: str) -> Optional[np.ndarray]:
    """Load audio from *path* as mono 16 kHz float32, returning ``None`` on failure."""
    try:
        import soundfile as sf
    except ImportError:
        logger.warning("soundfile not installed; cannot load audio for local ASR.")
        return None

    try:
        data, sr = sf.read(path, dtype="float32")
    except Exception:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        path,
                        "-ar",
                        str(SAMPLING_RATE),
                        "-ac",
                        "1",
                        "-f",
                        "wav",
                        wav_path,
                    ],
                    check=True,
                    capture_output=True,
                )
                data, sr = sf.read(wav_path, dtype="float32")
            finally:
                Path(wav_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Failed to load or convert audio %s: %s", path, e)
            return None

    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLING_RATE:
        from scipy.signal import resample

        n = int(len(data) * SAMPLING_RATE / sr)
        data = resample(data, n).astype(np.float32)
    return data


def write_wav_16k(audio: np.ndarray, path: str) -> None:
    """Write a mono 16 kHz float32 array to *path* as a WAV file."""
    import soundfile as sf

    sf.write(path, audio, SAMPLING_RATE, subtype="FLOAT")
