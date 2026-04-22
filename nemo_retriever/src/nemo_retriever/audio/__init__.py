# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Audio pipeline: media chunking (MediaChunkActor) and ASR (TranscriptionActor).

Provides the same semantics as nv-ingest-api dataloader + Parakeet for
batch, inprocess, fused, and online run modes.
"""

from __future__ import annotations

from nemo_retriever.audio.transcription_actor import TranscriptionActor, TranscriptionCPUActor, TranscriptionGPUActor
from nemo_retriever.audio.transcription_actor import transcription_params_from_env
from nemo_retriever.audio.chunk_actor import MediaChunkActor
from nemo_retriever.audio.media_interface import MediaInterface
from nemo_retriever.params import TranscriptionParams
from nemo_retriever.params import AudioChunkParams

from .cli import app

__all__ = [
    "TranscriptionActor",
    "TranscriptionCPUActor",
    "TranscriptionGPUActor",
    "TranscriptionParams",
    "app",
    "transcription_params_from_env",
    "AudioChunkParams",
    "MediaChunkActor",
    "MediaInterface",
]
