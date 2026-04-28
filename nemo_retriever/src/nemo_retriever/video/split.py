# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VideoSplitActor: one Ray stage that splits each video into both per-frame
rows and per-audio-chunk rows.

Emits a mixed-schema DataFrame: frame rows carry ``page_image`` (consumed
downstream by ``VideoFrameOCRActor``); audio-chunk rows carry ``bytes``
(consumed downstream by ``ASRActor``). Downstream OCR and ASR each pass
through rows that don't belong to them, so the whole pipeline stays a flat
sequential chain — no Ray fan-out / union — while exposing split, OCR and
ASR as three distinct progress bars. Mirrors the row-multiplying role of
``PDFSplitActor`` (PDF → per-page rows) for video inputs.
"""

from __future__ import annotations

import logging
from typing import Any, List

import pandas as pd

from nemo_retriever.audio.chunk_actor import AudioChunkActor
from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.graph.operator_archetype import ArchetypeOperator
from nemo_retriever.params import AudioChunkParams, VideoExtractParams
from nemo_retriever.video.frame_actor import VideoFrameExtractCPUActor

logger = logging.getLogger(__name__)


@designer_component(
    name="Video Split",
    category="Video",
    compute="cpu",
    description="Splits each video into frame rows + audio-chunk rows in one stage.",
    category_color="#ff6bbb",
)
class VideoSplitCPUActor(AbstractOperator, CPUOperator):
    """Split each ``path`` row into per-frame rows and per-audio-chunk rows.

    Both inner extractors are CPU-bound (ffmpeg subprocess), so this stage
    is CPU-only. Downstream OCR / ASR archetypes still resolve to their
    GPU variants when GPUs are available.
    """

    def __init__(
        self,
        video_params: VideoExtractParams | None = None,
        audio_chunk_params: AudioChunkParams | None = None,
    ) -> None:
        super().__init__(video_params=video_params, audio_chunk_params=audio_chunk_params)
        self._video_params = video_params or VideoExtractParams()
        # libsndfile (used by ASR) can't decode MP4 directly. Force the chunker
        # to demux video to MP3 before splitting so audio-only ASR works.
        chunk_params = audio_chunk_params or AudioChunkParams(
            split_type="time",
            split_interval=self._video_params.split_interval,
        )
        if hasattr(chunk_params, "model_copy") and not getattr(chunk_params, "audio_only", False):
            chunk_params = chunk_params.model_copy(update={"audio_only": True})
        self._audio_chunk_params = chunk_params

        self._frame_actor: VideoFrameExtractCPUActor | None = None
        self._chunk_actor: AudioChunkActor | None = None

    def _ensure_inner(self) -> None:
        if self._video_params.extract_frames and self._frame_actor is None:
            self._frame_actor = VideoFrameExtractCPUActor(params=self._video_params)
        if self._video_params.extract_audio and self._chunk_actor is None:
            self._chunk_actor = AudioChunkActor(params=self._audio_chunk_params)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, batch_df: Any, **kwargs: Any) -> pd.DataFrame:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return pd.DataFrame()
        self._ensure_inner()

        outputs: List[pd.DataFrame] = []
        if self._frame_actor is not None:
            try:
                frames = self._frame_actor.run(batch_df)
            except Exception as exc:
                logger.exception("Video frame extraction failed: %s", exc)
                frames = pd.DataFrame()
            if isinstance(frames, pd.DataFrame) and not frames.empty:
                outputs.append(frames)

        if self._chunk_actor is not None:
            try:
                chunks = self._chunk_actor.run(batch_df)
            except Exception as exc:
                logger.exception("Audio chunking failed: %s", exc)
                chunks = pd.DataFrame()
            if isinstance(chunks, pd.DataFrame) and not chunks.empty:
                outputs.append(chunks)

        if not outputs:
            return pd.DataFrame()
        return pd.concat(outputs, ignore_index=True, sort=False)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        return self.run(batch_df)


class VideoSplitActor(ArchetypeOperator):
    """Graph-facing archetype resolving to the CPU video-split actor."""

    _cpu_variant_class = VideoSplitCPUActor

    def __init__(
        self,
        video_params: VideoExtractParams | None = None,
        audio_chunk_params: AudioChunkParams | None = None,
    ) -> None:
        resolved_video = video_params or VideoExtractParams()
        super().__init__(video_params=resolved_video, audio_chunk_params=audio_chunk_params)
        self._video_params = resolved_video
        self._audio_chunk_params = audio_chunk_params
