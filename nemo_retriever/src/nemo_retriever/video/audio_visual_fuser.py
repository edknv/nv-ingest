# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AudioVisualFuser: combine ASR rows and frame OCR/VLM rows into ``audio_visual`` rows.

Three modes via :attr:`AudioVisualFuseParams.mode`:

- ``per_utterance`` (default): each ASR utterance pairs with the single
  most-overlapping concurrent frame; output is
  ``"[AUDIO] <audio> | [VISUAL] <visual>"`` with the visual portion capped at
  :data:`FRAME_TEXT_MAX_CHARS`. The source audio row is dropped whenever a
  fused row was produced for its window so retrieval doesn't see two
  near-identical embeddings (audio-only and audio+visual) for the same
  utterance; audio rows whose window has no concurrent frame are preserved.

- ``per_scene``: one fused row per ``(source_path, scene_id)``; all audio
  rows whose window intersects the scene window plus all frame texts in
  the scene are concatenated. Visual content is capped at
  :attr:`AudioVisualFuseParams.scene_visual_max_chars` (default 800).
  Requires :attr:`VideoFrameParams.scene_detection.enabled` so frames
  carry a ``scene_id``; missing metadata raises ``ValueError``.

- ``per_sentence``: one fused row per ASR sentence with all overlapping
  frame texts concatenated. Requires ASR was run with
  ``segment_audio=True`` (the audio rows must carry ``segment_count``
  metadata). Emits one fused row per sentence with all overlapping frame
  texts concatenated (per-frame cap then total cap).

In all modes, ``video_frame`` rows are consumed by the fusion and not
passed through downstream — every visual moment that mattered is already
represented inside an ``audio_visual`` row.

Set :attr:`AudioVisualFuseParams.enabled` to ``False`` to skip fusion.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.ocr.shared import concat_with_passthrough
from nemo_retriever.params import AudioVisualFuseParams
from nemo_retriever.video import _content_types as _CT

logger = logging.getLogger(__name__)

#: Cap on the visual portion of a fused row — keeps a long slide-OCR blob
#: from dominating the embedding over the (typically much shorter) audio
#: transcript. The value (100) was selected empirically from a cap-sweep
#: against the video_retrieval_pipeline benchmark, where it maximised
#: recall@5 across 25/50/75/100/125/150/200.
FRAME_TEXT_MAX_CHARS: int = 100


def _row_content_type(row: Any) -> str:
    md = row.get("metadata") if isinstance(row, dict) else getattr(row, "metadata", None)
    if isinstance(md, dict):
        ct = md.get("_content_type")
        if isinstance(ct, str):
            return ct
    direct = row.get("_content_type") if isinstance(row, dict) else getattr(row, "_content_type", None)
    return str(direct) if isinstance(direct, str) else ""


def _row_segment_window(row: Any) -> tuple[float, float] | None:
    md = row.get("metadata") if isinstance(row, dict) else getattr(row, "metadata", None)
    if not isinstance(md, dict):
        return None
    try:
        start = float(md["segment_start_seconds"])
        end = float(md["segment_end_seconds"])
    except (KeyError, TypeError, ValueError):
        return None
    return start, end


def _row_scene_id(row: Any) -> Optional[int]:
    md = row.get("metadata") if isinstance(row, dict) else getattr(row, "metadata", None)
    if isinstance(md, dict):
        sid = md.get("scene_id")
        if sid is not None:
            try:
                return int(sid)
            except (TypeError, ValueError):
                return None
    return None


def _keep_upstream(row: Any, fused_window_keys: set[tuple[str, float, float]]) -> bool:
    """Drop ``video_frame`` rows and audio rows whose window already fused."""
    kind = _row_content_type(row)
    if kind == _CT.VIDEO_FRAME:
        return False
    if kind != _CT.AUDIO:
        return True
    window = _row_segment_window(row)
    if window is None:
        return True
    source = getattr(row, "source_path", None)
    if not isinstance(source, str):
        return True
    return (source, float(window[0]), float(window[1])) not in fused_window_keys


def _filter_upstream(batch_df: pd.DataFrame, fused_window_keys: set[tuple[str, float, float]]) -> pd.DataFrame:
    mask = [_keep_upstream(row, fused_window_keys) for row in batch_df.itertuples(index=False)]
    return batch_df[mask].reset_index(drop=True)


@designer_component(
    name="Audio-Visual Fuser",
    category="Video",
    compute="cpu",
    description="Fuses audio utterances with the most-representative concurrent video frame OCR text",
)
class AudioVisualFuser(AbstractOperator, CPUOperator):
    """Replace audio rows with fused audio+visual rows where frames overlap.

    Self-join semantics: needs *all* rows for a given source (audio
    utterances + frame OCR) to be co-located in a single batch. The
    ``REQUIRES_GLOBAL_BATCH`` marker tells :class:`RayDataExecutor` to
    force a single block + ``batch_size=None`` for this stage, so the
    fuser sees the whole dataset in one ``process()`` call.
    """

    #: Read by ``RayDataExecutor`` to force a global view (one block, one batch).
    REQUIRES_GLOBAL_BATCH: bool = True
    #: Per-source self-join — ``source_path`` co-locates all needed rows.
    GLOBAL_BATCH_GROUP_KEYS: tuple[str, ...] = ("source_path",)

    def __init__(self, params: AudioVisualFuseParams | None = None) -> None:
        super().__init__(params=params)
        self._params = params or AudioVisualFuseParams()

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, batch_df: Any, **kwargs: Any) -> Any:
        if not self._params.enabled:
            return batch_df
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return batch_df

        mode = self._params.mode
        if mode == "per_scene":
            return self._process_per_scene(batch_df)
        if mode == "per_sentence":
            return self._process_per_sentence(batch_df)
        return self._process_per_utterance(batch_df)

    def _process_per_utterance(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        # Bucket frame rows by source_path so we can self-join cheaply.
        # Each entry is (frame_start_seconds, frame_end_seconds, text). Storing
        # the window (rather than the midpoint timestamp) means dedup-merged
        # rows with wide windows still fuse with utterances inside that window.
        frames_by_source: Dict[str, List[tuple[float, float, str]]] = {}
        for row in batch_df.itertuples(index=False):
            if _row_content_type(row) != _CT.VIDEO_FRAME:
                continue
            window = _row_segment_window(row)
            text = getattr(row, "text", None)
            if window is None or not isinstance(text, str) or not text.strip():
                continue
            source = getattr(row, "source_path", None)
            if not isinstance(source, str):
                continue
            f_start, f_end = window
            frames_by_source.setdefault(source, []).append((float(f_start), float(f_end), text.strip()))

        if not frames_by_source:
            return _filter_upstream(batch_df, set())

        fused_rows: List[Dict[str, Any]] = []
        for row in batch_df.itertuples(index=False):
            if _row_content_type(row) != _CT.AUDIO:
                continue
            window = _row_segment_window(row)
            if window is None:
                continue
            u_start, u_end = window
            source = getattr(row, "source_path", None)
            if not isinstance(source, str):
                continue
            audio_text = getattr(row, "text", None)
            if not isinstance(audio_text, str) or not audio_text.strip():
                continue
            frame_entries = frames_by_source.get(source, [])
            # Window-overlap: a frame fuses when its visibility window
            # intersects the utterance window. Handles narrow per-frame windows
            # (single frame) and wide merged windows (text-dedup output) alike.
            concurrent_entries = [
                (f_start, f_end, text)
                for f_start, f_end, text in frame_entries
                if max(u_start, f_start) <= min(u_end, f_end)
            ]
            if not concurrent_entries:
                continue

            # Pick the frame whose midpoint is closest to the utterance
            # midpoint; tiebreak by longer OCR (favours slides that actually
            # carry content vs. near-blank ones).
            u_mid = (u_start + u_end) / 2.0
            best = min(
                concurrent_entries,
                key=lambda fe: (abs((fe[0] + fe[1]) / 2.0 - u_mid), -len(fe[2])),
            )
            visual_text = best[2]
            if len(visual_text) > FRAME_TEXT_MAX_CHARS:
                visual_text = visual_text[:FRAME_TEXT_MAX_CHARS].rstrip()

            row_dict = row._asdict() if hasattr(row, "_asdict") else dict(zip(batch_df.columns, row))
            metadata = dict(row_dict.get("metadata") or {})
            metadata.update(
                {
                    "segment_start_seconds": float(u_start),
                    "segment_end_seconds": float(u_end),
                    "modality": _CT.AUDIO_VISUAL,
                    "_content_type": _CT.AUDIO_VISUAL,
                    "fused_concurrent_total": len(concurrent_entries),
                }
            )
            fused_text = f"[AUDIO] {audio_text.strip()} | [VISUAL] {visual_text}".strip()
            fused_row = dict(row_dict)
            fused_row["text"] = fused_text
            fused_row["metadata"] = metadata
            fused_row["_content_type"] = _CT.AUDIO_VISUAL
            fused_rows.append(fused_row)

        if not fused_rows:
            return _filter_upstream(batch_df, set())

        fused_window_keys = {
            (
                str(row["source_path"] or ""),
                float(row["metadata"]["segment_start_seconds"]),
                float(row["metadata"]["segment_end_seconds"]),
            )
            for row in fused_rows
        }
        return concat_with_passthrough(
            pd.DataFrame(fused_rows),
            _filter_upstream(batch_df, fused_window_keys),
        )

    def _process_per_scene(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """One fused row per (source_path, scene_id), audio+visual concatenated."""
        from collections import defaultdict

        if "_content_type" not in batch_df.columns:
            return batch_df

        is_frame = batch_df["_content_type"].astype(str) == _CT.VIDEO_FRAME
        is_audio = batch_df["_content_type"].astype(str) == _CT.AUDIO

        if is_frame.any():
            for _, row in batch_df[is_frame].iterrows():
                if _row_scene_id(row) is None:
                    raise ValueError(
                        "per_scene mode requires scene_id on frame rows; "
                        "enable VideoFrameParams.scene_detection.enabled."
                    )

        cap = int(self._params.scene_visual_max_chars)

        frames_by_scene: dict[tuple[str, int], list[tuple[float, float, str]]] = defaultdict(list)
        scene_windows: dict[tuple[str, int], tuple[float, float]] = {}
        for _, row in batch_df[is_frame].iterrows():
            md = row.get("metadata") or {}
            source = str(row.get("source_path") or "")
            scene_id = int(md.get("scene_id"))
            text = str(row.get("text") or "").strip()
            window = _row_segment_window(row)
            if window is None or not text:
                continue
            frames_by_scene[(source, scene_id)].append((window[0], window[1], text))
            f_scene_start = (
                window[0]
                if md.get("scene_start_seconds") is None
                else float(md["scene_start_seconds"])
            )
            f_scene_end = (
                window[1]
                if md.get("scene_end_seconds") is None
                else float(md["scene_end_seconds"])
            )
            existing = scene_windows.get((source, scene_id))
            if existing is None:
                scene_windows[(source, scene_id)] = (f_scene_start, f_scene_end)
            else:
                scene_windows[(source, scene_id)] = (
                    min(existing[0], f_scene_start),
                    max(existing[1], f_scene_end),
                )

        fused_rows: list[dict[str, Any]] = []
        consumed_audio_keys: set[tuple[str, float, float]] = set()
        for (source, scene_id), entries in frames_by_scene.items():
            scene_start, scene_end = scene_windows[(source, scene_id)]
            audio_texts: list[tuple[float, str]] = []
            for _, row in batch_df[is_audio].iterrows():
                if str(row.get("source_path") or "") != source:
                    continue
                window = _row_segment_window(row)
                if window is None:
                    continue
                if max(window[0], scene_start) <= min(window[1], scene_end):
                    text = str(row.get("text") or "").strip()
                    if text:
                        audio_texts.append((window[0], text))
                        consumed_audio_keys.add((source, float(window[0]), float(window[1])))
            if not audio_texts:
                continue
            audio_texts.sort()
            visual_texts = sorted(entries)
            audio_blob = " ".join(t for _, t in audio_texts)
            visual_blob = " | ".join(t for _, _, t in visual_texts)
            if cap and len(visual_blob) > cap:
                visual_blob = visual_blob[:cap].rstrip()
            fused_text = f"[AUDIO] {audio_blob} | [VISUAL] {visual_blob}".strip()
            fused_rows.append(
                {
                    "path": source,
                    "source_path": source,
                    "_content_type": _CT.AUDIO_VISUAL,
                    "text": fused_text,
                    "metadata": {
                        "_content_type": _CT.AUDIO_VISUAL,
                        "modality": _CT.AUDIO_VISUAL,
                        "segment_start_seconds": float(scene_start),
                        "segment_end_seconds": float(scene_end),
                        "scene_id": int(scene_id),
                        "fusion_mode": "per_scene",
                    },
                }
            )

        kept = _filter_upstream(batch_df, consumed_audio_keys)
        if not fused_rows:
            return kept
        return concat_with_passthrough(pd.DataFrame(fused_rows), kept)

    def _process_per_sentence(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """One fused row per ASR sentence; all overlapping frames concatenated.

        Operates on rows produced by ``MediaASRActor`` with
        ``segment_audio=True``: each audio row is then a punctuation-bounded
        sentence carrying ``segment_count`` in its metadata.  Absence of
        the marker raises ``ValueError`` so misconfiguration fails loudly.
        """
        if "_content_type" not in batch_df.columns:
            return batch_df

        is_audio = batch_df["_content_type"].astype(str) == _CT.AUDIO
        is_frame = batch_df["_content_type"].astype(str) == _CT.VIDEO_FRAME

        for _, arow in batch_df[is_audio].iterrows():
            amd = arow.get("metadata") or {}
            if not isinstance(amd, dict) or amd.get("segment_count") is None:
                raise ValueError(
                    "per_sentence mode requires asr_params.segment_audio=True; "
                    "audio rows lack segment_count metadata."
                )

        per_frame_cap = int(self._params.per_sentence_per_frame_max_chars)
        total_cap = int(self._params.per_sentence_total_visual_max_chars)

        # Bucket frame entries by source_path: (window_start, window_end, text).
        frames_by_source: dict[str, list[tuple[float, float, str]]] = {}
        for _, row in batch_df[is_frame].iterrows():
            window = _row_segment_window(row)
            text = str(row.get("text") or "").strip()
            source = str(row.get("source_path") or "")
            if window is None or not text or not source:
                continue
            frames_by_source.setdefault(source, []).append((window[0], window[1], text))

        fused_rows: list[dict[str, Any]] = []
        consumed_audio_keys: set[tuple[str, float, float]] = set()
        for _, row in batch_df[is_audio].iterrows():
            window = _row_segment_window(row)
            source = str(row.get("source_path") or "")
            audio_text = str(row.get("text") or "").strip()
            if window is None or not source or not audio_text:
                continue
            u_start, u_end = window
            entries = frames_by_source.get(source, [])
            overlapping = sorted(
                (e for e in entries if max(u_start, e[0]) <= min(u_end, e[1])),
                key=lambda e: e[0],
            )
            if not overlapping:
                continue
            visual_pieces: list[str] = []
            for _, _, t in overlapping:
                clipped = t if len(t) <= per_frame_cap else t[:per_frame_cap].rstrip()
                visual_pieces.append(clipped)
            visual_blob = " | ".join(visual_pieces)
            if total_cap and len(visual_blob) > total_cap:
                visual_blob = visual_blob[:total_cap].rstrip()
            fused_text = f"[AUDIO] {audio_text} | [VISUAL] {visual_blob}".strip()
            md = dict(row.get("metadata") or {})
            md.update(
                {
                    "_content_type": _CT.AUDIO_VISUAL,
                    "modality": _CT.AUDIO_VISUAL,
                    "fusion_mode": "per_sentence",
                }
            )
            consumed_audio_keys.add((source, float(u_start), float(u_end)))
            fused_rows.append(
                {
                    "path": row.get("path"),
                    "source_path": source,
                    "_content_type": _CT.AUDIO_VISUAL,
                    "text": fused_text,
                    "metadata": md,
                }
            )

        kept = _filter_upstream(batch_df, consumed_audio_keys)
        if not fused_rows:
            return kept
        return concat_with_passthrough(pd.DataFrame(fused_rows), kept)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
