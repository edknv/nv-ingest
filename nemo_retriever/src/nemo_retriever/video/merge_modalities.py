# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audio-anchored merge of video-frame OCR rows with audio-transcript rows.

The video pipeline produces two row kinds in parallel — image rows from the
frame extractor (with OCR'd text) and audio rows from the ASR chain. With this
merge step, each audio row absorbs the OCR text of the frame whose span
contains the audio's midpoint, producing a single ``video_segment`` row whose
time span (and embedding granularity) matches the audio's.

Frames whose span contains no overlapping audio are emitted as standalone
``video_frame`` rows so silent visual content isn't dropped.

Why audio-anchored: the recall harness scores against per-utterance gold spans,
so the row granularity must follow audio timestamps. A frame-anchored merge
would collapse many utterances into one frame-sized row and lose that
granularity.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


_TEXT_JOIN = "\n"


def _meta(row: Any) -> Dict[str, Any]:
    m = row.get("metadata") if hasattr(row, "get") else None
    return m if isinstance(m, dict) else {}


def _modality(row: Any) -> str:
    return str(_meta(row).get("modality") or "")


def _span_seconds(meta: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    s = meta.get("segment_start_seconds")
    e = meta.get("segment_end_seconds")
    if s is None or e is None:
        return None
    try:
        return float(s), float(e)
    except (TypeError, ValueError):
        return None


def _source_key(meta: Dict[str, Any]) -> str:
    return str(meta.get("source_path") or meta.get("source_video") or "")


def _pick_frame(
    audio_span: Tuple[float, float],
    frames_for_source: List[Tuple[int, float, float, str, pd.Series]],
) -> Optional[Tuple[int, str, pd.Series]]:
    """Pick the best frame for an audio span.

    Prefer the frame whose span contains the audio midpoint. If none does, fall
    back to the frame with the largest temporal overlap. Return ``None`` if no
    frame overlaps at all.
    """
    a_s, a_e = audio_span
    a_mid = (a_s + a_e) / 2.0
    for f_idx, f_s, f_e, ocr, f_row in frames_for_source:
        if f_s <= a_mid < f_e:
            return f_idx, ocr, f_row
    best: Optional[Tuple[int, str, pd.Series]] = None
    best_overlap = 0.0
    for f_idx, f_s, f_e, ocr, f_row in frames_for_source:
        overlap = max(0.0, min(a_e, f_e) - max(a_s, f_s))
        if overlap > best_overlap:
            best_overlap = overlap
            best = (f_idx, ocr, f_row)
    return best


def merge_video_frame_audio_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Audio-anchor merge: stamp each audio row with the matching frame's OCR.

    ``df`` is the union of the frame and audio branches in the video pipeline.
    Both kinds carry ``metadata.segment_start_seconds`` / ``segment_end_seconds``
    in wall-clock seconds.

    Returns a new DataFrame:
      - one ``video_segment`` row per audio input row, with ``text`` = audio
        transcript + ``\\n`` + OCR text of the frame whose window contains the
        audio's midpoint (or has the most overlap). The audio's span is kept,
        so embedding granularity follows the audio timestamps.
      - audio rows with no overlapping frame stay as standalone
        ``audio_segment`` rows.
      - frames that no audio row claimed are emitted as standalone
        ``video_frame`` rows, so silent visual content is preserved.
      - rows of any other modality pass through.
    """
    if df is None or len(df) == 0:
        return df
    if "metadata" not in df.columns:
        return df

    frames: List[pd.Series] = []
    audios: List[pd.Series] = []
    others: List[pd.Series] = []
    for _, row in df.iterrows():
        modality = _modality(row)
        if modality == "video_frame":
            frames.append(row)
        elif modality == "audio_segment":
            audios.append(row)
        else:
            others.append(row)

    if not audios:
        # Nothing to anchor to. Pass everything through unchanged.
        return df

    frames_by_source: Dict[str, List[Tuple[int, float, float, str, pd.Series]]] = {}
    for frame_idx, row in enumerate(frames):
        meta = _meta(row)
        span = _span_seconds(meta)
        if span is None:
            continue
        f_s, f_e = span
        ocr_text = str(row.get("text", "") or "")
        frames_by_source.setdefault(_source_key(meta), []).append((frame_idx, f_s, f_e, ocr_text, row))

    consumed_frame_indices: set[int] = set()
    out_records: List[Dict[str, Any]] = []

    for row in audios:
        meta = _meta(row)
        span = _span_seconds(meta)
        audio_text = str(row.get("text", "") or "").strip()
        if span is None:
            out_records.append(row.to_dict())
            continue

        candidates = frames_by_source.get(_source_key(meta), [])
        pick = _pick_frame(span, candidates)
        if pick is None:
            # No frame overlapped; keep the audio row as-is.
            out_records.append(row.to_dict())
            continue

        frame_idx, ocr_text, frame_row = pick
        consumed_frame_indices.add(frame_idx)
        ocr_text = (ocr_text or "").strip()

        merged_text_parts = [t for t in (audio_text, ocr_text) if t]
        merged_text = _TEXT_JOIN.join(merged_text_parts)

        new_meta = copy.deepcopy(meta)
        new_meta["modality"] = "video_segment"
        new_meta["_content_type"] = "video_segment"
        if audio_text:
            new_meta["audio_text"] = audio_text
        if ocr_text:
            new_meta["ocr_text"] = ocr_text
        f_meta = _meta(frame_row)
        f_seg_idx = f_meta.get("segment_index")
        if isinstance(f_seg_idx, (int, float)):
            new_meta["frame_segment_index"] = int(f_seg_idx)
        f_pos = f_meta.get("frame_position_seconds")
        if f_pos is not None:
            new_meta["frame_position_seconds"] = f_pos

        new_row = row.to_dict()
        new_row["metadata"] = new_meta
        new_row["text"] = merged_text
        new_row["_content_type"] = "video_segment"
        # Carry the frame's image / stored URI through for downstream
        # inspection. Embedding stays text-only (the merged text already has
        # the OCR concatenated), matching the harness's embed_modality=text
        # and the query-side encoder so retrieval is same-modal.
        for col in ("page_image", "images", "stored_image_uri", "_stored_image_uri", "_bbox_xyxy_norm"):
            if col in frame_row.index:
                value = frame_row[col]
                if value is not None and not (hasattr(value, "__len__") and len(value) == 0):
                    new_row[col] = value
        out_records.append(new_row)

    # Emit unmatched frames as standalone video_frame rows (silent visual content).
    # Embedded as text via their OCR text (matches the harness's text mode); the
    # frame image is preserved on the row for downstream inspection but is not
    # forced into the embedder.
    for frame_idx, row in enumerate(frames):
        if frame_idx in consumed_frame_indices:
            continue
        out_records.append(row.to_dict())

    for row in others:
        out_records.append(row.to_dict())

    if not out_records:
        return df.iloc[0:0]
    return pd.DataFrame(out_records).reset_index(drop=True)
