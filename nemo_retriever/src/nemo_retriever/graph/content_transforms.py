# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-friendly content row transforms used by example pipelines."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from nemo_retriever.io.image_store import resolve_image_b64
from nemo_retriever.ocr.ocr import _crop_b64_image_by_norm_bbox
from nemo_retriever.params.models import IMAGE_MODALITIES

_CONTENT_COLUMNS = ("table", "chart", "infographic")


def _combine_text_with_content(row: Any, text_column: str, content_columns: Sequence[str]) -> str:
    """Combine page text with OCR content text for embedding."""
    parts = []
    base = row.get(text_column)
    if isinstance(base, str) and base.strip():
        parts.append(base.strip())
    for col in content_columns:
        content_list = row.get(col)
        if isinstance(content_list, list):
            for item in content_list:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
                    caption = item.get("caption", "")
                    if isinstance(caption, str) and caption.strip():
                        parts.append(caption.strip())
    return "\n\n".join(parts) if parts else ""


def _deep_copy_row(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Copy row dicts without sharing nested mutable values across exploded rows."""
    import copy

    out: Dict[str, Any] = {}
    for key, value in row_dict.items():
        if isinstance(value, (dict, list)):
            out[key] = copy.deepcopy(value)
        else:
            out[key] = value
    return out


def explode_content_to_rows(
    batch_df: Any,
    *,
    text_column: str = "text",
    content_columns: Sequence[str] = _CONTENT_COLUMNS,
    modality: str = "text",
    text_elements_modality: Optional[str] = None,
    structured_elements_modality: Optional[str] = None,
) -> Any:
    """Expand each page row into multiple rows for per-element embedding."""
    text_mod = text_elements_modality or modality
    struct_mod = structured_elements_modality or modality

    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df

    any_images = text_mod in IMAGE_MODALITIES or struct_mod in IMAGE_MODALITIES

    if not any(column in batch_df.columns for column in content_columns):
        batch_df = batch_df.copy()
        if text_mod in IMAGE_MODALITIES and "page_image" in batch_df.columns:
            batch_df["_image_b64"] = batch_df["page_image"].apply(
                lambda page_image: resolve_image_b64(page_image) if isinstance(page_image, dict) else None
            )
        if "page_image" in batch_df.columns:
            batch_df["_stored_image_uri"] = batch_df["page_image"].apply(
                lambda page_image: page_image.get("stored_image_uri") if isinstance(page_image, dict) else None
            )
        batch_df["_embed_modality"] = text_mod
        return batch_df

    new_rows: List[Dict[str, Any]] = []
    for _, row in batch_df.iterrows():
        row_dict = row.to_dict()
        exploded_any = False

        page_image = row_dict.get("page_image")
        page_image_b64: Optional[str] = None
        page_stored_uri: Optional[str] = None
        if isinstance(page_image, dict):
            page_stored_uri = page_image.get("stored_image_uri")
            if any_images:
                page_image_b64 = resolve_image_b64(page_image)

        page_text = row_dict.get(text_column)
        if isinstance(page_text, str) and page_text.strip():
            page_row = _deep_copy_row(row_dict)
            page_row["_embed_modality"] = text_mod
            page_row["_content_type"] = "text"
            if text_mod in IMAGE_MODALITIES:
                page_row["_image_b64"] = page_image_b64
            page_row["_stored_image_uri"] = page_stored_uri
            page_row["_bbox_xyxy_norm"] = None
            new_rows.append(page_row)
            exploded_any = True

        for column in content_columns:
            content_list = row_dict.get(column)
            if not isinstance(content_list, list):
                continue
            for item in content_list:
                if not isinstance(item, dict):
                    continue
                item_b64 = resolve_image_b64(item) if struct_mod in IMAGE_MODALITIES else None
                # Emit rows for text and (optionally) caption fields.
                for field, content_type in [("text", column), ("caption", f"{column}_caption")]:
                    value = item.get(field, "")
                    if not isinstance(value, str) or not value.strip():
                        continue
                    content_row = _deep_copy_row(row_dict)
                    content_row[text_column] = value.strip()
                    content_row["_embed_modality"] = struct_mod
                    content_row["_content_type"] = content_type
                    if struct_mod in IMAGE_MODALITIES:
                        if item_b64:
                            content_row["_image_b64"] = item_b64
                        elif page_image_b64:
                            bbox = item.get("bbox_xyxy_norm")
                            if bbox and len(bbox) == 4:
                                cropped_b64, _ = _crop_b64_image_by_norm_bbox(page_image_b64, bbox_xyxy_norm=bbox)
                                content_row["_image_b64"] = cropped_b64
                            else:
                                content_row["_image_b64"] = page_image_b64
                        else:
                            content_row["_image_b64"] = None
                    content_row["_stored_image_uri"] = item.get("stored_image_uri") or page_stored_uri
                    content_row["_bbox_xyxy_norm"] = item.get("bbox_xyxy_norm")
                    new_rows.append(content_row)
                    exploded_any = True

        if not exploded_any:
            preserved = _deep_copy_row(row_dict)
            preserved["_embed_modality"] = text_mod
            preserved["_content_type"] = "text"
            if text_mod in IMAGE_MODALITIES:
                preserved["_image_b64"] = page_image_b64
            preserved["_stored_image_uri"] = page_stored_uri
            preserved["_bbox_xyxy_norm"] = None
            new_rows.append(preserved)

    return pd.DataFrame(new_rows).reset_index(drop=True)


def collapse_content_to_page_rows(
    batch_df: Any,
    *,
    text_column: str = "text",
    content_columns: Sequence[str] = _CONTENT_COLUMNS,
    modality: str = "text",
) -> Any:
    """Collapse each page into a single row for page-level embedding."""
    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df

    batch_df = batch_df.copy()
    batch_df[text_column] = batch_df.apply(
        lambda row: _combine_text_with_content(row, text_column, content_columns),
        axis=1,
    )

    if modality in IMAGE_MODALITIES:
        if "page_image" in batch_df.columns:
            batch_df["_image_b64"] = batch_df["page_image"].apply(
                lambda page_image: resolve_image_b64(page_image) if isinstance(page_image, dict) else None
            )
        else:
            batch_df["_image_b64"] = None

    if "page_image" in batch_df.columns:
        batch_df["_stored_image_uri"] = batch_df["page_image"].apply(
            lambda page_image: page_image.get("stored_image_uri") if isinstance(page_image, dict) else None
        )

    batch_df["_embed_modality"] = modality
    return batch_df


def merge_video_frame_text_into_audio_rows(batch_df: Any) -> Any:
    """Concat each video frame's OCR text into the audio row whose window contains
    the frame, then drop the standalone frame rows.

    Frame rows (with a ``page_image`` dict) carry per-segment time windows
    (e.g. 60-75s); audio rows (with ``bytes`` and a transcript) carry per-utterance
    windows from ASR fan-out (e.g. 62.3-64.1s). For ``recall_match_mode=audio_segment``
    the matcher checks midpoint-in-window with a small tolerance, so frame rows
    seldom score on their own. Folding their OCR text into the temporally
    overlapping audio row preserves the OCR signal at the audio's fine-grained
    timestamps, freeing top-K slots that would otherwise hold un-matchable frames.

    Mixed-mode batches (e.g. heterogeneous "auto" with PDFs alongside videos)
    are safe: rows with no ``page_image`` and no time window are passed through
    unchanged. Within one batch, rows are grouped by ``metadata.source_path``,
    which both extractors set to the original video path.
    """
    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df

    if "page_image" not in batch_df.columns:
        return batch_df  # nothing to merge — no frame rows present

    has_frame_image = batch_df["page_image"].apply(lambda v: isinstance(v, dict))
    if not has_frame_image.any():
        return batch_df

    frames = batch_df[has_frame_image]
    other = batch_df[~has_frame_image]

    # Group frames by source video for O(1) lookup per audio row.
    frames_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in frames.iterrows():
        meta = row.get("metadata")
        if not isinstance(meta, dict):
            continue
        source = meta.get("source_path") or meta.get("source_video")
        if not source:
            continue
        frames_by_source.setdefault(str(source), []).append(
            {
                "text": row.get("text") or "",
                "start": meta.get("segment_start_seconds"),
                "end": meta.get("segment_end_seconds"),
            }
        )

    if not frames_by_source:
        # All frame rows lacked usable metadata — pass everything through unchanged.
        return batch_df

    new_texts: List[str] = []
    for _, row in other.iterrows():
        existing_text = row.get("text") or ""
        meta = row.get("metadata")
        if not isinstance(meta, dict):
            new_texts.append(existing_text)
            continue
        source = meta.get("source_path") or meta.get("source_video")
        a_start = meta.get("segment_start_seconds")
        a_end = meta.get("segment_end_seconds")
        if not source or a_start is None or a_end is None:
            new_texts.append(existing_text)
            continue

        try:
            a_mid = (float(a_start) + float(a_end)) / 2.0
        except (TypeError, ValueError):
            new_texts.append(existing_text)
            continue

        matches: List[str] = []
        for f in frames_by_source.get(str(source), ()):
            f_start, f_end = f["start"], f["end"]
            if f_start is None or f_end is None:
                continue
            try:
                if float(f_start) <= a_mid <= float(f_end):
                    text = (f["text"] or "").strip()
                    if text:
                        matches.append(text)
            except (TypeError, ValueError):
                continue

        if matches:
            ocr_blob = "\n".join(matches)
            new_texts.append(f"{existing_text}\n\n{ocr_blob}" if existing_text else ocr_blob)
        else:
            new_texts.append(existing_text)

    if other.empty:
        # Frames-only batch (audio disabled): drop frames since they can't be
        # merged into anything. Caller can keep frames=on if they want frame-only
        # rows; typical usage of this UDF is with both modalities enabled.
        return batch_df.iloc[:0]

    merged = other.copy()
    merged["text"] = new_texts
    return merged.reset_index(drop=True)
