# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assemble retrieval hits into the recall-layer evidence pack: minimal span,
fidelity, modality, citation, deterministic handle, confidence, plus coverage."""
from __future__ import annotations

import os
from typing import Any, Sequence

from nemo_retriever.evidence.confidence import corroboration, hit_confidence, normalize_scores
from nemo_retriever.evidence.coverage import coverage
from nemo_retriever.evidence.spans import extract_span
from nemo_retriever.vdb.records import _derive_fidelity


def _page_key(hit: dict[str, Any]) -> str:
    # Resolve source the same way the citation/handle do, so corroboration
    # groups hits consistently when only ``source`` (not ``source_id``) is set.
    source = hit.get("source_id") or hit.get("source") or ""
    return f"{source}:{hit.get('page_number', '')}"


def build_evidence_pack(
    hits: Sequence[dict[str, Any]],
    query: str,
    *,
    max_tokens: int = 120,
) -> dict[str, Any]:
    """Shape ranked ``RetrievalHit`` dicts (highest-scored first) into the pack.
    Pure: no retrieval or I/O. ``hits`` must already be in rank order."""
    # Prefer the rerank ``_score`` (higher = better). ``_distance`` is only a
    # degraded fallback: it is a similarity *distance* (lower = better), so when
    # it is the only signal the normalized confidence is unreliable.
    scores = [float(h.get("_score", h.get("_distance", 0.0)) or 0.0) for h in hits]
    norm = normalize_scores(scores)
    keys = [_page_key(h) for h in hits]

    evidence: list[dict[str, Any]] = []
    for i, hit in enumerate(hits):
        text = str(hit.get("text", ""))
        metadata = hit.get("metadata") or {}
        content_metadata = metadata.get("content_metadata") or {}
        # Resolve modality/fidelity the way the existing _evidence_item does: the
        # normalized hit carries them in metadata ("type"/"fidelity"), not under
        # a top-level content_type. Fall back to deriving fidelity from modality.
        modality = hit.get("content_type") or metadata.get("type") or metadata.get("content_type")
        fidelity = metadata.get("fidelity") or _derive_fidelity(modality, metadata, content_metadata)
        src_raw = str(hit.get("source_id") or hit.get("source") or "")
        source = os.path.basename(src_raw)
        if source.lower().endswith(".pdf"):
            source = source[:-4]
        page = hit.get("page_number")
        if page is None:
            page = metadata.get("page_number")
        evidence.append(
            {
                "span": extract_span(text, query, max_tokens=max_tokens),
                "fidelity": fidelity,
                "modality": modality,
                "citation": {"source": source, "locator": f"p.{page}" if page is not None else ""},
                "handle": f"{source}|{page}|{i}",
                "confidence": hit_confidence(norm[i], corroboration(i, keys)),
            }
        )

    best_norm = norm[0] if norm else 0.0
    cov = coverage([str(h.get("text", "")) for h in hits], query, best_norm_score=best_norm)
    return {"evidence": evidence, "coverage": cov}
