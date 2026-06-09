# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pick a bounded, query-relevant window from a chunk so evidence items carry
minimal spans, not full chunks.

Uses a TOKEN-budget sliding window (not sentence-splitting): the returned span is
the contiguous run of <= ``max_tokens`` whitespace tokens with the highest overlap
with the query's content terms. Sentence-splitting fails on tabular/numeric content
(financial tables have almost no sentence punctuation, so a "sentence" is the whole
chunk); a token budget bounds the span regardless of punctuation."""
from __future__ import annotations

import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\w+")


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split(text.strip())]
    return [s for s in parts if s]


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2}


def _tok_matches(token: str, q_terms: set[str]) -> int:
    return 1 if any(w in q_terms for w in _WORD.findall(token.lower())) else 0


def extract_span(text: str, query: str, *, max_tokens: int = 120) -> str:
    """Return the <= ``max_tokens``-token window with the most query-term hits.

    Ties resolve to the earliest window (stable). Whole text is returned when it is
    already within budget. Window selection is O(n) via a sliding sum."""
    text = text.strip()
    if not text:
        return ""
    toks = text.split()
    if len(toks) <= max_tokens:
        return text

    q_terms = _terms(query)
    match = [_tok_matches(t, q_terms) for t in toks]

    w = max_tokens
    running = sum(match[:w])
    best_i, best_score = 0, running
    for i in range(1, len(toks) - w + 1):
        running += match[i + w - 1] - match[i - 1]
        if running > best_score:
            best_score, best_i = running, i
    return " ".join(toks[best_i : best_i + w])
