# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage signals over a retrieved hit set: exact-term thin spots and a weak
flag. Lets the consumer tell 'broaden the search' from 'out of corpus'."""
from __future__ import annotations

import re
from typing import Any, Sequence

_WORD = re.compile(r"\w+")
_WEAK_THRESHOLD = 0.15


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2}


def coverage(texts: Sequence[str], query: str, *, best_norm_score: float) -> dict[str, Any]:
    q_terms = _terms(query)
    corpus_terms: set[str] = set()
    for t in texts:
        corpus_terms |= _terms(t)
    thin = sorted(q_terms - corpus_terms)
    weak = (len(texts) == 0) or (best_norm_score < _WEAK_THRESHOLD)
    return {"thin_spots": thin, "weak": bool(weak), "n_hits": len(texts)}
