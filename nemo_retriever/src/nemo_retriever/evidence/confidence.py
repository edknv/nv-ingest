# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cheap, deterministic per-item confidence: normalized rerank score plus a
corroboration bonus. No LLM. The score->confidence mapping is the first
calibration knob (see spec risks)."""
from __future__ import annotations

from typing import Sequence

_W_SCORE = 0.7
_W_CORROB = 0.3


def normalize_scores(scores: Sequence[float]) -> list[float]:
    """Min-max normalize to [0, 1]. All-equal (incl. single) maps to 1.0."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def corroboration(idx: int, keys: Sequence[str]) -> float:
    """Fraction of the *other* hits that share this hit's page key."""
    others = [k for j, k in enumerate(keys) if j != idx]
    if not others:
        return 0.0
    return sum(1 for k in others if k == keys[idx]) / len(others)


def hit_confidence(norm_score: float, corrob: float) -> float:
    return round(_W_SCORE * norm_score + _W_CORROB * corrob, 3)
