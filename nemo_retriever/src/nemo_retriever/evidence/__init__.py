# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recall-layer output shaping: turn retrieval hits into an evidence pack of
minimal spans with fidelity, citations, handles, confidence, and coverage."""
from __future__ import annotations

from nemo_retriever.evidence.pack import build_evidence_pack

__all__ = ["build_evidence_pack"]
