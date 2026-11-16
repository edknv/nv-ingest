# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental bounded search policy and SystemOne HTTP decision adapter.

Controller excerpts are independent of the full candidates retained for selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any
from urllib.parse import urlparse

import httpx

from nemo_retriever._agentic.nemo_agent.llm.helpers import resolve_api_key


@dataclass(frozen=True)
class SystemOneConfig:
    """Opt-in retrieval controller settings; endpoint is a full POST URL."""

    endpoint: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    max_actions: int = 12
    max_expansions: int = 2
    max_thinks: int = 1
    max_subqueries: int = 4
    evidence_chars: int = 4000
    sufficiency_threshold: float = 0.7
    selection_pool_k: int | None = None
    timeout: float = 30.0

    def __post_init__(self) -> None:
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("system_one.endpoint must be an HTTP(S) URL without embedded credentials")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("system_one.model must be non-empty")
        for name in ("max_actions", "max_expansions", "max_thinks", "max_subqueries", "evidence_chars"):
            value = getattr(self, name)
            minimum = 0 if name in {"max_expansions", "max_thinks"} else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"system_one.{name} must be an integer >= {minimum}")
        if self.selection_pool_k is not None and (
            isinstance(self.selection_pool_k, bool)
            or not isinstance(self.selection_pool_k, int)
            or self.selection_pool_k < 1
        ):
            raise ValueError("system_one.selection_pool_k must be a positive integer or None")
        _probability(self.sufficiency_threshold)
        if (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("system_one.timeout must be finite and positive")


def _probability(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("SystemOne probabilities must be finite numbers in [0, 1]")
    return float(value)


@dataclass(frozen=True)
class Decision:
    action: str
    probabilities: dict[str, float]
    sufficiency: float
    usage: dict[str, Any] = field(default_factory=dict)
    model: str | None = None


class SystemOneClient:
    """Score an action Choice and evidence-sufficiency Noul in one request."""

    def __init__(self, config: SystemOneConfig) -> None:
        self.config = config

    def decide(self, state: dict[str, Any], actions: list[str]) -> Decision:
        descriptions = {
            "expand": "Generate new search queries to find missing supporting evidence.",
            "think": "Reason over the current evidence and identify unresolved gaps before searching.",
            "finish": "Stop searching and send the retained candidates to final selection.",
        }
        payload = {
            "model": self.config.model,
            "state": state,
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": "Which allowed action best advances retrieval for the original query?",
                    "criteria": {action: descriptions[action] for action in actions},
                },
                "sufficient": {
                    "type": "noul",
                    "instructions": "Does the retrieved evidence support all requirements of the original query?",
                },
            },
        }
        headers = {"Accept": "application/json"}
        token = resolve_api_key(self.config.api_key)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        with httpx.Client(timeout=self.config.timeout) as client:
            response = client.post(self.config.endpoint, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
        try:
            answer = body["answers"]["action"]
            sufficient = body["answers"]["sufficient"]
            if answer["type"] != "choice" or sufficient["type"] != "noul":
                raise ValueError("Unexpected SystemOne answer types")
            probabilities = {key: _probability(value) for key, value in answer["probabilities"].items()}
            if set(probabilities) != set(actions) or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-4):
                raise ValueError("SystemOne action distribution does not match allowed actions")
            action = answer["choice"]
            if action not in actions or probabilities[action] != max(probabilities.values()):
                raise ValueError("SystemOne choice does not match its action distribution")
            usage = body.get("usage") or {}
            if not isinstance(usage, dict):
                raise ValueError("SystemOne usage must be an object")
            return Decision(action, probabilities, _probability(sufficient["noul"]), usage, body.get("model"))
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("Malformed SystemOne decision response") from exc
