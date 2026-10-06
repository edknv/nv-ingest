# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Bound native agent requests without rewriting the query or synthesizing evidence."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Callable

from .llm.errors import ContextLimitError

_BUDGET_NOTE = "Document text truncated to fit the context budget; retrieve it again for evidence."


def validate_context_budget(window: int | None, output: int, margin: int) -> None:
    for name, value, minimum in (
        ("context_window_tokens", window, 1),
        ("context_output_tokens", output, 1),
        ("context_safety_margin_tokens", margin, 0),
    ):
        if value is None and name == "context_window_tokens":
            continue
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if window is not None and window <= output + margin:
        raise ValueError("context_window_tokens must exceed context_output_tokens + context_safety_margin_tokens")


def estimate_prompt_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> int:
    """Conservative text estimate: UTF-8 bytes plus protocol framing, not a tokenizer."""
    wire = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") not in ("text", "input_text") for block in content
        ):
            raise ContextLimitError("Context budgeting needs a model-specific token counter for non-text content.")
        wire.append({key: value for key, value in message.items() if not key.startswith("__")})
    return (
        len(json.dumps({"messages": wire, "tools": tools}, ensure_ascii=False).encode("utf-8")) + 64 * len(wire) + 256
    )


def fit_context(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    budget: int,
    count_tokens: Callable[[list[dict[str, Any]], list[dict[str, Any]]], int],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep the original instructions/question and the latest complete tool transaction.

    Remove older assistant turns together with every tool response and auto-continue
    message belonging to them. Retrieved document JSON is shortened only in its text
    field; identifiers and scores of retained documents stay intact. If even an
    empty document cannot fit, discard its block. Tool schemas and tool-call
    arguments always remain intact.
    """
    history = deepcopy(messages)
    before = count_tokens(history, tools)
    removed = shortened = discarded = 0
    while count_tokens(history, tools) > budget:
        starts = [index for index, message in enumerate(history) if message.get("role") == "assistant"]
        if len(starts) < 2:
            break
        del history[starts[0] : starts[1]]
        removed += 1
    while count_tokens(history, tools) > budget:
        candidates = []
        for message in history:
            if message.get("role") == "system":
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in reversed(content):
                if not isinstance(block, dict) or block.get("type") != "text":
                    continue
                try:
                    document = json.loads(block.get("text", ""))
                except (ValueError, TypeError):
                    continue
                if (
                    isinstance(document, dict)
                    and "id" in document
                    and "score" in document
                    and isinstance(document.get("text"), str)
                ):
                    candidates.append((content, block, document))
        if not candidates:
            raise ContextLimitError(
                "The context budget cannot fit the protected instructions, original question, "
                "tool schemas, and latest tool-call metadata. Increase context_window_tokens "
                "or reduce the request; no oversized completion was sent."
            )
        content, block, document = candidates[0]
        original = document["text"]
        document["note"] = _BUDGET_NOTE

        def set_prefix(length: int) -> None:
            document["text"] = original[:length]
            block["text"] = json.dumps(document, ensure_ascii=False)

        set_prefix(0)
        if count_tokens(history, tools) <= budget:
            low, high = 0, len(original)
            while low < high:
                middle = (low + high + 1) // 2
                set_prefix(middle)
                if count_tokens(history, tools) <= budget:
                    low = middle
                else:
                    high = middle - 1
            set_prefix(low)
        else:
            content.remove(block)
            discarded += 1
        shortened += 1
    return history, {
        "estimated_prompt_tokens_before": before,
        "estimated_prompt_tokens_after": count_tokens(history, tools),
        "removed_turns": removed,
        "shortened_documents": shortened - discarded,
        "discarded_documents": discarded,
    }
