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


def retrieval_blocks(messages):
    """Yield structured retrieval blocks, excluding instructions and plain text."""
    for message in messages:
        if message.get("role") == "system" or not isinstance(message.get("content"), list):
            continue
        rank = 0
        for block in message["content"]:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            try:
                document = json.loads(block.get("text", ""))
            except (ValueError, TypeError):
                continue
            if not isinstance(document, dict) or "id" not in document or "score" not in document:
                continue
            if "text" in document and not isinstance(document["text"], str):
                continue
            yield message, block, document, rank
            rank += 1


def visible_document_texts(messages):
    """Full evidence still visible to the next completion, keyed by identity/text."""
    return {
        (doc["id"], doc["text"])
        for _, _, doc, _ in retrieval_blocks(messages)
        if doc.get("text") and "truncated" not in str(doc.get("note", "")).lower()
    }


def _remove_block(message, block):
    message["content"].remove(block)
    if not message["content"] and message.get("role") == "tool":
        message["content"] = [{"type": "text", "text": _BUDGET_NOTE}]


def _refresh_references(history):
    full_ids = {identifier for identifier, _ in visible_document_texts(history)}
    for _, block, document, _ in retrieval_blocks(history):
        if "text" not in document and document["id"] not in full_ids:
            document["note"] = _BUDGET_NOTE
            block["text"] = json.dumps(document, ensure_ascii=False)


def fit_context(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    budget: int,
    count_tokens: Callable[[list[dict[str, Any]], list[dict[str, Any]]], int],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Reduce redundant/lower-ranked evidence before evicting conversation turns.

    Preserve the question, instructions and tool transactions. Evict whole older
    turns only when their protected non-document content itself cannot fit.
    """
    history = deepcopy(messages)
    before = count_tokens(history, tools)
    removed = shortened = discarded = deduplicated = 0
    if before > budget:
        seen = set()
        full_ids = {identifier for identifier, _ in visible_document_texts(history)}
        for message, block, document, _ in list(retrieval_blocks(history)):
            identity = (document["id"], document.get("text"))
            duplicate = identity in seen or ("text" not in document and document["id"] in full_ids)
            if duplicate:
                _remove_block(message, block)
                deduplicated += 1
            else:
                seen.add(identity)
    while count_tokens(history, tools) > budget:
        protected = deepcopy(history)
        for message, block, _, _ in list(retrieval_blocks(protected)):
            _remove_block(message, block)
        if count_tokens(protected, tools) > budget:
            starts = [i for i, message in enumerate(history) if message.get("role") == "assistant"]
            if len(starts) >= 2:
                del history[starts[0] : starts[1]]
                removed += 1
                _refresh_references(history)
                continue
            raise ContextLimitError(
                "The context budget cannot fit the protected instructions, original question, "
                "tool schemas, and latest tool-call metadata. Increase context_window_tokens "
                "or reduce the request; no oversized completion was sent."
            )
        candidates = list(retrieval_blocks(history))
        if not candidates:
            raise ContextLimitError("The protected request cannot fit the context budget.")
        # Bootstrap evidence is expendable once research starts. Within tool
        # results, remove lower-ranked blocks before any turn's leading evidence.
        message, block, document, rank = min(
            candidates, key=lambda item: (item[0].get("role") != "user", item[3] == 0, -item[3])
        )
        original = document.get("text", "")
        if rank > 0 or not original:
            _remove_block(message, block)
            discarded += 1
        else:
            document["note"] = _BUDGET_NOTE

            def set_prefix(length):
                document["text"] = original[:length]
                block["text"] = json.dumps(document, ensure_ascii=False)

            set_prefix(0)
            _refresh_references(history)
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
                shortened += 1
            else:
                _remove_block(message, block)
                discarded += 1
        _refresh_references(history)
    return history, {
        "estimated_prompt_tokens_before": before,
        "estimated_prompt_tokens_after": count_tokens(history, tools),
        "removed_turns": removed,
        "shortened_documents": shortened,
        "discarded_documents": discarded,
        "deduplicated_documents": deduplicated,
    }
