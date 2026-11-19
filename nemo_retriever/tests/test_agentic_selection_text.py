# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Preserving selection excerpts must not change the retrieval conversation."""

import json

import pandas as pd
import pytest

from nemo_retriever.operators.graph_ops.react_agent_operator import ReActAgentOperator
from nemo_retriever.operators.graph_ops.rrf_aggregator_operator import RRFAggregatorOperator
from nemo_retriever.query.agentic import AgenticRetrievalConfig


def test_distinct_excerpts_preserve_ranking_and_skip_synthetic_final_rows():
    rows = pd.DataFrame(
        {
            "query_id": ["q"] * 6,
            "query_text": ["question"] * 6,
            "step_idx": [0, 1, 2, 3, 4, 5],
            "doc_id": ["a", "a", "a", "a", "b", "a"],
            "text": ["first", " second ", "second", "", "other", "synthetic"],
            "rank": [1] * 6,
            "is_final_result": [False] * 5 + [True],
            "has_valid_final_results": [True] * 6,
        }
    )
    original = RRFAggregatorOperator().run(rows)
    enriched = RRFAggregatorOperator(text_policy="all_retrieved").run(rows)
    pd.testing.assert_frame_equal(original.drop(columns="text"), enriched.drop(columns="text"))
    assert original.iloc[0]["text"] == "first"
    assert enriched.iloc[0]["text"] == "first\n\n[Additional retrieved excerpt]\n\nsecond"
    assert enriched.iloc[1]["text"] == "other"


def _reply(name, arguments, call):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call-{call}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def test_repeated_document_preserves_selection_excerpt_without_changing_loop_messages(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def run(policy):
        requests = []
        fetches = []

        def retrieve(query, top_k):
            fetches.append(query)
            return [{"id": "a", "score": 1.0, "text": "first" if len(fetches) == 1 else "later-excerpt"}]

        def complete(**kwargs):
            requests.append(json.loads(json.dumps(kwargs["messages"])))
            if len(requests) == 1:
                return _reply("retrieve", {"query": "follow-up", "top_k": 1}, len(requests))
            return _reply("final_results", {"doc_ids": ["a"], "search_successful": "true"}, len(requests))

        operator = ReActAgentOperator(
            llm_model="test-model",
            retriever_fn=retrieve,
            retriever_top_k=1,
            target_top_k=1,
            max_steps=3,
            chat_completion_fn=complete,
            selection_text_policy=policy,
        )
        rows = operator.run(pd.DataFrame({"query_id": ["q"], "query_text": ["question"]}))
        result = RRFAggregatorOperator(text_policy=policy).run(rows)
        return requests, fetches, result

    original, original_fetches, original_result = run("first")
    preserved, preserved_fetches, preserved_result = run("all_retrieved")
    assert len(original) == 2
    assert len(original_fetches) == 2
    assert original_fetches == preserved_fetches
    assert original == preserved
    assert "_selection_text" not in json.dumps(preserved)
    assert "later-excerpt" not in json.dumps(preserved)
    assert original_result.iloc[0]["text"] == "first"
    assert preserved_result.iloc[0]["text"] == "first\n\n[Additional retrieved excerpt]\n\nlater-excerpt"
    pd.testing.assert_frame_equal(original_result.drop(columns="text"), preserved_result.drop(columns="text"))


@pytest.mark.parametrize("value", ["unknown", None, [], {}])
def test_invalid_selection_text_policy(value):
    with pytest.raises(ValueError, match="selection_text_policy"):
        AgenticRetrievalConfig(selection_text_policy=value)


def test_selection_text_policy_defaults_to_first():
    assert AgenticRetrievalConfig().selection_text_policy == "first"
