# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Selection must return distinct candidates or use the existing correction policy."""

import json

import pandas as pd
import pytest

from nemo_retriever._agentic.nemo_agent.tools import LogSelectedDocs, ToolError
from nemo_retriever.operators.graph_ops.selection_agent_operator import SelectionAgentOperator


@pytest.mark.parametrize("include_msg", [False, True])
def test_selected_documents_reject_duplicates_and_preserve_valid_order(include_msg):
    tool = LogSelectedDocs(top_k=3, candidate_docids=["a", "b", "c"], include_msg=include_msg)
    with pytest.raises(ToolError, match="3 distinct documents"):
        tool._validate_payload(["a", "b", "a"], message="selected evidence")
    assert tool._validate_payload(["c", "b", "a"], message="selected evidence")["doc_ids"] == ["c", "b", "a"]


@pytest.mark.parametrize("corrected", [True, False])
def test_duplicate_selection_receives_feedback_then_corrects_or_falls_back(tmp_path, monkeypatch, corrected):
    monkeypatch.chdir(tmp_path)
    requests = []

    def complete(**kwargs):
        requests.append(json.loads(json.dumps(kwargs["messages"])))
        doc_ids = ["b", "a"] if corrected and len(requests) == 2 else ["a", "a"]
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"call-{len(requests)}",
                                "type": "function",
                                "function": {
                                    "name": "log_selected_documents",
                                    "arguments": json.dumps({"doc_ids": doc_ids}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    operator = SelectionAgentOperator(
        llm_model="test-model",
        top_k=2,
        max_steps=2,
        chat_completion_fn=complete,
        length_retry_extra_body={"max_tokens": 32768},
    )
    candidates = pd.DataFrame(
        {
            "query_id": ["q", "q"],
            "query_text": ["question", "question"],
            "doc_id": ["a", "b"],
            "text": ["evidence a", "evidence b"],
            "rrf_score": [0.9, 0.8],
        }
    )
    result = operator.run(candidates)

    assert len(requests) == 2
    assert requests[1][:2] == requests[0][:2]
    assert "duplicate document IDs" in json.dumps(requests[1])
    assert result["doc_id"].tolist() == (["b", "a"] if corrected else ["a", "b"])
    assert set(result["result_source"]) == ({"selection_agent"} if corrected else {"rrf"})
