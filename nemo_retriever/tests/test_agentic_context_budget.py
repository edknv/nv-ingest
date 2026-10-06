# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy

import pytest
from nemo_retriever._agentic.nemo_agent import Agent, AgentConfig, create_retrieve_tool
from nemo_retriever._agentic.nemo_agent.context_budget import (
    estimate_prompt_tokens,
    fit_context,
)
from nemo_retriever._agentic.nemo_agent.llm import (
    ContextLimitError,
    create_llm,
    create_llm_config,
)


def _document(identifier, text):
    return {"type": "text", "text": json.dumps({"id": identifier, "score": 1, "text": text}, ensure_ascii=False)}


def _response(name, arguments):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }


def test_oversized_bootstrap_preserves_query_schemas_and_document_identity():
    history = [
        {"role": "system", "content": "Follow instructions."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Query: original question"},
                _document("d1", 'é"\\ evidence ' * 10000),
            ],
        },
    ]
    original = deepcopy(history)
    tools = [{"type": "function", "function": {"name": "retrieve", "parameters": {"type": "object"}}}]
    fitted, metrics = fit_context(history, tools, 3000, estimate_prompt_tokens)
    assert estimate_prompt_tokens(fitted, tools) <= 3000
    assert fitted[0] == history[0]
    assert fitted[1]["content"][0] == history[1]["content"][0]
    document = json.loads(fitted[1]["content"][1]["text"])
    assert document["id"] == "d1" and document["score"] == 1
    assert "truncated" in document["note"]
    assert original[1]["content"][1]["text"] != fitted[1]["content"][1]["text"]
    assert original == history
    assert metrics["shortened_documents"] == 1


def test_eviction_keeps_parallel_tool_responses_paired_with_latest_assistant():
    history = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "old" * 3000, "tool_calls": [{"id": "old"}]},
        {"role": "tool", "tool_call_id": "old", "content": "old output"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "new-1"}, {"id": "new-2"}]},
        {"role": "tool", "tool_call_id": "new-1", "content": [_document("d1", "new evidence")]},
        {"role": "tool", "tool_call_id": "new-2", "content": "second output"},
    ]
    fitted, metrics = fit_context(history, [], 2500, estimate_prompt_tokens)
    assert fitted == history[:2] + history[4:]
    assert metrics["removed_turns"] == 1
    assert {m["tool_call_id"] for m in fitted if m["role"] == "tool"} == {"new-1", "new-2"}


def test_old_evidence_is_shortened_before_recent_evidence():
    history = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": [{"type": "text", "text": "question"}, _document("old", "O" * 10000)]},
        {"role": "assistant", "tool_calls": [{"id": "new"}], "content": ""},
        {"role": "tool", "tool_call_id": "new", "content": [_document("recent", "important" * 200)]},
    ]
    fitted, _ = fit_context(history, [], 3500, estimate_prompt_tokens)
    assert json.loads(fitted[-1]["content"][0]["text"])["text"] == "important" * 200
    assert len(json.loads(fitted[1]["content"][1]["text"])["text"]) < 10000
    assert estimate_prompt_tokens(fitted, []) <= 3500


def test_default_counter_includes_tool_schemas_and_rejects_unknown_multimodal_cost():
    messages = [{"role": "user", "content": "query"}]
    assert estimate_prompt_tokens(messages, [{"name": "tool", "schema": "x" * 1000}]) > estimate_prompt_tokens(
        messages, []
    )
    with pytest.raises(ContextLimitError, match="model-specific token counter"):
        estimate_prompt_tokens(
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "image"}}]}], []
        )


def test_custom_counter_is_used_for_every_budget_check():
    history = [{"role": "user", "content": [{"type": "text", "text": "question"}, _document("d", "0123456789" * 100)]}]

    def count(messages, tools):
        return sum(len(str(message.get("content", ""))) for message in messages) + len(str(tools))

    fitted, metrics = fit_context(history, [], 400, count)
    assert count(fitted, []) <= 400
    assert metrics["estimated_prompt_tokens_after"] == count(fitted, [])


@pytest.mark.parametrize("window,output,margin", [(True, 1, 0), (1000, True, 0), (1000, 1, -1)])
def test_public_and_private_configs_reject_invalid_budgets(window, output, margin):
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    for config in (AgentConfig, AgenticRetrievalConfig):
        with pytest.raises(ValueError):
            config(context_window_tokens=window, context_output_tokens=output, context_safety_margin_tokens=margin)


@pytest.mark.parametrize("maximum,window", [(128, 4096), ("128", 4096), (8192, 4609)])
def test_small_window_uses_effective_completion_cap(maximum, window):
    from nemo_retriever.operators.graph_ops.react_agent_operator import ReActAgentOperator
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    cfg = AgenticRetrievalConfig(context_window_tokens=window, max_tokens=maximum)
    assert cfg.context_output_tokens == 4096
    operator = ReActAgentOperator(
        llm_model="test",
        retriever_fn=lambda q, k: [],
        context_window_tokens=window,
        max_tokens=cfg.max_tokens,
    )
    agent = Agent(
        config=AgentConfig(context_window_tokens=window),
        llm=operator._build_llm(),
        retrieve_tool=create_retrieve_tool("default", lambda q, k: []),
    )
    assert agent.config.context_output_tokens == 4096
    assert agent.llm.config.max_completion_tokens == int(maximum)


@pytest.mark.parametrize("maximum,window", [(None, 4096), (128, 640), (8192, 4608)])
def test_unfit_effective_reservation_rejected_at_public_and_agent_construction(maximum, window):
    from nemo_retriever.operators.graph_ops.react_agent_operator import ReActAgentOperator
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    with pytest.raises(ValueError, match="effective output reservation"):
        AgenticRetrievalConfig(context_window_tokens=window, max_tokens=maximum)
    with pytest.raises(ValueError, match="effective output reservation"):
        ReActAgentOperator(
            llm_model="test",
            retriever_fn=lambda q, k: [],
            context_window_tokens=window,
            max_tokens=maximum,
        )
    with pytest.raises(ValueError, match="effective output reservation"):
        Agent(
            config=AgentConfig(context_window_tokens=window),
            llm=create_llm(
                create_llm_config("callable", model="test", max_completion_tokens=maximum),
                completion_fn=lambda **kw: {},
            ),
            retrieve_tool=create_retrieve_tool("default", lambda q, k: []),
        )


@pytest.mark.parametrize("maximum", [None, 128])
def test_native_loop_bounds_requests_restores_retrieved_evidence_and_reserves_output(maximum):
    seen = []

    def completion(**kwargs):
        seen.append(deepcopy(kwargs))
        if len(seen) == 1:
            return _response("retrieve", {"query": "again", "top_k": 1})
        return _response("log_answer", {"answer": "42", "citations": ["d1"]})

    llm = create_llm(
        create_llm_config("callable", model="test", max_completion_tokens=maximum), completion_fn=completion
    )
    config = AgentConfig(
        mode="answer",
        context_window_tokens=20000,
        context_output_tokens=256,
        context_safety_margin_tokens=128,
        max_steps=2,
        user_msg_type="with_results",
        end_tool_with_msg=False,
    )
    agent = Agent(
        config=config,
        llm=llm,
        retrieve_tool=create_retrieve_tool(
            "default", lambda query, top_k: [{"id": "d1", "score": 1, "text": "Evidence " * 30000}]
        ),
    )
    result = agent.run_sync("original question", query_id="q")
    assert result.succeeded and result.answer == "42" and result.citations == ["d1"]
    assert len(seen) == 2
    reserve = maximum or 256
    for call in seen:
        assert call["max_tokens"] == reserve
        assert estimate_prompt_tokens(call["messages"], call["tools"]) + reserve + 128 <= 20000
        assert "original question" in str(call["messages"][1])
    repeated = json.loads(seen[1]["messages"][-1]["content"])
    assert repeated["id"] == "d1" and repeated["text"]
    assert len(result.extra_data["context_budget"]) == 2


def test_unfit_protected_prompt_fails_before_completion_without_truncating_question():
    calls = []
    llm = create_llm(create_llm_config("callable", model="test"), completion_fn=lambda **kw: calls.append(kw))
    agent = Agent(
        config=AgentConfig(
            mode="answer",
            user_msg_type="simple",
            context_window_tokens=6000,
            context_output_tokens=256,
            context_safety_margin_tokens=128,
        ),
        llm=llm,
        retrieve_tool=create_retrieve_tool("default", lambda q, k: []),
    )
    result = agent.run_sync("original question " * 10000)
    assert result.error.category == "context_limit"
    assert not calls and not result.succeeded


def test_query_options_forward_budget_to_both_native_operator_modes():
    from nemo_retriever.query.agentic import AgenticRetriever
    from nemo_retriever.query.options import QueryAgenticOptions, QueryRequest
    from nemo_retriever.query.workflow import build_agentic_config

    options = QueryAgenticOptions(
        enabled=True,
        llm_model="model",
        invoke_url="http://localhost/v1/chat/completions",
        context_window_tokens=65536,
        context_output_tokens=8192,
        context_safety_margin_tokens=1024,
    )
    cfg = build_agentic_config(QueryRequest(query="question", agentic=options))
    assert cfg.context_window_tokens == 65536
    retriever = AgenticRetriever.__new__(AgenticRetriever)
    retriever._cfg = cfg
    for mode in ("select", "answer"):
        operator = retriever._build_react_operator(mode=mode, chat_completion_fn=lambda **kw: {})
        assert operator._context_window_tokens == 65536
        assert operator._context_output_tokens == 8192
        assert operator._context_safety_margin_tokens == 1024


@pytest.mark.parametrize("text", ["", "evidence " * 100])
def test_large_retrieval_batch_discards_lower_ranked_blocks_without_orphaning_tools(text):
    documents = [_document(f"d{i}", text) for i in range(500)]
    history = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "original question"},
        {"role": "assistant", "tool_calls": [{"id": "retrieve"}], "content": ""},
        {"role": "tool", "tool_call_id": "retrieve", "content": documents},
    ]
    fitted, metrics = fit_context(history, [], 4000, estimate_prompt_tokens)
    assert estimate_prompt_tokens(fitted, []) <= 4000
    assert fitted[:3] == history[:3]
    assert fitted[-1]["tool_call_id"] == "retrieve"
    kept = [json.loads(block["text"]) for block in fitted[-1]["content"]]
    assert kept and kept[0]["id"] == "d0"
    assert [doc["id"] for doc in kept] == [f"d{i}" for i in range(len(kept))]
    assert metrics["discarded_documents"] == 500 - len(kept)
    assert len(history[-1]["content"]) == 500


def test_bootstrap_reduction_keeps_prior_year_evidence_and_tool_history():
    history = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": [{"type": "text", "text": "question"}, _document("bootstrap", "B" * 30000)]},
        {"role": "assistant", "tool_calls": [{"id": "old"}]},
        {"role": "tool", "tool_call_id": "old", "content": [_document("1952", "E" * 20000)]},
        {"role": "assistant", "tool_calls": [{"id": "new"}]},
        {"role": "tool", "tool_call_id": "new", "content": [_document("1954", "F" * 20000)]},
    ]
    fitted, metrics = fit_context(history, [], 60928, estimate_prompt_tokens)
    assert metrics["removed_turns"] == 0
    assert fitted[2:] == history[2:]
    assert estimate_prompt_tokens(fitted, []) <= 60928


def test_visible_repeat_is_not_resent_but_trace_keeps_evicted_transactions():
    seen = []

    def completion(**kwargs):
        seen.append(deepcopy(kwargs))
        if len(seen) < 4:
            response = _response("retrieve", {"query": "again", "top_k": 1})
            response["choices"][0]["message"]["content"] = "reasoning " * 900
            response["choices"][0]["message"]["tool_calls"][0]["id"] = f"call-{len(seen)}"
            return response
        return _response("log_answer", {"answer": "42", "citations": ["d"]})

    agent = Agent(
        config=AgentConfig(
            mode="answer",
            user_msg_type="with_results",
            end_tool_with_msg=False,
            context_window_tokens=20000,
            context_output_tokens=256,
            context_safety_margin_tokens=128,
            max_steps=4,
        ),
        llm=create_llm(create_llm_config("callable", model="test"), completion_fn=completion),
        retrieve_tool=create_retrieve_tool(
            "default", lambda q, k: [{"id": "d", "score": 1, "text": "original evidence"}]
        ),
    )
    result = agent.run_sync("question", query_id="trace")
    assert result.succeeded
    assert "This document is retrieved before" in seen[1]["messages"][-1]["content"]
    assert sum(m["removed_turns"] for m in result.extra_data["context_budget"]) > 0
    assert result.atif_trace["final_metrics"]["extra"]["llm_call_count"] == len(seen) == 4
    assert result.atif_trace["final_metrics"]["total_prompt_tokens"] == 40
    assert result.atif_trace["final_metrics"]["total_completion_tokens"] == 8


def test_length_failure_is_audited_with_its_usage():
    response = {
        "choices": [{"message": {"role": "assistant", "content": "partial reasoning"}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 123, "completion_tokens": 16384},
    }
    agent = Agent(
        config=AgentConfig(mode="answer", user_msg_type="simple"),
        llm=create_llm(create_llm_config("callable", model="test"), completion_fn=lambda **kwargs: response),
        retrieve_tool=create_retrieve_tool("default", lambda q, k: []),
    )
    result = agent.run_sync("question")
    assert result.error.category == "bad_finish_reason"
    assert result.atif_trace["final_metrics"]["extra"]["llm_call_count"] == 1
    assert result.atif_trace["final_metrics"]["total_completion_tokens"] == 16384


def test_deduplicated_tool_response_remains_valid_text_content():
    history = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": [{"type": "text", "text": "question"}, _document("d", "evidence" * 500)]},
        {"role": "assistant", "tool_calls": [{"id": "repeat"}]},
        {"role": "tool", "tool_call_id": "repeat", "content": [_document("d", "evidence" * 500)]},
    ]
    fitted, metrics = fit_context(history, [], 6000, estimate_prompt_tokens)
    assert metrics["deduplicated_documents"] == 1
    assert fitted[-1]["content"]
    assert fitted[-1]["content"][0]["text"]
    assert fitted[-1]["tool_call_id"] == "repeat"
    assert estimate_prompt_tokens(fitted, []) <= 6000
