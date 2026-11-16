# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Offline integration coverage for separate evidence and answer models."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from nemo_retriever.query.agentic import AgenticAnswerConfig, AgenticRetrievalConfig, AgenticRetriever
from .test_agentic_eval import FakeRetriever, _make_tool_call_response

EVIDENCE_USAGE = {
    "prompt_tokens": 20,
    "completion_tokens": 7,
    "total_tokens": 27,
    "prompt_tokens_details": {"cached_tokens": 10},
    "completion_tokens_details": {"reasoning_tokens": 4},
}
ANSWER_USAGE = {
    "prompt_tokens": 30,
    "completion_tokens": 11,
    "total_tokens": 41,
    "prompt_tokens_details": {"cached_tokens": 5},
    "completion_tokens_details": {"reasoning_tokens": 6},
}


def _cfg(**kwargs):
    return AgenticRetrievalConfig(
        llm_model="evidence-model",
        invoke_url="http://evidence/v1/chat/completions",
        api_key="evidence-key",
        reasoning_effort="low",
        temperature=0.1,
        max_tokens=123,
        answer_model=AgenticAnswerConfig(
            llm_model="answer-model",
            invoke_url="http://answer/v1/chat/completions",
            api_key="answer-key",
            reasoning_effort="high",
            temperature=0.4,
            max_tokens=456,
        ),
        top_k=1,
        **kwargs,
    )


def _evidence_response(usage=EVIDENCE_USAGE):
    response = _make_tool_call_response(
        "final_results",
        {"doc_ids": ["doc_1"], "search_successful": "true"},
        usage=usage,
    )
    response["choices"][0]["message"]["reasoning_content"] = "PRIVATE EVIDENCE REASONING"
    return response


def _answer_response(citations=None, answer="Supported answer", usage=ANSWER_USAGE):
    return _make_tool_call_response(
        "log_answer",
        {"answer": answer, "citations": citations or ["doc_1"]},
        usage=usage,
    )


@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_split_handoff_settings_usage_and_citations():
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        if kwargs["model"] == "evidence-model":
            assert {tool["function"]["name"] for tool in kwargs["tools"]} == {"retrieve", "final_results"}
            assert "untrusted" in kwargs["messages"][0]["content"]
            return _evidence_response()
        assert [tool["function"]["name"] for tool in kwargs["tools"]] == ["log_answer"]
        handoff = json.loads(kwargs["messages"][1]["content"])
        assert handoff["question"] == "Find doc; answer in French."
        assert handoff["evidence"][0] == {
            "citation_id": "doc_1",
            "text": "matching",
            "source": {"source": {"source_id": "/tmp/clip.wav"}, "source_id": "/tmp/doc.pdf", "page_number": 1},
        }
        assert "PRIVATE EVIDENCE REASONING" not in json.dumps(kwargs["messages"])
        assert "untrusted" in kwargs["messages"][0]["content"]
        return _answer_response(["doc_1", "doc_1"])

    with patch("nemo_retriever.models.nim.chat_completions.invoke_chat_completion_step", side_effect=complete):
        retriever = AgenticRetriever(_cfg(text_truncation=8))
        result = retriever.answer_with_usage(["q"], ["Find doc; answer in French."])
    assert result.answers.iloc[0]["succeeded"]
    assert result.answers.iloc[0]["citations"] == ["doc_1"]
    assert result.answers.iloc[0]["citation_hits"][0]["text"] == "matching document"
    assert len(retriever._retriever.query_calls) == 1
    assert result.usage == {"q": {"evidence_gathering": EVIDENCE_USAGE, "answer_synthesis": ANSWER_USAGE}}
    for call, model, key, effort, temp, budget, url in zip(
        calls,
        ["evidence-model", "answer-model"],
        ["evidence-key", "answer-key"],
        ["low", "high"],
        [0.1, 0.4],
        [123, 456],
        ["http://evidence/v1/chat/completions", "http://answer/v1/chat/completions"],
    ):
        assert call["model"] == model
        assert call["api_key"] == key
        assert call["extra_body"]["reasoning_effort"] == effort
        assert call["temperature"] == temp
        assert call["max_tokens"] == budget
        assert call["invoke_url"] == url


@pytest.mark.parametrize("failure", ["exception", "unknown_citation", "empty", "no_call", "retrieval_call", "length"])
@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_synthesis_failure_is_explicit_and_retains_usage(failure):
    def complete(**kwargs):
        if kwargs["model"] == "evidence-model":
            return _evidence_response()
        if failure == "exception":
            raise RuntimeError("answer endpoint unavailable")
        response = _answer_response(
            citations=["unknown"] if failure == "unknown_citation" else None,
            answer="" if failure == "empty" else "answer",
        )
        if failure == "no_call":
            response["choices"][0]["message"]["tool_calls"] = []
        elif failure == "retrieval_call":
            response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = "retrieve"
        elif failure == "length":
            response["choices"][0]["finish_reason"] = "length"
        return response

    with patch("nemo_retriever.models.nim.chat_completions.invoke_chat_completion_step", side_effect=complete):
        result = AgenticRetriever(_cfg()).answer_with_usage(["q"], ["question"])
    row = result.answers.iloc[0]
    assert not row["succeeded"]
    assert row["answer"] is None
    assert row["citations"] is None
    assert row["citation_hits"] == []
    assert row["error_category"] == "answer_synthesis.failed"
    assert row["error_message"]
    assert row["error_exception_class"]
    assert result.usage["q"]["evidence_gathering"] == EVIDENCE_USAGE
    if failure != "exception":
        assert result.usage["q"]["answer_synthesis"] == ANSWER_USAGE
    else:
        assert "answer_synthesis" not in result.usage["q"]


@pytest.mark.parametrize("failure", ["endpoint", "retrieval", "max_steps"])
@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_evidence_errors_do_not_synthesize(failure):
    calls = []

    def complete(**kwargs):
        calls.append(kwargs["model"])
        assert kwargs["model"] == "evidence-model"
        if failure == "endpoint":
            raise RuntimeError("evidence endpoint unavailable")
        return _make_tool_call_response("retrieve", {"query": "another search"}, usage=EVIDENCE_USAGE)

    retriever = AgenticRetriever(_cfg(react_max_steps=2))
    if failure == "retrieval":
        retriever._retriever.query = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("VDB unavailable"))
    with patch("nemo_retriever.models.nim.chat_completions.invoke_chat_completion_step", side_effect=complete):
        result = retriever.answer_with_usage(["q"], ["question"])
    row = result.answers.iloc[0]
    assert not row["succeeded"]
    assert row["error_category"].startswith("evidence_gathering.")
    assert row["error_message"]
    if failure == "max_steps":
        assert row["error_category"] == "evidence_gathering.max_steps"
        assert len(calls) == 2
        assert result.usage["q"]["evidence_gathering"]["total_tokens"] == 54
    else:
        assert not result.usage


@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_split_absent_usage_and_empty_input():
    def complete(**kwargs):
        if kwargs["model"] == "evidence-model":
            return _evidence_response(usage=None)
        return _answer_response(usage=None)

    with patch("nemo_retriever.models.nim.chat_completions.invoke_chat_completion_step", side_effect=complete):
        retriever = AgenticRetriever(_cfg())
        result = retriever.answer_with_usage(["q"], ["question"])
        empty = retriever.answer_with_usage([], [])
    assert result.answers.iloc[0]["succeeded"]
    assert result.usage == {}
    assert empty.answers.empty
    assert "citation_hits" in empty.answers.columns


@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_concurrency_is_bounded_across_stages_and_invocations():
    active = 0
    peak = 0
    lock = threading.Lock()

    def complete(**kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(active, peak)
        try:
            time.sleep(0.01)
            if kwargs["model"] == "evidence-model":
                return _evidence_response()
            handoff = json.loads(kwargs["messages"][1]["content"])
            question = handoff["question"]
            assert handoff["evidence"][0]["text"] == f"evidence {question}"
            return _answer_response(answer=question)
        finally:
            with lock:
                active -= 1

    retriever = AgenticRetriever(_cfg(num_concurrent=2))
    original_query = retriever._retriever.query

    def query(text, **kwargs):
        return [{**hit, "text": f"evidence {text}"} for hit in original_query(text, **kwargs)]

    retriever._retriever.query = query
    with patch("nemo_retriever.models.nim.chat_completions.invoke_chat_completion_step", side_effect=complete):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(retriever.answer_with_usage, ["same", "q2"], [f"{i} first", f"{i} second"])
                for i in range(2)
            ]
            results = [future.result() for future in futures]
    assert peak == 2
    for i, result in enumerate(results):
        assert result.answers["answer"].tolist() == [f"{i} first", f"{i} second"]
        assert result.answers["succeeded"].tolist() == [True, True]
        assert result.usage == {
            qid: {"evidence_gathering": EVIDENCE_USAGE, "answer_synthesis": ANSWER_USAGE} for qid in ("same", "q2")
        }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"llm_model": ""},
        {"invoke_url": ""},
        {"max_tokens": 0},
        {"max_tokens": True},
        {"temperature": float("nan")},
        {"temperature": 2.1},
        {"llm_client": "unknown"},
    ],
)
def test_answer_config_validation(kwargs):
    options = {"llm_model": "answer", "invoke_url": "http://answer/v1/chat/completions", **kwargs}
    with pytest.raises(ValueError):
        AgenticAnswerConfig(**options)


@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_partial_evidence_usage_survives_later_retrieval_failure_and_batch_continues():
    retriever = AgenticRetriever(_cfg(num_concurrent=2))
    original_query = retriever._retriever.query

    def query(text, **kwargs):
        if text == "broken search":
            raise RuntimeError("VDB failed on second hop")
        return original_query(text, **kwargs)

    retriever._retriever.query = query

    def complete(**kwargs):
        if kwargs["model"] == "answer-model":
            return _answer_response()
        if "broken question" in json.dumps(kwargs["messages"]):
            return _make_tool_call_response("retrieve", {"query": "broken search"}, usage=EVIDENCE_USAGE)
        return _evidence_response()

    with patch("nemo_retriever.models.nim.chat_completions.invoke_chat_completion_step", side_effect=complete):
        result = retriever.answer_with_usage(["broken", "good"], ["broken question", "good question"])
    assert result.answers["succeeded"].tolist() == [False, True]
    assert result.answers.iloc[0]["error_category"] == "evidence_gathering.tool_failed"
    assert result.usage["broken"] == {"evidence_gathering": EVIDENCE_USAGE}
    assert result.usage["good"] == {"evidence_gathering": EVIDENCE_USAGE, "answer_synthesis": ANSWER_USAGE}


@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_split_with_local_evidence_model():
    cfg = AgenticRetrievalConfig(
        llm_model="nemotron-8b",
        top_k=1,
        answer_model=AgenticAnswerConfig(llm_model="answer", invoke_url="http://answer/v1/chat/completions"),
    )
    with patch(
        "nemo_retriever.query.agentic._build_agent_chat_completion_fn", return_value=lambda **_: _evidence_response()
    ), patch("nemo_retriever.models.nim.chat_completions.invoke_chat_completion_step", return_value=_answer_response()):
        result = AgenticRetriever(cfg).answer_with_usage(["q"], ["question"])
    assert result.answers.iloc[0]["succeeded"]
    assert result.usage == {"q": {"evidence_gathering": EVIDENCE_USAGE, "answer_synthesis": ANSWER_USAGE}}
