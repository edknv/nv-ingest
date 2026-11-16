# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Offline controller, provider contract, and graph integration coverage."""

from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from nemo_retriever.operators.graph_ops.system_one_operator import SelectionPoolOperator, SystemOneSearchOperator
from nemo_retriever.query.system_one import Decision, SystemOneClient, SystemOneConfig

CFG = SystemOneConfig(endpoint="https://decision.example/v1/systemone", model="test-decision")


def _decision(action="finish", sufficient=0.9):
    probabilities = {"expand": 0.1, "think": 0.1, "finish": 0.1}
    probabilities[action] = 0.8
    total = sum(probabilities.values())
    return Decision(action, {k: v / total for k, v in probabilities.items()}, sufficient, {"input_tokens": 5})


def _operator(config=CFG, outputs=None, decisions=None, retrieve=None):
    llm = MagicMock()
    llm.completion.side_effect = [
        SimpleNamespace(message={"content": json.dumps(value)}, finish_reason="stop")
        for value in (outputs or [{"queries": []}])
    ]
    llm.pop_query_usage.return_value = {}
    client = MagicMock()
    client.decide.side_effect = decisions or [_decision()]
    if retrieve is None:
        retrieve = MagicMock(return_value=[{"doc_id": "d", "text": "full evidence " * 1000}])
    return SystemOneSearchOperator(config=config, retriever_fn=retrieve, llm=llm, decision_client=client)


def _run(op):
    return op.run(pd.DataFrame({"query_id": ["customer"], "query_text": ["original"]}))


def test_sufficient_stop_retains_full_text_and_caps_controller_evidence():
    op = _operator(config=replace(CFG, evidence_chars=17), outputs=[{"queries": ["new", "new", "original"]}])
    result = _run(op)
    assert result.step_idx.tolist() == [0, 1]
    assert len(result.text.iloc[0]) > 17
    assert not result.has_valid_final_results.any()
    state = op._decision_client.decide.call_args.args[0]
    assert sum(len(item["text"]) for item in state["evidence"]) <= 17
    assert state["searches"] == ["original", "new"]
    assert op.pop_query_trace("customer")["stop_reason"] == "sufficient"
    assert op.pop_query_usage("customer")["system_one_decision"]["input_tokens"] == 5
    assert op.pop_query_usage("customer") == {}


def test_rejected_finish_uses_progress_action_and_action_budget():
    op = _operator(
        config=replace(CFG, max_actions=1),
        outputs=[{"queries": []}, {"queries": ["follow up"]}],
        decisions=[_decision(sufficient=0.2)],
    )
    _run(op)
    trace = op.pop_query_trace("customer")
    assert trace["stop_reason"] == "action_budget"
    assert trace["decisions"][0]["applied_action"] == "expand"
    assert [entry["query"] for entry in trace["searches"]] == ["original", "follow up"]


def test_think_notes_reach_next_round_and_exhausted_actions_are_masked():
    op = _operator(
        config=replace(CFG, max_thinks=1),
        outputs=[{"queries": []}, {"notes": "missing date"}],
        decisions=[_decision("think"), _decision()],
    )
    _run(op)
    state, actions = op._decision_client.decide.call_args.args
    assert state["notes"] == "missing date"
    assert "think" not in actions
    assert op.pop_query_usage("customer")["system_one_decision"]["input_tokens"] == 10


def test_operation_budget_stops_without_another_provider_call():
    op = _operator(config=replace(CFG, max_thinks=0, max_expansions=0))
    _run(op)
    op._decision_client.decide.assert_not_called()
    assert op.pop_query_trace("customer")["stop_reason"] == "operation_budget"


@pytest.mark.parametrize("failure_stage", ["plan", "decision", "expand"])
def test_provider_failure_preserves_candidates(failure_stage):
    op = _operator(decisions=[_decision("expand")])
    if failure_stage == "plan":
        op._llm.completion.side_effect = ValueError("malformed provider content")
    elif failure_stage == "decision":
        op._decision_client.decide.side_effect = httpx.ConnectError("unavailable")
    else:
        op._llm.completion.side_effect = [
            SimpleNamespace(message={"content": '{"queries": []}'}, finish_reason="stop"),
            ValueError("bad"),
        ]
    assert _run(op).doc_id.tolist() == ["d"]
    assert op.pop_query_trace("customer")["stop_reason"] == "provider_error"


def test_retrieval_failure_propagates_during_later_round():
    retrieve = MagicMock(side_effect=[[{"doc_id": "d", "text": "evidence"}], ValueError("broken index")])
    op = _operator(outputs=[{"queries": ["new"]}], retrieve=retrieve)
    with pytest.raises(ValueError, match="broken index"):
        _run(op)


def test_candidate_pool_limit_is_per_query_and_optional():
    df = pd.DataFrame(
        {"query_id": ["a", "a", "b", "b"], "doc_id": ["z", "a", "c", "d"], "rrf_score": [1.0, 1.0, 2.0, 1.0]}
    )
    pd.testing.assert_frame_equal(SelectionPoolOperator().run(df), df)
    assert SelectionPoolOperator(top_k=1).run(df).doc_id.tolist() == ["c", "a"]


def _body():
    return {
        "model": "pinned-model",
        "answers": {
            "action": {"type": "choice", "choice": "finish", "probabilities": {"expand": 0.2, "finish": 0.8}},
            "sufficient": {"type": "noul", "noul": 0.9},
        },
        "usage": {"input_tokens": 42},
    }


def test_http_contract_and_environment_key(monkeypatch):
    monkeypatch.setenv("S1_TEST_KEY", "secret")
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json=_body())

    real_client = httpx.Client
    with patch(
        "nemo_retriever.query.system_one.httpx.Client",
        side_effect=lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw),
    ):
        result = SystemOneClient(replace(CFG, api_key="os.environ/S1_TEST_KEY")).decide(
            {"query": "q"}, ["expand", "finish"]
        )
    assert result.action == "finish" and result.sufficiency == 0.9
    payload = json.loads(seen[0].content)
    assert payload["questions"]["action"]["criteria"].keys() == {"expand", "finish"}
    assert payload["questions"]["sufficient"]["type"] == "noul"
    assert seen[0].headers["Authorization"] == "Bearer secret"
    assert "secret" not in repr(replace(CFG, api_key="secret"))


@pytest.mark.parametrize("invalid", ["nan", "missing", "sum", "choice", "boolean", "type"])
def test_http_rejects_malformed_decisions(invalid):
    body = _body()
    if invalid == "nan":
        body["answers"]["sufficient"]["noul"] = float("nan")
    elif invalid == "missing":
        del body["answers"]["sufficient"]
    elif invalid == "sum":
        body["answers"]["action"]["probabilities"]["expand"] = 0.9
    elif invalid == "choice":
        body["answers"]["action"]["choice"] = "expand"
    elif invalid == "boolean":
        body["answers"]["sufficient"]["noul"] = True
    else:
        body["answers"]["action"]["type"] = "noul"
    with patch("nemo_retriever.query.system_one.httpx.Client") as client:
        response = client.return_value.__enter__.return_value.post.return_value
        response.json.return_value = body
        with pytest.raises(ValueError):
            SystemOneClient(CFG).decide({}, ["expand", "finish"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_actions": True},
        {"max_thinks": -1},
        {"evidence_chars": 0},
        {"timeout": float("inf")},
        {"sufficiency_threshold": float("nan")},
        {"endpoint": "file:///tmp/a"},
        {"selection_pool_k": 0},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        replace(CFG, **kwargs)


@pytest.mark.parametrize("pool_k", [None, 1])
def test_graph_integration_selects_and_rehydrates_with_caller_ids(pool_k):
    from nemo_retriever.query.agentic import AgenticRetrievalConfig, AgenticRetriever
    from .test_agentic_eval import FakeRetriever, _make_tool_call_response

    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        if kwargs.get("tools"):
            return _make_tool_call_response(
                "log_selected_documents", {"doc_ids": ["doc_1"], "message": "best"}, usage={"prompt_tokens": 9}
            )
        return {
            "choices": [{"message": {"content": '{"queries": []}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3},
        }

    cfg = AgenticRetrievalConfig(
        llm_model="nemotron-8b", top_k=1, system_one=replace(CFG, selection_pool_k=pool_k), text_truncation=1
    )
    with patch("nemo_retriever.query.agentic.Retriever", FakeRetriever), patch(
        "nemo_retriever.query.agentic._build_agent_chat_completion_fn", return_value=complete
    ), patch.object(SystemOneClient, "decide", side_effect=lambda *args: _decision()):
        retriever = AgenticRetriever(cfg)
        result = retriever.retrieve_with_usage(["customer-a", "customer-b"], ["first", "second"])
        with pytest.raises(ValueError, match="retrieval only"):
            retriever.answer(["q"], ["question"])
    assert result.documents.query_id.tolist() == ["customer-a", "customer-b"]
    assert result.documents.result_source.tolist() == ["selection_agent", "selection_agent"]
    assert result.documents.hit.iloc[0]["text"] == "matching document"
    for query_id in ["customer-a", "customer-b"]:
        assert result.traces[query_id]["stop_reason"] == "sufficient"
        assert result.usage[query_id]["system_one_plan"]["prompt_tokens"] == 3
        assert result.usage[query_id]["system_one_decision"]["input_tokens"] == 5
        assert result.usage[query_id]["top1_agent"]["prompt_tokens"] == 9
    assert "matching document" in json.dumps(calls[1]["messages"])


def test_selection_pool_cannot_be_smaller_than_final_count():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    with pytest.raises(ValueError, match="selection_pool_k must be >= top_k"):
        AgenticRetrievalConfig(top_k=10, system_one=replace(CFG, selection_pool_k=5))


def test_empty_search_emits_rrf_schema_and_trace():
    op = _operator(retrieve=MagicMock(return_value=[]))
    result = _run(op)
    assert result.empty
    assert {"query_id", "query_text", "step_idx", "doc_id", "text", "rank"} <= set(result.columns)
    assert op.pop_query_trace("customer")["candidate_count"] == 0


def test_model_output_limits_bound_search_calls():
    op = _operator(config=replace(CFG, max_subqueries=1, max_actions=1), outputs=[{"queries": ["new", "discard"]}])
    _run(op)
    assert [call.args[0] for call in op._retriever_fn.call_args_list] == ["original", "new"]


def test_new_evidence_reaches_controller_after_expansion():
    retrieve = MagicMock(
        side_effect=[
            [{"doc_id": f"old-{i}", "text": "old text"} for i in range(10)],
            [{"doc_id": "new", "text": "new supporting evidence"}],
        ]
    )
    op = _operator(
        retrieve=retrieve,
        outputs=[{"queries": []}, {"queries": ["new search"]}],
        decisions=[_decision("expand"), _decision()],
    )
    _run(op)
    state = op._decision_client.decide.call_args.args[0]
    assert "new" in [item["doc_id"] for item in state["evidence"]]
    assert state["candidate_count"] == 11


@pytest.mark.parametrize("sufficiency,expected_stop,search_count", [(0.69, "action_budget", 2), (0.7, "sufficient", 1)])
def test_companion_noul_controls_stopping_independently_of_choice(sufficiency, expected_stop, search_count):
    """Exercise the prompt payload and stop gate together through the HTTP adapter."""
    requests = []

    def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert set(payload["questions"]) == {"action", "sufficient"}
        # The Choice routes search actions.
        assert set(payload["questions"]["action"]["criteria"]) == {"expand", "finish"}
        noul = payload["questions"]["sufficient"]
        assert noul["type"] == "noul"
        assert set(noul["criteria"]) == {"true", "false"}
        body = _body()
        body["answers"]["sufficient"]["noul"] = sufficiency
        return httpx.Response(200, json=body)

    op = _operator(
        config=replace(CFG, max_actions=1, max_expansions=1, max_thinks=0),
        outputs=[{"queries": []}, {"queries": ["missing connection"]}],
    )
    op._decision_client = SystemOneClient(op._config)
    real_client = httpx.Client
    with patch(
        "nemo_retriever.query.system_one.httpx.Client",
        side_effect=lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw),
    ):
        _run(op)
    trace = op.pop_query_trace("customer")
    assert trace["stop_reason"] == expected_stop
    assert len(trace["searches"]) == search_count
    assert len(requests) == 1
    assert trace["decisions"][0]["action"] == "finish"
