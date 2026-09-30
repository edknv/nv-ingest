# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise cached cleanup ownership without loading models or using a GPU."""

from threading import Lock
from unittest.mock import MagicMock

import pytest

from nemo_retriever.graph.pipeline_graph import Graph, Node
from nemo_retriever.graph.retriever import Retriever
from nemo_retriever.models.local.llama_nemotron_embed_vl_1b_v2_embedder import (
    LlamaNemotronEmbedVL1BV2VLLMEmbedder,
)
from nemo_retriever.operators.embed.gpu_operator import _BatchEmbedActor as GPUEmbedActor
from nemo_retriever.operators.embed.operators import _BatchEmbedActor
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.query.agentic import AgenticRetriever


def _cached_chain():
    embedder = LlamaNemotronEmbedVL1BV2VLLMEmbedder.__new__(LlamaNemotronEmbedVL1BV2VLLMEmbedder)
    llm = MagicMock()
    embedder._llm = llm
    operator = GPUEmbedActor.__new__(GPUEmbedActor)
    operator._model = embedder
    archetype = _BatchEmbedActor(params=None)
    archetype._resolved_delegate = operator
    archetype._resolved_delegate_key = (1, 1)
    inner = Retriever()
    inner._cached_graph = Graph()
    inner._cached_graph.add_root(Node(archetype, operator_kwargs={}))
    inner._cache_key = ("cached",)
    agent = AgenticRetriever.__new__(AgenticRetriever)
    agent._lock = Lock()
    agent._chat_completion_fn = None
    agent._retriever = inner
    return agent, inner, archetype, operator, embedder, llm


@pytest.mark.parametrize("cuda_available", [False, True])
def test_agentic_unload_shuts_down_cached_embedding_chain(monkeypatch, cuda_available):
    agent, inner, archetype, operator, embedder, llm = _cached_chain()
    monkeypatch.setattr("torch.cuda.is_available", lambda: cuda_available)
    empty_cache = MagicMock()
    monkeypatch.setattr("torch.cuda.empty_cache", empty_cache)
    resolve = MagicMock(side_effect=AssertionError("cleanup must not resolve operators"))
    monkeypatch.setattr(archetype, "_resolve_delegate", resolve)

    def shutdown(*, timeout):
        assert timeout == 30.0
        assert embedder._llm is llm
        empty_cache.assert_not_called()

    llm.llm_engine.engine_core.shutdown.side_effect = shutdown
    agent.unload()
    agent.unload()

    llm.llm_engine.engine_core.shutdown.assert_called_once_with(timeout=30.0)
    assert empty_cache.call_count == int(cuda_available)
    assert embedder._llm is None
    assert operator._model is None
    assert archetype._resolved_delegate is None
    assert archetype._resolved_delegate_key is None
    assert inner._cached_graph is None
    assert inner._cache_key is None
    resolve.assert_not_called()


def test_shutdown_failure_preserves_entire_chain_for_retry(monkeypatch):
    agent, inner, archetype, operator, embedder, llm = _cached_chain()
    graph = inner._cached_graph
    empty_cache = MagicMock()
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.empty_cache", empty_cache)
    llm.llm_engine.engine_core.shutdown.side_effect = [RuntimeError("shutdown failed"), None]

    with pytest.raises(RuntimeError, match="shutdown failed"):
        agent.unload()

    assert inner._cached_graph is graph
    assert inner._cache_key == ("cached",)
    assert archetype._resolved_delegate is operator
    assert archetype._resolved_delegate_key == (1, 1)
    assert operator._model is embedder
    assert embedder._llm is llm
    empty_cache.assert_not_called()

    agent.unload()
    assert llm.llm_engine.engine_core.shutdown.call_count == 2
    empty_cache.assert_called_once_with()
    assert inner._cached_graph is None


def test_unload_unused_retriever_and_archetype_does_not_construct_operators(monkeypatch):
    inner = Retriever()
    archetype = _BatchEmbedActor(params=None)
    build = MagicMock(side_effect=AssertionError("cleanup must not build a graph"))
    resolve = MagicMock(side_effect=AssertionError("cleanup must not resolve operators"))
    monkeypatch.setattr(inner, "_build_default_graph", build)
    monkeypatch.setattr(archetype, "_resolve_delegate", resolve)
    inner.unload()
    archetype.unload()
    build.assert_not_called()
    resolve.assert_not_called()


def test_unload_leaves_custom_graph_owned_by_caller():
    graph = MagicMock()
    inner = Retriever(graph=graph)
    inner.unload()
    assert inner.graph is graph
    assert graph.mock_calls == []


def test_unload_visits_shared_graph_operators_once():
    operator = MagicMock(spec=AbstractOperator)
    operator.unload = MagicMock()
    node = Node(operator, operator_kwargs={})
    child = Node(operator, operator_kwargs={})
    node.add_child(child)
    inner = Retriever()
    inner._cached_graph = Graph()
    inner._cached_graph.roots = [node, child, node]
    inner.unload()
    operator.unload.assert_called_once_with()


def test_agent_llm_failure_retains_owner_and_still_cleans_inner_retriever():
    agent = AgenticRetriever.__new__(AgenticRetriever)
    agent._lock = Lock()
    chat_fn = MagicMock()
    chat_fn.unload.side_effect = [RuntimeError("agent shutdown failed"), None]
    agent._chat_completion_fn = chat_fn
    agent._retriever = MagicMock()
    with pytest.raises(RuntimeError, match="agent shutdown failed"):
        agent.unload()
    assert agent._chat_completion_fn is chat_fn
    agent._retriever.unload.assert_called_once_with()
    agent.unload()
    assert agent._chat_completion_fn is None
