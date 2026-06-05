# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

from fastmcp import Client

import nemo_retriever.adapters.cli.mcp_server as mcp_server


def _run(coro):
    return asyncio.run(coro)


def test_mcp_exposes_query_and_verify_tools() -> None:
    async def go():
        async with Client(mcp_server.mcp) as c:
            return [t.name for t in await c.list_tools()]

    names = _run(go())
    assert "query" in names
    assert "verify" in names


def test_mcp_query_tool_returns_shaped_hits(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_server,
        "query_documents",
        lambda *a, **k: [
            {"text": "passage", "source": "d.pdf", "page_number": 1,
             "metadata": {"type": "text"}, "_distance": 0.2}
        ],
    )

    async def go():
        async with Client(mcp_server.mcp) as c:
            return await c.call_tool("query", {"question": "q"})

    res = _run(go())
    assert res.data[0]["source"] == "d.pdf"
    assert res.data[0]["modality"] == "text"
    assert res.data[0]["page_number"] == 1


def test_mcp_query_tool_honors_max_text_chars(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_server,
        "query_documents",
        lambda *a, **k: [
            {"text": "abcdefghij", "source": "d.pdf", "page_number": 1,
             "metadata": {"type": "text"}, "_distance": 0.2}
        ],
    )

    async def go():
        async with Client(mcp_server.mcp) as c:
            return await c.call_tool("query", {"question": "q", "max_text_chars": 0})

    res = _run(go())
    assert res.data[0]["text"] == ""


def test_mcp_verify_tool(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_server,
        "verify_claim",
        lambda claim, source, **k: {
            "claim": claim, "source": source, "page": k.get("page"),
            "evidence": [], "independent_evidence_found": False,
            "matched_terms": [], "unmatched_terms": [],
        },
    )

    async def go():
        async with Client(mcp_server.mcp) as c:
            return await c.call_tool("verify", {"claim": "c", "source": "doc"})

    res = _run(go())
    assert res.data["claim"] == "c"
    assert res.data["source"] == "doc"


def test_mcp_exposes_retrieve_tool() -> None:
    async def go():
        async with Client(mcp_server.mcp) as c:
            return [t.name for t in await c.list_tools()]

    names = _run(go())
    assert "retrieve" in names
    assert "query" in names and "verify" in names  # still present


def test_mcp_retrieve_tool_returns_contract_shape(monkeypatch) -> None:
    import nemo_retriever.adapters.cli.sdk_workflow as sw

    monkeypatch.setattr(
        sw, "query_documents",
        lambda *a, **k: [
            {"text": "p", "pdf_basename": "doc", "source": "doc.pdf", "page_number": 2,
             "content_type": "text", "metadata": {"type": "text", "fidelity": "verbatim"}, "_score": 0.4}
        ],
    )

    async def go():
        async with Client(mcp_server.mcp) as c:
            return await c.call_tool("retrieve", {"question": "q", "hybrid": False})

    res = _run(go())
    ev = res.data["evidence"]
    assert ev and ev[0]["fidelity"] == "verbatim" and ev[0]["citation"] == "doc p.2"
    assert res.data["coverage"]["strategies_used"] == ["semantic"]
