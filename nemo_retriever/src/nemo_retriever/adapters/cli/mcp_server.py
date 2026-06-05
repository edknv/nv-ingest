# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP surface for the retriever: exposes read ops (query, verify) as MCP tools."""

from __future__ import annotations

from fastmcp import FastMCP

from nemo_retriever.adapters.cli._hit_format import _query_cli_hit
from nemo_retriever.adapters.cli.sdk_workflow import (
    DEFAULT_LANCEDB_URI,
    DEFAULT_TABLE_NAME,
    query_documents,
    verify_claim,
)

mcp = FastMCP("nemo-retriever")


@mcp.tool
def query(
    question: str,
    top_k: int = 10,
    hybrid: bool = False,
    max_text_chars: int | None = None,
    content_types: str | None = None,
    lancedb_uri: str = DEFAULT_LANCEDB_URI,
    table_name: str = DEFAULT_TABLE_NAME,
    embed_model_name: str | None = None,
    rerank: bool = False,
) -> list[dict]:
    """Search the corpus. Returns hits with source, page_number, text, modality, score.

    Set max_text_chars=0 for a metadata-only summary, or N for N-char snippets.
    Set hybrid=True to combine vector + full-text retrieval (needs a --hybrid index).
    """
    hits = query_documents(
        question,
        top_k=top_k,
        hybrid=hybrid,
        content_types=content_types,
        lancedb_uri=lancedb_uri,
        table_name=table_name,
        embed_model_name=embed_model_name,
        rerank=rerank,
    )
    return [_query_cli_hit(h, max_text_chars) for h in hits]


@mcp.tool
def verify(
    claim: str,
    source: str,
    page: int | None = None,
    against: str = "text,table",
    lancedb_uri: str = DEFAULT_LANCEDB_URI,
    table_name: str = DEFAULT_TABLE_NAME,
) -> dict:
    """Fetch independent text/table evidence for a claim's (source, page) location.

    Returns the evidence plus a mechanical term/number-overlap signal. Does NOT
    judge agreement — you decide whether the evidence confirms the claim.
    """
    return verify_claim(
        claim,
        source,
        page=page,
        against=against,
        lancedb_uri=lancedb_uri,
        table_name=table_name,
    )


@mcp.tool
def retrieve(
    question: str,
    top_k: int = 10,
    hybrid: bool = True,
    lancedb_uri: str = DEFAULT_LANCEDB_URI,
    table_name: str = DEFAULT_TABLE_NAME,
    embed_model_name: str | None = None,
) -> dict:
    """Answer-ready, fidelity-tagged, cited evidence + coverage for a question.

    Returns {evidence:[{text,source,locator,modality,fidelity,score,citation}], coverage}.
    Warm when `retriever serve-models` is running (honors EMBED_INVOKE_URL).
    """
    from nemo_retriever.adapters.cli.sdk_workflow import retrieve as _retrieve

    return _retrieve(
        question,
        top_k=top_k,
        hybrid=hybrid,
        lancedb_uri=lancedb_uri,
        table_name=table_name,
        embed_model_name=embed_model_name,
    )
