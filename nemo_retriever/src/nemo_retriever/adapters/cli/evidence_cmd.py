# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`retriever evidence "<question>"` -> evidence pack JSON (recall-layer output)."""
from __future__ import annotations

import json

import typer

from nemo_retriever.adapters.cli.main import DEFAULT_LANCEDB_URI, DEFAULT_TABLE_NAME, app
from nemo_retriever.adapters.cli.sdk_workflow import query_documents
from nemo_retriever.evidence import build_evidence_pack


@app.command("evidence")
def evidence_command(
    question: str = typer.Argument(..., help="Natural-language question."),
    lancedb_uri: str = typer.Option(DEFAULT_LANCEDB_URI, "--lancedb-uri"),
    table_name: str = typer.Option(DEFAULT_TABLE_NAME, "--table-name"),
    top_k: int = typer.Option(10, "--top-k", min=1),
    candidate_k: int = typer.Option(40, "--candidate-k", min=1, help="Wide pre-rerank pool."),
    rerank: bool = typer.Option(False, "--rerank/--no-rerank", help="Cross-encoder rerank the candidate pool (off by default to match the retrieve() path)."),
    max_tokens: int = typer.Option(120, "--max-tokens", min=1, help="Span length cap (whitespace tokens)."),
) -> None:
    # Reuse the verified query path (sdk_workflow.query_documents): it builds the
    # Retriever with hybrid + optional rerank and returns RetrievalHit dicts.
    hits = query_documents(
        question,
        top_k=top_k,
        candidate_k=candidate_k,
        hybrid=True,
        rerank=rerank,
        lancedb_uri=lancedb_uri,
        table_name=table_name,
    )
    pack = build_evidence_pack(hits, question, max_tokens=max_tokens)
    typer.echo(json.dumps(pack, indent=2, ensure_ascii=False, default=str))
