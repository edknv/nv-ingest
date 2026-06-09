# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`retriever answer "<question>"` -> a short, server-side-synthesized grounded answer.

The "thin-agent / fat-service" path: retrieval AND synthesis happen here (on a cheap
NIM model), so the calling agent makes ONE call and gets a short answer + citations
instead of reading chunks and looping. Minimizes agent round-trips (the dominant
turn-cost term). Synthesis needs ``NVIDIA_API_KEY`` in the environment."""
from __future__ import annotations

import json
import os

import typer

from nemo_retriever.adapters.cli.main import DEFAULT_LANCEDB_URI, DEFAULT_TABLE_NAME, app
from nemo_retriever.adapters.cli.sdk_workflow import query_documents

_DEFAULT_SYNTH_MODEL = "nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5"


def _citation(hit: dict) -> dict:
    meta = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    raw = str(hit.get("source_id") or hit.get("source") or "")
    src = os.path.basename(raw)
    if src.lower().endswith(".pdf"):
        src = src[:-4]
    page = hit.get("page_number")
    if page is None:
        page = meta.get("page_number")
    return {"source": src, "page": page}


@app.command("answer")
def answer_command(
    question: str = typer.Argument(..., help="Natural-language question."),
    lancedb_uri: str = typer.Option(DEFAULT_LANCEDB_URI, "--lancedb-uri"),
    table_name: str = typer.Option(DEFAULT_TABLE_NAME, "--table-name"),
    top_k: int = typer.Option(10, "--top-k", min=1),
    candidate_k: int = typer.Option(40, "--candidate-k", min=1),
    model: str = typer.Option(_DEFAULT_SYNTH_MODEL, "--model", help="Server-side synthesis model (NIM)."),
) -> None:
    from nemo_retriever.llm.clients.litellm import LiteLLMClient

    # Warm path: if a model server is up (EMBED_INVOKE_URL), embed the query over HTTP
    # so there's no per-call model cold-load. The embed client defaults to the VL model
    # name, but serve-models may host a different id — resolve the served id from
    # /v1/models so the request matches (avoids a 404 "model does not exist").
    embed_url = os.environ.get("EMBED_INVOKE_URL") or None
    embed_kwargs: dict = {}
    if embed_url:
        # query_documents resolves the server's served model name (avoids a 404).
        embed_kwargs["embed_invoke_url"] = embed_url
    else:
        embed_kwargs["query_embed_backend"] = "hf"  # faster cold start than vllm for single-query CLI

    def _retrieve(use_hybrid: bool):
        return query_documents(
            question, top_k=top_k, candidate_k=candidate_k, hybrid=use_hybrid,
            lancedb_uri=lancedb_uri, table_name=table_name, **embed_kwargs,
        )

    try:
        hits = _retrieve(True)
    except Exception:  # noqa: BLE001 — index may have no FTS index; degrade to vector-only
        hits = _retrieve(False)
    chunks = [str(h.get("text", "")) for h in hits]

    client = LiteLLMClient.from_kwargs(model=model, api_key=os.environ.get("NVIDIA_API_KEY"))
    gen = client.generate(question, chunks)

    # Ranked, de-duplicated citations (source+page) for the agent's selected_chunks.
    cites: list[dict] = []
    seen = set()
    for h in hits:
        c = _citation(h)
        key = (c["source"], c["page"])
        if key in seen:
            continue
        seen.add(key)
        cites.append({"rank": len(cites) + 1, **c})

    out = {
        "answer": gen.answer,
        "model": gen.model,
        "error": gen.error,
        "citations": cites,
        "coverage": {"n_hits": len(hits), "n_docs": len({c["source"] for c in cites})},
    }
    typer.echo(json.dumps(out, indent=2, ensure_ascii=False, default=str))
