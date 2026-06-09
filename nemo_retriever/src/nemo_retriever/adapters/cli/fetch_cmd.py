# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`retriever fetch "<handle>"` -> the full chunk text behind an evidence-pack span.

A handle is ``"<source>|<page>|<rank>"`` (from the evidence pack). Fetch resolves it
to the full indexed text for that source+page by a pure metadata lookup on LanceDB
(no embedding model, no PDF re-parse) — the cheap escalation path when a span lacks
the detail needed, so the agent never has to re-read the source document."""
from __future__ import annotations

import json
import os

import typer

from nemo_retriever.adapters.cli.main import DEFAULT_LANCEDB_URI, DEFAULT_TABLE_NAME, app


def _as_dict(v: object) -> dict:
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.strip():
        try:
            return json.loads(v)
        except Exception:
            return {}
    return {}


def _row_source_page_text(row: dict) -> tuple[str, object, str]:
    meta = _as_dict(row.get("metadata"))
    src = _as_dict(row.get("source"))
    name = src.get("source_name") or src.get("source_id") or row.get("source_id") or row.get("source") or ""
    base = os.path.basename(str(name))
    if base.lower().endswith(".pdf"):
        base = base[:-4]
    page = meta.get("page_number")
    if page is None:
        page = row.get("page_number")
    return base, page, str(row.get("text") or "")


@app.command("fetch")
def fetch_command(
    handle: str = typer.Argument(..., help='Evidence handle "<source>|<page>|<rank>".'),
    lancedb_uri: str = typer.Option(DEFAULT_LANCEDB_URI, "--lancedb-uri"),
    table_name: str = typer.Option(DEFAULT_TABLE_NAME, "--table-name"),
) -> None:
    parts = str(handle).split("|")
    source = parts[0] if parts else ""
    page = parts[1] if len(parts) > 1 else ""

    import lancedb

    table = lancedb.connect(lancedb_uri).open_table(table_name)
    df = table.to_pandas()
    chunks: list[str] = []
    for _, row in df.iterrows():
        base, pg, text = _row_source_page_text(row.to_dict())
        if base == source and str(pg) == str(page) and text:
            chunks.append(text)

    out = {
        "handle": handle,
        "source": source,
        "locator": f"p.{page}" if page else "",
        "n_chunks": len(chunks),
        "text": "\n\n".join(chunks),
    }
    typer.echo(json.dumps(out, indent=2, ensure_ascii=False, default=str))
