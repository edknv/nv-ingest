# Objective 3b — MCP surface (`retriever mcp`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the retriever's read ops as MCP tools — a `retriever mcp` command serving `query` + `verify` over stdio via FastMCP — so agent harnesses consume them directly with typed I/O. Bump contract to 1.5.0.

**Architecture:** Move the `_query_cli_hit` shaper into a tiny shared module, then add `adapters/cli/mcp_server.py` (a `FastMCP` instance with two thin tool wrappers over `query_documents`/`verify_claim`) and a lazy-importing `@app.command("mcp")` launcher. No new retrieval logic, no new deps (`fastmcp` already a declared dep). Editable install ⇒ edits take effect immediately.

**Tech Stack:** Python 3.12, FastMCP 3.2.4 (`@mcp.tool`, `mcp.run()` stdio default, `fastmcp.Client` in-memory test client whose `CallToolResult.data` holds the structured return), Typer, pytest + `asyncio.run`.

## Ground truth (verified)
- `_query_cli_hit(hit, max_text_chars=None)` lives in `adapters/cli/main.py` (post-3c); query prints `[_query_cli_hit(hit, max_text_chars) for hit in hits]`.
- `query_documents` accepts `top_k, candidate_k, page_dedup, content_types, lancedb_uri, table_name, embed_invoke_url, embed_model_name, reranker_*, rerank, hybrid` (first positional arg is the query string). `verify_claim(claim, source, *, page, lancedb_uri, table_name, against)` exists. `DEFAULT_LANCEDB_URI`/`DEFAULT_TABLE_NAME` are in `sdk_workflow.py`.
- main.py imports SDK fns in a block ending `    verify_claim,\n)`, then `from nemo_retriever.vdb.records import RetrievalHit`.
- CLI: commands via `@app.command(...)`; `verify_command` ends with `typer.echo(json.dumps(result, indent=2, sort_keys=True, default=str))` immediately before `@app.callback()`.
- FastMCP: `@mcp.tool` (bare), `mcp.run()` (stdio default); `Client(mcp)` in-memory; `await client.call_tool(name, args)` → `CallToolResult` with `.data` = structured return; `await client.list_tools()` → tools with `.name`. `retriever mcp --help` shows help without running the server (Typer).

All commits `--no-gpg-sign`. Unit tests are no-GPU; `doctor` needs GPU for its live probe.

---

### Task 1: Move `_query_cli_hit` to a shared module (behavior-preserving refactor)

**Files:** create `nemo_retriever/src/nemo_retriever/adapters/cli/_hit_format.py`; modify `adapters/cli/main.py`

- [ ] **Step 1: Create `_hit_format.py`** with the current shaper verbatim:
```python
# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-sourced query-hit shaping, shared by the CLI and the MCP surface."""

from __future__ import annotations

from nemo_retriever.vdb.records import RetrievalHit


def _query_cli_hit(hit: RetrievalHit, max_text_chars: int | None = None) -> dict[str, object]:
    metadata = hit.get("metadata") or {}
    modality = hit.get("content_type") or metadata.get("type") or "text"
    # Relevance the engine ranked by: rerank/hybrid score if present, else the
    # vector distance, else null. Hit ORDER is authoritative; score is informational.
    if "_score" in hit:
        score: object = hit["_score"]
    elif "_distance" in hit:
        score = hit["_distance"]
    else:
        score = None
    text = hit.get("text", "")
    # Compact output: truncate text to max_text_chars (0 = omit -> metadata-only
    # summary). None/negative = full text (default, backward-compatible).
    if max_text_chars is not None and max_text_chars >= 0 and len(text) > max_text_chars:
        text = text[:max_text_chars] + ("…" if max_text_chars > 0 else "")
    return {
        "source": hit.get("source", ""),
        "page_number": hit.get("page_number"),
        "text": text,
        "modality": modality,
        "score": score,
    }
```

- [ ] **Step 2: Import it in `main.py`** — after the line `from nemo_retriever.vdb.records import RetrievalHit`, add:
```python
from nemo_retriever.adapters.cli._hit_format import _query_cli_hit
```

- [ ] **Step 3: Remove the old `_query_cli_hit` def from `main.py`.** Replace the entire block:
```python
def _query_cli_hit(hit: RetrievalHit, max_text_chars: int | None = None) -> dict[str, object]:
    metadata = hit.get("metadata") or {}
    modality = hit.get("content_type") or metadata.get("type") or "text"
    # Relevance the engine ranked by: rerank/hybrid score if present, else the
    # vector distance, else null. Hit ORDER is authoritative; score is informational.
    if "_score" in hit:
        score: object = hit["_score"]
    elif "_distance" in hit:
        score = hit["_distance"]
    else:
        score = None
    text = hit.get("text", "")
    # Compact output: truncate text to max_text_chars (0 = omit -> metadata-only
    # summary). None/negative = full text (default, backward-compatible).
    if max_text_chars is not None and max_text_chars >= 0 and len(text) > max_text_chars:
        text = text[:max_text_chars] + ("…" if max_text_chars > 0 else "")
    return {
        "source": hit.get("source", ""),
        "page_number": hit.get("page_number"),
        "text": text,
        "modality": modality,
        "score": score,
    }
```
with:
```python
# _query_cli_hit moved to adapters/cli/_hit_format.py (shared with the MCP surface)
```

- [ ] **Step 4: Verify the refactor is behavior-preserving (no GPU)**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python -m pytest nemo_retriever/tests/test_root_query_cli.py -q`
Expected: all pass (query output identical — `_query_cli_hit` is now imported, same code). If an import error occurs, confirm Step 2's import path.

- [ ] **Step 5: Commit**
```bash
git add nemo_retriever/src/nemo_retriever/adapters/cli/_hit_format.py nemo_retriever/src/nemo_retriever/adapters/cli/main.py
git commit --no-gpg-sign -m "refactor(retriever): single-source _query_cli_hit in _hit_format (for CLI + MCP reuse)"
```

---

### Task 2: MCP server module + `retriever mcp` command + tests

**Files:** create `nemo_retriever/src/nemo_retriever/adapters/cli/mcp_server.py`; modify `adapters/cli/main.py`; create `nemo_retriever/tests/test_mcp_server.py`

- [ ] **Step 1: Create `mcp_server.py`**
```python
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
```

- [ ] **Step 2: Add the `mcp` command to `main.py`** — replace:
```python
    typer.echo(json.dumps(result, indent=2, sort_keys=True, default=str))


@app.callback()
```
with:
```python
    typer.echo(json.dumps(result, indent=2, sort_keys=True, default=str))


@app.command("mcp")
def mcp_command() -> None:
    """Serve the retriever's read tools (query, verify) over MCP (stdio transport)."""
    from nemo_retriever.adapters.cli.mcp_server import mcp

    mcp.run()


@app.callback()
```

- [ ] **Step 3: Create `nemo_retriever/tests/test_mcp_server.py`**
```python
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
```

- [ ] **Step 4: Run the MCP tests (no GPU)**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python -m pytest nemo_retriever/tests/test_mcp_server.py -q`
Expected: 4 passed. (If `res.data` is not the structured payload on this FastMCP build, fall back to `res.structured_content`; the probe confirmed `.data` works.)

- [ ] **Step 5: Commit**
```bash
git add nemo_retriever/src/nemo_retriever/adapters/cli/mcp_server.py nemo_retriever/src/nemo_retriever/adapters/cli/main.py nemo_retriever/tests/test_mcp_server.py
git commit --no-gpg-sign -m "feat(retriever): add MCP surface (retriever mcp; query + verify tools over stdio)"
```

---

### Task 3: Contract 1.5.0 + skill note

**Files:** `skills/nemo-retriever/contract/cli-contract.json`, `.../CONTRACT.md`, `skills/nemo-retriever/SKILL.md`

- [ ] **Step 1: Add `mcp` to `subcommands` + bump version** — in `cli-contract.json`, replace:
```json
  "contract_version": "1.4.0",
  "subcommands": ["ingest", "query", "verify"],
```
with:
```json
  "contract_version": "1.5.0",
  "subcommands": ["ingest", "query", "verify", "mcp"],
```

- [ ] **Step 2: Changelog in `CONTRACT.md`** — replace:
```markdown
## Changelog
- **1.4.0** — `query` gains `--max-text-chars N`: truncate each hit's text to N chars (`0` = metadata-only summary; unset = full). Compact output for token economy.
```
with:
```markdown
## Changelog
- **1.5.0** — `mcp` subcommand added: serves the read tools (`query`, `verify`) over MCP (stdio) via FastMCP, so agent harnesses can call the engine directly.
- **1.4.0** — `query` gains `--max-text-chars N`: truncate each hit's text to N chars (`0` = metadata-only summary; unset = full). Compact output for token economy.
```

- [ ] **Step 3: Bump version + add MCP note in `SKILL.md`** — replace:
```
This skill targets engine **contract_version 1.4.0** (`contract/cli-contract.json`).
```
with:
```
This skill targets engine **contract_version 1.5.0** (`contract/cli-contract.json`). The engine can also be driven over MCP (`retriever mcp`, stdio) for harnesses that prefer MCP tools (`query`, `verify`) over CLI calls.
```

- [ ] **Step 4: Validate JSON + commit**
```bash
/home/edwardk/git/nv-ingest/retriever/bin/python -c "import json; json.load(open('skills/nemo-retriever/contract/cli-contract.json')); print('JSON OK')"
git add skills/nemo-retriever/contract/ skills/nemo-retriever/SKILL.md
git commit --no-gpg-sign -m "docs(skill): contract 1.5.0 + note MCP surface (retriever mcp)"
```
Expected: `JSON OK`.

---

### Task 4: Live validation

**Files:** none.

- [ ] **Step 1: Full unit suites (no GPU)**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python -m pytest nemo_retriever/tests/test_mcp_server.py nemo_retriever/tests/test_root_query_cli.py nemo_retriever/tests/test_verify.py -q`
Expected: all pass.

- [ ] **Step 2: `retriever mcp --help` works without starting the server (no GPU)**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/retriever mcp --help; echo "exit=$?"`
Expected: prints the command help, `exit=0` (Typer shows help without invoking `mcp.run()`).

- [ ] **Step 3: `doctor` asserts the `mcp` subcommand (GPU for live probe)**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python skills/nemo-retriever/scripts/doctor.py 2>&1 | grep -iE "subcommand|checks passed|FAIL"; echo "exit=${PIPESTATUS[0]}"` (run the full `doctor.py`).
Expected: ``subcommand `mcp` exists`` `[PASS]` (plus ingest/query/verify), final `N/N checks passed`, `exit=0`.

- [ ] **Step 4: No commit (validation only).** If green, slice 3b is complete.

---

## Self-review

**Spec coverage (3b):**
- `mcp_server.py` with `query` + `verify` FastMCP tools wrapping the SDK fns + `_query_cli_hit` → Task 2 Step 1. ✓
- `_query_cli_hit` single-sourced in a shared module (no circular/heavy import) → Task 1. ✓
- `retriever mcp` command, stdio, lazy import → Task 2 Step 2. ✓
- read-only (no ingest tool), stdio only → only `query`/`verify` defined. ✓
- contract 1.5.0 + `mcp` in subcommands + doctor asserts → Task 3 + Task 4 Step 3. ✓
- SKILL.md MCP note → Task 3 Step 3. ✓
- in-memory FastMCP tests (list + call query incl. max_text_chars + verify) → Task 2 Step 3. ✓

**Placeholder scan:** No TBD/TODO; all code is concrete.

**Type consistency:** the `query` tool's params (`top_k, hybrid, max_text_chars, content_types, lancedb_uri, table_name, embed_model_name, rerank`) are all accepted by `query_documents`; `verify` tool params match `verify_claim`'s signature (`claim, source, page, against, lancedb_uri, table_name`). Tests patch `mcp_server.query_documents`/`mcp_server.verify_claim` (the names bound in the MCP module, which the tools call) — the same import-binding subtlety handled in 2c. `CallToolResult.data` is the structured payload (probe-confirmed). `mcp` added to the contract `subcommands` list that `doctor` iterates; `retriever mcp --help` exits 0 without serving.
