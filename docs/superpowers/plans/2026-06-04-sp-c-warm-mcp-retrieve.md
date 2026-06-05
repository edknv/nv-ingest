# SP-C — warm MCP `retrieve` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `retrieve` honor a warm `serve-models` endpoint (via `EMBED_INVOKE_URL`) and expose `retrieve` as an MCP tool — so the skill's primary tool is warm and reachable over MCP.

**Architecture:** (1) Add `embed_invoke_url` to the SP-B `retrieve` (default from env), threaded into `query_documents`. (2) Add a `retrieve` MCP tool mirroring the existing `query`/`verify` wrappers. Pure wiring; no daemon, no retrieval/ingest change; `index` stays CLI.

**Tech Stack:** Python 3.12, editable `nemo_retriever` (`./retriever` venv), FastMCP 3.2.4, pytest. Unit/MCP tests no-GPU; light-live (`--help`/`doctor`).

## Ground truth (verified)
- `retrieve` (SP-B, `adapters/cli/sdk_workflow.py`) currently threads only `embed_model_name`; its `_run(use_hybrid)` calls `query_documents(question, top_k=, hybrid=, lancedb_uri=, table_name=, embed_model_name=)`. `query_documents` accepts `embed_invoke_url`.
- The SDK does not read `EMBED_INVOKE_URL`; only `main.py:643` (CLI `query`) does.
- `mcp_server.py` (3b) has `query` + `verify` tools, imports `DEFAULT_LANCEDB_URI`/`DEFAULT_TABLE_NAME` + SDK fns; `fastmcp.Client(mcp)` in-memory test pattern established in `tests/test_mcp_server.py` (`CallToolResult.data`).

All commits `--no-gpg-sign`.

---

### Task 1: `retrieve` honors `EMBED_INVOKE_URL`

**Files:** Modify `nemo_retriever/src/nemo_retriever/adapters/cli/sdk_workflow.py`; Modify `nemo_retriever/tests/test_retrieve.py`

- [ ] **Step 1: Write the failing tests** — append to `nemo_retriever/tests/test_retrieve.py`:
```python


def test_retrieve_threads_embed_invoke_url_from_env(monkeypatch):
    captured = {}

    def fake_qd(question, **k):
        captured.update(k)
        return [_hit("p")]

    monkeypatch.setattr(sw, "query_documents", fake_qd)
    monkeypatch.setenv("EMBED_INVOKE_URL", "http://127.0.0.1:8081/v1/embeddings")
    sw.retrieve("q", lancedb_uri="x", table_name="t")
    assert captured.get("embed_invoke_url") == "http://127.0.0.1:8081/v1/embeddings"


def test_retrieve_explicit_embed_invoke_url_overrides_env(monkeypatch):
    captured = {}
    monkeypatch.setattr(sw, "query_documents", lambda question, **k: captured.update(k) or [_hit("p")])
    monkeypatch.setenv("EMBED_INVOKE_URL", "http://env/v1/embeddings")
    sw.retrieve("q", lancedb_uri="x", table_name="t", embed_invoke_url="http://explicit/v1/embeddings")
    assert captured.get("embed_invoke_url") == "http://explicit/v1/embeddings"


def test_retrieve_no_env_means_no_endpoint(monkeypatch):
    captured = {}
    monkeypatch.setattr(sw, "query_documents", lambda question, **k: captured.update(k) or [_hit("p")])
    monkeypatch.delenv("EMBED_INVOKE_URL", raising=False)
    sw.retrieve("q", lancedb_uri="x", table_name="t")
    assert captured.get("embed_invoke_url") is None
```

- [ ] **Step 2: Run to confirm failure** — `/home/edwardk/git/nv-ingest/retriever/bin/python -m pytest nemo_retriever/tests/test_retrieve.py -q -k embed_invoke_url` → fails (`retrieve` doesn't pass `embed_invoke_url`).

- [ ] **Step 3: Thread `embed_invoke_url` through `retrieve`** — in `sdk_workflow.py`, replace:
```python
    embed_model_name: str | None = None,
) -> dict[str, Any]:
    """Skill-first retrieve: one fused query -> answer-ready, fidelity-tagged, cited evidence + coverage.

    Single hybrid (vector+BM25) query; if the index has no FTS index, gracefully
    falls back to vector-only. Returns the ``retrieve_result`` contract shape.
    """
    def _run(use_hybrid: bool) -> list:
        return query_documents(
            question,
            top_k=top_k,
            hybrid=use_hybrid,
            lancedb_uri=lancedb_uri,
            table_name=table_name,
            embed_model_name=embed_model_name,
        )
```
with:
```python
    embed_model_name: str | None = None,
    embed_invoke_url: str | None = None,
) -> dict[str, Any]:
    """Skill-first retrieve: one fused query -> answer-ready, fidelity-tagged, cited evidence + coverage.

    Single hybrid (vector+BM25) query; if the index has no FTS index, gracefully
    falls back to vector-only. Returns the ``retrieve_result`` contract shape.

    ``embed_invoke_url`` defaults from ``EMBED_INVOKE_URL`` (set by ``retriever
    serve-models``) so retrieval is warm when a model server is running.
    """
    import os as _os

    endpoint = embed_invoke_url if embed_invoke_url is not None else (_os.environ.get("EMBED_INVOKE_URL") or None)

    def _run(use_hybrid: bool) -> list:
        return query_documents(
            question,
            top_k=top_k,
            hybrid=use_hybrid,
            lancedb_uri=lancedb_uri,
            table_name=table_name,
            embed_model_name=embed_model_name,
            embed_invoke_url=endpoint,
        )
```

- [ ] **Step 4: Run tests to pass** — `/home/edwardk/git/nv-ingest/retriever/bin/python -m pytest nemo_retriever/tests/test_retrieve.py -q` → all pass (the 3 new + the existing 6 = 9).

- [ ] **Step 5: Commit**
```bash
git add nemo_retriever/src/nemo_retriever/adapters/cli/sdk_workflow.py nemo_retriever/tests/test_retrieve.py
git commit --no-gpg-sign -m "feat(retriever): retrieve honors EMBED_INVOKE_URL (warm via serve-models)"
```

---

### Task 2: `retrieve` MCP tool

**Files:** Modify `nemo_retriever/src/nemo_retriever/adapters/cli/mcp_server.py`; Modify `nemo_retriever/tests/test_mcp_server.py`

- [ ] **Step 1: Add the tool** — append to `mcp_server.py` (after the `verify` tool):
```python


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
```

- [ ] **Step 2: Add MCP tool tests** — append to `nemo_retriever/tests/test_mcp_server.py`:
```python


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
```
(The MCP `retrieve` tool calls `sdk_workflow.retrieve`, which calls `sdk_workflow.query_documents`; patching `query_documents` exercises the real shaping end-to-end.)

- [ ] **Step 3: Run** — `/home/edwardk/git/nv-ingest/retriever/bin/python -m pytest nemo_retriever/tests/test_mcp_server.py -q` → 6 passed (4 existing + 2 new).

- [ ] **Step 4: Commit**
```bash
git add nemo_retriever/src/nemo_retriever/adapters/cli/mcp_server.py nemo_retriever/tests/test_mcp_server.py
git commit --no-gpg-sign -m "feat(retriever): add warm 'retrieve' MCP tool (alongside query/verify)"
```

---

### Task 3: Docs + light-live validation

**Files:** Modify `docs/superpowers/contracts/retriever/README.md`

- [ ] **Step 1: Document the warm-MCP composition** — append to `docs/superpowers/contracts/retriever/README.md`:
```markdown

## Warm MCP (today's wiring)
The library already supports the contract over MCP, warm:
1. `retriever serve-models` — warm embedder; export the printed `EMBED_INVOKE_URL`.
2. Point a harness at `retriever mcp` — its `retrieve`/`query`/`verify` tools then run
   warm (`retrieve` honors `EMBED_INVOKE_URL`), no per-call cold-load.
`index` remains a CLI/setup step (`retriever ingest`), not an MCP tool.
```

- [ ] **Step 2: Light-live checks (no full warm round-trip)**

Run:
```bash
cd /home/edwardk/git/nv-ingest
./retriever/bin/python -m pytest nemo_retriever/tests/test_retrieve.py nemo_retriever/tests/test_mcp_server.py -q 2>&1 | tail -2
./retriever/bin/retriever mcp --help >/dev/null && echo "mcp cmd OK"
./retriever/bin/python skills/nemo-retriever/scripts/doctor.py 2>&1 | grep -iE "subcommand .mcp|checks passed|FAIL"; echo "doctor_exit=${PIPESTATUS[0]}"
```
Expected: all unit tests pass; `mcp cmd OK`; doctor shows ``subcommand `mcp` exists`` `[PASS]` and `N/N checks passed`, exit 0.

- [ ] **Step 3: Commit docs**
```bash
git add docs/superpowers/contracts/retriever/README.md
git commit --no-gpg-sign -m "docs(retriever-contract): warm-MCP wiring (serve-models + EMBED_INVOKE_URL + retriever mcp)"
```

---

## Self-review

**Spec coverage (SP-C):**
- `retrieve` honors `EMBED_INVOKE_URL` (env default, explicit override, env-unset) → Task 1 (3 tests + the env-default + pass-through). ✓
- `retrieve` MCP tool added; query/verify kept; index not exposed → Task 2 (tool + `test_mcp_exposes_retrieve_tool` asserts retrieve+query+verify present; no index tool added). ✓
- Returns `retrieve_result` shape over MCP → `test_mcp_retrieve_tool_returns_contract_shape` (real shaping via patched `query_documents`). ✓
- Warm-MCP docs → Task 3 Step 1. ✓
- Light-live (mcp --help, doctor) → Task 3 Step 2. ✓
- No daemon / ingest / ranking change; no contract-version bump → only `sdk_workflow.py` (1 param), `mcp_server.py` (1 tool), README touched. ✓

**Placeholder scan:** No TBD/TODO; all code complete.

**Type consistency:** `embed_invoke_url` is the exact `query_documents` param name (verified). The MCP `retrieve` tool's params match the SP-B `retrieve` signature (it omits `embed_invoke_url`, relying on the env default — intentional, so the agent doesn't pass URLs). Tests patch `sdk_workflow.query_documents` (the name `retrieve` calls). `res.data` is the FastMCP structured payload (per 3b probe). `retrieve` tool returns the `retrieve_result` dict from SP-B unchanged.
