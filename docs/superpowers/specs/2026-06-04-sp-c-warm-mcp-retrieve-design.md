# SP-C — warm MCP `retrieve` (design)

**Date:** 2026-06-04
**Type:** Engine change to `nemo_retriever` (third library sub-project of the skill-first design)
**Parent:** `2026-06-04-retriever-skill-first-design.md`. Depends on SP-B (`retrieve`), composes Objective-3a (`serve-models`) + 3b (`retriever mcp`).

## Problem

The skill calls `retrieve` over MCP, ideally **warm** (no per-call cold-load). Two gaps:
1. SP-B's `retrieve` threads only `embed_model_name` — it does **not** accept/honor a warm endpoint, and the SDK never reads `EMBED_INVOKE_URL` from env (only the CLI's `query_command` does). So `retrieve` always cold-loads even when `serve-models` is up. (`main.py:743` already *claims* "query/verify/MCP honor that env var" — currently false for the SDK/MCP.)
2. The MCP server (3b) exposes `query` + `verify` but **no `retrieve`** — the skill-first primitive isn't reachable over MCP.

## Decisions (from brainstorming)
- Add a `retrieve` MCP tool; keep `query`/`verify`; **`index` stays a CLI/setup step** (heavy/mutating, not an agent tool).
- Warmth via env: `retrieve` honors `EMBED_INVOKE_URL` (set by `serve-models`).
- Validation: unit + light-live (no full serve-models warm round-trip — warmth already proven in 3a).

## Design

### 1. `retrieve` honors the warm endpoint (`adapters/cli/sdk_workflow.py`)
Add `embed_invoke_url: str | None = None` to `retrieve`; when `None`, default it from `os.environ.get("EMBED_INVOKE_URL")`; pass it into the `query_documents(...)` calls (both hybrid and fallback). Effect: `retrieve` (CLI, SDK, MCP) is **warm** whenever `serve-models` is running and `EMBED_INVOKE_URL` is exported — no per-call cold-load. Default (env unset) is unchanged behavior. The CLI `retrieve` verb (SP-B) needs no change (env default flows through); optionally it gains an explicit `--embed-invoke-url` passthrough for parity with `query`.

### 2. `retrieve` MCP tool (`adapters/cli/mcp_server.py`)
Add a third tool mirroring the existing `query`/`verify` wrappers, calling the SP-B `retrieve`:
```python
@mcp.tool
def retrieve(question, top_k=10, hybrid=True, lancedb_uri=DEFAULT_LANCEDB_URI,
            table_name=DEFAULT_TABLE_NAME, embed_model_name=None) -> dict:
    """Answer-ready, fidelity-tagged, cited evidence + coverage for a question.
    Warm when serve-models is running (honors EMBED_INVOKE_URL)."""
    from nemo_retriever.adapters.cli.sdk_workflow import retrieve as _retrieve
    return _retrieve(question, top_k=top_k, hybrid=hybrid, lancedb_uri=lancedb_uri,
                     table_name=table_name, embed_model_name=embed_model_name)
```
It returns the `retrieve_result` contract shape; warmth is automatic via #1's env default.

### 3. Docs
- The contracts README / `serve-models` docs note the warm-MCP composition: run `serve-models`, export `EMBED_INVOKE_URL`, then point a harness at `retriever mcp` — its `retrieve` tool is warm.

## Components & boundaries
- `retrieve` (SDK) — one new optional param + env default + pass-through. No logic change.
- `retrieve` MCP tool — thin wrapper (mirrors query/verify), no new logic.
- **Boundary:** SP-C wires warmth + exposure; it does not add a daemon (3a) or change retrieval/ingest. `index` is not exposed over MCP.

## Testing
- **Unit (no GPU):**
  - `retrieve` env-threading: monkeypatch `query_documents` to capture kwargs; set `EMBED_INVOKE_URL`; assert `retrieve(...)` passes that `embed_invoke_url`; assert an explicit `embed_invoke_url=` arg overrides env; assert no env → `embed_invoke_url is None`.
  - MCP `retrieve` tool: `fastmcp.Client(mcp)` in-memory — `list_tools()` includes `retrieve`; `call_tool("retrieve", {"question": "q"})` returns the (monkeypatched) `retrieve_result`.
- **Light live:** `retriever mcp --help` exits 0; `doctor` stays green (still asserts the `mcp` subcommand). (Full `serve-models`→warm-MCP round-trip skipped — 3a already proved warmth.)

## Non-goals
- No `index` MCP tool (CLI/setup only).
- No new daemon / serving code (reuse `serve-models`).
- No reranker-warm (deferred; `retrieve` doesn't rerank by default).
- No contract-version bump (additive wiring; CLI flag/subcommand surface unchanged).

## Open questions
- Whether to also thread `RERANKER_INVOKE_URL` — deferred with reranker-warm.
- Whether the CLI `retrieve` verb should expose `--embed-invoke-url` explicitly (parity) or rely solely on the env default — lean: add the flag for parity, env remains the default.
