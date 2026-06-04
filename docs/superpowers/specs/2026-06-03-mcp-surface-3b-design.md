# Objective 3b — MCP surface (`retriever mcp`)

**Date:** 2026-06-03
**Type:** Engine addition to `nemo_retriever` (second slice of rebuild Objective 3)
**Parent:** `2026-06-03-skill-first-retriever-rebuild-design.md` (Objective 3 = agent-economics). Follows 3c (compact I/O).

## Context

Objective 3 decomposes into 3c (compact I/O — done), **3b (this spec — MCP surface)**, and 3a (warm query daemon — later, decided: build in-process warm daemon). 3b exposes the retriever's read operations as MCP tools so any agent harness consumes them directly with typed I/O, reducing the skill toward tool-wiring.

## Feasibility (verified)

- `fastmcp` **3.2.4** is installed AND a declared dependency (`pyproject.toml: "fastmcp>=2.0.0"`). `FastMCP`, the `@m.tool` decorator, `m.run(transport=…)` (defaults to **stdio**), and an in-memory `fastmcp.Client` (for tests) are all available. No dependency work.
- The CLI mounts subapps via `app.add_typer(...)` and registers commands with `@app.command(...)` (`adapters/cli/main.py:43,68`).
- SDK functions to wrap already exist: `query_documents`, `verify_claim` (`adapters/cli/sdk_workflow.py`).

So 3b is a thin wrapper: no new retrieval logic, no new deps.

## Design

### New module: `adapters/cli/mcp_server.py`
```python
mcp = FastMCP("nemo-retriever")

@mcp.tool
def query(question, *, top_k=10, hybrid=False, max_text_chars=None,
          content_types=None, lancedb_uri=DEFAULT_LANCEDB_URI,
          table_name=DEFAULT_TABLE_NAME, embed_model_name=None, rerank=False) -> list[dict]:
    hits = query_documents(question, top_k=top_k, hybrid=hybrid, content_types=content_types,
                           lancedb_uri=lancedb_uri, table_name=table_name,
                           embed_model_name=embed_model_name, rerank=rerank)
    return [_query_cli_hit(h, max_text_chars) for h in hits]

@mcp.tool
def verify(claim, source, *, page=None, against="text,table",
           lancedb_uri=DEFAULT_LANCEDB_URI, table_name=DEFAULT_TABLE_NAME) -> dict:
    return verify_claim(claim, source, page=page, against=against,
                        lancedb_uri=lancedb_uri, table_name=table_name)
```
FastMCP auto-generates each tool's input schema from the typed signature. The `query` tool reuses `_query_cli_hit` (from 2a/2c) so MCP output carries the same `modality`/`score` fields and honors `max_text_chars`. To avoid a circular import, `_query_cli_hit` is the one helper `mcp_server` imports from `main` (or, cleaner, `_query_cli_hit` is moved to a small shared module both import — see Components).

### CLI command (`main.py`)
```python
@app.command("mcp")
def mcp_command() -> None:
    """Serve the retriever's read tools (query, verify) over MCP (stdio)."""
    from nemo_retriever.adapters.cli.mcp_server import mcp
    mcp.run()  # stdio transport by default
```

### Tools exposed (read-only, v1)
- `query` → `query_documents` + `_query_cli_hit` shaping.
- `verify` → `verify_claim`.
- **No `ingest` tool** — ingest is a heavy/mutating GPU setup step; exposing it as an agent-callable tool is a larger, riskier surface left out of v1 (easy to add later).

### Transport
stdio (`mcp.run()` default) — the transport local agent harnesses (Claude Code, etc.) use. HTTP/SSE is a future option, not in this slice.

## Components & boundaries

- `mcp_server.py` — defines the `FastMCP` instance + the two tool wrappers; depends only on the SDK functions + `_query_cli_hit`. One clear responsibility (the MCP surface).
- `_query_cli_hit` — currently a private helper in `main.py`. To let `mcp_server` reuse it without importing `main` (which builds the whole Typer app), **move `_query_cli_hit` into a tiny shared module** `adapters/cli/_hit_format.py` and import it from both `main.py` and `mcp_server.py`. This is a targeted refactor that keeps the hit-shaping single-sourced (avoids drift between CLI and MCP output).
- `mcp_command` in `main.py` — thin launcher; imports `mcp` lazily inside the function so normal CLI startup doesn't pay the FastMCP import cost.

## Contract + skill (consistent with prior slices)

- `cli-contract.json`: bump `contract_version` 1.4.0 → **1.5.0**; add `"mcp"` to the `subcommands` list so `doctor` asserts `retriever mcp --help` works.
- `CONTRACT.md`: changelog for 1.5.0.
- `SKILL.md`: declare contract 1.5.0 + a one-line note that the engine can be driven over MCP via `retriever mcp` (stdio) for harnesses that prefer MCP tools over CLI calls.

## Testing

- **Unit (no GPU)**, `nemo_retriever/tests/test_mcp_server.py`:
  - Monkeypatch the SDK functions the tools call; use `fastmcp.Client(mcp)` in-memory via `asyncio.run`:
    - `list_tools()` includes `query` and `verify`.
    - `call_tool("query", {"question": "q"})` returns the (mocked) hits, shaped through `_query_cli_hit` (has `modality`/`score`); passing `max_text_chars=0` yields empty `text`.
    - `call_tool("verify", {"claim": "c", "source": "doc"})` returns the (mocked) verdict dict.
- **Live gate:** `doctor` stays green with `mcp` asserted in `subcommands` (`retriever mcp --help` exits 0).

## Non-goals

- No `ingest` tool (read-only v1).
- No HTTP/SSE transport (stdio only).
- No new retrieval/verify logic — pure wrapping.
- Not the warm daemon (3a); MCP tools still construct models in-process per call until 3a lands (MCP + 3a compose later: the daemon makes these tool calls warm).

## Open question

- Whether `query`'s MCP default should be compact (`max_text_chars` small) for agent economics. Kept at parity with the CLI (full text default) for least surprise; can revisit when MCP is used in anger.
