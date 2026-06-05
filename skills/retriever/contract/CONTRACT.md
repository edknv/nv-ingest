# nemo-retriever skill↔engine contract

`contract_version` (see `cli-contract.json`) is the semver the **skill** asserts
about the installed **engine**. Run `scripts/doctor.py` to verify the installed
`retriever` satisfies it.

## Files
- `cli-contract.json` — required/forbidden CLI flags, default table name, table columns.
- `actual-hit.schema.json` — the shape `retriever query` emits TODAY; what the skill parses.
- `target-hit.schema.json` — the agent-native hit we are migrating toward (doc_id, modality,
  fidelity, provenance). Not yet emitted; documented so the skill and engine evolve together.

## Versioning
- Bump **patch** for clarifications, **minor** for additive engine capabilities the skill can
  use, **major** when the engine changes something the skill relies on (a hit field, a flag,
  the default table). A major bump means the skill must be updated in the same change.
- The skill declares the version it needs in `SKILL.md`. `doctor.py` fails if the installed
  engine no longer matches `cli-contract.json` / `actual-hit.schema.json`.

## How drift gets caught
`doctor.py` runs on the skill's setup turn and in CI (`tests/test_contract.py`). It performs a
LIVE probe — ingest a tiny fixture, run a query, validate the hit against `actual-hit.schema.json`
— plus static `--help` flag-surface checks. Any divergence (e.g. a renamed field, a removed flag,
`--input-type` reappearing) fails loudly with a remediation hint.

## Changelog
- **1.6.0** — `serve-models` subcommand added: launches a warm vLLM embeddings server (`--runner pooling`) and prints `export EMBED_INVOKE_URL=…`, so `query` avoids the per-query cold-load. (Reranker-warm deferred — vLLM lacks the `/v1/ranking` path `rerank.py` needs; see the 3a spec.)
- **1.5.0** — `mcp` subcommand added: serves the read tools (`query`, `verify`) over MCP (stdio) via FastMCP, so agent harnesses can call the engine directly.
- **1.4.0** — `query` gains `--max-text-chars N`: truncate each hit's text to N chars (`0` = metadata-only summary; unset = full). Compact output for token economy.
- **1.3.0** — `verify` subcommand added: fetches independent `text`/`table` evidence for a claim's (source, page) + a mechanical term/number-overlap signal. Engine retrieves; caller judges agreement.
- **1.2.0** — `--hybrid` flag added to `ingest` (builds the BM25/FTS index) and `query` (vector + full-text retrieval). Opt-in; a `--hybrid` query needs a `--hybrid`-built index.
- **1.1.0** — query hits now carry `modality` (required) and `score` (optional); see `actual-hit.schema.json`. First step from `actual-hit` toward `target-hit`.
- **1.0.0** — initial contract (lean `{page_number, source, text}` hit, flag surface, table conventions).
