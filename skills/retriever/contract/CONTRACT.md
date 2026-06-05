# retriever skill↔engine contract

`contract_version` (see `cli-contract.json`) is the semver the **skill** asserts
about the installed **engine**. Run `scripts/doctor.py` to verify the installed
`retriever` satisfies it.

The skill's one primitive is **`retriever retrieve <question>`** → `{ evidence, coverage }`.
The contract is defined around that primitive — not around the engine's CLI flag
surface. `query`/`verify`/`mcp` still ship but the skill does not depend on them;
they are documented under "Legacy" and are NOT gated by `doctor.py`.

## Files
- `cli-contract.json` — the gated surface: required subcommands, `retrieve`'s
  required + forbidden flags, `ingest`'s flags, and a `legacy` block for the
  ungated commands. `default_table_name` is the engine's table-name constant
  (operator config), not the skill name.
- `retrieve-result.schema.json` — the shape `retriever retrieve` emits and the
  skill reasons over: `evidence[]` (each with `text, source, locator, modality,
  fidelity, score, citation`) + `coverage`. This is THE contract the skill relies on.
- `legacy-query-hit.schema.json` — the shape the deprecated `retriever query`
  emits. Kept for reference only; the skill no longer parses it.

## Versioning
- Bump **patch** for clarifications, **minor** for additive engine capabilities the
  skill can use, **major** when the engine changes something the skill relies on
  (a `retrieve` evidence/coverage field, the `retrieve` flag surface, or the gated
  primitive). A major bump means the skill must be updated in the same change.
- `doctor.py` fails if the installed engine no longer matches `cli-contract.json` /
  `retrieve-result.schema.json`.

## How drift gets caught
`doctor.py` runs on the skill's setup turn and in CI (`tests/test_contract.py`). It
performs a LIVE probe — ingest a tiny fixture, run `retrieve`, validate
`{evidence, coverage}` (including the `fidelity` enum) against
`retrieve-result.schema.json` — plus static `--help` checks: the required
subcommands exist, `retrieve` exposes its required flags, and `retrieve` does NOT
expose strategy knobs (`--content-types`, `--rerank`, …). Any divergence (a renamed
evidence field, a missing `fidelity`, a strategy knob leaking onto `retrieve`,
`--input-type` reappearing on `ingest`) fails loudly with a remediation hint.

## Legacy (not gated)
`query`, `verify`, and `mcp`'s `query`/`verify` tools still exist for callers that
want raw hits, but the `retriever` skill routes everything through `retrieve`. If a
future skill revision adopts `verify` as a first-class move, promote it out of the
`legacy` block and add a gated check + schema.

## Changelog
- **2.0.0** — contract re-centered on the `retrieve` primitive. `doctor.py` now
  live-probes `retrieve` and validates `retrieve-result.schema.json` (evidence +
  coverage, `fidelity` enum) instead of `query`/`actual-hit`. `retrieve`'s
  strategy knobs are forbidden-checked. `query`/`verify`/`mcp` demoted to a
  `legacy` block; `actual-hit.schema.json` renamed `legacy-query-hit.schema.json`;
  the never-emitted `target-hit.schema.json` dropped (superseded by
  `retrieve-result.schema.json`). `table_columns` removed (only the retired
  `grep_corpus.py` read the physical table). MAJOR: the gated primitive changed.
- **1.6.0** — `serve-models` subcommand added: launches a warm vLLM embeddings server (`--runner pooling`) and prints `export EMBED_INVOKE_URL=…`, so `retrieve`/`query` avoid the per-query cold-load. (Reranker-warm deferred — vLLM lacks the `/v1/ranking` path `rerank.py` needs; see the 3a spec.)
- **1.5.0** — `mcp` subcommand added: serves the read tools over MCP (stdio) via FastMCP, so agent harnesses can call the engine directly.
- **1.4.0** — `query` gains `--max-text-chars N`: truncate each hit's text to N chars (`0` = metadata-only summary; unset = full). Compact output for token economy.
- **1.3.0** — `verify` subcommand added: fetches independent `text`/`table` evidence for a claim's (source, page) + a mechanical term/number-overlap signal. Engine retrieves; caller judges agreement.
- **1.2.0** — `--hybrid` flag added to `ingest` (builds the BM25/FTS index) and `query` (vector + full-text retrieval). Opt-in; a `--hybrid` query needs a `--hybrid`-built index.
- **1.1.0** — query hits gained `modality` (required) and `score` (optional).
- **1.0.0** — initial contract (lean `{page_number, source, text}` hit, flag surface, table conventions).
