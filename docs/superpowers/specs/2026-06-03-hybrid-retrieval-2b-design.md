# Objective 2b — opt-in hybrid retrieval (`--hybrid` on ingest + query)

**Date:** 2026-06-03
**Type:** Engine change to `nemo_retriever` (second slice of rebuild Objective 2)
**Parent:** `2026-06-03-skill-first-retriever-rebuild-design.md` (Objective 2). Follows 2a (typed-hit surfacing).

## Context & decision

Objective 2's multi-strategy work, per the earlier decision, **leans on the existing LanceDB hybrid (vector + full-text/BM25)** rather than building a new RRF fusion operator. The adoption decision is **opt-in `--hybrid` flag on both `ingest` and `query`, default OFF** — safe, explicit, zero behavior change for existing callers.

## Key finding (from a code trace)

The engine **already supports hybrid end-to-end**; only the CLI surface is missing:
- `LanceDB.__init__(hybrid=…)` exists; `create_index()` builds the FTS index `if hybrid` (`vdb/lancedb.py:~515`, `table.create_fts_index("text", …)`).
- `retrieval()` runs the hybrid branch and **requires `query_texts`** (`vdb/lancedb.py:~670`, raises without it).
- `RetrieveVdbOperator.process()` **already threads `query_texts`** to `retrieval()` whenever hybrid is effective (`vdb/operators.py:224-225`), and `query_texts` is already passed through graph execution (`retriever.py:~298`).
- `Retriever` forwards `vdb_kwargs` verbatim to the VDB constructor via `_coerce_vdb_init` (`retriever.py:~242`), so a `hybrid` key in `vdb_kwargs` reaches `LanceDB(hybrid=True)` with no new threading.

So there is **no `query_texts` risk** and **no internal plumbing to add** — 2b is wiring two CLI flags into the `vdb_kwargs` that already flow down.

## Dependency

Hybrid query needs an FTS index, which is built **only by a `--hybrid` ingest**. So the skill workflow becomes: `ingest --hybrid` (setup turn) → `query --hybrid` (query turns). A `--hybrid` query against a non-hybrid-ingested table has no FTS index; that is an accepted limitation of the opt-in model (documented), not handled by fallback in this slice.

## Design

### Query path
- Add `--hybrid` (`typer.Option(False)`) to `query_command` (`adapters/cli/main.py`).
- Pass `hybrid=hybrid` into the `query_documents(...)` call.
- In `query_documents` (`adapters/cli/sdk_workflow.py`), build `vdb_kwargs` and **add `"hybrid": True` only when `hybrid` is truthy**. When off, `vdb_kwargs` stays exactly `{"uri", "table_name"}` — preserving current behavior and the existing exact-match tests.
- No change to `Retriever`, `RetrieveVdbOperator`, or `LanceDB` (the kwarg + query_texts paths already exist).

### Ingest path
- Add `--hybrid` (`typer.Option(False)`) to `ingest_command`.
- Thread `hybrid` through `ingest_documents(...)` → `resolve_ingest_plan(...)`.
- In `resolve_ingest_plan`, **add `"hybrid": True` to `VdbUploadParams.vdb_kwargs` only when set** (current dict is `{"uri","table_name","overwrite"}`; off-path unchanged).
- `LanceDB(hybrid=True)` then builds the FTS index during `create_index()`.

### Conditional injection (load-bearing)
Both paths add the `hybrid` key **only when the flag is set**, so the default-off `vdb_kwargs` dicts are byte-identical to today. This keeps `test_root_query_cli.py` and `test_root_cli_workflow.py` exact-match assertions valid and guarantees zero behavior change when the flag is absent.

### Contract + skill (consistent with 2a)
- `cli-contract.json`: bump `contract_version` 1.1.0 → **1.2.0**; add `--hybrid` to `query.required_flags` and `ingest.required_flags` (so `doctor`'s static flag-surface check asserts the flag now exists on both subcommands).
- `CONTRACT.md`: changelog entry for 1.2.0.
- `SKILL.md`: declare contract 1.2.0.
- `setup.md`: note that `--hybrid` on ingest builds the FTS index enabling hybrid query (lexical + semantic recall).
- `query.md`: document `--hybrid` for queries (combines vector + full-text; catches exact terms semantic search misses) and that it requires a `--hybrid`-ingested index.

## The win

With `--hybrid`, the skill's single disciplined query becomes vector + BM25 — folding the exact-term recall that our workflow's separate `keyword` angle provided into one engine call, opt-in. No orchestration, no subagents.

## Testing

- **Query CLI** (`tests/test_root_query_cli.py`): a new test passing `--hybrid` asserts `vdb_kwargs == {"uri":…, "table_name":…, "hybrid": True}`; confirm an existing no-flag test still asserts the dict **without** a `hybrid` key.
- **Ingest CLI** (`tests/test_root_cli_workflow.py`): a new test passing `--hybrid` asserts the ingest `vdb_kwargs` includes `"hybrid": True`; the existing no-flag test still asserts the dict without it.
- **Live round-trip** (the real gate): `retriever ingest --hybrid` a tiny corpus (must build the FTS index without error), then `retriever query --hybrid "<q>"` returns hits and does not raise. Plus `doctor.py` stays green with `--hybrid` now asserted in the flag surface.

## Non-goals

- No RRF fusion operator, no new "strategies" registry (that was the *other* 2b option; explicitly out per the lean-on-hybrid decision).
- No auto-detection / vector fallback when an index lacks an FTS index (the opt-in model documents the ingest+query pairing instead).
- No distinct visual/tabular strategies (those remain `--content-types` filters).
- No change to ranking internals, reranking, or `_query_cli_hit` (2a already did the hit shape).

## Open question

- Hybrid relevance comes back as LanceDB's fused `_score`; 2a's `score` field already surfaces it. No score-normalization work needed here; revisit only if a later slice introduces cross-strategy RRF.
