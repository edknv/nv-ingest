# SP-B — the `retrieve` primitive (design)

**Date:** 2026-06-04
**Type:** Engine change to `nemo_retriever` (second library sub-project of the skill-first design)
**Parent:** `2026-06-04-retriever-skill-first-design.md`. Depends on SP-A (stored `fidelity`).

## Problem

The skill calls one tool, `retrieve(question)`, expecting answer-ready, fidelity-tagged, cited evidence + coverage (the contract's `retrieve_result`). Today the agent must run `query`, know flags, and parse lean hits. SP-B builds the single primitive that returns the contract shape — composing the fused query + SP-A's stored fidelity + citations + coverage.

## Decisions (from brainstorming)

- **Fusion = single hybrid query** (vector + BM25): one call returns mixed-modality, fused results. Lean; reuses 2b.
- **Graceful fallback**: try hybrid; if the index lacks an FTS index (not built with `--hybrid`), catch and retry vector-only. Works regardless of how the corpus was ingested.
- **Surface = SDK fn + CLI `retrieve` verb.** MCP exposure + warm serving are SP-C.

## Grounding (verified)

`query_documents(question, hybrid=…, top_k=…)` returns full `RetrievalHit`s. `_normalize_hit` exposes per hit: `text`, `source`/`source_id`, `page_number`, `pdf_basename`, `content_type`, `_distance`/`_score`, and **`metadata`** (the parsed stored content_metadata — contains SP-A's `fidelity`, plus `type`, `bbox_xyxy_norm`, segment/timestamp). So every `evidence_item` field is derivable from a hit.

## Design

### SDK: `retrieve` in `adapters/cli/sdk_workflow.py`
```
retrieve(question, *, top_k=10, hybrid=True, lancedb_uri, table_name, embed_model_name=None) -> dict
```
1. **Fused query with fallback:**
   ```
   try: hits = query_documents(question, top_k=top_k, hybrid=hybrid, lancedb_uri=…, table_name=…, embed_model_name=…)
        strategies = ["semantic", "lexical"] if hybrid else ["semantic"]
   except Exception: hits = query_documents(question, top_k=top_k, hybrid=False, …)  # FTS absent → vector-only
        strategies = ["semantic"]
   ```
2. **Shape each hit → `evidence_item`:**
   - `text` = hit text; `source` = `pdf_basename` or basename(source).
   - `locator`: page_number → `{kind:"page", value:N}`; else `metadata.segment_start_seconds` → `{kind:"segment", value}`; else `metadata.frame_timestamp_seconds` → `{kind:"timestamp", value}`; else `metadata.bbox_xyxy_norm` → `{kind:"bbox", value}`; else `{kind:"page", value:None}`.
   - `modality` = `content_type` (or `metadata.type`), default `"text"`.
   - `fidelity` = `metadata.fidelity` if present (SP-A), **else** `_derive_fidelity(modality, metadata, metadata)` (reuse SP-A's mapping — handles pre-SP-A indexes).
   - `score` = `_score` if present, else `_distance`, else `None`.
   - `citation` = `f"{source} p.{N}"` (page) / `f"{source} @{value}"` (segment/timestamp) / `f"{source}"`.
3. **`coverage`:**
   - `strategies_used` = the strategies that actually ran (per fallback).
   - `n_docs_seen` = count of distinct `source` across evidence.
   - `thin_spots` (heuristics): `[]` normally; `["no matches — likely out of corpus"]` if empty; `+["single source"]` if `n_docs_seen == 1` and evidence non-empty; `+["only low-fidelity (chart/image) evidence"]` if all `fidelity == "vlm_caption"`.
4. Return `{"evidence": [...], "coverage": {...}}` — matching `contracts/retriever/contract.schema.json` `$defs.retrieve_result`.

### CLI: `@app.command("retrieve")`
`retriever retrieve "<question>" --top-k --hybrid/--no-hybrid --lancedb-uri --table-name --embed-model-name` → prints the result dict as JSON (same clean-stdout convention as `query`).

## Components & boundaries
- `retrieve` (SDK) — pure composition over `query_documents` + hit-shaping + coverage; reuses `_derive_fidelity` (SP-A) for the fidelity fallback. No new retrieval logic.
- `retrieve` CLI verb — thin wrapper printing JSON.
- **Boundary:** SP-B returns the contract shape over the *existing* retrieval; it does not add warm serving or MCP (SP-C), and does not change ingest or ranking.

## Testing
- **Unit (no GPU):** monkeypatch `query_documents` to return crafted hits (with/without `metadata.fidelity`, mixed modalities, page vs segment locators); assert `retrieve` produces the right `evidence_item`s (locator kind, citation string, fidelity incl. fallback) and `coverage` (strategies_used, n_docs_seen, thin_spots for empty/single-source/low-fidelity). Assert each evidence item validates against `contract.schema.json` `$defs.evidence_item` (load the schema, check required keys + enum membership).
- **Fallback unit:** make the first `query_documents(hybrid=True)` raise, assert it retries with `hybrid=False` and `strategies_used == ["semantic"]`.
- **CLI unit:** monkeypatch `retrieve`, assert the `retrieve` command prints the JSON and exits 0.
- **Live (GPU):** `retriever retrieve "<q>"` against a real index returns evidence with `fidelity`/`citation`/`coverage` populated; confirm graceful fallback on a non-hybrid index (no crash).

## Non-goals
- No warm serving / MCP (SP-C). No skill activation (SP-D).
- No explicit RRF / separate-strategy fusion (single hybrid call).
- No answer synthesis or verification inside `retrieve` (engine retrieves; the skill judges; `verify` is its own op).
- No ingest or ranking changes.

## Open questions
- Whether `retrieve` should also accept an `intent` hint (contract has it optional). Lean: omit in v1; add when a use appears.
- `coverage.thin_spots` heuristics are a starting set; refine as the skill is exercised.
