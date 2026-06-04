# Objective 2a — surface the typed hit (modality + score)

**Date:** 2026-06-03
**Type:** Engine change to `nemo_retriever` (first slice of rebuild Objective 2)
**Parent:** `2026-06-03-skill-first-retriever-rebuild-design.md` (Objective 2 = native agent-quality retrieval)

## Context

Objective 2 (native multi-strategy fusion + verification) decomposes into three shippable slices:
- **2a — surface the typed hit** (this spec): expose modality + score in query output.
- **2b — multi-strategy RRF fusion** (later; will *lean on the existing LanceDB hybrid* per decision).
- **2c — `verify` engine op** (later).

2a is first because it leverages data the engine *already computes*, unblocks 2b (fusion needs scores) and 2c (verify needs modality), and advances the Objective-1 contract from `actual-hit` toward `target-hit` — all with a small additive change.

## Problem

The engine's internal hit (`RetrievalHit` TypedDict, `vdb/records.py:15`) already carries `content_type`, `metadata` (with `type`), `_distance`, and `_score`. But the CLI throws this away: `_query_cli_hit()` (`adapters/cli/main.py:74`) flattens every hit to `{source, page_number, text}`. Consequently the **skill cannot tell a chart/image hit from a text hit** — which is the exact reason `query.md` carries an elaborate manual "Charts and images" hedging-and-reverification recipe. Surfacing modality removes the guesswork.

## Goal

`retriever query` emits, per hit: `{source, page_number, text, modality, score}` — additively, without breaking existing consumers.

## Design

### The change

Extend `_query_cli_hit(hit)` (`adapters/cli/main.py:74`) to add:

- **`modality`** (string, always present): `hit.get("content_type") or hit.get("metadata", {}).get("type") or "text"`. Values mirror what the engine stores (`text | table | chart | image` today; A/V types pass through unchanged).
- **`score`** (number or null): `hit.get("_score")` if present (rerank/hybrid relevance), else `hit.get("_distance")` if present, else `null`. Documented meaning: when it comes from `_score`, higher = more relevant; when from `_distance`, lower = closer. **Hit order remains authoritative** (the engine already returns hits ranked); `score` is informational and never used by the skill to re-sort.

Nothing else about the query path changes — no new retrieval, no reranking changes. This is purely a serialization-boundary addition.

### Why minimal (modality + score, not the full hit)

`bbox_xyxy_norm`, `source_id`, `stored_image_uri`, and the raw `metadata` blob are deliberately **not** surfaced, to preserve the skill's compact-output economics (the skill's hard-limits exist to keep per-turn token/cache cost down). `modality` + `score` are two small scalars that deliver the value (modality-awareness) without bloating output. Richer fields can arrive with later slices if a concrete need appears (YAGNI).

### Backward compatibility

Purely additive: existing keys (`source`, `page_number`, `text`) are unchanged, so the skill, the workflow, and existing tests keep working. The Objective-1 contract's `actual-hit.schema.json` is `additionalProperties: true`, so `doctor.py` passes both before and after.

### Contract update (ties back to Objective 1)

- `actual-hit.schema.json`: add `modality` (required) and `score` (optional, `["number","null"]`) — `modality` becomes a guaranteed field of the engine's real output.
- `cli-contract.json`: bump `contract_version` `1.0.0` → **`1.1.0`** (minor: additive engine capability the skill can now use).
- `CONTRACT.md`: note the 1.1.0 addition.
- `doctor.py`: the live-probe hit assertion now also checks `modality` is present (via the updated `actual-hit.schema.json` — the existing `validate()` already enforces `required`, so this is driven by the schema, not new doctor code).
- `SKILL.md`: bump the declared contract version to 1.1.0.

### Skill doc update

`query.md`: document the new `modality`/`score` fields, and **soften the manual chart recipe** — the skill can now branch on `modality == "chart" | "image"` directly instead of inferring it. (The recipe stays as the *verification* step until 2c lands a `verify` op; only the detection guesswork is removed.)

## Components & boundaries

- `_query_cli_hit()` — the single serialization point; one focused function changes.
- Contract files — declare the new reality; independent of the code change but updated in lockstep.
- `doctor.py` — unchanged code; its assertion tightens because the schema tightens.

## Testing

- `tests/test_root_query_cli.py` — extend a query-CLI test to assert each emitted hit has `modality` (one of the known values) and a `score` key (number or null). Follow the existing mock-the-graph / inject-fake-results pattern (`tests/test_retriever_queries.py:18-40`).
- Live gate: re-run `doctor.py` (now asserting `modality`) — must stay green against the real engine.

## Non-goals

- No multi-strategy retrieval or fusion (that is 2b).
- No `verify` operation (that is 2c).
- No surfacing of bbox/source_id/the metadata blob.
- No change to ranking, reranking, or the LanceDB search itself.

## Open question

- `score` semantics mix `_score` (higher-better) and `_distance` (lower-better) under one key. Resolved for 2a by documenting the dual meaning and keeping hit order authoritative; a cleaner normalized `relevance` can be introduced in 2b when fusion makes scores load-bearing.
