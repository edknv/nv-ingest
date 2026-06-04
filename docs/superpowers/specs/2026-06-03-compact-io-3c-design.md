# Objective 3c — compact query output (`--max-text-chars`)

**Date:** 2026-06-03
**Type:** Engine change to `nemo_retriever` (first slice of rebuild Objective 3)
**Parent:** `2026-06-03-skill-first-retriever-rebuild-design.md` (Objective 3 = agent-economics).

## Context

Objective 3 (agent-economics: daemon + MCP + compact I/O) decomposes into:
- **3c — compact I/O** (this spec): smallest, lowest-risk, directly serves the skill's token economics.
- **3b — MCP surface** (later; net-new FastMCP wrapper).
- **3a — warm query daemon** (later; decided approach: build an in-process warm daemon to kill the per-query cold-load).

3c is first because it leverages an existing serialization point and gives an immediate token win with no architectural change.

## Problem

`query.md` is built around *not* dumping hit text into context: it pipes `retriever query` through a python one-liner that prints only rank/page/source, then fetches full text selectively, and repeatedly warns that "pulling all 10 hits' text into context inflates cached prompt size on every subsequent turn." That compaction is the skill doing the engine's job. The engine should be able to return compact output directly.

## Goal

`retriever query` gains `--max-text-chars N` to control per-hit `text` size — additively, without breaking existing consumers.

## Design

### The change
Add `--max-text-chars` (`int | None`, default `None`) to `query_command`, threaded into the existing hit shaper `_query_cli_hit`:
- **unset / `None` (default):** emit full `text` — backward-compatible; existing tests and behavior unchanged.
- **`N > 0`:** truncate each hit's `text` to `N` characters and append `…` if it was longer (snippets).
- **`0`:** emit empty `text` — a **metadata-only summary** (`source`, `page_number`, `modality`, `score`) in one call, replacing the skill's hand-rolled summary pipeline.

`_query_cli_hit(hit, max_text_chars=None)` applies the truncation at the serialization boundary; nothing else in the query path (embedding, retrieval, rerank, dedup) changes.

### Behavior detail
- Truncation is character-based on the already-shaped `text`. Negative values are treated as `None` (full text). The `…` ellipsis is appended only when truncation actually removed characters and `N > 0`.
- All other hit fields (`source`, `page_number`, `modality`, `score`) are always emitted unchanged — only `text` is affected.

### Backward compatibility
Default `None` ⇒ full text, so the 2a/2b query-CLI tests (which assert full `text`) keep passing. The flag is purely additive.

### Contract + skill (consistent with 2a/2b/2c)
- `cli-contract.json`: bump `contract_version` 1.3.0 → **1.4.0**; add `--max-text-chars` to `query.required_flags` so `doctor` asserts it exists.
- `CONTRACT.md`: changelog for 1.4.0.
- `SKILL.md`: declare contract 1.4.0.
- `query.md`: document `--max-text-chars` — note that `--max-text-chars 0` yields a metadata-only summary in one call (the agent can then fetch full text for a specific hit), and snippets via `N > 0`.

## Components & boundaries
- `_query_cli_hit` — the single serialization point gains an optional `max_text_chars` param; one focused function changes.
- `query_command` — adds one Typer option and passes it to the shaper.
- Contract files + `doctor` — declare/assert the new flag. No code change to `doctor` (the flag is added to `query.required_flags`, which the existing static check iterates).

## Testing
- **Query CLI** (`tests/test_root_query_cli.py`): a new test passes `--max-text-chars 5` and asserts each hit's `text` is ≤ 6 chars (5 + `…`) and truncated; a `--max-text-chars 0` assertion shows empty `text` with `source`/`modality`/`score` still present; the existing no-flag test (full text) still passes.
- **Live gate:** `doctor` stays green with `--max-text-chars` now asserted in the query flag surface.

## Non-goals
- `verify` stays full-text — it's a single opt-in call whose purpose is to give the agent evidence to **judge**; truncating it would defeat it.
- No `--fields` selector (text size is the dominant token cost; field selection is YAGNI for now).
- No change to ranking, retrieval, hit ordering, or any non-`text` field.
- Not the daemon (3a) or MCP (3b).

## Open question
- Truncation is by raw characters, which can cut mid-word/mid-number. Acceptable for a summary/snippet whose purpose is triage (the agent fetches full text when it needs precision); a future refinement could truncate on a token/word boundary.
