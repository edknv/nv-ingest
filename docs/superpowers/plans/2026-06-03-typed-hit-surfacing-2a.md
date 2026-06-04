# Objective 2a — Surface Typed Hit (modality + score) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `retriever query` emit `modality` and `score` per hit (additively), so the skill can see chart/image/table vs text directly, and bump the skill contract to 1.1.0.

**Architecture:** One serialization-boundary change in the engine (`_query_cli_hit`), the two CLI tests that pin output shape updated, the Objective-1 contract bumped to require `modality`, and `doctor`'s mini-validator hardened for the `score` union type. The on-PATH `retriever` is an **editable install** of `nemo_retriever/src`, so source edits take effect immediately (no reinstall).

**Tech Stack:** Python 3.12, Typer CLI, pytest (`typer.testing.CliRunner`), the `retriever` venv at `/home/edwardk/git/nv-ingest/retriever`, JSON Schema (mini-validator in `doctor.py`).

---

## Ground truth (verified)

- `_query_cli_hit()` is the only output-shaping point: `nemo_retriever/src/nemo_retriever/adapters/cli/main.py:74-79`, currently returns `{source, page_number, text}`.
- Internal hit (`vdb/records.py:15` `RetrievalHit`) carries `content_type`, `metadata` (dict with `type`), `_distance`, `_score` (all optional). `records.py:84-86` keeps `metadata["type"]` populated from `content_type`.
- Two tests pin the output shape and **will break**: `nemo_retriever/tests/test_root_query_cli.py` — `test_root_query_passes_query_options_and_prints_json` (uses `expected_output`, hits carry `_distance` 0.2 / 0.4 and `metadata.type` text/table) and `test_root_query_passes_candidate_dedup_and_content_filters` (inline expected, one hit `metadata.type` text, no distance/score).
- Contract lives in `skills/nemo-retriever/contract/`; `doctor.py` validates the live query hit against `actual-hit.schema.json` with a mini-validator whose `types` map is keyed by single strings (a list `type` would raise `TypeError: unhashable type: 'list'`).

## File structure

- Modify: `nemo_retriever/src/nemo_retriever/adapters/cli/main.py` (`_query_cli_hit`).
- Modify: `nemo_retriever/tests/test_root_query_cli.py` (two expected outputs).
- Modify: `skills/nemo-retriever/contract/actual-hit.schema.json`, `cli-contract.json`, `CONTRACT.md`.
- Modify: `skills/nemo-retriever/scripts/doctor.py` (`validate` union-type hardening).
- Modify: `skills/nemo-retriever/SKILL.md` (declared contract version), `skills/nemo-retriever/references/query.md` (document new fields).

---

### Task 1: Engine change + update the two breaking CLI tests

**Files:**
- Modify: `nemo_retriever/src/nemo_retriever/adapters/cli/main.py`
- Modify: `nemo_retriever/tests/test_root_query_cli.py`

- [ ] **Step 1: Update the two test expectations to the new shape (they will fail until the code changes)**

In `nemo_retriever/tests/test_root_query_cli.py`, replace:
```python
    expected_output = [
        {"source": "doc.pdf", "page_number": 1, "text": "passage"},
        {"source": "other.pdf", "page_number": 2, "text": "other"},
    ]
```
with:
```python
    expected_output = [
        {"source": "doc.pdf", "page_number": 1, "text": "passage", "modality": "text", "score": 0.2},
        {"source": "other.pdf", "page_number": 2, "text": "other", "modality": "table", "score": 0.4},
    ]
```
and replace:
```python
    assert json.loads(result.output) == [
        {"page_number": 1, "source": "doc.pdf", "text": "text row"},
    ]
```
with:
```python
    assert json.loads(result.output) == [
        {"page_number": 1, "source": "doc.pdf", "text": "text row", "modality": "text", "score": None},
    ]
```

- [ ] **Step 2: Run the tests to confirm they FAIL (red)**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python -m pytest nemo_retriever/tests/test_root_query_cli.py -q`
Expected: failures in the two updated tests (output still missing `modality`/`score`).

- [ ] **Step 3: Change `_query_cli_hit` to surface modality + score**

In `nemo_retriever/src/nemo_retriever/adapters/cli/main.py`, replace:
```python
def _query_cli_hit(hit: RetrievalHit) -> dict[str, object]:
    return {
        "source": hit.get("source", ""),
        "page_number": hit.get("page_number"),
        "text": hit.get("text", ""),
    }
```
with:
```python
def _query_cli_hit(hit: RetrievalHit) -> dict[str, object]:
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
    return {
        "source": hit.get("source", ""),
        "page_number": hit.get("page_number"),
        "text": hit.get("text", ""),
        "modality": modality,
        "score": score,
    }
```

- [ ] **Step 4: Run the tests to confirm they PASS (green)**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python -m pytest nemo_retriever/tests/test_root_query_cli.py -q`
Expected: all pass. (These tests mock `Retriever`; no GPU.)

- [ ] **Step 5: Commit**
```bash
git add nemo_retriever/src/nemo_retriever/adapters/cli/main.py nemo_retriever/tests/test_root_query_cli.py
git commit --no-gpg-sign -m "feat(retriever): surface modality + score in query CLI output"
```

---

### Task 2: Bump the contract + harden doctor's validator

**Files:**
- Modify: `skills/nemo-retriever/contract/actual-hit.schema.json`
- Modify: `skills/nemo-retriever/contract/cli-contract.json`
- Modify: `skills/nemo-retriever/contract/CONTRACT.md`
- Modify: `skills/nemo-retriever/scripts/doctor.py`

- [ ] **Step 1: Add `modality` (required) + `score` to `actual-hit.schema.json`**

Replace the whole file `skills/nemo-retriever/contract/actual-hit.schema.json` with:
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "nemo-retriever/actual-hit",
  "title": "Actual retriever query hit (engine v26.x, contract 1.1.0)",
  "type": "object",
  "required": ["page_number", "source", "text", "modality"],
  "properties": {
    "page_number": { "type": "integer", "description": "1-indexed page, or segment/timestamp for A/V" },
    "source": { "type": "string", "description": "source file path (flattened from the table's source JSON)" },
    "text": { "type": "string" },
    "modality": { "type": "string", "description": "content type: text | table | chart | image | audio | video_frame" },
    "score": { "type": ["number", "null"], "description": "relevance: _score (higher=better) or _distance (lower=closer), null if neither; hit ORDER is authoritative" }
  },
  "additionalProperties": true
}
```

- [ ] **Step 2: Bump `contract_version` and document `modality`/`score` in `cli-contract.json`**

In `skills/nemo-retriever/contract/cli-contract.json`, change:
```json
  "contract_version": "1.0.0",
```
to:
```json
  "contract_version": "1.1.0",
```

- [ ] **Step 3: Note 1.1.0 in `CONTRACT.md`**

In `skills/nemo-retriever/contract/CONTRACT.md`, append at the end of the file:
```markdown

## Changelog
- **1.1.0** — query hits now carry `modality` (required) and `score` (optional); see `actual-hit.schema.json`. First step from `actual-hit` toward `target-hit`.
- **1.0.0** — initial contract (lean `{page_number, source, text}` hit, flag surface, table conventions).
```

- [ ] **Step 4: Harden `doctor.validate()` for union (`["number","null"]`) types**

In `skills/nemo-retriever/scripts/doctor.py`, replace:
```python
def validate(obj, schema):
    """Tiny dependency-free validator for the subset of JSON Schema we use."""
    if not isinstance(obj, dict):
        return False, "hit is not an object"
    for req in schema.get("required", []):
        if req not in obj:
            return False, f"missing required field '{req}'"
    types = {"integer": int, "string": str, "number": (int, float), "object": dict, "array": list}
    for name, spec in schema.get("properties", {}).items():
        if name in obj and "type" in spec:
            py = types.get(spec["type"])
            if py and not isinstance(obj[name], py):
                return False, f"field '{name}' should be {spec['type']}, got {type(obj[name]).__name__}"
    return True, ""
```
with:
```python
def validate(obj, schema):
    """Tiny dependency-free validator for the subset of JSON Schema we use.

    Handles both a single type string and a union list (e.g. ["number", "null"]).
    """
    if not isinstance(obj, dict):
        return False, "hit is not an object"
    for req in schema.get("required", []):
        if req not in obj:
            return False, f"missing required field '{req}'"
    types = {"integer": int, "string": str, "number": (int, float), "object": dict,
             "array": list, "null": type(None), "boolean": bool}
    for name, spec in schema.get("properties", {}).items():
        if name not in obj or "type" not in spec:
            continue
        allowed = spec["type"] if isinstance(spec["type"], list) else [spec["type"]]
        pytypes = []
        for key in allowed:
            mapped = types.get(key)
            if mapped is None:
                continue
            pytypes.extend(mapped if isinstance(mapped, tuple) else [mapped])
        if pytypes and not isinstance(obj[name], tuple(pytypes)):
            return False, f"field '{name}' should be {spec['type']}, got {type(obj[name]).__name__}"
    return True, ""
```

- [ ] **Step 5: Validate JSON + syntax, then commit**
```bash
/home/edwardk/git/nv-ingest/retriever/bin/python -c "import json; [json.load(open(f)) for f in ['skills/nemo-retriever/contract/actual-hit.schema.json','skills/nemo-retriever/contract/cli-contract.json']]; import ast; ast.parse(open('skills/nemo-retriever/scripts/doctor.py').read()); print('OK')"
git add skills/nemo-retriever/contract/ skills/nemo-retriever/scripts/doctor.py
git commit --no-gpg-sign -m "feat(skill): contract 1.1.0 (modality required + score); harden doctor validator for union types"
```
Expected: `OK`.

---

### Task 3: Update skill docs

**Files:**
- Modify: `skills/nemo-retriever/SKILL.md`
- Modify: `skills/nemo-retriever/references/query.md`

- [ ] **Step 1: Bump the declared contract version in `SKILL.md`**

In `skills/nemo-retriever/SKILL.md`, replace:
```
This skill targets engine **contract_version 1.0.0** (`contract/cli-contract.json`).
```
with:
```
This skill targets engine **contract_version 1.1.0** (`contract/cli-contract.json`).
```

- [ ] **Step 2: Document `modality`/`score` in `query.md` and soften the chart-detection guesswork**

In `skills/nemo-retriever/references/query.md`, replace:
```
Each hit emitted by `retriever query` has exactly: `page_number` (int, **1-indexed**), `source` (the file path), and `text`. (There is no `pdf_basename`/`metadata`/`_distance` in the query output — those live only in the LanceDB table, not the query result. Derive a display name from `source`. Schema asserted by `contract/actual-hit.schema.json` / `scripts/doctor.py`.)
```
with:
```
Each hit emitted by `retriever query` has: `page_number` (int, **1-indexed**), `source` (the file path), `text`, `modality` (`text`|`table`|`chart`|`image`|`audio`|`video_frame`), and `score` (number or null — relevance/distance; hit order is authoritative, so don't re-sort on it). (There is no `pdf_basename` in the query output — derive a display name from `source`. Schema asserted by `contract/actual-hit.schema.json` / `scripts/doctor.py`.) Use `modality` to tell chart/image hits from prose **directly** — no need to guess from content.
```

- [ ] **Step 3: Commit**
```bash
git add skills/nemo-retriever/SKILL.md skills/nemo-retriever/references/query.md
git commit --no-gpg-sign -m "docs(skill): document modality/score hit fields; declare contract 1.1.0"
```

---

### Task 4: Live validation

**Files:** none (validation only)

- [ ] **Step 1: Full query-CLI unit test (no GPU)**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python -m pytest nemo_retriever/tests/test_root_query_cli.py -q`
Expected: all pass.

- [ ] **Step 2: Live `doctor` probe (now asserts `modality` via the 1.1.0 schema; needs GPU)**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python skills/nemo-retriever/scripts/doctor.py; echo "exit=$?"`
Expected: all `[PASS]`, including `hit matches actual-hit.schema.json` (the probe hit must now contain `modality`); `exit=0`. If `modality` is missing on the probe hit, the editable engine change in Task 1 did not take effect — confirm `_query_cli_hit` was edited in `nemo_retriever/src/...` (not a build copy).

- [ ] **Step 3: Eyeball a real query's new fields**

Run (reusing any existing index, or skip if none is handy):
```bash
mkdir -p /tmp/2a_probe && cp skills/nemo-retriever/tests/fixtures/contract_probe.txt /tmp/2a_probe/ && \
/home/edwardk/git/nv-ingest/retriever/bin/retriever ingest /tmp/2a_probe/ --table-name p2a --lancedb-uri /tmp/2a_lancedb --embed-model-name nvidia/llama-nemotron-embed-1b-v2 --quiet && \
/home/edwardk/git/nv-ingest/retriever/bin/retriever query "capital of the test corpus" --top-k 2 --table-name p2a --lancedb-uri /tmp/2a_lancedb --embed-model-name nvidia/llama-nemotron-embed-1b-v2 | /home/edwardk/git/nv-ingest/retriever/bin/python -c "import json,sys; h=json.load(sys.stdin)[0]; print('keys=',sorted(h)); assert 'modality' in h and 'score' in h, 'missing new fields'; print('modality=',h['modality'],'score=',h['score'])"; rm -rf /tmp/2a_probe /tmp/2a_lancedb
```
Expected: prints `keys=` including `modality` and `score`, then `modality= text score= <number>` with no assertion error.

- [ ] **Step 4: No commit (validation only).** If all green, slice 2a is complete.

---

## Self-review

**Spec coverage (2a design):**
- "emit `{source, page_number, text, modality, score}`" → Task 1 `_query_cli_hit`. ✓
- `modality` from `content_type` → `metadata.type` → `"text"` → Task 1 Step 3. ✓
- `score` = `_score` else `_distance` else null → Task 1 Step 3. ✓
- additive / backward compatible → only adds keys; existing keys unchanged. The two tests that pinned the old shape are updated (Task 1) — that's the expected breakage, handled. ✓
- contract bump (actual-hit + modality required, cli-contract 1.1.0, CONTRACT.md, SKILL.md, doctor asserts modality) → Task 2 + Task 3 Step 1. ✓
- query.md documents fields + softens chart detection → Task 3 Step 2. ✓
- doctor live gate → Task 4 Step 2. ✓

**Placeholder scan:** No TBD/TODO. All code blocks are complete and exact.

**Type consistency:** `modality`/`score` keys match across `_query_cli_hit` (Task 1), the two test expectations (Task 1), `actual-hit.schema.json` (Task 2), `query.md` (Task 3). `contract_version` `1.1.0` matches between `cli-contract.json`, `CONTRACT.md` changelog, and `SKILL.md`. The `doctor.validate` rewrite (Task 2 Step 4) handles the `["number","null"]` union that `score` introduces (the reason this task is bundled with the schema bump) — without it, the mini-validator would `TypeError` on the unhashable list `type`. The `pytypes` loop flattens `"number"` (which maps to the tuple `(int, float)`) and appends single types, then `isinstance` checks against the union.
