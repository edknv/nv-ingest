# Skill↔Library Contract (Objective 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `nemo-retriever` skill drift-proof against the installed `retriever` engine by shipping a versioned, machine-checkable contract + a `doctor` that live-probes the engine, so the class of mismatch that broke work this session (lean hit schema, nonexistent `--input-type`, wrong default table) is caught immediately instead of at runtime.

**Architecture:** Co-versioned with the skill under `skills/nemo-retriever/`. A `contract/` dir holds (a) the **actual** engine I/O contract the skill relies on today and (b) the documented **target** schema. A `scripts/doctor.py` runs static flag-surface checks plus a live ingest→query probe and asserts the output matches the actual contract. A pytest wraps `doctor` for CI. The skill declares the contract version it needs and tells the agent to run `doctor` on the setup turn; two known-wrong skill doc statements are corrected.

**Tech Stack:** the `retriever` CLI (venv at `/home/edwardk/git/nv-ingest/retriever`), LanceDB, Python 3.12, pytest, JSON Schema (validated with the stdlib-friendly `jsonschema` package shipped in the venv, or a minimal inline validator if absent).

---

## Ground truth (verified this session — the contract pins these)

- **Query output hit** = exactly `{ "page_number": int, "source": str, "text": str }`. No `pdf_basename`, `metadata.type`, `rank`, or `_distance`. (`--content-types` filters server-side but does not add fields.)
- **`retriever query` flags** include: `--top-k`, `--candidate-k`, `--content-types`, `--rerank/--no-rerank`, `--embed-model-name`, `--reranker-model`, `--lancedb-uri` (default `lancedb`), `--table-name` (default **`nemo-retriever`**), `--page-dedup/--no-page-dedup`.
- **`retriever ingest`** has **no `--input-type`** flag; it auto-detects mixed formats in one pass. It has `--overwrite` (default) / `--append`, `--ocr-version [v1|v2]`, `--ocr-lang [multi|english]`, `--table-name`, `--lancedb-uri`, `--embed-model-name`.
- **LanceDB table columns** = `vector`, `text`, `metadata` (JSON string `{page_number,type,bbox_xyxy_norm?}`), `source` (JSON string `{source_id,source_name}`).
- Default table name (`nemo-retriever`) differs from what the skill/workflow use (`nv-ingest`), so callers must pass `--table-name`.

## File structure

- Create: `skills/nemo-retriever/contract/CONTRACT.md` — human-readable contract + versioning rules.
- Create: `skills/nemo-retriever/contract/actual-hit.schema.json` — JSON Schema for today's query-output hit.
- Create: `skills/nemo-retriever/contract/target-hit.schema.json` — documented richer target hit (doc_id/modality/fidelity/provenance).
- Create: `skills/nemo-retriever/contract/cli-contract.json` — required flag surface + table conventions + contract version.
- Create: `skills/nemo-retriever/scripts/doctor.py` — static checks + live probe; exits nonzero on any failure.
- Create: `skills/nemo-retriever/tests/test_contract.py` — pytest wrapping `doctor`.
- Create: `skills/nemo-retriever/tests/fixtures/contract_probe.txt` — tiny fixture corpus for the live probe.
- Modify: `skills/nemo-retriever/SKILL.md` — declare required contract version + "run doctor on setup".
- Modify: `skills/nemo-retriever/references/setup.md` — remove the nonexistent `--input-type` recipe.
- Modify: `skills/nemo-retriever/references/query.md` — correct the documented hit schema to match `actual-hit.schema.json`.

Each file has one responsibility: schemas describe data, `cli-contract.json` describes the CLI surface, `doctor.py` is the single executable checker, the test just invokes it, the skill docs reference it.

---

### Task 1: Contract artifacts

**Files:**
- Create: `skills/nemo-retriever/contract/actual-hit.schema.json`
- Create: `skills/nemo-retriever/contract/target-hit.schema.json`
- Create: `skills/nemo-retriever/contract/cli-contract.json`
- Create: `skills/nemo-retriever/contract/CONTRACT.md`

- [ ] **Step 1: Create `actual-hit.schema.json`** (the shape the skill relies on TODAY)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "nemo-retriever/actual-hit",
  "title": "Actual retriever query hit (engine v26.x)",
  "type": "object",
  "required": ["page_number", "source", "text"],
  "properties": {
    "page_number": { "type": "integer", "description": "1-indexed page, or segment/timestamp for A/V" },
    "source": { "type": "string", "description": "source file path (flattened from the table's source JSON)" },
    "text": { "type": "string" }
  },
  "additionalProperties": true
}
```

- [ ] **Step 2: Create `target-hit.schema.json`** (documented goal — NOT yet emitted by the engine)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "nemo-retriever/target-hit",
  "title": "Target agent-native hit (design goal, not yet emitted)",
  "type": "object",
  "required": ["doc_id", "source_uri", "locator", "modality", "text", "score", "provenance"],
  "properties": {
    "doc_id": { "type": "string" },
    "source_uri": { "type": "string" },
    "locator": { "type": "object", "description": "page | segment | timestamp | bbox" },
    "modality": { "type": "string", "enum": ["text", "table", "chart", "image", "audio", "video_frame"] },
    "text": { "type": "string" },
    "score": { "type": "number" },
    "provenance": {
      "type": "object",
      "required": ["extractor", "fidelity"],
      "properties": {
        "extractor": { "type": "string" },
        "model": { "type": "string" },
        "fidelity": { "type": "string", "enum": ["verbatim", "ocr", "transcribed", "vlm_caption"] },
        "confidence": { "type": "number" }
      }
    }
  },
  "additionalProperties": false
}
```

- [ ] **Step 3: Create `cli-contract.json`** (flag surface + conventions the skill depends on)

```json
{
  "contract_version": "1.0.0",
  "query": {
    "required_flags": ["--top-k", "--content-types", "--rerank", "--embed-model-name", "--lancedb-uri", "--table-name"],
    "forbidden_flags": [],
    "default_table_name": "nemo-retriever",
    "hit_schema": "actual-hit.schema.json"
  },
  "ingest": {
    "required_flags": ["--append", "--overwrite", "--ocr-version", "--ocr-lang", "--table-name", "--lancedb-uri", "--embed-model-name"],
    "forbidden_flags": ["--input-type"],
    "single_pass_multiformat": true
  },
  "table_columns": ["vector", "text", "metadata", "source"]
}
```

- [ ] **Step 4: Create `CONTRACT.md`**

```markdown
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
```

- [ ] **Step 5: Validate JSON parses and commit**

```bash
/home/edwardk/git/nv-ingest/retriever/bin/python -c "import json; [json.load(open(f)) for f in ['skills/nemo-retriever/contract/actual-hit.schema.json','skills/nemo-retriever/contract/target-hit.schema.json','skills/nemo-retriever/contract/cli-contract.json']]; print('JSON OK')"
git add skills/nemo-retriever/contract/
git commit --no-gpg-sign -m "feat(skill): add nemo-retriever skill<->engine contract (actual + target schemas, cli-contract)"
```
Expected: `JSON OK`.

---

### Task 2: `doctor.py` (static checks + live probe)

**Files:**
- Create: `skills/nemo-retriever/scripts/doctor.py`
- Create: `skills/nemo-retriever/tests/fixtures/contract_probe.txt`

- [ ] **Step 1: Create the fixture corpus**

Create `skills/nemo-retriever/tests/fixtures/contract_probe.txt` with:
```
Contract probe document.
The capital of the test corpus is Probeville.
This single short text file exists only so doctor.py can ingest one tiny document and run one query to assert the live hit schema.
```

- [ ] **Step 2: Create `doctor.py`**

```python
#!/usr/bin/env python
"""Verify the installed `retriever` engine satisfies the skill's contract.

Usage: <RETRIEVER_VENV>/bin/python skills/nemo-retriever/scripts/doctor.py
Exits 0 if all checks pass, 1 otherwise. Always runs a LIVE ingest+query probe.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CONTRACT_DIR = os.path.join(os.path.dirname(HERE), "contract")
FIXTURE = os.path.join(os.path.dirname(HERE), "tests", "fixtures", "contract_probe.txt")
EMBED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"

results = []  # (ok: bool, label: str, detail: str)


def check(ok, label, detail=""):
    results.append((bool(ok), label, detail))


def retriever_bin():
    path = shutil.which("retriever")
    if not path:
        return None
    return path


def help_text(bin_path, subcmd):
    try:
        out = subprocess.run([bin_path, subcmd, "--help"], capture_output=True, text=True, timeout=60)
        return (out.stdout or "") + (out.stderr or "")
    except Exception as e:  # noqa: BLE001
        return f"__ERROR__ {e}"


def main():
    contract = json.load(open(os.path.join(CONTRACT_DIR, "cli-contract.json")))
    hit_schema = json.load(open(os.path.join(CONTRACT_DIR, "actual-hit.schema.json")))

    bin_path = retriever_bin()
    check(bin_path is not None, "retriever CLI on PATH",
          "" if bin_path else "run skills/nemo-retriever/references/install.md")
    if not bin_path:
        return report()

    # --- Static flag-surface checks (no GPU) ---
    qhelp = help_text(bin_path, "query")
    for flag in contract["query"]["required_flags"]:
        check(flag in qhelp, f"query has {flag}")
    ihelp = help_text(bin_path, "ingest")
    for flag in contract["ingest"]["required_flags"]:
        check(flag in ihelp, f"ingest has {flag}")
    for flag in contract["ingest"]["forbidden_flags"]:
        check(flag not in ihelp, f"ingest does NOT have {flag}",
              "engine changed: skill assumes single-pass auto-detect")

    # --- Live probe: ingest tiny fixture, query, validate hit schema (GPU) ---
    tmp = tempfile.mkdtemp(prefix="retriever_doctor_")
    try:
        corpus = os.path.join(tmp, "corpus")
        os.makedirs(corpus)
        shutil.copy(FIXTURE, corpus)
        uri = os.path.join(tmp, "lancedb")
        table = "contract_probe"
        ing = subprocess.run(
            [bin_path, "ingest", corpus + "/", "--table-name", table, "--lancedb-uri", uri,
             "--embed-model-name", EMBED_MODEL, "--quiet"],
            capture_output=True, text=True, timeout=900)
        check(ing.returncode == 0, "live ingest of fixture", ing.stderr.strip()[-300:])

        q = subprocess.run(
            [bin_path, "query", "What is the capital of the test corpus?", "--top-k", "3",
             "--table-name", table, "--lancedb-uri", uri, "--embed-model-name", EMBED_MODEL],
            capture_output=True, text=True, timeout=600)
        check(q.returncode == 0, "live query", q.stderr.strip()[-300:])
        hits = []
        if q.returncode == 0:
            try:
                hits = json.loads(q.stdout)
                check(isinstance(hits, list) and len(hits) > 0, "query returned hits")
            except Exception as e:  # noqa: BLE001
                check(False, "query stdout is JSON", str(e))
        # validate first hit against the actual-hit schema
        if hits:
            ok, why = validate(hits[0], hit_schema)
            check(ok, "hit matches actual-hit.schema.json", why)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return report()


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


def report():
    failed = [r for r in results if not r[0]]
    for ok, label, detail in results:
        mark = "PASS" if ok else "FAIL"
        line = f"[{mark}] {label}"
        if detail and not ok:
            line += f"  -- {detail}"
        print(line)
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Syntax-check `doctor.py`**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python -c "import ast; ast.parse(open('skills/nemo-retriever/scripts/doctor.py').read()); print('SYNTAX OK')"`
Expected: `SYNTAX OK`.

- [ ] **Step 4: Run `doctor.py` live (the real validation — needs GPU)**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python skills/nemo-retriever/scripts/doctor.py; echo "exit=$?"`
Expected: all checks print `[PASS]`, final line `N/N checks passed`, `exit=0`. In particular `ingest does NOT have --input-type` and `hit matches actual-hit.schema.json` must PASS (these encode the two biggest drifts found this session). If any check FAILS, read the detail — either the contract is wrong (fix the JSON in Task 1) or the engine genuinely diverged (record it).

- [ ] **Step 5: Commit**

```bash
git add skills/nemo-retriever/scripts/doctor.py skills/nemo-retriever/tests/fixtures/contract_probe.txt
git commit --no-gpg-sign -m "feat(skill): add doctor.py contract checker (static flag checks + live ingest/query probe)"
```

---

### Task 3: Contract pytest

**Files:**
- Create: `skills/nemo-retriever/tests/test_contract.py`

- [ ] **Step 1: Create the test**

```python
"""Contract test: the installed retriever engine must satisfy the skill contract.

Runs the live doctor probe (ingest + query + schema check). Requires the retriever
venv with GPU access; skips cleanly if the CLI is absent.
"""
import shutil
import subprocess
import sys
import os

import pytest

DOCTOR = os.path.join(os.path.dirname(__file__), "..", "scripts", "doctor.py")


@pytest.mark.skipif(shutil.which("retriever") is None, reason="retriever CLI not installed")
def test_engine_satisfies_contract():
    proc = subprocess.run([sys.executable, DOCTOR], capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, f"doctor reported contract violations:\n{proc.stdout}\n{proc.stderr}"
```

- [ ] **Step 2: Run the test**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python -m pytest skills/nemo-retriever/tests/test_contract.py -v`
Expected: `1 passed` (or `1 skipped` only if `retriever` is not on PATH — on this machine it is, so expect `1 passed`).

- [ ] **Step 3: Commit**

```bash
git add skills/nemo-retriever/tests/test_contract.py
git commit --no-gpg-sign -m "test(skill): contract pytest wrapping the doctor live probe"
```

---

### Task 4: Wire the contract into the skill + correct known drifts

**Files:**
- Modify: `skills/nemo-retriever/SKILL.md`
- Modify: `skills/nemo-retriever/references/setup.md`
- Modify: `skills/nemo-retriever/references/query.md`

- [ ] **Step 1: Declare the contract + doctor in `SKILL.md`**

After the `## Install (if `retriever` is missing)` section in `skills/nemo-retriever/SKILL.md`, insert:
```markdown
## Contract (run once on the setup turn)

This skill targets engine **contract_version 1.0.0** (`contract/cli-contract.json`). On the
setup turn, after the index is built, verify the installed engine matches:

`<RETRIEVER_VENV>/bin/python <skill_dir>/scripts/doctor.py`

If `doctor.py` reports any `[FAIL]`, the installed `retriever` has drifted from what this skill
assumes — read `contract/CONTRACT.md` and the failing check before trusting query results.
```

- [ ] **Step 2: Remove the nonexistent `--input-type` recipes in `setup.md`**

In `skills/nemo-retriever/references/setup.md`, replace the line:
```
<RETRIEVER_VENV>/bin/retriever ingest ./images/ --input-type image --ocr-version v2 --ocr-lang english
```
with:
```
<RETRIEVER_VENV>/bin/retriever ingest ./images/ --ocr-version v2 --ocr-lang english
```
and replace:
```
<RETRIEVER_VENV>/bin/retriever ingest ./office/ --input-type doc
```
with:
```
<RETRIEVER_VENV>/bin/retriever ingest ./office/
```
and replace:
```
<RETRIEVER_VENV>/bin/retriever ingest ./media/ --input-type audio   # or --input-type video
```
with:
```
<RETRIEVER_VENV>/bin/retriever ingest ./media/   # audio/video auto-detected; needs [multimedia] + ffmpeg
```
Then add one sentence under the "Other input shapes" heading: `The installed CLI auto-detects formats in a single pass — there is no \`--input-type\` flag (verified by \`scripts/doctor.py\`).`

- [ ] **Step 3: Correct the documented hit schema in `query.md`**

In `skills/nemo-retriever/references/query.md`, replace the sentence beginning:
```
Each hit has: `text`, `pdf_basename`, `page_number` (int, **1-indexed**: the first page of a PDF is page `1`), `pdf_page` (string composite key `"<basename>_<page_number>"` — not a number, don't use it as one), `_distance`, and `metadata` (JSON with `type` ∈ `text|table|chart|image`).
```
with:
```
Each hit emitted by `retriever query` has exactly: `page_number` (int, **1-indexed**), `source` (the file path), and `text`. (There is no `pdf_basename`/`metadata`/`_distance` in the query output — those live only in the LanceDB table, not the query result. Derive a display name from `source`. Schema asserted by `contract/actual-hit.schema.json` / `scripts/doctor.py`.)
```

- [ ] **Step 4: Commit**

```bash
git add skills/nemo-retriever/SKILL.md skills/nemo-retriever/references/setup.md skills/nemo-retriever/references/query.md
git commit --no-gpg-sign -m "docs(skill): declare contract v1.0.0 + run doctor on setup; correct --input-type and hit-schema drifts"
```

---

### Task 5: Final verification

**Files:** none (validation only)

- [ ] **Step 1: Re-run doctor end-to-end**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python skills/nemo-retriever/scripts/doctor.py; echo "exit=$?"`
Expected: `exit=0`, all `[PASS]`.

- [ ] **Step 2: Confirm the corrected docs no longer reference `--input-type`**

Run: `grep -rn "input-type" skills/nemo-retriever/references/ | grep -v "no .--input-type. flag" || echo "no stale --input-type recipes"`
Expected: `no stale --input-type recipes` (the only remaining mention is the explanatory sentence).

- [ ] **Step 3: Confirm the test passes from a clean invocation**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python -m pytest skills/nemo-retriever/tests/test_contract.py -q`
Expected: `1 passed`.

---

## Self-review

**Spec coverage (Objective 1 of the rebuild spec):**
- "Versioned typed contract" → Task 1 (`cli-contract.json` `contract_version`, `actual-hit`/`target-hit` schemas, CONTRACT.md versioning rules). ✓
- "reality now + target documented" (the chosen scope) → `actual-hit.schema.json` (reality) + `target-hit.schema.json` (documented goal). ✓
- "`retriever doctor` asserts the installed engine satisfies the contract" → Task 2 `doctor.py`. ✓
- "always live probe" (chosen) → `doctor.py` always ingests fixture + queries + validates; Task 2 Step 4 and Task 3 run it live. ✓
- "co-versioned skill + contract tests" → Task 3 pytest + Task 4 SKILL.md version declaration. ✓
- "skill stops compensating / fix drift root cause" → Task 4 corrects the `--input-type` and hit-schema falsehoods that caused this session's failures. ✓

**Placeholder scan:** No TBD/TODO. `<RETRIEVER_VENV>` and `<skill_dir>` appear only inside SKILL.md prose (the skill's existing substitution convention), not as plan gaps. All code blocks are complete.

**Type consistency:** `contract_version` `"1.0.0"` matches between `cli-contract.json` (Task 1), CONTRACT.md, and the SKILL.md declaration (Task 4). `doctor.py` reads `cli-contract.json` keys exactly as defined (`query.required_flags`, `ingest.required_flags`, `ingest.forbidden_flags`). The `actual-hit.schema.json` `required` fields (`page_number`/`source`/`text`) are exactly what `doctor.validate()` checks and what `query.md` is corrected to describe. The forbidden `--input-type` check (Task 2) matches the `setup.md` correction (Task 4) and the ground-truth findings.

**Note on cost:** every `doctor` run and the pytest cold-load the embedder (~30–60s GPU) by design (live probe was the chosen option). If this lands in CI without GPU, the test self-skips only when `retriever` is absent — on a CPU-only-but-installed runner it would attempt the probe and fail; gate it with a CI marker at that point (out of scope here).
