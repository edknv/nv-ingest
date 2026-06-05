# retriever Skill + Contract Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Materialize the skill-first design driver as committed artifacts: the judgment-only `retriever` `SKILL.md`, a machine-readable `retrieve`/`verify`/`index` contract schema, and a README mapping the library sub-projects (SP-A→D) that must satisfy it.

**Architecture:** Documentation/contract artifacts only — no engine code, no GPU. They live in a **non-activating** location (`docs/superpowers/contracts/retriever/`) so the skill isn't loaded-but-broken before its `retrieve` backend exists; SP-D later moves `SKILL.md` into `skills/retriever/`. The JSON contract is validated only for parse-correctness + schema self-consistency.

**Tech Stack:** Markdown, JSON Schema (draft 2020-12), the `retriever` venv python for JSON validation.

## File structure
- Create: `docs/superpowers/contracts/retriever/contract.schema.json` — machine-readable `retrieve`/`verify`/`index` result shapes (`$defs`).
- Create: `docs/superpowers/contracts/retriever/SKILL.md` — the intended judgment-only skill (verbatim from the spec).
- Create: `docs/superpowers/contracts/retriever/README.md` — what this is, the skill↔library boundary, and the SP-A→D conformance map.

All commits use `--no-gpg-sign` (GPG signing fails in this environment).

---

### Task 1: Machine-readable contract schema

**Files:** Create `docs/superpowers/contracts/retriever/contract.schema.json`

- [ ] **Step 1: Create the schema**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "retriever/contract",
  "title": "retriever skill-first contract (retrieve / verify / index)",
  "$defs": {
    "locator": {
      "type": "object",
      "required": ["kind", "value"],
      "properties": {
        "kind": { "type": "string", "enum": ["page", "segment", "timestamp", "bbox"] },
        "value": {}
      },
      "additionalProperties": false
    },
    "evidence_item": {
      "type": "object",
      "required": ["text", "source", "locator", "modality", "fidelity", "score", "citation"],
      "properties": {
        "text": { "type": "string" },
        "source": { "type": "string", "description": "doc id / name" },
        "locator": { "$ref": "#/$defs/locator" },
        "modality": { "type": "string", "enum": ["text", "table", "chart", "image", "audio", "video_frame"] },
        "fidelity": { "type": "string", "enum": ["verbatim", "ocr", "transcribed", "vlm_caption"],
          "description": "trust as a literal source: verbatim > ocr > transcribed > vlm_caption" },
        "score": { "type": "number" },
        "citation": { "type": "string", "description": "render-ready, e.g. 'doc p.3'" }
      },
      "additionalProperties": true
    },
    "retrieve_result": {
      "type": "object",
      "required": ["evidence", "coverage"],
      "properties": {
        "evidence": { "type": "array", "items": { "$ref": "#/$defs/evidence_item" } },
        "coverage": {
          "type": "object",
          "required": ["strategies_used", "n_docs_seen", "thin_spots"],
          "properties": {
            "strategies_used": { "type": "array", "items": { "type": "string" } },
            "n_docs_seen": { "type": "integer" },
            "thin_spots": { "type": "array", "items": { "type": "string" } }
          },
          "additionalProperties": false
        }
      },
      "additionalProperties": false
    },
    "verify_result": {
      "type": "object",
      "required": ["evidence", "corroborated_signal"],
      "properties": {
        "evidence": { "type": "array", "items": { "$ref": "#/$defs/evidence_item" } },
        "corroborated_signal": { "type": "boolean" }
      },
      "additionalProperties": false
    },
    "index_result": {
      "type": "object",
      "required": ["ingested", "skipped"],
      "properties": {
        "ingested": { "type": "array", "items": { "type": "string" } },
        "skipped": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["item", "reason"],
            "properties": { "item": { "type": "string" }, "reason": { "type": "string" } },
            "additionalProperties": false
          }
        }
      },
      "additionalProperties": false
    }
  }
}
```

- [ ] **Step 2: Validate it parses and the `$defs`/`$ref`s resolve**

Run:
```bash
/home/edwardk/git/nv-ingest/retriever/bin/python - <<'PY'
import json
s = json.load(open("docs/superpowers/contracts/retriever/contract.schema.json"))
defs = s["$defs"]
assert set(defs) == {"locator", "evidence_item", "retrieve_result", "verify_result", "index_result"}, sorted(defs)
# every $ref points at an existing $def
import re
refs = re.findall(r'"#/\$defs/([a-z_]+)"', json.dumps(s))
assert all(r in defs for r in refs), [r for r in refs if r not in defs]
# spot-check the load-bearing enums
ev = defs["evidence_item"]["properties"]
assert ev["fidelity"]["enum"] == ["verbatim", "ocr", "transcribed", "vlm_caption"]
assert ev["modality"]["enum"] == ["text", "table", "chart", "image", "audio", "video_frame"]
print("CONTRACT OK")
PY
```
Expected: `CONTRACT OK`.

- [ ] **Step 3: Commit**
```bash
git add docs/superpowers/contracts/retriever/contract.schema.json
git commit --no-gpg-sign -m "feat(retriever-contract): machine-readable retrieve/verify/index schema (skill-first driver)"
```

---

### Task 2: The judgment-only `SKILL.md` artifact

**Files:** Create `docs/superpowers/contracts/retriever/SKILL.md`

- [ ] **Step 1: Create the skill** (verbatim from the approved spec — the judgment-only playbook)
```markdown
---
name: retriever
description: Answer, quote, verify, or aggregate over a document corpus (PDF,
  image, Office, HTML, TXT, audio, video). Use for any multi-file or non-text
  question instead of native Read/Grep.
---

# retriever — reasoning over retrieved evidence

You have one tool: `retrieve(question)` → cited, fidelity-tagged evidence.
(Corpus not indexed? `index(path)` once.) You never build queries, choose
strategies, or parse output — you reason about what comes back.

## 1. Pick the move
- fact / number / date → retrieve; read the top evidence
- "list / count / every / across" → aggregate; do not sample
- exact quote → quote verbatim with its citation
- compare across docs → retrieve per entity, then contrast
- image / chart / audio / video → evidence is a transcription; treat per §2

## 2. Trust by fidelity  ← the core skill
verbatim > ocr > table > vlm_caption. A number or directional claim resting
ONLY on a `vlm_caption` (chart/image) is unconfirmed: call
`verify(claim, source, locator)`; assert confidently only if an independent
higher-fidelity passage corroborates, else quote it and tag "(chart-derived,
unconfirmed)". Never upgrade a low-fidelity reading to a confident fact.

## 3. Answer honestly
- Cite source + locator for every claim.
- Re-read the question: address every entity / year / category — even "not provided".
- If the answer isn't in the evidence, say so. Never fabricate from adjacent text.
- Use `coverage` to know if a thin/empty result means "broaden" vs "out-of-corpus".

## 4. When retrieval falls short
exact-term miss → broaden / rephrase; nothing relevant → likely out-of-corpus,
say so; `coverage` flags a stale or partial index → re-`index`.
```

- [ ] **Step 2: Verify it is judgment-only (no leaked mechanics)**

Run:
```bash
grep -niE -- "--[a-z-]+|/v1/|venv|stdout|bash|lancedb|vllm|escap|\.json" docs/superpowers/contracts/retriever/SKILL.md && echo "LEAK FOUND — remove mechanics" || echo "CLEAN: no CLI/mechanics leaked"
```
Expected: `CLEAN: no CLI/mechanics leaked` (the skill must reference only `retrieve`/`verify`/`index` and judgment).

- [ ] **Step 3: Commit**
```bash
git add docs/superpowers/contracts/retriever/SKILL.md
git commit --no-gpg-sign -m "feat(retriever-contract): judgment-only retriever SKILL.md (design driver, not yet activated)"
```

---

### Task 3: README — boundary + conformance map

**Files:** Create `docs/superpowers/contracts/retriever/README.md`

- [ ] **Step 1: Create the README**
```markdown
# retriever — skill-first design artifacts

These are the **design driver** for a from-scratch `retriever` skill. The skill is
the product; the library is its backend, specified to satisfy `contract.schema.json`.
Design rationale: `docs/superpowers/specs/2026-06-04-retriever-skill-first-design.md`.

**Not yet activated.** `SKILL.md` here is intentionally NOT in `skills/` — it depends
on a `retrieve`/`verify`/`index` backend that does not exist yet, so loading it now
would be a broken skill. SP-D moves it into `skills/retriever/` once `retrieve` is live.

## Files
- `SKILL.md` — the judgment-only skill (the irreducible retrieval wisdom).
- `contract.schema.json` — machine-readable `retrieve` / `verify` / `index` result shapes
  the library must satisfy.

## Skill ↔ library boundary
- **Library** (backend): retrieve, fuse strategies, tag `fidelity`, cite, `verify`,
  report `coverage`, serve warm. Mechanical, typed, testable.
- **Skill** (the model): choose the move, judge trust/sufficiency, decide when to
  verify, compose an honest answer, decide when to refuse. Judgment only.

## Library sub-projects (built to satisfy the contract; dependency order)
- **SP-A — ingest provenance → `fidelity`**: record extractor/OCR/ASR/caption provenance
  per chunk so a true `fidelity` exists. Foundational (today only `modality` is stored).
- **SP-B — `retrieve` primitive**: fuse strategies → attach `fidelity`+`citation` →
  compute `coverage` → return answer-ready evidence. Composes existing hybrid/content-type/
  verify work + SP-A's fidelity.
- **SP-C — serving + MCP**: expose `retrieve`/`verify`/`index` warm (build on `serve-models`)
  and as MCP tools (build on `retriever mcp`), re-pointed to `retrieve`.
- **SP-D — ship the skill**: move `SKILL.md` into `skills/retriever/`; retire the old
  CLI `nemo-retriever` skill.
```

- [ ] **Step 2: Commit**
```bash
git add docs/superpowers/contracts/retriever/README.md
git commit --no-gpg-sign -m "docs(retriever-contract): README — skill/library boundary + SP-A..D conformance map"
```

---

### Task 4: Final check

- [ ] **Step 1: Confirm artifacts exist, JSON valid, skill clean, non-activating location**

Run:
```bash
ls docs/superpowers/contracts/retriever/
/home/edwardk/git/nv-ingest/retriever/bin/python -c "import json; json.load(open('docs/superpowers/contracts/retriever/contract.schema.json')); print('JSON OK')"
test ! -e skills/retriever && echo "not activated in skills/ (correct)" || echo "WARNING: skill is in skills/ — would load broken"
```
Expected: three files listed; `JSON OK`; `not activated in skills/ (correct)`.

- [ ] **Step 2: No commit (validation only).** If green, the skill-first design driver is materialized.

---

## Self-review

**Spec coverage (2026-06-04 skill-first spec):**
- Judgment-only `SKILL.md` (the exact playbook) → Task 2. ✓
- `retrieve`/`verify`/`index` contract, machine-readable → Task 1 (`$defs` for each result + `evidence_item`/`locator`). ✓
- Fidelity model (`verbatim>ocr>transcribed>vlm_caption`) → encoded in the schema enum + skill §2; Task 1 Step 2 asserts the enum. ✓
- Skill↔library boundary + SP-A→D enumeration → Task 3 README. ✓
- "Not activated until retrieve exists" → non-activating location + Task 4 Step 1 assertion. ✓

**Placeholder scan:** No TBD/TODO. The README references SP-A→D as future specs — that's the spec's own decomposition, not a plan gap (this plan delivers only the artifacts).

**Type consistency:** the schema `evidence_item` fields (`text, source, locator, modality, fidelity, score, citation`) match the contract block in the spec and the fields the `SKILL.md` references (`fidelity`, `citation`, `coverage`). `coverage` keys (`strategies_used, n_docs_seen, thin_spots`) match between schema (Task 1) and skill §3/§4 usage (Task 2). The `fidelity`/`modality` enums match between the schema and the skill's §2 trust order.
