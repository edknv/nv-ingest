# SP-D — Activate `retriever` Skill, Retire `nemo-retriever` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the from-scratch `retriever` skill live — a CLI-backed `skills/retriever/SKILL.md` (its one primitive is `retriever retrieve`) with the still-useful infra migrated in — and retire the old `nemo-retriever` skill.

**Architecture:** Mostly `git mv`/`git rm` + one new `SKILL.md`. `skills/` is symlinked from `.claude/skills`, so a new `skills/retriever/SKILL.md` becomes loadable. No engine code changes.

**Tech Stack:** git, Markdown, the `retriever` venv (for the doctor/live checks).

## Ground truth (verified)
- `skills/nemo-retriever/` (20 tracked files). **Keep:** `contract/`, `scripts/doctor.py`, `scripts/grep_corpus.py`, `scripts/filename_fast_path.py`, `references/install.md`, `tests/fixtures/contract_probe.txt`, `tests/test_contract.py`. **Drop:** `SKILL.md`, `references/{query.md,setup.md,troubleshooting.md,cli/}`, `BENCHMARK.md`, `evals/`, `skill-card.md`, `skill.oms.sig`, `tests/__pycache__/`.
- `doctor.py` computes its contract dir relative to `scripts/` (`os.path.dirname(HERE)/contract`) → after moving `scripts/` + `contract/` together, paths still resolve. `test_contract.py` invokes `doctor.py` by a path relative to the test file → moving both together holds.
- `.claude/skills -> ../skills`; `skills/retriever` does not exist yet.
- `retriever retrieve "<q>"` (SP-B/C) exists, warm via `EMBED_INVOKE_URL`.

All commits `--no-gpg-sign`.

---

### Task 1: Migrate the still-useful infra into `skills/retriever/`

**Files:** `git mv` within `skills/`.

- [ ] **Step 1: Create the dir and move infra**
```bash
cd /home/edwardk/git/nv-ingest
mkdir -p skills/retriever/scripts skills/retriever/tests/fixtures skills/retriever/references
git mv skills/nemo-retriever/contract skills/retriever/contract
git mv skills/nemo-retriever/scripts/doctor.py skills/retriever/scripts/doctor.py
git mv skills/nemo-retriever/scripts/grep_corpus.py skills/retriever/scripts/grep_corpus.py
git mv skills/nemo-retriever/scripts/filename_fast_path.py skills/retriever/scripts/filename_fast_path.py
git mv skills/nemo-retriever/references/install.md skills/retriever/references/install.md
git mv skills/nemo-retriever/tests/fixtures/contract_probe.txt skills/retriever/tests/fixtures/contract_probe.txt
git mv skills/nemo-retriever/tests/test_contract.py skills/retriever/tests/test_contract.py
```

- [ ] **Step 2: Verify doctor + contract test still resolve paths (no GPU for the static parts)**
```bash
./retriever/bin/python -c "import ast; ast.parse(open('skills/retriever/scripts/doctor.py').read()); print('doctor parse OK')"
./retriever/bin/python -c "import json,os; d='skills/retriever/contract'; [json.load(open(os.path.join(d,f))) for f in ('cli-contract.json','actual-hit.schema.json','target-hit.schema.json')]; print('contract JSON OK')"
ls skills/retriever/contract skills/retriever/scripts skills/retriever/tests/fixtures
```
Expected: `doctor parse OK`, `contract JSON OK`, files listed. (Full live doctor runs in Task 4.)

- [ ] **Step 3: Commit**
```bash
git add -A skills/retriever skills/nemo-retriever
git commit --no-gpg-sign -m "refactor(skill): migrate contract/doctor/grep_corpus/install/fixtures into skills/retriever"
```

---

### Task 2: Create the activated `skills/retriever/SKILL.md`

**Files:** Create `skills/retriever/SKILL.md`

- [ ] **Step 1: Write the skill** (CLI-backed, judgment-only)
```markdown
---
name: retriever
description: Answer, quote, verify, or aggregate over a document corpus (PDF, image,
  Office, HTML, TXT, audio, video). Use for any multi-file or non-text question instead
  of native Read / Grep. Not for editing files, web browsing, or single-file plain-text lookups.
---

# retriever — reasoning over retrieved evidence

Retrieval is one command:
`<RETRIEVER_VENV>/bin/retriever retrieve "<question>"` → JSON
`{ evidence: [ { text, source, locator, modality, fidelity, score, citation } ], coverage: {...} }`.

You never build queries, choose strategies, or parse a vector DB — you run that one command
and reason about what it returns. (If your harness exposes the retriever MCP tools, call the
`retrieve` tool instead — same result, no Bash.)

## Setup (one-time, operator)
- If `command -v retriever` is empty, install per `references/install.md` (it prints `RETRIEVER_VENV`).
- Index the corpus once: `<RETRIEVER_VENV>/bin/retriever ingest <dir>` (add `--hybrid` for exact-term recall).
- Optional warm querying: `<RETRIEVER_VENV>/bin/retriever serve-models`, then export the printed
  `EMBED_INVOKE_URL` — `retrieve` is then warm (no per-call cold-load).
- `<RETRIEVER_VENV>/bin/python scripts/doctor.py` confirms the installed engine matches the contract.

## 1. Pick the move
- fact / number / date → retrieve; read the top evidence
- "list / count / every / across" → aggregate; do not sample
- exact quote → quote verbatim with its citation
- compare across docs → retrieve per entity, then contrast
- image / chart / audio / video → the evidence is a transcription; treat per §2

## 2. Trust by fidelity  ← the core skill
Each evidence item carries `fidelity`: verbatim > ocr > transcribed > vlm_caption. A number or
directional claim resting ONLY on a `vlm_caption` (chart/image) is unconfirmed — quote it and tag
"(chart-derived, unconfirmed)" unless a higher-fidelity item states the same fact. Prefer
verbatim/ocr/table evidence over captions for exact values. Never upgrade a low-fidelity reading
to a confident fact.

## 3. Answer honestly
- Cite each claim with the item's `citation` (source + locator).
- Re-read the question: address every entity / year / category — even "not provided".
- If the answer isn't in the evidence, say so. Never fabricate from adjacent text.
- Read `coverage.thin_spots` to tell "broaden the search" from "out of corpus".

## 4. When retrieval falls short
exact-term miss → re-`retrieve` with the exact term (or re-`ingest` with `--hybrid`); nothing
relevant → likely out-of-corpus, say so; `coverage` flags a thin/stale index → re-`ingest`.
```

- [ ] **Step 2: Mechanics-lean check** — the body must reference only `retrieve`/`ingest`/`serve-models`/`doctor` + judgment, no flag-spelling/escaping/stdout/vector-DB internals (the frontmatter `---` and `EMBED_INVOKE_URL`/`RETRIEVER_VENV` setup tokens are allowed):
```bash
grep -niE -- "--table-name|--lancedb-uri|--content-types|/v1/|stdout|json.load|jq |lancedb|vllm|escap|tee /tmp" skills/retriever/SKILL.md && echo "LEAK — remove" || echo "CLEAN: judgment + minimal setup only"
```
Expected: `CLEAN: judgment + minimal setup only`.

- [ ] **Step 3: Commit**
```bash
git add skills/retriever/SKILL.md
git commit --no-gpg-sign -m "feat(skill): activate judgment-only retriever SKILL.md (CLI-backed, one retrieve primitive)"
```

---

### Task 3: Retire `nemo-retriever`

**Files:** `git rm -r skills/nemo-retriever`

- [ ] **Step 1: Remove the remainder**
```bash
cd /home/edwardk/git/nv-ingest
git rm -r skills/nemo-retriever
rm -rf skills/nemo-retriever  # clear any leftover untracked (e.g. __pycache__)
```

- [ ] **Step 2: Verify retirement + activation**
```bash
test ! -e skills/nemo-retriever && echo "nemo-retriever retired" || echo "STILL PRESENT"
test -f skills/retriever/SKILL.md && echo "retriever active" || echo "MISSING"
ls skills/retriever
```
Expected: `nemo-retriever retired`, `retriever active`, and `skills/retriever` lists `SKILL.md contract references scripts tests`.

- [ ] **Step 3: Commit**
```bash
git add -A skills
git commit --no-gpg-sign -m "refactor(skill): retire old nemo-retriever skill (superseded by retriever)"
```

---

### Task 4: Validation

**Files:** none.

- [ ] **Step 1: Migrated contract test + doctor (live, GPU)**

Run:
```bash
cd /home/edwardk/git/nv-ingest
./retriever/bin/python -m pytest skills/retriever/tests/test_contract.py -q 2>&1 | tail -3
./retriever/bin/python skills/retriever/scripts/doctor.py 2>&1 | tail -2; echo "doctor_exit=${PIPESTATUS[0]}"
```
Expected: contract test `1 passed`; doctor `N/N checks passed`, exit 0 (paths resolved after the move).

- [ ] **Step 2: The skill's actual command works (live, GPU)**

```bash
mkdir -p /tmp/sd && cp data/multimodal_test.pdf /tmp/sd/
./retriever/bin/retriever ingest /tmp/sd/ --table-name sd --lancedb-uri /tmp/sd_db --embed-model-name nvidia/llama-nemotron-embed-1b-v2 --quiet
./retriever/bin/retriever retrieve "most expensive gadget in Chart 1" --top-k 3 --no-hybrid --table-name sd --lancedb-uri /tmp/sd_db --embed-model-name nvidia/llama-nemotron-embed-1b-v2 | ./retriever/bin/python -c "import json,sys; r=json.load(sys.stdin); e=r['evidence']; assert e and {'fidelity','citation','modality'} <= set(e[0]); print('skill command OK:', len(e), 'items; top fidelity=', e[0]['fidelity'])"
rm -rf /tmp/sd /tmp/sd_db
```
Expected: `skill command OK: N items; top fidelity= ...` — the exact command the activated SKILL.md tells the agent to run returns the contract shape.

- [ ] **Step 3: No commit (validation only).** If green, the `retriever` skill is live and `nemo-retriever` is retired — SP-D, and the skill-first build, complete.

---

## Self-review

**Spec coverage (SP-D):**
- New CLI-backed `skills/retriever/SKILL.md` (one `retrieve` primitive + MCP note + minimal setup + judgment §1–4) → Task 2. ✓
- Migrate infra (contract/doctor/grep_corpus/filename_fast_path/install/fixtures/contract-test) → Task 1. ✓
- Retire `nemo-retriever` (git rm remainder) → Task 3. ✓
- doctor/contract paths still resolve post-move → Task 1 Step 2 + Task 4 Step 1. ✓
- Mechanics-lean skill body → Task 2 Step 2. ✓
- Skill's actual command works live → Task 4 Step 2. ✓
- No engine code change; activation via symlinked `skills/` → only `skills/` touched. ✓

**Placeholder scan:** No TBD/TODO. `<RETRIEVER_VENV>`/`<question>`/`<dir>` are the skill's own substitution tokens (its documented convention), not plan gaps.

**Type consistency:** the SKILL.md evidence fields (`text,source,locator,modality,fidelity,score,citation`) + `coverage` match the SP-B `retrieve` output and the `contract.schema.json`; the `retriever retrieve` command + flags match SP-B/SP-C. The migrate/drop file lists partition the 20 tracked files with no overlap. `doctor.py`'s relative `contract/` resolution holds because `scripts/` and `contract/` move together (Task 1 Step 1).
