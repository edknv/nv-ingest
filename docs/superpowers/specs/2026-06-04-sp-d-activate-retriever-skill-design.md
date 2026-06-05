# SP-D — activate the `retriever` skill, retire `nemo-retriever` (design)

**Date:** 2026-06-04
**Type:** Skill activation + retirement (final library sub-project of the skill-first design)
**Parent:** `2026-06-04-retriever-skill-first-design.md`. Depends on SP-A/B/C (the backend the skill calls).

## Problem

The from-scratch `retriever` skill + contract were committed as the design driver (`docs/superpowers/contracts/retriever/`); the backend (`fidelity`, `retrieve`, warm MCP) is now built (SP-A/B/C). SP-D makes the skill **real**: a loadable `skills/retriever/` skill, and retires the old CLI-manual `nemo-retriever` skill it supersedes.

## Key finding (grounding)

**No `retriever` MCP server is registered in this harness** (no `.mcp.json`, nothing in `.claude/`). So a tool-first skill ("call the `retrieve` tool") would be **broken when activated** — the tools don't exist. The activated skill must be **CLI-backed**: its one primitive is the `retriever retrieve "<q>"` CLI verb (SP-B, warm via `EMBED_INVOKE_URL` per SP-C), with a note to prefer the MCP tools where a harness provides them.

The old `skills/nemo-retriever/` (20 tracked files) splits into: **mechanics to drop** (`SKILL.md`, `references/{query,setup,troubleshooting,cli}`, `BENCHMARK.md`, `evals/`, `skill-card.md`, `skill.oms.sig`) and **infra to keep** (`contract/`, `scripts/doctor.py`, `scripts/grep_corpus.py`, `scripts/filename_fast_path.py`, `references/install.md`, `tests/{fixtures,test_contract.py}`).

## Decisions (from brainstorming)
- **CLI-backed** activation (+ "use MCP tools if your harness has them" note).
- **Migrate infra, then `git rm` the old skill** — consolidate onto one active skill, keep the useful infra.

## Design

### 1. New `skills/retriever/SKILL.md` (CLI-backed, judgment-only)
The playbook from the design driver, with the invocation made concrete:
- **One primitive:** `<RETRIEVER_VENV>/bin/retriever retrieve "<question>"` → `{evidence:[{text,source,locator,modality,fidelity,score,citation}], coverage}`. (If the harness exposes the retriever MCP tools, prefer the `retrieve` tool — same result, no Bash.)
- **Setup (one-time, operator):** install per `references/install.md` if `retriever` is missing; `retriever ingest <dir>` once (`--hybrid` for exact-term recall); optionally `retriever serve-models` + export `EMBED_INVOKE_URL` for warm queries; `scripts/doctor.py` to confirm the engine matches the contract.
- **Judgment (unchanged):** §1 pick the move; §2 trust by fidelity (verbatim > ocr > transcribed > vlm_caption; verify/hedge low-fidelity numbers); §3 answer honestly (cite via `citation`, cover every entity, refuse if not in evidence, read `coverage`); §4 failure-mode policy.

Per-turn the skill is one `retrieve` call + reasoning; the only "mechanics" are the one-time operator setup lines.

### 2. Migrate infra into `skills/retriever/`
`git mv` from `skills/nemo-retriever/`: `contract/`, `scripts/doctor.py`, `scripts/grep_corpus.py`, `scripts/filename_fast_path.py`, `references/install.md`, `tests/fixtures/contract_probe.txt`, `tests/test_contract.py`. `doctor.py` resolves its `contract/` relative to `scripts/` → paths hold after the move.

### 3. Retire `nemo-retriever`
`git rm -r` the remainder of `skills/nemo-retriever/` (the dropped mechanics + the now-empty dir). After SP-D the only active retrieval skill is `retriever`.

## Components & boundaries
- `skills/retriever/SKILL.md` — the activated skill (judgment + minimal setup + one CLI primitive).
- Migrated infra — `contract/` + `doctor.py` (engine drift-check), `grep_corpus.py` (keyword fallback), `install.md`, contract test/fixture.
- **Boundary:** SP-D is activation + file moves; no engine/retrieval code changes. `skills/` is symlinked from `.claude/skills`, so creating `skills/retriever/SKILL.md` makes it loadable.

## Testing / validation
- **Mechanical:** after migration, `skills/retriever/scripts/doctor.py` still runs (paths resolve) and the migrated contract test passes; `skills/nemo-retriever/` no longer exists; `skills/retriever/SKILL.md` present.
- **Skill-path live:** the skill's actual command works — `retriever retrieve "<q>"` against a tiny index returns the contract shape (already proven in SP-B; re-confirm via the new skill's exact command form).
- **Mechanics-lean check:** `SKILL.md` body references only `retrieve`/`ingest`/`serve-models`/`doctor` + judgment — no flag-spelling, escaping, stdout discipline, or vector-DB internals.

## Non-goals
- No registration of an MCP server (harness/operator concern; the skill notes it).
- No engine/retrieval/ingest code changes.
- Not preserving the old skill's `evals/`/`BENCHMARK.md`/`skill-card.md` (retired; recoverable from git).
- `skill.oms.sig` not regenerated (signing out of scope, per standing guidance).

## Open questions
- Whether the new skill should carry a trimmed `setup.md`/`troubleshooting.md` or fold the few needed lines into `SKILL.md`. Lean: fold into `SKILL.md` + keep only `install.md` as a reference (smallest surface).
- The skill-first `retrieve_result` contract lives at `docs/superpowers/contracts/retriever/contract.schema.json`; whether to also copy it under `skills/retriever/contract/` for colocation. Lean: leave it in docs (single source); `SKILL.md` need not cite it.
