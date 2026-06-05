# `retriever` — skill-first design (skill + `retrieve` contract)

**Date:** 2026-06-04
**Type:** Skill-first design. The skill is the product; the library is its backend.
**Supersedes framing of:** `2026-06-03-skill-first-retriever-rebuild-design.md` (that rebuilt the *library* and bolted onto the old CLI skill; this designs the *skill* from scratch and specifies the library to serve it).

## Principle

Design top-down: **skill → the one primitive it needs → the library work that primitive requires.** We do not constrain the design to the current `nemo_retriever` library's shape; the library is whatever satisfies the contract below. This is the explicit reframe: the skill comes first.

## The skill (the product)

A judgment playbook an agent uses to answer / quote / verify / aggregate over a document corpus. It contains **only** retrieval *wisdom* — the capability lives in one typed primitive. Full intended `SKILL.md`:

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

That is the whole skill: no flags, escaping, stdout discipline, venv paths, per-format recipes, or cache hard-limits — none of it exists at the skill layer.

## The contract the skill depends on

```
retrieve(question: str, intent?: "lookup|aggregate|quote|compare|describe|transcribe")
  -> {
       evidence: [ {
         text: str,
         source: str,                     # doc id / name
         locator: { kind: "page|segment|timestamp|bbox", value: ... },
         modality: "text|table|chart|image|audio|video_frame",
         fidelity: "verbatim|ocr|transcribed|vlm_caption",
         score: number,
         citation: str                    # render-ready, e.g. "doc p.3"
       } ],
       coverage: { strategies_used: [str], n_docs_seen: int, thin_spots: [str] }
     }

verify(claim: str, source: str, locator?) -> { evidence: [...], corroborated_signal: bool }

index(path: str) -> { ingested: [...], skipped: [...] }
```

Contract properties (the library must provide):
- **Fused multi-strategy** — semantic + lexical + visual + tabular, fused in-engine; the agent never picks strategies.
- **True `fidelity`** on every hit — the differentiator §2 reasons over (requires ingest provenance; see SP-A).
- **Citations + `coverage`** built in; **compact by default** with drill-down.
- **Served warm** (no per-call cold-load) and **available as an MCP tool** (the skill's preferred surface) and a CLI verb.

## Fidelity model (the differentiator)

`fidelity` ranks how trustworthy a hit's text is as a literal source:
`verbatim` (extracted text layer) > `ocr` (OCR'd) > `transcribed` (ASR for audio/video) > `vlm_caption` (chart/image model caption — error-prone for numbers/directions). The skill's trust ordering and verify-trigger are driven entirely by this field. It must be assigned at **ingest** from the extractor used (not derivable post-hoc from modality alone — that's why SP-A is foundational).

## Library sub-projects (downstream; specified *against* this contract)

In skill-first dependency order — each its own spec → plan → build:
- **SP-A — ingest provenance → `fidelity`:** record extractor/OCR/ASR/caption provenance per chunk so a true `fidelity` exists. *Foundational new capability; today only `modality` is stored.*
- **SP-B — `retrieve` primitive:** fuse strategies → attach `fidelity` + `citation` → compute `coverage` → return answer-ready evidence. Composes existing hybrid/content-type/verify work + SP-A's fidelity.
- **SP-C — serving + MCP:** expose `retrieve`/`verify`/`index` warm (build on Objective-3a `serve-models`) and as MCP tools (build on Objective-3b), re-pointed to `retrieve`.
- **SP-D — ship the skill** (the markdown above) once `retrieve` is live; retire the old CLI skill.

## Boundaries (what's skill vs library)

- **Library:** retrieve, fuse, tag fidelity, cite, verify, report coverage, serve warm. Mechanical, typed, testable.
- **Skill (the model):** choose the move, decide when evidence is trustworthy, decide when to verify, compose an honest answer, decide when to refuse. Judgment only.
- The line: the library *gives trustworthy evidence*; the skill *reasons about trust and sufficiency*. The library never decides confirmed/refuted or composes the final answer.

## Non-goals
- Not extending or compatible-with the existing CLI SKILL.md — it is retired (SP-D), not maintained.
- The library does not synthesize final answers or judge claims (engine retrieves/tags; agent judges).
- Not a general RAG framework — one corpus-retrieval contract.

## This spec's deliverable & next step
This document is the **design driver**: the skill markdown + the `retrieve`/`verify`/`index` contract. Its implementation plan commits these as artifacts — the skill `SKILL.md` and a machine-readable contract schema — which the library SP-A→SP-C are then built to satisfy. The skill goes live (SP-D) once `retrieve` exists.

## Open questions
- `intent` hint: explicit param vs engine-inferred from the question. (Lean: optional hint; engine works without it.)
- `index` as a skill tool vs an operator-run setup step (ingest is heavy/one-time). (Lean: setup step, surfaced but not a per-turn tool.)
- Whether `coverage.thin_spots` is engine-computed or heuristic — pin in SP-B.
