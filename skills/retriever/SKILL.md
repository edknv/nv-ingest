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
