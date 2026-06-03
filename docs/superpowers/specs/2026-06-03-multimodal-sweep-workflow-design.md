# multimodal-sweep — design

**Date:** 2026-06-03
**Type:** Reusable Claude Code workflow (named, lives in `.claude/workflows/`)
**Pattern:** Multi-modal retrieval sweep over a `nemo-retriever` corpus

## Problem

The `nemo-retriever` skill answers a question with **one** semantic query per turn. That single angle has known blind spots:

- Pure semantic search misses **exact terms / identifiers** a keyword search would catch.
- Numbers and directional claims often live in **figures and tables**, not prose.
- The skill itself documents that **chart/image transcriptions hallucinate** — they reverse direction words (increase↔decrease) and misread percentages — and must be verified against prose.

A single agent also can't widen coverage without blowing up cache cost: in a normal session every query turn re-reads all prior turns (quadratic). The skill's hard limits (≤2 Bash calls, no narration between calls, no subagents) exist to contain exactly this.

## Insight: workflows are the right host

A workflow spawns N **isolated** subagents, each with its own context. So:

- Each leaf agent does **one disciplined `nemo-retriever` query turn** and returns structured output — exactly what the skill is optimized for.
- The quadratic cache problem disappears: concurrent queries don't re-read each other.
- The workflow (not the leaf) owns fan-out, dedupe, and verification — none of which a single query turn is allowed to do.

**Shared-state rule:** the LanceDB index is built **once** (`retriever ingest`) and is concurrent-safe to *read*. Never ingest in parallel. Ingest is a setup agent because workflow scripts have no Bash access — only agents do.

## Inputs (`args`)

```
{
  question:    string,   // required — the hard question to answer
  corpusDir:   string,   // default "./pdfs"  — source docs (ingested only if index missing)
  indexDir:    string,   // default "./lancedb"
  tableName:   string,   // default "nv-ingest"
  topK:        number,   // default 10
  embedModel:  string,   // default "nvidia/llama-nemotron-embed-1b-v2"
  angles:      string[], // default all five below
  verify:      boolean,  // default true  — Phase 3 adversarial verification
  writeReport: boolean   // default true  — write markdown report to repo
}
```

## Phases

### Phase 0 — Setup (1 agent, barrier)

Resolve the `retriever` venv path (`command -v retriever`; if missing, follow the skill's `references/install.md`). If `indexDir` has no index, run `retriever ingest corpusDir`. Returns structured:

```
{ retrieverVenv: string, indexReady: boolean, docCount: number, distinctDocs: string[] }
```

True barrier — every downstream agent needs `retrieverVenv`. If `indexReady` is false after this phase, abort with a clear message.

### Phase 1 — Sweep (N parallel angle agents, barrier)

Each angle agent receives: the question, the **exact CLI command** for its angle, and the resolved `retrieverVenv`. Each does one disciplined query and returns:

```
{ angle: string,
  candidateAnswer: string,
  hits: [{ doc: string, page: number, type: "text"|"table"|"chart"|"image", snippet: string, rank: number }],
  confidence: "high"|"medium"|"low" }
```

Default angles:

| Angle | Mechanism |
|-------|-----------|
| `semantic` | `retriever query "<question>" --top-k <topK> --rerank --embed-model-name <embedModel>` |
| `reformulated` | agent rephrases the question 2–3 ways (incl. a HyDE-style hypothetical-answer phrasing), runs a query per phrasing, unions hits |
| `keyword` | agent extracts key terms/identifiers, runs `scripts/grep_corpus.py "<regex>"` |
| `visual` | semantic query, then keep only hits with `metadata.type ∈ {chart, image}` |
| `tabular` | semantic query, then keep only hits with `metadata.type == table` |

Barrier is justified (not laziness): the merge needs **all** hits to dedupe, and we early-exit if the total hit count is zero.

### Phase 2 — Merge (1 agent)

Dedupe hits by `(doc, page, type)`. Synthesize a **draft** `final_answer` with `[doc p.N]` citations. Emit `claims_to_verify`: any number or directional claim resting **only** on a `chart`/`image` hit with no corroborating `text`/`table` hit covering the same fact. Returns:

```
{ draftAnswer: string,
  citations: [{ doc, page, type }],
  claims_to_verify: [{ claim: string, doc: string, page: number }],
  confidence: string }
```

### Phase 3 — Adversarial verify (parallel, conditional on `verify` and non-empty `claims_to_verify`)

One agent per flagged claim runs the targeted prose extract on that page:

```
retriever pdf stage page-elements <corpusDir> --method pdfium --json-output-dir /tmp/pdf_text --compact-json
```

then reads the page's extraction JSON and returns `{ claim, verdict: "confirmed"|"refuted"|"not_found", evidence: string }`.

A short final-synthesis step folds verdicts into the answer: confirmed → assert confidently; refuted/not_found → hedge with the verbatim chart phrase ("chart-derived, not verified against prose"). This turns the skill's #1 documented accuracy failure into a verification gate.

### Output

Workflow returns:

```
{ final_answer: string,
  citations: [{ doc, page, type }],
  confidence: string,
  byAngle: [...Phase 1 results...],
  verified: [...Phase 3 verdicts...] }
```

If `writeReport`, also write a markdown report (`final_answer`, per-angle hit tables, verification verdicts) to the repo.

## Indexing note

`retriever` `page_number` is **1-indexed** (first page = 1). Citations and report use 1-indexed pages as-is. (Downstream consumers that expect 0-indexed must subtract 1 — out of scope here.)

## Non-goals (YAGNI)

- Not a general RAG framework — it wraps the existing `retriever` CLI only.
- No re-ingest / incremental indexing logic — ingest once if missing, otherwise reuse.
- No 0-indexed page conversion, no eval/scoring (that's Pattern D, a separate artifact).
- Angle agents do not spawn their own subagents (the skill bans it; the workflow owns fan-out).

## Components & boundaries

- **Workflow script** (`.claude/workflows/multimodal-sweep.js`) — pure orchestration: phases, fan-out, dedupe glue, conditional verify, return shape. No Bash.
- **Angle-agent prompts** — self-contained: each carries its exact CLI template + structured-output contract. Swapping an angle = editing one prompt + one `angles` entry.
- **Schemas** — `READER`/`MERGE`/`VERDICT` JSON schemas drive `agent({schema})` so returns are validated, not parsed.

Each unit is independently understandable: an angle prompt says what command to run and what to return; the script says how results flow; schemas say the contract between them.
