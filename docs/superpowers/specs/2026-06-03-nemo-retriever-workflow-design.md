# nemo-retriever-workflow — design

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
  question:     string,   // required — the hard question to answer
  corpusDir:    string,   // default "./pdfs"  — source docs (any supported format; ingested only if index missing)
  indexDir:     string,   // default "./lancedb"
  tableName:    string,   // default "nv-ingest"
  topK:         number,   // default 10
  embedModel:   string,   // default "nvidia/llama-nemotron-embed-1b-v2"
  angles:       string[], // default all five below
  verify:       boolean,  // default true  — Phase 3 adversarial verification
  writeReport:  boolean,  // default true  — return reportMarkdown for the caller to write
  ocrLang:      string,   // default "english" — OCR language for image ingest (or "multi")
  installExtras: boolean  // default false — may Phase 0 attempt to install missing host deps (libreoffice/ffmpeg/[multimedia])?
}
```

`args` may arrive as a JSON-encoded string from the Workflow tool; the script normalizes string-or-object before reading.

## Phases

### Phase 0 — Setup (1 agent, barrier) — format-aware multi-pass ingest

Resolve the `retriever` venv path (`command -v retriever`; if missing, follow the skill's `references/install.md`). If `indexDir` already has the table, **reuse it** (no ingest). Otherwise build it with format-aware multi-pass ingest:

1. Inventory file extensions in `corpusDir` and group into buckets:

   | Bucket | Extensions | Ingest flags |
   |---|---|---|
   | default | `.pdf .html .txt` | *(none — base install)* |
   | image | `.jpg .png .tiff .bmp` | `--input-type image --ocr-version v2 --ocr-lang <ocrLang>` |
   | doc | `.docx .pptx` | `--input-type doc` (needs libreoffice) |
   | audio | `.mp3 .wav .m4a` | `--input-type audio` (needs `[multimedia]` + ffmpeg) |
   | video | `.mp4 .mov .mkv` | `--input-type video` (needs `[multimedia]` + ffmpeg) |

2. Run one ingest pass per non-empty bucket into the **same table**: the **first pass uses `--overwrite`** (default — creates a fresh table), **every subsequent pass uses `--append`** (adds rows without dup-checks). Explicit `--input-type X` over a mixed folder processes only the matching subset, so passes don't need per-type subdirs.
3. For a bucket whose host deps are missing: if `installExtras` is true, attempt the install from `references/install.md`; otherwise **skip it and record the reason** (never silently drop — see the skill's "Unsupported file types" warning).

Returns structured:

```
{ retrieverVenv: string, indexReady: boolean, docCount: number, distinctDocs: string[],
  ingestedTypes: string[], skippedTypes: [{ type: string, reason: string }] }
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
| `semantic` | `retriever query "<question>" --top-k <topK> --rerank --table-name <t> --lancedb-uri <dir> --embed-model-name <m>` |
| `reformulated` | agent rephrases the question 2–3 ways (incl. a HyDE-style hypothetical-answer phrasing), runs a query per phrasing, unions hits |
| `keyword` | agent extracts key terms/identifiers, runs `scripts/grep_corpus.py "<regex>" --lancedb-uri <dir> --table-name <t>` |
| `visual` | query with `--content-types chart,image` (server-side filter); tag hits chart/image |
| `tabular` | query with `--content-types table`; tag hits table |

**Schema tolerance:** the installed `retriever query` emits lean hits (`page_number`, `source`, `text`) — no per-hit `type` or `pdf_basename`. So all queries pass `--table-name`/`--lancedb-uri` explicitly (the CLI default table differs), `doc` is derived from `pdf_basename`-or-`basename(source)`, and `type` is inferred from the producing angle (visual→chart/image, tabular→table, others→text) when the CLI omits it. For audio/video corpora `visual`/`tabular` simply return empty (which the merge tolerates).

Barrier is justified (not laziness): the merge needs **all** hits to dedupe, and we early-exit if the total hit count is zero.

### Phase 2 — Merge (1 agent)

Dedupe hits by `(doc, page, type)`. Synthesize a **draft** `final_answer` with `[doc p.N]` citations. Emit `claims_to_verify`: any number or directional claim resting **only** on a `chart`/`image` hit with no corroborating `text`/`table` hit covering the same fact. Returns:

```
{ draftAnswer: string,
  citations: [{ doc, page, type }],
  claims_to_verify: [{ claim: string, doc: string, page: number }],
  confidence: string }
```

### Phase 3 — Adversarial verify (parallel, conditional on `verify` and non-empty `claims_to_verify`) — format-aware

One agent per flagged claim verifies it against an independent modality, branching on the flagged source's extension:

- **`.pdf`** → targeted prose re-extract (strongest), then read the page JSON:
  ```
  retriever pdf stage page-elements <corpusDir> --method pdfium --json-output-dir /tmp/sweep_verify --compact-json
  ```
- **non-PDF** (image / office / audio / video) → re-query the index restricted to that `source` with `--content-types text,table` and judge whether genuine (non-caption) text corroborates the claim.

Returns `{ claim, verdict: "confirmed"|"refuted"|"not_found"|"unverifiable", evidence: string }` — `unverifiable` when no independent modality exists (e.g. a number that lives only in an image with no transcript).

A short final-synthesis step folds verdicts into the answer: confirmed → assert confidently; refuted/not_found/unverifiable → hedge with the verbatim chart phrase ("chart-derived, not verified against prose"). This turns the skill's #1 documented accuracy failure into a verification gate.

### Output

Workflow returns:

```
{ final_answer: string,
  citations: [{ doc, page, type }],
  confidence: string,
  byAngle: [...Phase 1 results...],
  verified: [...Phase 3 verdicts...] }
```

The workflow returns `reportMarkdown` (a full report string: `final_answer`, per-angle hit tables, verification verdicts) and `reportPath` (the intended path, or null when `writeReport` is false). Workflow scripts have no filesystem access, so the **caller writes** `reportMarkdown` to `reportPath` after the run returns.

## Indexing note

`retriever` `page_number` is **1-indexed** (first page = 1). Citations and report use 1-indexed pages as-is. (Downstream consumers that expect 0-indexed must subtract 1 — out of scope here.) For **audio/video** sources `page_number` is a segment index / timestamp rather than a page; it is cited verbatim as returned.

## Multi-format support

| Format | Auto-ingest (Phase 0) | Query + sweep | visual/tabular | Verify (Phase 3) |
|---|---|---|---|---|
| PDF | yes | yes | yes | pdfium prose re-extract |
| Image (`.jpg .png .tiff .bmp`) | yes (`--input-type image` + OCR) | yes | yes | index re-query |
| Office (`.docx .pptx`) | yes if libreoffice present | yes | yes if charts/tables extracted | index re-query |
| HTML / TXT | yes | yes | n/a | index re-query |
| Audio / Video | yes if `[multimedia]`+ffmpeg present | yes (over transcript) | n/a (returns empty) | index re-query |

Buckets whose host deps are absent (and `installExtras` is false) are skipped and reported in `skippedTypes`, never silently dropped.

## Non-goals (YAGNI)

- Not a general RAG framework — it wraps the existing `retriever` CLI only.
- No incremental/delta indexing — build once if missing (multi-pass for mixed formats), otherwise reuse the whole table.
- No 0-indexed page conversion, no eval/scoring (that's Pattern D, a separate artifact).
- Angle agents do not spawn their own subagents (the skill bans it; the workflow owns fan-out).

## Components & boundaries

- **Workflow script** (`workflows/nemo-retriever-workflow.js`, with `.claude/workflows` a symlink to `../workflows` — mirrors the `skills/` layout) — pure orchestration: phases, fan-out, dedupe glue, conditional verify, return shape. No Bash.
- **Angle-agent prompts** — self-contained: each carries its exact CLI template + structured-output contract. Swapping an angle = editing one prompt + one `angles` entry.
- **Schemas** — `READER`/`MERGE`/`VERDICT` JSON schemas drive `agent({schema})` so returns are validated, not parsed.

Each unit is independently understandable: an angle prompt says what command to run and what to return; the script says how results flow; schemas say the contract between them.
