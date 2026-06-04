# Skill-first agent-native retriever — ground-up rebuild design

**Date:** 2026-06-03
**Type:** Architecture design (greenfield rebuild of the nemo-retriever library, reoriented around the skill)
**Status:** Design exploration — not yet planned/scheduled for implementation

## Motivation

"Skill" here means **`SKILL.md`** — the disciplined single-query-turn `nemo-retriever` skill — not any workflow built on top of it. Today that skill is a thick adapter that compensates for a **library** not built for agent consumption. The compensations are visible *in the skill's own files*:

- **Schema gymnastics** — `query.md` documents hits as `pdf_basename` / `metadata.type` / `rank` / `_distance`, but the installed CLI emits lean `{page_number, source, text}`; the skill must reconcile the difference and derive a display name from `source`.
- **A manual verification recipe** — `query.md`'s entire "Charts and images" section, including "run ONE `pdf stage page-elements --method pdfium` call to confirm the number," is a hand-rolled `verify` the skill performs because the engine offers none.
- **Cache-cost hard limits** — the skill's "≤2 Bash calls, no narration between calls, quadratic cache cost" rules exist purely to manage agent economics the engine forces on it.
- **Drift-prone setup recipes** — `setup.md` documents an `--input-type` flag per format that the installed CLI **does not have** (it auto-detects in one pass), and a default table name that differs from what the skill uses, so every query needs explicit `--table-name`/`--lancedb-uri`.
- **Cold-load cost** — every `retriever query` cold-loads the embedder/reranker on GPU via vLLM (~30–60s), which `troubleshooting.md` and the hard-limits exist to work around.

**Thesis:** "Reorient everything for the skill" = push these `SKILL.md` compensations *down into the library* behind a stable typed contract, so `SKILL.md` collapses to a thin, declarative page and stops doing the engine's job. *(The `nemo-retriever-workflow` built earlier is one optional consumer that independently hit the same gaps — a useful corroboration, not the design's subject.)*

## Objectives (ranked, with dependencies)

1. **Thin, drift-proof skill (foundation).** A versioned typed contract, co-versioned skill, and contract tests so the skill stops compensating. Everything else depends on stable typed I/O.
2. **Native agent-quality retrieval (differentiator).** Multi-angle fusion + chart/image verification as first-class library operations. Depends on the contract's modality/fidelity fields.
3. **Agent-economics (multiplier).** Persistent daemon (no per-query cold-load), MCP surface, compact-by-default I/O. Orthogonal to #2; depends only on the contract. Ranked third as an optimization, **but pull forward if cost is the felt pain** — per-query cold-load was empirically the worst thing observed.

## Architectural shape

Three shapes were considered:
- **(A) Layered engine + thin co-versioned skill** — a core engine fronted by one stable typed contract; CLI, daemon, and skill are thin clients.
- **(B) Service-first** — everything is a long-running service; CLI/skill are RPC clients. Best for cost, heaviest ops.
- **(C) MCP-native** — the library *is* an MCP server; the skill is mostly tool definitions + a doc.

**Decision:** (A) is the spine; (B) and (C) are two I/O surfaces hanging off the *same* contract. A makes the skill thin and drift-proof; B and C are how agents reach it cheaply.

## Components

### 1. The Contract (the spine — Objective 1)

A semver'd, typed schema that **is** the public surface. Core types:

```
Hit {
  doc_id, source_uri,
  locator,        // page | segment | timestamp | bbox (modality-appropriate)
  modality,       // text | table | chart | image | audio | video_frame
  text, score,
  provenance { extractor, model, fidelity, confidence }
}
IngestManifest { ingested:[{format,count}], skipped:[{path,format,reason}], index_id }
Verdict { claim, verdict, basis_modality, evidence }   // verdict: confirmed | refuted | unverifiable
```

Key innovation — **`fidelity` ∈ {verbatim, ocr, transcribed, vlm_caption}**: it bakes the chart/image-hallucination caveat into the data model. A `vlm_caption` hit is structurally lower-fidelity than a `verbatim` text hit, so ranking, fusion, and "does this claim need verification?" become data-driven rather than skill-side prose warnings.

Rules: every field always present (no version-dependent optionals — that is the exact bug class hit this session). Schema is semver'd; the skill declares the contract version it needs; `retriever doctor` asserts the installed engine satisfies it.

### 2. Ingest — one verb, idempotent, loud (Objectives 1–2)

Keep the single-pass auto-detect (already the real behavior), but rebuilt to be:
- **Idempotent & incremental:** content-hash dedup, upsert-by-default — removes the `--overwrite`/`--append` footgun.
- **Loud about gaps:** always emit `IngestManifest` (skipped formats + reasons + install hints); never silently drop a format.
- **Modality-faithful at extraction:** every chunk tagged with `modality` + `fidelity` at ingest time, so downstream honesty is structural.

### 3. Multi-strategy fusion *inside* the single query (Objective 2 — the big one)

The skill issues **one disciplined query per turn** — it does not fan out. So the engine makes that single call strong internally: multi-strategy retrieval + fusion happen *inside* `query()`, transparently to the skill.

```
query(q, strategies=[semantic, lexical, visual, tabular], fuse=rrf, verify=auto) -> Hit[]
```

The engine owns fan-out + reciprocal-rank fusion + dedup-by-`(doc, locator, modality)`. Strategies are a pluggable registry (`semantic` = embed+rerank, `lexical` = keyword/regex, `visual`/`tabular` = modality filters), defaulting to all of them so the skill's one call is automatically as strong as a hand-built sweep — with no orchestration, no per-angle CLI flags, and no subagents (which the skill bans anyway). Single-strategy stays available for callers who want it; the skill simply takes the strong default.

### 4. Verification as an engine op (Objective 2)

```
verify(claim, hit) -> Verdict
```

Auto-selects the strongest independent modality (prose re-extract for PDF; cross-modal corroboration otherwise). Because the engine knows `fidelity`, it can auto-flag claims resting only on a `vlm_caption` and verify them inside `query(..., verify=auto)`, returning answer-ready, trust-tagged evidence. This **replaces `query.md`'s hand-rolled "run ONE `pdf stage page-elements` call to confirm the chart number" recipe** — the skill stops carrying a verification procedure the engine should own.

### 5. Persistent serving (Objective 3 — the cost multiplier)

A `retriever serve` daemon holds embedder/reranker/extractor models warm; CLI and skill become thin clients over a local socket — killing the per-query vLLM cold-load. The same daemon exposes an **MCP server** surface (`ingest`/`query`/`verify` tools) so any agent harness consumes the retriever directly with typed I/O, reducing the skill to tool wiring + a usage doc.

### 6. Agent-economics-aware I/O (Objective 3)

Compact-by-default responses (summaries + fetch handles), opt-in detail fetch — making the skill's cache discipline ("don't dump 10 hits") a library default. Clean JSON stdout; `troubleshooting.md` recipes become typed error codes with recovery hints.

### 7. The skill, reborn thin & co-versioned (Objective 1)

SKILL.md collapses to ~a page: start daemon (or point at MCP) → ingest once → `query` with verify → cite via provenance. The `references/` (setup/query/troubleshooting/cli) largely evaporate because the engine stops surprising. The skill ships **inside the library repo, co-versioned**, with contract tests in CI + `retriever doctor` asserting the installed engine matches the skill's declared contract version — fixing the drift root-cause structurally.

## Data flow (end to end)

```
corpus/ --ingest--> [engine: extract+modality+fidelity tag] --> index (+ IngestManifest)
                                                                     |
agent ── query(q, strategies, verify=auto) ──> [daemon: fan-out strategies
                                                  -> RRF fuse -> dedup
                                                  -> auto-verify low-fidelity claims]
                                                                     |
                                                <── Hit[] + Verdict[] (typed, provenance-tagged, compact)
```

## Migration (strangler, even though greenfield)

Stand up the contract API + daemon **alongside** the current engine; move the skill onto the new typed surface; then rebuild internals behind the stable contract. The skill never breaks during the rewrite.

## Trade-offs / risks

- Daemon/MCP adds a stateful ops surface (model lifecycle, memory residency).
- In-engine strategy fusion trades some agent flexibility for simplicity — mitigated by keeping strategies composable/overridable.
- Co-versioning couples skill and library release cadence — intended, but a real constraint.
- Greenfield cost is large; the lighter "contract + conform" subset was explicitly *not* chosen, so the strangler path is how cost is contained.

## The tell that this design is right

`SKILL.md` collapses from a multi-file adapter (SKILL + setup/query/troubleshooting/cli references) to roughly a **single declarative page**: start daemon → ingest once → `query` with verify → cite via provenance. The schema gymnastics, the chart/image verification recipe, the cache-cost hard-limits, and the per-format setup recipes all disappear because the engine no longer surprises the skill. *(Secondary corroboration: the `nemo-retriever-workflow` also becomes obsolete or a ~20-line wrapper, since the multi-strategy sweep and verification it hand-assembled are now single engine calls — but the skill, not the workflow, is the design's measure of success.)*

## Non-goals

- Not a re-ranking/embedding-model research effort — this is about the *interface and orchestration* the library exposes to agents, not new model quality.
- Not a general-purpose vector DB — it remains a document-retrieval engine over LanceDB.
- Does not prescribe the MCP vs daemon choice as mutually exclusive — both ride the same contract.

## Open questions (for a future planning pass)

- Daemon transport: local Unix socket + MCP only, or also a network service for multi-client deployments?
- Strategy fusion default: RRF vs learned fusion — start with RRF (transparent, no training).
- Index identity across re-ingest: content-hash per-doc upsert vs whole-table rebuild semantics.
- How much of `verify` can be precomputed at ingest (fidelity is known then) vs must be query-time.
