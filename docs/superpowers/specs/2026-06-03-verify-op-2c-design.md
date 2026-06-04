# Objective 2c — `verify` (index-lookup evidence fetch)

**Date:** 2026-06-03
**Type:** Engine change to `nemo_retriever` (third/final slice of rebuild Objective 2)
**Parent:** `2026-06-03-skill-first-retriever-rebuild-design.md` (Objective 2). Follows 2a (typed hits) and 2b (hybrid).

## Problem

`query.md` carries a hand-rolled verification recipe: when a chart/image hit gives a number, "run ONE `pdf stage page-elements --method pdfium` call" to confirm it against prose. That recipe is the skill doing the engine's job. 2c promotes it into a `verify` engine op — but as a **retrieval** primitive, not a judge.

## Decisions (from brainstorming)

- **Mechanism: index-lookup only.** Fetch the stored `text`/`table` chunks for a claim's `(source, page)` directly from LanceDB. No re-extraction, no GPU, all formats. Sufficient because ingest already stores a page's prose as a distinct `text` chunk separate from its `chart`/`image` chunk — the corroborating evidence is already indexed.
- **Returns evidence + a mechanical signal, NOT a confirmed/refuted verdict.** A retrieval engine fetches; judging agreement vs contradiction is LLM work (what the workflow's Phase 3 agent did). `verify` returns the independent-modality evidence and a purely mechanical signal (do the claim's numbers/terms appear in it?); the caller judges.

## Interface

### SDK (`adapters/cli/sdk_workflow.py`)
```
verify_claim(
    claim: str,
    source: str,
    *,
    page: int | None = None,
    lancedb_uri: str = DEFAULT_LANCEDB_URI,
    table_name: str = DEFAULT_TABLE_NAME,
    against: Sequence[str] = ("text", "table"),
) -> dict
```
Direct LanceDB table access (mirrors the existing `_count_lancedb_rows` direct-access pattern — not the graph/retrieval path). Opens the table, parses each row's `source` JSON (`source_name`) and `metadata` JSON (`page_number`, `type`), and:
- filters to rows whose `source_name` matches `source` (basename-tolerant: compare basenames with/without extension),
- if `page` is given, filters to that `page_number`,
- keeps only rows whose `type` ∈ `against` (default `text`,`table` — i.e. **excludes** the chart/image modality being checked, so evidence is genuinely independent),
- returns those chunks as `evidence`, plus a mechanical term/number overlap signal.

### CLI (`adapters/cli/main.py`, new `verify` subcommand)
```
retriever verify "<claim>" --source <doc> [--page N] --table-name <t> --lancedb-uri <dir> [--against text,table]
```
Prints the result dict as JSON (same clean-stdout convention as `query`).

### Output
```
{
  "claim": str,
  "source": str,
  "page": int | null,
  "evidence": [ { "text": str, "modality": str, "page": int } ],
  "independent_evidence_found": bool,     # any text/table chunk at the location
  "matched_terms": [str],                  # claim numbers/terms found verbatim in evidence
  "unmatched_terms": [str]                 # claim numbers/terms NOT found
}
```

## Mechanical signal (no judgment)

Extract from `claim`: numbers via regex `\d[\d.,%]*`, and salient terms (tokens length ≥ 4, minus stopwords). For each, check **verbatim presence** in the concatenated evidence text. `matched_terms`/`unmatched_terms` report the split; `independent_evidence_found` is whether any `against`-modality chunk exists at the location. No agreement/contradiction inference — that is explicitly the caller's job.

## Components & boundaries

- `verify_claim` — a self-contained SDK function doing a deterministic table lookup; depends only on `lancedb` + JSON parsing (same surface as `_count_lancedb_rows`). Independent of the retrieval graph, embedder, and reranker.
- `verify` CLI command — thin Typer wrapper: parse args → call `verify_claim` → print JSON.
- No change to `Retriever`, operators, `LanceDB.retrieval`, or ingest.

## Contract + skill (consistent with 2a/2b)

- `cli-contract.json`: bump `contract_version` 1.2.0 → **1.3.0**; record the `verify` subcommand (e.g. a `subcommands` list including `verify`).
- `CONTRACT.md`: changelog for 1.3.0.
- `doctor.py`: assert the `verify` subcommand exists (e.g. `retriever verify --help` exits 0 / appears in `retriever --help`). Static, no GPU.
- `SKILL.md`: declare contract 1.3.0.
- `query.md`: replace the manual "run ONE pdfium `page-elements` call" step in the chart/image section with: run `retriever verify "<claim>" --source <doc> --page <N>` to fetch the page's independent prose/table evidence, then judge the chart number against it (engine fetches, you judge).

## Testing

- **Unit** (no GPU): construct a tiny LanceDB table in a temp dir with a chart chunk + a text chunk for the same `(source, page)`; assert `verify_claim` returns the text chunk as evidence (not the chart), `independent_evidence_found == True`, and that a number present in the text lands in `matched_terms` while an absent one lands in `unmatched_terms`. A second case with no text/table chunk asserts `independent_evidence_found == False`.
- **CLI** (no GPU): mock `verify_claim` (or point at the temp table) and assert the `verify` command prints the expected JSON and exits 0.
- **Live round-trip** (GPU for ingest only): ingest `multimodal_test` (has chart + text on p.1) → `retriever verify "Premium desk fan costs $150" --source multimodal_test --page 1` → returns the page's text/table evidence with the term-overlap signal; `doctor` stays green with `verify` asserted.

## Non-goals

- No re-extraction / pdfium / YOLOX path (index-lookup only).
- No confirmed/refuted/agreement judgment in the engine (caller judges).
- No `query(..., verify=auto)` integration (that auto-flags + auto-verifies; a larger follow-on needing claim extraction — out of scope).
- No semantic matching of evidence to claim (the signal is verbatim term/number presence only).

## Open question

- Basename-tolerant `source` matching could be ambiguous if two docs share a basename. Acceptable for v1 (single-corpus skill use); a future slice could match on full `source_id` path when provided.
