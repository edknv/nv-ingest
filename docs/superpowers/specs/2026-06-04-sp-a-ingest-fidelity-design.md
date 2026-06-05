# SP-A — ingest provenance → `fidelity` (design)

**Date:** 2026-06-04
**Type:** Engine change to `nemo_retriever` ingest (first library sub-project of the skill-first design)
**Parent:** `2026-06-04-retriever-skill-first-design.md` (the `retriever` skill + contract). SP-A is the foundational capability the contract's `fidelity` field needs.

## Problem

The skill's core judgment (§2 "trust by fidelity") and the contract's `evidence_item.fidelity` (`verbatim|ocr|transcribed|vlm_caption`) require knowing **how each chunk's text was produced**. Today the index stores `modality` (`type`: text/table/chart/image/audio) but **not** fidelity — so a verbatim PDF-text-layer chunk is indistinguishable from an error-prone chart caption. SP-A makes the real provenance a stored, queryable field. (SP-A only *stores* it; *surfacing* it in `retrieve` output is SP-B.)

## Key finding (grounding)

True fidelity is derivable at **one funnel**: `_client_record_from_graph_row` (`vdb/records.py:69`), where `content_metadata["type"]` is already stamped before LanceDB write. The real provenance signals are reachable there:
- `_content_type` / `type` (text/table/chart/image/audio/video) — always present.
- `needs_ocr_for_text` (the verbatim-vs-OCR signal) — set at `pdf/extract.py:377`, `image/load.py:53,132`, lives in the page `metadata`.
- `subtype == "page_image"` (page-OCR vs standalone VLM caption for images) — in `content_metadata`.
- audio/video — deterministically transcribed.

Extra metadata keys survive to the stored `metadata` JSON and are queryable (confirmed by the existing `_content_type` persistence test). So SP-A is a **single-point derivation**, not a per-stage rewrite.

## Design

Add a pure helper `_derive_fidelity(content_type, metadata, content_metadata) -> str | None` and call it in `_client_record_from_graph_row`, stamping `content_metadata.setdefault("fidelity", fidelity)` right after the existing `type` stamp.

### Mapping (→ contract enum `verbatim|ocr|transcribed|vlm_caption`)
| modality (`type`) | signal | fidelity |
|---|---|---|
| text | `needs_ocr_for_text` true | `ocr` |
| text | else (PDF text layer) | `verbatim` |
| image | `subtype == "page_image"` | `ocr` (page raster OCR) |
| image | else (standalone) | `vlm_caption` |
| table, chart, infographic | — (YOLOX region + OCR) | `ocr` |
| audio, video, video_frame | — | `transcribed` |
| unknown / missing type | — | `None` (omit field) |

Modality is **unchanged and still stored separately**, so the skill sees both `modality` (e.g. `table`) and `fidelity` (e.g. `ocr`).

### Signal-survival fallback
`needs_ocr_for_text` lives in page `metadata`; if it doesn't reach `row["metadata"]` at the funnel (the OCR stage may overwrite text without propagating the flag), text fidelity falls back to **`verbatim`** (conservative — the common case is the PDF text layer). The verbatim/ocr split is then *best-effort* in SP-A; threading the flag from the OCR stage for full correctness is a documented follow-up. Validation checks which fidelity values actually appear on a real ingested corpus.

## Components & boundaries
- `_derive_fidelity` — pure function of (content_type, metadata, content_metadata); independently unit-testable.
- `_client_record_from_graph_row` — one added call; no other behavior changes.
- **Boundary:** SP-A *stores* fidelity in index metadata. It does **not** surface it in query/`retrieve` output (that's SP-B) and does not change retrieval, ranking, or modality.

## Testing
- **Unit (no GPU):** call `_client_record_from_graph_row` (or `_derive_fidelity`) with constructed rows covering each mapping case — text(verbatim), text(needs_ocr→ocr), image(page_image→ocr), image(standalone→vlm_caption), table/chart→ocr, audio→transcribed, unknown→absent — asserting `content_metadata["fidelity"]`. Follow the `tests/test_lancedb_row_metadata.py` SimpleNamespace pattern.
- **Live (GPU, ingest):** ingest a mixed corpus (`multimodal_test.pdf` has text+table+chart; add a `.wav`), then read the LanceDB `metadata` and assert `fidelity` is present and sane per modality. **Record which fidelity values actually appear** (confirms whether `needs_ocr_for_text` survived).

## Non-goals
- No per-extraction-stage `_fidelity` stamping (single-point only; deeper threading is a follow-up).
- No query/`retrieve`-output surfacing of fidelity (SP-B).
- No new modality values, no ranking/retrieval change.
- No new contract enum value (table/chart map to `ocr`, not a separate tier).

## Open questions
- Does `needs_ocr_for_text` survive to `row["metadata"]` at the funnel? Pinned by the live validation; if not, text=`verbatim` fallback + a follow-up to thread it.
- `infographic` modality maps to `ocr` here; revisit if a distinct treatment is wanted.

## SP-A live result (executed 2026-06-04)

Ingested `multimodal_test.pdf` (+ a `.wav`) → 11 rows. Stored `(type, fidelity)`:
`(chart, ocr)×3`, `(table, ocr)×2`, `(text, verbatim)×6`. **Fidelity is stored and
queryable — SP-A's goal met.**

Findings:
- **`needs_ocr_for_text` did NOT survive to the funnel** — all text chunks got `verbatim`
  (no `text→ocr`). The verbatim/ocr split is best-effort in SP-A; threading the flag from
  the OCR stage to `row["metadata"]` is the follow-up to make it fully correct.
- The `.wav` produced no rows this run (ASR likely needs an endpoint/extra not active), so
  the `audio→transcribed` branch wasn't exercised live — it remains unit-tested only.
- `chart`/`table`→`ocr` and `text`→`verbatim` confirmed end-to-end.
