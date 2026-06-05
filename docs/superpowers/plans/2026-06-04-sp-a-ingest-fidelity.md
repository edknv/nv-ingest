# SP-A — Ingest Provenance → `fidelity` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store a true per-chunk `fidelity` (`verbatim|ocr|transcribed|vlm_caption`) in the index, derived at the single store funnel from real provenance signals.

**Architecture:** Add a pure `_derive_fidelity(...)` helper in `vdb/records.py` and call it inside `_client_record_from_graph_row` right after the existing `type` stamp, putting `content_metadata["fidelity"]` into the record (→ persisted in the stored `metadata` JSON, queryable). One file changes; SP-A only *stores* fidelity (surfacing in `retrieve` is SP-B).

**Tech Stack:** Python 3.12, the editable `nemo_retriever` (`./retriever` venv), pytest. Unit tests are no-GPU; one live ingest check uses GPU.

## Ground truth (verified)
- Funnel: `_client_record_from_graph_row(row)` (`nemo_retriever/src/nemo_retriever/vdb/records.py:69`), called by public `to_client_vdb_records(rows)` (`:121`). It builds `content_metadata`, stamps `content_metadata.setdefault("type", content_type)` (`:85-86`), and returns `{"document_type": ..., "metadata": record_metadata}` where `record_metadata["content_metadata"] = content_metadata` (`:115`).
- Signals reachable there: `content_type = row.get("_content_type") or row.get("content_type")`; the row's `metadata` (`row["metadata"]`, holds `needs_ocr_for_text` when present); `content_metadata` (holds `subtype`).
- Extra `content_metadata` keys persist to the stored LanceDB `metadata` JSON and are queryable (existing `tests/test_lancedb_row_metadata.py` confirms the pattern for `_content_type`).
- Content types seen in the pipeline include `_caption` variants (`table_caption`, `chart_caption`, `infographic_caption`).

All commits `--no-gpg-sign`.

---

### Task 1: `_derive_fidelity` + funnel stamp + unit tests

**Files:**
- Modify: `nemo_retriever/src/nemo_retriever/vdb/records.py`
- Create: `nemo_retriever/tests/test_fidelity.py`

- [ ] **Step 1: Write the failing tests**

Create `nemo_retriever/tests/test_fidelity.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.vdb.records import _client_record_from_graph_row, _derive_fidelity


def _fidelity_of(row: dict) -> object:
    rec = _client_record_from_graph_row(row)
    assert rec is not None
    return rec["metadata"]["content_metadata"].get("fidelity")


def _row(content_type, *, needs_ocr=None, subtype=None) -> dict:
    meta: dict = {"embedding": [0.1, 0.2]}
    if needs_ocr is not None:
        meta["needs_ocr_for_text"] = needs_ocr
    cm: dict = {"page_number": 1}
    if subtype is not None:
        cm["subtype"] = subtype
    meta["content_metadata"] = cm
    return {"text": "x", "metadata": meta, "_content_type": content_type}


def test_derive_fidelity_pure_mapping() -> None:
    assert _derive_fidelity("text", {}, {}) == "verbatim"
    assert _derive_fidelity("text", {"needs_ocr_for_text": True}, {}) == "ocr"
    assert _derive_fidelity("image", {}, {}) == "vlm_caption"
    assert _derive_fidelity("image", {}, {"subtype": "page_image"}) == "ocr"
    assert _derive_fidelity("table", {}, {}) == "ocr"
    assert _derive_fidelity("chart_caption", {}, {}) == "ocr"
    assert _derive_fidelity("audio", {}, {}) == "transcribed"
    assert _derive_fidelity("video", {}, {}) == "transcribed"
    assert _derive_fidelity("", {}, {}) is None
    assert _derive_fidelity("mystery", {}, {}) is None


def test_fidelity_stamped_into_stored_record() -> None:
    assert _fidelity_of(_row("text")) == "verbatim"
    assert _fidelity_of(_row("text", needs_ocr=True)) == "ocr"
    assert _fidelity_of(_row("image")) == "vlm_caption"
    assert _fidelity_of(_row("image", subtype="page_image")) == "ocr"
    assert _fidelity_of(_row("table")) == "ocr"
    assert _fidelity_of(_row("audio")) == "transcribed"


def test_fidelity_absent_for_unknown_type() -> None:
    # Unknown/empty content_type -> no fidelity key (don't guess)
    rec = _client_record_from_graph_row(_row("mystery"))
    assert "fidelity" not in rec["metadata"]["content_metadata"]
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python -m pytest nemo_retriever/tests/test_fidelity.py -q`
Expected: ImportError / fail — `_derive_fidelity` doesn't exist yet.

- [ ] **Step 3: Add `_derive_fidelity` to `records.py`**

Insert this function immediately **before** `def _client_record_from_graph_row(` (i.e., after `_dict_or_empty`, around `records.py:67`):
```python
def _derive_fidelity(content_type: Any, metadata: dict[str, Any], content_metadata: dict[str, Any]) -> str | None:
    """Map a chunk's modality + real provenance signals to a trust tier.

    verbatim (PDF text layer) > ocr (scanned/region OCR) > transcribed (ASR) >
    vlm_caption (chart/image model caption). Returns None for unknown types so
    the field is omitted rather than guessed.
    """
    t = str(content_type or "").lower()
    if t in ("audio", "video", "video_frame"):
        return "transcribed"
    if t == "image":
        return "ocr" if content_metadata.get("subtype") == "page_image" else "vlm_caption"
    if t.startswith(("table", "chart", "infographic")):
        return "ocr"
    if t == "text":
        return "ocr" if metadata.get("needs_ocr_for_text") is True else "verbatim"
    return None
```

- [ ] **Step 4: Call it in the funnel** — in `_client_record_from_graph_row`, replace:
```python
    content_type = row.get("_content_type") or row.get("content_type")
    if content_type:
        content_metadata.setdefault("type", content_type)
```
with:
```python
    content_type = row.get("_content_type") or row.get("content_type")
    if content_type:
        content_metadata.setdefault("type", content_type)
    fidelity = _derive_fidelity(content_type, metadata, content_metadata)
    if fidelity:
        content_metadata.setdefault("fidelity", fidelity)
```

- [ ] **Step 5: Run tests to confirm pass**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python -m pytest nemo_retriever/tests/test_fidelity.py -q`
Expected: 3 passed.

- [ ] **Step 6: Regression — existing records/metadata tests still pass**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python -m pytest nemo_retriever/tests/test_lancedb_row_metadata.py nemo_retriever/tests/ -q -k "record or metadata or vdb" 2>&1 | tail -5`
Expected: all pass (fidelity is additive; `type` and other fields unchanged).

- [ ] **Step 7: Commit**
```bash
git add nemo_retriever/src/nemo_retriever/vdb/records.py nemo_retriever/tests/test_fidelity.py
git commit --no-gpg-sign -m "feat(ingest): derive + store per-chunk fidelity (verbatim/ocr/transcribed/vlm_caption)"
```

---

### Task 2: Live ingest validation (does fidelity land, and does needs_ocr survive?)

**Files:** none (validation only)

- [ ] **Step 1: Ingest a mixed corpus and inspect stored fidelity**

```bash
cd /home/edwardk/git/nv-ingest
mkdir -p /tmp/fid && cp data/multimodal_test.pdf data/multimodal_test.wav /tmp/fid/ 2>/dev/null; cp data/multimodal_test.pdf /tmp/fid/ 2>/dev/null
./retriever/bin/retriever ingest /tmp/fid/ --table-name fid --lancedb-uri /tmp/fid_db --embed-model-name nvidia/llama-nemotron-embed-1b-v2 --quiet
./retriever/bin/python -c "
import lancedb, json, collections
df = lancedb.connect('/tmp/fid_db').open_table('fid').to_pandas()
def m(x):
    try: return json.loads(x) if isinstance(x,str) else (x or {})
    except Exception: return {}
pairs = collections.Counter((m(r).get('type'), m(r).get('fidelity')) for r in df['metadata'])
print('rows:', len(df))
print('(type, fidelity) counts:')
for k,v in sorted(pairs.items(), key=lambda kv: str(kv[0])): print('  ', k, '->', v)
assert any(f for (_,f) in pairs), 'no fidelity stored!'
"
rm -rf /tmp/fid /tmp/fid_db```
Expected: `rows: N`; a `(type, fidelity)` table where text→`verbatim` (or `ocr`), table/chart→`ocr`, image→`vlm_caption`/`ocr`, audio→`transcribed`; assertion passes (fidelity stored). **Record whether any text chunk got `ocr`** — if all text is `verbatim`, `needs_ocr_for_text` did not survive to the funnel (expected per the spec's caveat), and the verbatim/ocr split is a documented follow-up. That's a finding, not a failure.

- [ ] **Step 2: Record the survival finding in the spec**

Append a one-line "## SP-A live result" note to `docs/superpowers/specs/2026-06-04-sp-a-ingest-fidelity-design.md` stating the observed `(type, fidelity)` distribution and whether `needs_ocr_for_text` survived (i.e., whether any text→`ocr`). Commit:
```bash
git add docs/superpowers/specs/2026-06-04-sp-a-ingest-fidelity-design.md
git commit --no-gpg-sign -m "docs(SP-A): record live fidelity distribution + needs_ocr survival finding"
```

- [ ] **Step 3: No further commit (validation only).** If fidelity is stored, SP-A is complete.

---

## Self-review

**Spec coverage (SP-A design):**
- Single-point derivation at `_client_record_from_graph_row` funnel → Task 1 Steps 3-4. ✓
- Mapping (text→verbatim/ocr via needs_ocr; image→ocr/vlm_caption via subtype; table/chart/infographic→ocr; audio/video→transcribed; unknown→None) → `_derive_fidelity` (Step 3) + asserted in `test_derive_fidelity_pure_mapping` (Step 1). ✓
- Persists to LanceDB metadata, queryable → Task 2 reads it back from the stored `metadata` JSON. ✓
- `needs_ocr_for_text` survival caveat / verbatim fallback → Task 2 Step 1 records it; conservative `verbatim` default is in `_derive_fidelity`. ✓
- SP-A stores only (no query surfacing) → no query-path files touched. ✓
- No contract-version bump (additive metadata; CLI surface unchanged) → no contract file touched. ✓

**Placeholder scan:** No TBD/TODO. Task 2 Step 2's recorded text is genuinely run-dependent (the observed distribution), which is the point of a validation-finding step.

**Type consistency:** `_derive_fidelity(content_type, metadata, content_metadata)` signature matches its call in the funnel (Step 4) and the unit-test calls (Step 1). Returned values are exactly the contract enum (`verbatim|ocr|transcribed|vlm_caption`) or `None`. The funnel stamps `content_metadata["fidelity"]`, which the test reads via `rec["metadata"]["content_metadata"]["fidelity"]` (matching the funnel's `record_metadata["content_metadata"] = content_metadata`).
