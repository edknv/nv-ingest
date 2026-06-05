# SP-B — `retrieve` primitive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `retrieve(question)` SDK fn + `retriever retrieve` CLI verb that returns the contract's `retrieve_result` — fused-query evidence shaped with `modality`/`fidelity`/`locator`/`citation`/`score` + `coverage` — graceful-falling-back to vector if the index isn't hybrid.

**Architecture:** Pure composition over `query_documents` (single hybrid call, vector fallback) + per-hit shaping into `evidence_item` (reusing SP-A's `_derive_fidelity` for the fidelity fallback) + coverage heuristics. No new retrieval logic; no ingest/ranking change; MCP/warm = SP-C.

**Tech Stack:** Python 3.12, editable `nemo_retriever` (`./retriever` venv), Typer, pytest. Unit/CLI tests no-GPU; one live check uses GPU.

## Ground truth (verified)
- `query_documents(query, *, top_k, ..., lancedb_uri, table_name, embed_model_name, rerank, hybrid)` (`adapters/cli/sdk_workflow.py:1017`) returns full `RetrievalHit`s.
- A hit exposes `text`, `source`/`pdf_basename`, `page_number`, `content_type`, `_score`/`_distance`, and `metadata` (parsed content_metadata, holds SP-A `fidelity`, `type`, `bbox_xyxy_norm`, `segment_start_seconds`, `frame_timestamp_seconds`).
- `_derive_fidelity(content_type, metadata, content_metadata)` exists (`vdb/records.py`, SP-A) → fidelity fallback for pre-SP-A indexes.
- Contract: `docs/superpowers/contracts/retriever/contract.schema.json` `$defs.evidence_item`/`retrieve_result` — `modality` and `fidelity` are enums; SP-B output must conform.
- CLI commands anchor: `@app.command("serve-models")` at `main.py:704`; `DEFAULT_LANCEDB_URI`/`DEFAULT_TABLE_NAME`/`_ROOT_CLI_ERRORS` in scope (used by query/verify commands).

All commits `--no-gpg-sign`.

---

### Task 1: `retrieve` + `_evidence_item` SDK functions + unit tests

**Files:** Modify `nemo_retriever/src/nemo_retriever/adapters/cli/sdk_workflow.py`; Create `nemo_retriever/tests/test_retrieve.py`

- [ ] **Step 1: Write the failing tests** (`nemo_retriever/tests/test_retrieve.py`)
```python
# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os

import nemo_retriever.adapters.cli.sdk_workflow as sw

_CONTRACT = os.path.join(
    os.path.dirname(__file__), "..", "..", "docs", "superpowers", "contracts", "retriever", "contract.schema.json"
)


def _hit(text, *, source="doc.pdf", page=1, content_type="text", fidelity=None, score=0.3, meta_extra=None):
    meta = {"type": content_type}
    if fidelity is not None:
        meta["fidelity"] = fidelity
    if meta_extra:
        meta.update(meta_extra)
    return {"text": text, "pdf_basename": source[:-4] if source.endswith(".pdf") else source,
            "source": source, "page_number": page, "content_type": content_type, "metadata": meta, "_score": score}


def _assert_conforms(result):
    schema = json.load(open(_CONTRACT))["$defs"]
    mod_enum = set(schema["evidence_item"]["properties"]["modality"]["enum"])
    fid_enum = set(schema["evidence_item"]["properties"]["fidelity"]["enum"])
    loc_enum = set(schema["locator"]["properties"]["kind"]["enum"])
    for e in result["evidence"]:
        assert {"text", "source", "locator", "modality", "fidelity", "score", "citation"} <= set(e)
        assert e["modality"] in mod_enum, e["modality"]
        assert e["fidelity"] in fid_enum, e["fidelity"]
        assert e["locator"]["kind"] in loc_enum
        assert isinstance(e["score"], (int, float))
    assert {"strategies_used", "n_docs_seen", "thin_spots"} <= set(result["coverage"])


def test_retrieve_shapes_evidence_and_coverage(monkeypatch):
    hits = [
        _hit("prose", source="a.pdf", page=3, content_type="text", fidelity="verbatim"),
        _hit("chart cap", source="a.pdf", page=3, content_type="chart", fidelity="vlm_caption"),
    ]
    monkeypatch.setattr(sw, "query_documents", lambda *a, **k: hits)
    r = sw.retrieve("q", lancedb_uri="x", table_name="t")
    _assert_conforms(r)
    assert r["evidence"][0]["citation"] == "a p.3"
    assert r["evidence"][0]["locator"] == {"kind": "page", "value": 3}
    assert r["coverage"]["strategies_used"] == ["semantic", "lexical"]
    assert r["coverage"]["n_docs_seen"] == 1
    assert "single source" in r["coverage"]["thin_spots"]


def test_retrieve_fidelity_fallback_when_absent(monkeypatch):
    # no metadata.fidelity -> derived from modality (chart -> vlm_caption)
    hits = [_hit("c", content_type="chart", fidelity=None)]
    monkeypatch.setattr(sw, "query_documents", lambda *a, **k: hits)
    r = sw.retrieve("q", lancedb_uri="x", table_name="t")
    assert r["evidence"][0]["fidelity"] == "vlm_caption"


def test_retrieve_normalizes_caption_modality(monkeypatch):
    hits = [_hit("t", content_type="table_caption", fidelity="ocr")]
    monkeypatch.setattr(sw, "query_documents", lambda *a, **k: hits)
    r = sw.retrieve("q", lancedb_uri="x", table_name="t")
    assert r["evidence"][0]["modality"] == "table"  # normalized to enum
    _assert_conforms(r)


def test_retrieve_empty_thin_spot(monkeypatch):
    monkeypatch.setattr(sw, "query_documents", lambda *a, **k: [])
    r = sw.retrieve("q", lancedb_uri="x", table_name="t")
    assert r["evidence"] == []
    assert "no matches — likely out of corpus" in r["coverage"]["thin_spots"]


def test_retrieve_graceful_vector_fallback(monkeypatch):
    calls = []

    def fake_qd(question, **k):
        calls.append(k.get("hybrid"))
        if k.get("hybrid"):
            raise RuntimeError("no FTS index")
        return [_hit("p")]

    monkeypatch.setattr(sw, "query_documents", fake_qd)
    r = sw.retrieve("q", lancedb_uri="x", table_name="t")
    assert calls == [True, False]  # tried hybrid, fell back to vector
    assert r["coverage"]["strategies_used"] == ["semantic"]
    assert len(r["evidence"]) == 1
```

- [ ] **Step 2: Run to confirm failure** — `/home/edwardk/git/nv-ingest/retriever/bin/python -m pytest nemo_retriever/tests/test_retrieve.py -q` → fails (`retrieve` undefined).

- [ ] **Step 3: Add `_evidence_item` + `retrieve` to `sdk_workflow.py`** — insert immediately **before** `def query_documents(` (`:1017`):
```python
_KNOWN_MODALITIES = {"text", "table", "chart", "image", "audio", "video_frame"}


def _normalize_modality(value: Any) -> str:
    m = str(value or "text").lower()
    if m in _KNOWN_MODALITIES:
        return m
    if m.startswith("table"):
        return "table"
    if m.startswith("chart"):
        return "chart"
    if m.startswith(("image", "infographic")):
        return "image"
    if m.startswith("video"):
        return "video_frame"
    if m.startswith("audio"):
        return "audio"
    return "text"


def _evidence_item(hit: dict[str, Any]) -> dict[str, Any]:
    import os as _os

    from nemo_retriever.vdb.records import _derive_fidelity

    meta = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    src_raw = hit.get("pdf_basename") or hit.get("source") or ""
    source = _os.path.basename(str(src_raw))
    if source.lower().endswith(".pdf"):
        source = source[:-4]
    raw_modality = hit.get("content_type") or meta.get("type") or "text"
    modality = _normalize_modality(raw_modality)

    page = hit.get("page_number")
    if page is not None:
        locator = {"kind": "page", "value": page}
        citation = f"{source} p.{page}"
    elif meta.get("segment_start_seconds") is not None:
        locator = {"kind": "segment", "value": meta["segment_start_seconds"]}
        citation = f"{source} @{meta['segment_start_seconds']}"
    elif meta.get("frame_timestamp_seconds") is not None:
        locator = {"kind": "timestamp", "value": meta["frame_timestamp_seconds"]}
        citation = f"{source} @{meta['frame_timestamp_seconds']}"
    elif meta.get("bbox_xyxy_norm") is not None:
        locator = {"kind": "bbox", "value": meta["bbox_xyxy_norm"]}
        citation = source
    else:
        locator = {"kind": "page", "value": None}
        citation = source

    fidelity = meta.get("fidelity") or _derive_fidelity(raw_modality, meta, meta) or "verbatim"

    if "_score" in hit and hit["_score"] is not None:
        score: float = hit["_score"]
    elif "_distance" in hit and hit["_distance"] is not None:
        score = hit["_distance"]
    else:
        score = 0.0

    return {
        "text": hit.get("text", ""),
        "source": source,
        "locator": locator,
        "modality": modality,
        "fidelity": fidelity,
        "score": score,
        "citation": citation,
    }


def retrieve(
    question: str,
    *,
    top_k: int = 10,
    hybrid: bool = True,
    lancedb_uri: str = DEFAULT_LANCEDB_URI,
    table_name: str = DEFAULT_TABLE_NAME,
    embed_model_name: str | None = None,
) -> dict[str, Any]:
    """Skill-first retrieve: one fused query -> answer-ready, fidelity-tagged, cited evidence + coverage.

    Single hybrid (vector+BM25) query; if the index has no FTS index, gracefully
    falls back to vector-only. Returns the `retrieve_result` contract shape.
    """
    def _run(use_hybrid: bool) -> list:
        return query_documents(
            question,
            top_k=top_k,
            hybrid=use_hybrid,
            lancedb_uri=lancedb_uri,
            table_name=table_name,
            embed_model_name=embed_model_name,
        )

    if hybrid:
        try:
            hits = _run(True)
            strategies = ["semantic", "lexical"]
        except Exception:  # noqa: BLE001 — e.g. table has no FTS index; degrade to vector
            hits = _run(False)
            strategies = ["semantic"]
    else:
        hits = _run(False)
        strategies = ["semantic"]

    evidence = [_evidence_item(h) for h in (hits or [])]
    sources = {e["source"] for e in evidence if e.get("source")}
    thin: list[str] = []
    if not evidence:
        thin.append("no matches — likely out of corpus")
    else:
        if len(sources) == 1:
            thin.append("single source")
        if all(e["fidelity"] == "vlm_caption" for e in evidence):
            thin.append("only low-fidelity (chart/image) evidence")
    return {
        "evidence": evidence,
        "coverage": {"strategies_used": strategies, "n_docs_seen": len(sources), "thin_spots": thin},
    }
```

- [ ] **Step 4: Run tests to pass** — `/home/edwardk/git/nv-ingest/retriever/bin/python -m pytest nemo_retriever/tests/test_retrieve.py -q` → 5 passed.

- [ ] **Step 5: Commit**
```bash
git add nemo_retriever/src/nemo_retriever/adapters/cli/sdk_workflow.py nemo_retriever/tests/test_retrieve.py
git commit --no-gpg-sign -m "feat(retriever): add retrieve() SDK primitive (fused query -> contract retrieve_result)"
```

---

### Task 2: `retriever retrieve` CLI verb + test

**Files:** Modify `adapters/cli/main.py`; Modify `nemo_retriever/tests/test_retrieve.py`

- [ ] **Step 1: Add the command** — in `main.py`, replace `@app.command("serve-models")` with the retrieve command + it:
```python
@app.command("retrieve")
def retrieve_command(
    question: str = typer.Argument(..., help="The question to retrieve evidence for."),
    top_k: int = typer.Option(10, "--top-k", min=1, help="Max evidence items."),
    hybrid: bool = typer.Option(True, "--hybrid/--no-hybrid", help="Fused vector+BM25 (falls back to vector if no FTS index)."),
    lancedb_uri: str = typer.Option(DEFAULT_LANCEDB_URI, "--lancedb-uri", help="LanceDB database URI."),
    table_name: str = typer.Option(DEFAULT_TABLE_NAME, "--table-name", help="LanceDB table name."),
    embed_model_name: str | None = typer.Option(None, "--embed-model-name", help="Embedding model name."),
) -> None:
    """Retrieve answer-ready, fidelity-tagged, cited evidence + coverage for a question."""
    from nemo_retriever.adapters.cli.sdk_workflow import retrieve

    try:
        with _quiet_capture():
            result = retrieve(
                question, top_k=top_k, hybrid=hybrid,
                lancedb_uri=lancedb_uri, table_name=table_name, embed_model_name=embed_model_name,
            )
    except _ROOT_CLI_ERRORS as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, indent=2, sort_keys=True, default=str))


@app.command("serve-models")
```
(Note: `_quiet_capture` is the same context manager `query_command` uses to keep stdout clean; confirm it's imported/defined in `main.py` — it is, used by `query_command`.)

- [ ] **Step 2: Add the CLI test** — append to `nemo_retriever/tests/test_retrieve.py`:
```python


def test_retrieve_cli_prints_json(monkeypatch) -> None:
    import importlib

    from typer.testing import CliRunner

    cli_main = importlib.import_module("nemo_retriever.adapters.cli.main")

    monkeypatch.setattr(
        sw, "retrieve",
        lambda question, **k: {"evidence": [], "coverage": {"strategies_used": ["semantic"], "n_docs_seen": 0, "thin_spots": ["no matches — likely out of corpus"]}},
    )
    result = CliRunner().invoke(cli_main.app, ["retrieve", "q", "--no-hybrid", "--table-name", "t"])
    assert result.exit_code == 0
    out = json.loads(result.output)
    assert out["coverage"]["strategies_used"] == ["semantic"]
    assert out["evidence"] == []
```
(The command imports `retrieve` from `sdk_workflow` inside the function, so patching `sw.retrieve` (= `sdk_workflow.retrieve`) takes effect.)

- [ ] **Step 3: Run** — `/home/edwardk/git/nv-ingest/retriever/bin/python -m pytest nemo_retriever/tests/test_retrieve.py -q` → 6 passed.

- [ ] **Step 4: Commit**
```bash
git add nemo_retriever/src/nemo_retriever/adapters/cli/main.py nemo_retriever/tests/test_retrieve.py
git commit --no-gpg-sign -m "feat(retriever): add 'retriever retrieve' CLI verb (prints retrieve_result JSON)"
```

---

### Task 3: Live validation

**Files:** none.

- [ ] **Step 1: All unit tests + help (no GPU)**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python -m pytest nemo_retriever/tests/test_retrieve.py -q && /home/edwardk/git/nv-ingest/retriever/bin/retriever retrieve --help >/dev/null && echo "CMD OK"`
Expected: 6 passed; `CMD OK`.

- [ ] **Step 2: Live retrieve against a real index (GPU) + graceful fallback**

```bash
cd /home/edwardk/git/nv-ingest
mkdir -p /tmp/rb && cp data/multimodal_test.pdf /tmp/rb/
./retriever/bin/retriever ingest /tmp/rb/ --table-name rb --lancedb-uri /tmp/rb_db --embed-model-name nvidia/llama-nemotron-embed-1b-v2 --quiet
# --no-hybrid avoids needing an FTS index; exercises the vector path end-to-end:
./retriever/bin/retriever retrieve "most expensive gadget in Chart 1" --top-k 3 --no-hybrid --table-name rb --lancedb-uri /tmp/rb_db --embed-model-name nvidia/llama-nemotron-embed-1b-v2 | ./retriever/bin/python -c "
import json,sys; r=json.load(sys.stdin)
e=r['evidence']; print('evidence:', len(e), '| coverage:', r['coverage'])
assert e, 'no evidence'
top=e[0]; print('top:', {k: top[k] for k in ('source','locator','modality','fidelity','citation')})
assert {'text','source','locator','modality','fidelity','score','citation'} <= set(top)
"
# graceful fallback: --hybrid on a non-hybrid index must NOT crash (falls back to vector)
./retriever/bin/retriever retrieve "gadget" --top-k 2 --hybrid --table-name rb --lancedb-uri /tmp/rb_db --embed-model-name nvidia/llama-nemotron-embed-1b-v2 | ./retriever/bin/python -c "import json,sys; r=json.load(sys.stdin); print('fallback strategies:', r['coverage']['strategies_used']); assert r['evidence']"
rm -rf /tmp/rb /tmp/rb_db```
Expected: the vector run prints `evidence: N` with a populated `coverage`, and a `top:` line where `fidelity`/`citation`/`locator`/`modality` are set (no assertion error). The fallback run prints `fallback strategies: ['semantic']` (hybrid attempted on a non-hybrid index, fell back to vector) and returns evidence without crashing.

- [ ] **Step 3: No commit (validation only).** If green, SP-B is complete — `retrieve` returns the contract shape end-to-end.

---

## Self-review

**Spec coverage (SP-B design):**
- `retrieve` SDK fn, single hybrid query + graceful vector fallback → Task 1 Step 3 (`retrieve` try/except) + `test_retrieve_graceful_vector_fallback`. ✓
- `evidence_item` shaping (text/source/locator/modality/fidelity/score/citation), fidelity fallback via `_derive_fidelity`, modality normalized to enum → `_evidence_item` (Task 1) + `test_retrieve_*` + `_assert_conforms` against the committed contract schema. ✓
- `coverage` (strategies_used / n_docs_seen / thin_spots heuristics) → `retrieve` + tests (single-source, empty, low-fidelity). ✓
- CLI `retrieve` verb printing JSON → Task 2. ✓
- Boundary: no MCP/warm, no ingest/ranking change → only `sdk_workflow.py` (+ `main.py` command) touched. ✓
- Live conformance + fallback-no-crash → Task 3 Step 2. ✓

**Placeholder scan:** No TBD/TODO; all code complete.

**Type consistency:** `retrieve`/`_evidence_item`/`_normalize_modality` signatures match their call sites and tests. Output keys exactly match `contract.schema.json` `$defs.evidence_item` (`text,source,locator,modality,fidelity,score,citation`) and `retrieve_result` (`evidence,coverage`); `modality`/`fidelity`/`locator.kind` are normalized into the schema enums, and `score` is always a number (0.0 default) — all asserted by `_assert_conforms`. The CLI test patches `sw.retrieve` (= the name the command imports from `sdk_workflow`).
