# Objective 2b — opt-in `--hybrid` (ingest + query) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `--hybrid` flag to `retriever ingest` and `retriever query` that threads into the existing `LanceDB` hybrid (vector + BM25) support, so the skill's single query can combine semantic + exact-term recall.

**Architecture:** Pure CLI/SDK plumbing — the engine already supports hybrid (FTS index at ingest, `query_texts` threading at query). Add the flag to both Typer commands and inject `"hybrid": True` into the `vdb_kwargs` dict **only when set** (so default-off behavior and existing exact-match tests are unchanged). Bump the skill contract to 1.2.0. The on-PATH `retriever` is an editable install of `nemo_retriever/src`, so edits take effect immediately.

**Tech Stack:** Python 3.12, Typer, pytest (`CliRunner`), the `retriever` venv at `/home/edwardk/git/nv-ingest/retriever`, LanceDB hybrid (`create_fts_index`).

---

## Ground truth (verified)

- Query option `--rerank` ends at `adapters/cli/main.py:637` (`) -> None:` at 638); `query_documents(...)` call is `main.py:647-660` (last arg `rerank=rerank,` at 660).
- Ingest option `--overwrite/--append` is `main.py:310-316`; `ingest_documents(...)` call is `main.py:497-...` with `overwrite=overwrite,` at `main.py:530`.
- `query_documents` signature `adapters/cli/sdk_workflow.py:926-940` (last param `rerank: bool = False,` at 939); builds `vdb_kwargs` at `sdk_workflow.py:948-951` (`"vdb_kwargs": {"uri": lancedb_uri, "table_name": table_name},` at 950).
- `ingest_documents` signature has `overwrite: bool = True,` at `sdk_workflow.py:738`; `resolve_ingest_plan` signature has `overwrite: bool = True,` at `sdk_workflow.py:537` (identical lines — both need the new param). `resolve_ingest_plan(...)` is called from `ingest_documents` at `sdk_workflow.py:785+`; `VdbUploadParams` built at `sdk_workflow.py:642-644`.
- Engine internals need NO change: `RetrieveVdbOperator.process` already threads `query_texts` when hybrid (`vdb/operators.py:224-225`); `_coerce_vdb_init` forwards `vdb_kwargs` to `LanceDB` (`retriever.py:~242`); `create_index` builds the FTS index `if hybrid` (`vdb/lancedb.py:~515`).
- Tests: query plumbing `tests/test_root_query_cli.py` (`test_root_query_passes_query_options_and_prints_json` asserts `retriever_calls == [{"top_k":3,"vdb_kwargs":{"uri":"/tmp/lancedb","table_name":"docs"}}]` — this no-flag assertion guards the conditional). Ingest plumbing `tests/test_root_cli_workflow.py` (`test_root_ingest_passes_vdb_options_and_run_mode` asserts `vdb_upload.call_args.args[0].vdb_kwargs == {"uri","table_name","overwrite":True}`).

## Validation note
All commits `--no-gpg-sign`. CLI tests mock the SDK (no GPU); the live hybrid round-trip and `doctor` need GPU.

---

### Task 1: Query `--hybrid`

**Files:** `nemo_retriever/src/nemo_retriever/adapters/cli/main.py`, `.../adapters/cli/sdk_workflow.py`, `nemo_retriever/tests/test_root_query_cli.py`

- [ ] **Step 1: Add the `--hybrid` Typer option to `query_command`**

In `adapters/cli/main.py`, replace the rerank option's closing + signature end:
```python
    rerank: bool = typer.Option(
        False,
        "--rerank/--no-rerank",
        help=(
            "Enable reranking after vector retrieval. Default off. Implicitly enabled when "
            "any of --reranker-invoke-url / --reranker-model-name / --reranker-backend is set."
        ),
    ),
) -> None:
```
with:
```python
    rerank: bool = typer.Option(
        False,
        "--rerank/--no-rerank",
        help=(
            "Enable reranking after vector retrieval. Default off. Implicitly enabled when "
            "any of --reranker-invoke-url / --reranker-model-name / --reranker-backend is set."
        ),
    ),
    hybrid: bool = typer.Option(
        False,
        "--hybrid",
        help="Combine vector + full-text (BM25) retrieval. Requires an index built with `ingest --hybrid`.",
    ),
) -> None:
```

- [ ] **Step 2: Pass `hybrid` into the `query_documents(...)` call**

In `adapters/cli/main.py`, replace:
```python
                reranker_backend=reranker_backend,
                rerank=rerank,
            )
```
with:
```python
                reranker_backend=reranker_backend,
                rerank=rerank,
                hybrid=hybrid,
            )
```

- [ ] **Step 3: Add `hybrid` param + conditional `vdb_kwargs` in `query_documents`**

In `adapters/cli/sdk_workflow.py`, replace:
```python
    reranker_backend: str | None = None,
    rerank: bool = False,
) -> list[RetrievalHit]:
```
with:
```python
    reranker_backend: str | None = None,
    rerank: bool = False,
    hybrid: bool = False,
) -> list[RetrievalHit]:
```
Then replace:
```python
    retriever_kwargs: dict[str, Any] = {
        "top_k": top_k,
        "vdb_kwargs": {"uri": lancedb_uri, "table_name": table_name},
    }
```
with:
```python
    vdb_kwargs: dict[str, Any] = {"uri": lancedb_uri, "table_name": table_name}
    if hybrid:
        vdb_kwargs["hybrid"] = True
    retriever_kwargs: dict[str, Any] = {
        "top_k": top_k,
        "vdb_kwargs": vdb_kwargs,
    }
```

- [ ] **Step 4: Add a query `--hybrid` test**

In `nemo_retriever/tests/test_root_query_cli.py`, append:
```python


def test_root_query_passes_hybrid_into_vdb_kwargs(monkeypatch) -> None:
    retriever_calls: list[dict[str, Any]] = []

    class FakeRetriever:
        def __init__(self, **kwargs: Any) -> None:
            retriever_calls.append(kwargs)

        def query(self, query: str, **_kwargs: Any) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(sdk_workflow, "Retriever", FakeRetriever)

    result = RUNNER.invoke(
        cli_main.app,
        ["query", "q", "--top-k", "5", "--lancedb-uri", "/tmp/lancedb", "--table-name", "docs", "--hybrid"],
    )

    assert result.exit_code == 0
    assert retriever_calls == [
        {"top_k": 5, "vdb_kwargs": {"uri": "/tmp/lancedb", "table_name": "docs", "hybrid": True}}
    ]
```

- [ ] **Step 5: Run query tests**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python -m pytest nemo_retriever/tests/test_root_query_cli.py -q`
Expected: all pass — the new hybrid test AND the existing no-flag test (whose assertion has `vdb_kwargs` *without* a `hybrid` key, proving the conditional leaves default behavior unchanged).

- [ ] **Step 6: Commit**
```bash
git add nemo_retriever/src/nemo_retriever/adapters/cli/main.py nemo_retriever/src/nemo_retriever/adapters/cli/sdk_workflow.py nemo_retriever/tests/test_root_query_cli.py
git commit --no-gpg-sign -m "feat(retriever): add opt-in --hybrid to query (vector + BM25)"
```

---

### Task 2: Ingest `--hybrid`

**Files:** `adapters/cli/main.py`, `adapters/cli/sdk_workflow.py`, `nemo_retriever/tests/test_root_cli_workflow.py`

- [ ] **Step 1: Add the `--hybrid` Typer option to `ingest_command`**

In `adapters/cli/main.py`, replace the overwrite option:
```python
    overwrite: bool = typer.Option(
        True,
        "--overwrite/--append",
        help=(
            "Overwrite the target LanceDB table by default. Use --append to add rows to an existing "
            "table without duplicate checks; rerunning the same inputs in append mode creates duplicates."
        ),
```
with:
```python
    hybrid: bool = typer.Option(
        False,
        "--hybrid",
        help="Build a full-text (BM25) index alongside vectors so `query --hybrid` can run hybrid search.",
    ),
    overwrite: bool = typer.Option(
        True,
        "--overwrite/--append",
        help=(
            "Overwrite the target LanceDB table by default. Use --append to add rows to an existing "
            "table without duplicate checks; rerunning the same inputs in append mode creates duplicates."
        ),
```
(Inserting `--hybrid` *before* the `overwrite` option keeps the edit anchored to a unique block.)

- [ ] **Step 2: Pass `hybrid` into the `ingest_documents(...)` call**

In `adapters/cli/main.py`, replace (this exact 16-space-indented line in the ingest call):
```python
                table_name=table_name,
                overwrite=overwrite,
```
with:
```python
                table_name=table_name,
                overwrite=overwrite,
                hybrid=hybrid,
```

- [ ] **Step 3: Add `hybrid` param to BOTH SDK signatures (`ingest_documents` and `resolve_ingest_plan`)**

In `adapters/cli/sdk_workflow.py`, first confirm the line appears exactly twice (the two signatures):
```bash
grep -c "^    overwrite: bool = True,$" nemo_retriever/src/nemo_retriever/adapters/cli/sdk_workflow.py
```
Expected: `2`. Then replace **all** occurrences of:
```python
    overwrite: bool = True,
```
with:
```python
    overwrite: bool = True,
    hybrid: bool = False,
```
(Use a replace-all; both signatures need the parameter. If the count is not 2, stop and disambiguate.)

- [ ] **Step 4: Pass `hybrid` into the `resolve_ingest_plan(...)` call inside `ingest_documents`**

In `adapters/cli/sdk_workflow.py`, the `resolve_ingest_plan(` call passes `overwrite=overwrite,`. Replace (the 8-space-indented call-site occurrence):
```python
        table_name=table_name,
        overwrite=overwrite,
```
with:
```python
        table_name=table_name,
        overwrite=overwrite,
        hybrid=hybrid,
```

- [ ] **Step 5: Conditionally inject `hybrid` into `VdbUploadParams`**

In `adapters/cli/sdk_workflow.py`, replace:
```python
    vdb_params = VdbUploadParams(
        vdb_kwargs={"uri": lancedb_uri, "table_name": table_name, "overwrite": bool(overwrite)}
    )
```
with:
```python
    vdb_upload_kwargs: dict[str, Any] = {"uri": lancedb_uri, "table_name": table_name, "overwrite": bool(overwrite)}
    if hybrid:
        vdb_upload_kwargs["hybrid"] = True
    vdb_params = VdbUploadParams(vdb_kwargs=vdb_upload_kwargs)
```

- [ ] **Step 6: Add an ingest `--hybrid` test**

In `nemo_retriever/tests/test_root_cli_workflow.py`, append a test modeled on `test_root_ingest_passes_vdb_options_and_run_mode`:
```python


def test_root_ingest_passes_hybrid_into_vdb_kwargs(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    doc = tmp_path / "a.pdf"
    doc.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(sdk_workflow, "create_ingestor", lambda **_: fake_ingestor)
    monkeypatch.setattr(sdk_workflow, "_count_lancedb_rows", lambda *_, **__: 1)

    result = RUNNER.invoke(
        cli_main.app,
        ["ingest", str(doc), "--lancedb-uri", "/tmp/lancedb", "--table-name", "docs", "--hybrid"],
    )

    assert result.exit_code == 0
    assert fake_ingestor.vdb_upload.call_args.args[0].vdb_kwargs == {
        "uri": "/tmp/lancedb",
        "table_name": "docs",
        "overwrite": True,
        "hybrid": True,
    }
```

- [ ] **Step 7: Run ingest tests**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python -m pytest nemo_retriever/tests/test_root_cli_workflow.py -q`
Expected: all pass — the new hybrid test AND the existing `test_root_ingest_passes_vdb_options_and_run_mode` (whose `vdb_kwargs` assertion has no `hybrid` key).

- [ ] **Step 8: Commit**
```bash
git add nemo_retriever/src/nemo_retriever/adapters/cli/main.py nemo_retriever/src/nemo_retriever/adapters/cli/sdk_workflow.py nemo_retriever/tests/test_root_cli_workflow.py
git commit --no-gpg-sign -m "feat(retriever): add opt-in --hybrid to ingest (builds FTS index)"
```

---

### Task 3: Contract bump + skill docs

**Files:** `skills/nemo-retriever/contract/cli-contract.json`, `.../CONTRACT.md`, `skills/nemo-retriever/SKILL.md`, `.../references/setup.md`, `.../references/query.md`

- [ ] **Step 1: Add `--hybrid` to the flag surface + bump version in `cli-contract.json`**

Replace:
```json
  "contract_version": "1.1.0",
  "query": {
    "required_flags": ["--top-k", "--content-types", "--rerank", "--embed-model-name", "--lancedb-uri", "--table-name"],
```
with:
```json
  "contract_version": "1.2.0",
  "query": {
    "required_flags": ["--top-k", "--content-types", "--rerank", "--hybrid", "--embed-model-name", "--lancedb-uri", "--table-name"],
```
and replace:
```json
    "required_flags": ["--append", "--overwrite", "--ocr-version", "--ocr-lang", "--table-name", "--lancedb-uri", "--embed-model-name"],
```
with:
```json
    "required_flags": ["--append", "--overwrite", "--hybrid", "--ocr-version", "--ocr-lang", "--table-name", "--lancedb-uri", "--embed-model-name"],
```

- [ ] **Step 2: Changelog in `CONTRACT.md`**

In `skills/nemo-retriever/contract/CONTRACT.md`, replace:
```markdown
## Changelog
- **1.1.0** — query hits now carry `modality` (required) and `score` (optional); see `actual-hit.schema.json`. First step from `actual-hit` toward `target-hit`.
```
with:
```markdown
## Changelog
- **1.2.0** — `--hybrid` flag added to `ingest` (builds the BM25/FTS index) and `query` (vector + full-text retrieval). Opt-in; a `--hybrid` query needs a `--hybrid`-built index.
- **1.1.0** — query hits now carry `modality` (required) and `score` (optional); see `actual-hit.schema.json`. First step from `actual-hit` toward `target-hit`.
```

- [ ] **Step 3: Bump declared version in `SKILL.md`**

In `skills/nemo-retriever/SKILL.md`, replace:
```
This skill targets engine **contract_version 1.1.0** (`contract/cli-contract.json`).
```
with:
```
This skill targets engine **contract_version 1.2.0** (`contract/cli-contract.json`).
```

- [ ] **Step 4: Document `--hybrid` in `setup.md`**

In `skills/nemo-retriever/references/setup.md`, immediately after the first `retriever ingest ./pdfs/ --embed-model-name ...` recipe's surrounding code fence (the `if [ "$TOTAL_PAGES" -le 800 ]` block), add a paragraph:
```markdown
**Hybrid recall (optional):** add `--hybrid` to the `ingest` command to also build a full-text (BM25) index. Then `retriever query --hybrid` combines semantic + exact-term matching in one query — useful when questions hinge on identifiers/codes that pure vector search misses. The FTS index is only built when you ingest with `--hybrid`.
```

- [ ] **Step 5: Document `--hybrid` in `query.md`**

In `skills/nemo-retriever/references/query.md`, after the main `retriever query` pipeline code block, add:
```markdown
**Hybrid query (optional):** if the index was built with `ingest --hybrid`, add `--hybrid` to the query to combine vector + full-text (BM25) retrieval — recovers exact-term/identifier matches semantic search alone misses. No effect (and no FTS index) on a non-hybrid index.
```

- [ ] **Step 6: Validate JSON + commit**
```bash
/home/edwardk/git/nv-ingest/retriever/bin/python -c "import json; json.load(open('skills/nemo-retriever/contract/cli-contract.json')); print('JSON OK')"
git add skills/nemo-retriever/contract/ skills/nemo-retriever/SKILL.md skills/nemo-retriever/references/setup.md skills/nemo-retriever/references/query.md
git commit --no-gpg-sign -m "docs(skill): contract 1.2.0 + document opt-in --hybrid (ingest + query)"
```
Expected: `JSON OK`.

---

### Task 4: Live validation

**Files:** none.

- [ ] **Step 1: Full CLI unit tests (no GPU)**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python -m pytest nemo_retriever/tests/test_root_query_cli.py nemo_retriever/tests/test_root_cli_workflow.py -q`
Expected: all pass.

- [ ] **Step 2: `doctor` (now asserts `--hybrid` on both subcommands; GPU)**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python skills/nemo-retriever/scripts/doctor.py; echo "exit=$?"`
Expected: all `[PASS]` including `query has --hybrid` and `ingest has --hybrid`; `exit=0`.

- [ ] **Step 3: Live hybrid round-trip (the real gate; GPU)**

```bash
mkdir -p /tmp/hy_probe && printf 'The widget model number is XJ-4417 and it ships in March.\nUnrelated filler sentence about logistics.\n' > /tmp/hy_probe/doc.txt && \
/home/edwardk/git/nv-ingest/retriever/bin/retriever ingest /tmp/hy_probe/ --table-name hy --lancedb-uri /tmp/hy_lancedb --embed-model-name nvidia/llama-nemotron-embed-1b-v2 --hybrid --quiet && \
/home/edwardk/git/nv-ingest/retriever/bin/retriever query "XJ-4417" --top-k 3 --table-name hy --lancedb-uri /tmp/hy_lancedb --embed-model-name nvidia/llama-nemotron-embed-1b-v2 --hybrid | /home/edwardk/git/nv-ingest/retriever/bin/python -c "import json,sys; h=json.load(sys.stdin); assert h, 'no hits'; print('hits=',len(h),'top modality=',h[0].get('modality'),'score=',h[0].get('score'))"; rm -rf /tmp/hy_probe /tmp/hy_lancedb
```
Expected: `ingest --hybrid` completes without error (FTS index built), `query --hybrid` returns ≥1 hit, prints `hits= N top modality= text score= <number>`, no assertion/traceback. (If ingest or query raises specifically about hybrid/FTS, capture the error — that is a real engine-integration finding.)

- [ ] **Step 4: No commit (validation only).** If all green, slice 2b is complete.

---

## Self-review

**Spec coverage (2b):**
- `--hybrid` on query (option + call + conditional vdb_kwargs) → Task 1. ✓
- `--hybrid` on ingest (option + call + both SDK signatures + resolve_ingest_plan call + conditional VdbUploadParams) → Task 2. ✓
- conditional injection preserves default behavior + existing exact-match tests → Tasks 1/2 keep the no-flag dicts unchanged; Steps 5/7 assert both new and existing tests pass. ✓
- contract 1.2.0 + `--hybrid` in flag surface + doctor asserts it → Task 3 Step 1 + Task 4 Step 2. ✓
- skill docs (setup.md ingest, query.md query) → Task 3 Steps 4-5. ✓
- live hybrid round-trip gate → Task 4 Step 3. ✓
- no engine-internal changes (query_texts already threaded) → confirmed in Ground truth; no task touches operators/retriever/lancedb. ✓

**Placeholder scan:** No TBD/TODO; all code/edits are concrete.

**Type consistency:** the `hybrid` key is `True` (bool) everywhere it's injected; the param is `hybrid: bool = False` in both `query_documents` and the two ingest SDK signatures and both Typer options. `contract_version` `1.2.0` matches across `cli-contract.json`, `CONTRACT.md`, `SKILL.md`. New tests assert `"hybrid": True` consistent with the conditional-injection code. The Task 2 Step 3 replace-all is guarded by a `grep -c == 2` check to avoid editing an unintended occurrence.
