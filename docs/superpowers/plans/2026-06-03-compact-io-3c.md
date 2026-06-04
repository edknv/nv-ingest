# Objective 3c — compact query output (`--max-text-chars`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--max-text-chars N` to `retriever query` (unset=full text, `N>0`=snippet+`…`, `0`=metadata-only) so the engine returns compact output, cutting the per-turn token/cache cost the skill works around. Bump contract to 1.4.0.

**Architecture:** One change at the serialization boundary — `_query_cli_hit` gains an optional `max_text_chars`, and `query_command` adds the Typer option + passes it. Backward-compatible (default `None`=full). Editable install ⇒ edits take effect immediately.

**Tech Stack:** Python 3.12, Typer, pytest (`CliRunner`), `retriever` venv at `/home/edwardk/git/nv-ingest/retriever`.

## Ground truth (verified)
- `_query_cli_hit` (`adapters/cli/main.py:74-`) currently returns `{source, page_number, text, modality, score}` (post-2a).
- Query prints: `typer.echo(json.dumps([_query_cli_hit(hit) for hit in hits], indent=2, sort_keys=True, default=str))` (`main.py:678`).
- `query_command`'s last option is the `--hybrid` block (added in 2b), immediately before `) -> None:`.
- Contract is `cli-contract.json` 1.3.0; `doctor` statically asserts each `query.required_flags` entry (with `COLUMNS=200` so full flag names show).

All commits `--no-gpg-sign`. Query CLI tests are no-GPU; `doctor` needs GPU for its live probe.

---

### Task 1: Engine change + tests

**Files:** `nemo_retriever/src/nemo_retriever/adapters/cli/main.py`, `nemo_retriever/tests/test_root_query_cli.py`

- [ ] **Step 1: Add `max_text_chars` to `_query_cli_hit`**

Replace:
```python
def _query_cli_hit(hit: RetrievalHit) -> dict[str, object]:
    metadata = hit.get("metadata") or {}
    modality = hit.get("content_type") or metadata.get("type") or "text"
    # Relevance the engine ranked by: rerank/hybrid score if present, else the
    # vector distance, else null. Hit ORDER is authoritative; score is informational.
    if "_score" in hit:
        score: object = hit["_score"]
    elif "_distance" in hit:
        score = hit["_distance"]
    else:
        score = None
    return {
        "source": hit.get("source", ""),
        "page_number": hit.get("page_number"),
        "text": hit.get("text", ""),
        "modality": modality,
        "score": score,
    }
```
with:
```python
def _query_cli_hit(hit: RetrievalHit, max_text_chars: int | None = None) -> dict[str, object]:
    metadata = hit.get("metadata") or {}
    modality = hit.get("content_type") or metadata.get("type") or "text"
    # Relevance the engine ranked by: rerank/hybrid score if present, else the
    # vector distance, else null. Hit ORDER is authoritative; score is informational.
    if "_score" in hit:
        score: object = hit["_score"]
    elif "_distance" in hit:
        score = hit["_distance"]
    else:
        score = None
    text = hit.get("text", "")
    # Compact output: truncate text to max_text_chars (0 = omit -> metadata-only
    # summary). None/negative = full text (default, backward-compatible).
    if max_text_chars is not None and max_text_chars >= 0 and len(text) > max_text_chars:
        text = text[:max_text_chars] + ("…" if max_text_chars > 0 else "")
    return {
        "source": hit.get("source", ""),
        "page_number": hit.get("page_number"),
        "text": text,
        "modality": modality,
        "score": score,
    }
```

- [ ] **Step 2: Add the `--max-text-chars` option to `query_command`**

Replace:
```python
    hybrid: bool = typer.Option(
        False,
        "--hybrid",
        help="Combine vector + full-text (BM25) retrieval. Requires an index built with `ingest --hybrid`.",
    ),
) -> None:
```
with:
```python
    hybrid: bool = typer.Option(
        False,
        "--hybrid",
        help="Combine vector + full-text (BM25) retrieval. Requires an index built with `ingest --hybrid`.",
    ),
    max_text_chars: int | None = typer.Option(
        None,
        "--max-text-chars",
        help="Truncate each hit's text to N chars (0 = omit text, metadata-only summary). Default: full text.",
    ),
) -> None:
```

- [ ] **Step 3: Pass `max_text_chars` into the print**

Replace:
```python
    typer.echo(json.dumps([_query_cli_hit(hit) for hit in hits], indent=2, sort_keys=True, default=str))
```
with:
```python
    typer.echo(json.dumps([_query_cli_hit(hit, max_text_chars) for hit in hits], indent=2, sort_keys=True, default=str))
```

- [ ] **Step 4: Add CLI tests**

In `nemo_retriever/tests/test_root_query_cli.py`, append:
```python


def test_root_query_max_text_chars_truncates_and_omits(monkeypatch) -> None:
    hits = [{"text": "abcdefghij", "source": "d.pdf", "page_number": 1,
             "metadata": {"type": "text"}, "_distance": 0.1}]

    class FakeRetriever:
        def __init__(self, **_: Any) -> None:
            pass

        def query(self, query: str, **_kwargs: Any) -> list[dict[str, Any]]:
            return hits

    monkeypatch.setattr(sdk_workflow, "Retriever", FakeRetriever)

    snip = RUNNER.invoke(cli_main.app, ["query", "q", "--max-text-chars", "5"])
    assert snip.exit_code == 0
    snip_hit = json.loads(snip.output)[0]
    assert snip_hit["text"] == "abcde…"
    assert snip_hit["modality"] == "text"  # non-text fields intact
    assert snip_hit["source"] == "d.pdf"

    meta = RUNNER.invoke(cli_main.app, ["query", "q", "--max-text-chars", "0"])
    meta_hit = json.loads(meta.output)[0]
    assert meta_hit["text"] == ""
    assert meta_hit["source"] == "d.pdf"
    assert meta_hit["page_number"] == 1
```

- [ ] **Step 5: Run query tests**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python -m pytest nemo_retriever/tests/test_root_query_cli.py -q`
Expected: all pass — the new truncation test AND the existing full-text tests (default `None` ⇒ no truncation, so their exact-output assertions still hold).

- [ ] **Step 6: Commit**
```bash
git add nemo_retriever/src/nemo_retriever/adapters/cli/main.py nemo_retriever/tests/test_root_query_cli.py
git commit --no-gpg-sign -m "feat(retriever): add --max-text-chars to query (compact/metadata-only output)"
```

---

### Task 2: Contract 1.4.0 + skill docs

**Files:** `skills/nemo-retriever/contract/cli-contract.json`, `.../CONTRACT.md`, `skills/nemo-retriever/SKILL.md`, `.../references/query.md`

- [ ] **Step 1: Add `--max-text-chars` to the query flag surface + bump version**

In `skills/nemo-retriever/contract/cli-contract.json`, replace:
```json
  "contract_version": "1.3.0",
  "subcommands": ["ingest", "query", "verify"],
  "query": {
    "required_flags": ["--top-k", "--content-types", "--rerank", "--hybrid", "--embed-model-name", "--lancedb-uri", "--table-name"],
```
with:
```json
  "contract_version": "1.4.0",
  "subcommands": ["ingest", "query", "verify"],
  "query": {
    "required_flags": ["--top-k", "--content-types", "--rerank", "--hybrid", "--max-text-chars", "--embed-model-name", "--lancedb-uri", "--table-name"],
```

- [ ] **Step 2: Changelog in `CONTRACT.md`**

Replace:
```markdown
## Changelog
- **1.3.0** — `verify` subcommand added: fetches independent `text`/`table` evidence for a claim's (source, page) + a mechanical term/number-overlap signal. Engine retrieves; caller judges agreement.
```
with:
```markdown
## Changelog
- **1.4.0** — `query` gains `--max-text-chars N`: truncate each hit's text to N chars (`0` = metadata-only summary; unset = full). Compact output for token economy.
- **1.3.0** — `verify` subcommand added: fetches independent `text`/`table` evidence for a claim's (source, page) + a mechanical term/number-overlap signal. Engine retrieves; caller judges agreement.
```

- [ ] **Step 3: Bump declared version in `SKILL.md`**

Replace:
```
This skill targets engine **contract_version 1.3.0** (`contract/cli-contract.json`).
```
with:
```
This skill targets engine **contract_version 1.4.0** (`contract/cli-contract.json`).
```

- [ ] **Step 4: Document `--max-text-chars` in `query.md`**

In `skills/nemo-retriever/references/query.md`, after the "Hybrid query (optional)" paragraph (added in 2b), add:
```markdown
**Compact output (optional):** add `--max-text-chars 0` to get a **metadata-only summary** (source/page/modality/score, no text) in one call — useful for triage before fetching full text for a specific hit; or `--max-text-chars 200` for short snippets. Default is full text.
```

- [ ] **Step 5: Validate JSON + commit**
```bash
/home/edwardk/git/nv-ingest/retriever/bin/python -c "import json; json.load(open('skills/nemo-retriever/contract/cli-contract.json')); print('JSON OK')"
git add skills/nemo-retriever/contract/ skills/nemo-retriever/SKILL.md skills/nemo-retriever/references/query.md
git commit --no-gpg-sign -m "docs(skill): contract 1.4.0 + document query --max-text-chars"
```
Expected: `JSON OK`.

---

### Task 3: Live validation

**Files:** none.

- [ ] **Step 1: Full query CLI unit tests (no GPU)**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python -m pytest nemo_retriever/tests/test_root_query_cli.py -q`
Expected: all pass.

- [ ] **Step 2: `doctor` (now asserts `--max-text-chars`; GPU for live probe)**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python skills/nemo-retriever/scripts/doctor.py 2>&1 | grep -iE "max-text-chars|checks passed|FAIL"; echo "exit=${PIPESTATUS[0]}"` (run the full `doctor.py`).
Expected: `[PASS] query has --max-text-chars`, final `N/N checks passed`, `exit=0`.

- [ ] **Step 3: Eyeball a real compact query (GPU for ingest)**

```bash
mkdir -p /tmp/c3 && printf 'A long-enough sentence about widget XJ-4417 shipping schedules and inventory levels.\n' > /tmp/c3/d.txt && \
/home/edwardk/git/nv-ingest/retriever/bin/retriever ingest /tmp/c3/ --table-name c3 --lancedb-uri /tmp/c3_db --embed-model-name nvidia/llama-nemotron-embed-1b-v2 --quiet && \
echo "--- metadata-only (0) ---" && \
/home/edwardk/git/nv-ingest/retriever/bin/retriever query "widget" --top-k 2 --table-name c3 --lancedb-uri /tmp/c3_db --embed-model-name nvidia/llama-nemotron-embed-1b-v2 --max-text-chars 0 | /home/edwardk/git/nv-ingest/retriever/bin/python -c "import json,sys; h=json.load(sys.stdin)[0]; assert h['text']=='', h; print('text empty OK, keys=',sorted(h))" && \
echo "--- snippet (8) ---" && \
/home/edwardk/git/nv-ingest/retriever/bin/retriever query "widget" --top-k 2 --table-name c3 --lancedb-uri /tmp/c3_db --embed-model-name nvidia/llama-nemotron-embed-1b-v2 --max-text-chars 8 | /home/edwardk/git/nv-ingest/retriever/bin/python -c "import json,sys; h=json.load(sys.stdin)[0]; assert len(h['text'])<=9 and h['text'].endswith('…'), h; print('snippet OK:', h['text'])"; rm -rf /tmp/c3 /tmp/c3_db
```
Expected: metadata-only prints `text empty OK`, snippet prints a ≤9-char string ending `…`. (No GPU used by the truncation itself; only ingest loads models.)

- [ ] **Step 4: No commit (validation only).** If green, slice 3c is complete.

---

## Self-review

**Spec coverage (3c):**
- `--max-text-chars` on query: unset=full / `N>0`=snippet+`…` / `0`=metadata-only → Task 1 Steps 1-3. ✓
- backward-compatible (default None=full; existing tests pass) → Task 1 Step 5 runs them. ✓
- contract 1.4.0 + `--max-text-chars` in flag surface + doctor asserts → Task 2 Step 1 + Task 3 Step 2. ✓
- query.md documents the option → Task 2 Step 4. ✓
- query-only (verify untouched) → no verify changes anywhere. ✓
- tests (truncate + omit + intact non-text fields) → Task 1 Step 4. ✓

**Placeholder scan:** No TBD/TODO; all edits concrete.

**Type consistency:** `max_text_chars: int | None` matches between the Typer option, `_query_cli_hit`'s new param, and the print call. The `…` ellipsis in the truncation matches the test's `"abcde…"` expectation (5 chars + ellipsis) and the snippet live-check (`endswith('…')`, `len ≤ 9` for N=8). `--max-text-chars` is added to `query.required_flags`, which `doctor` (with `COLUMNS=200`) asserts against `query --help`.
