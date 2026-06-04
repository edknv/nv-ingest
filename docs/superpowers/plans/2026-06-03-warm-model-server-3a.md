# Objective 3a — warm model server (`retriever serve-models`, embedder-first) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Kill the per-query embedder cold-load by adding `retriever serve-models`, which launches a warm vLLM OpenAI-compatible embeddings server and prints the `EMBED_INVOKE_URL` export the query path already consumes. Embedder-first; reranker-warm is investigated via a gating spike and deferred if `vllm serve` can't meet `rerank.py`'s `/v1/ranking` contract.

**Architecture:** A `serve-models` CLI command launches `vllm serve <embed-model>` as a supervised subprocess, health-gates readiness, prints the endpoint URL + `export EMBED_INVOKE_URL=…`, and stays foreground until Ctrl-C (clean teardown). Query/verify/MCP need **no change** — they already honor `--embed-invoke-url`/`EMBED_INVOKE_URL`. Pure helpers (argv builder, URL/export formatting, readiness poll) are unit-tested; the live launch is the gate.

**Tech Stack:** vLLM 0.20.0 (`vllm serve`), Typer, Python `subprocess`/`urllib`, pytest, the `retriever` venv.

## Ground truth (verified)
- `vllm` CLI present (0.20.0, has `serve`).
- Embed path POSTs OpenAI `/v1/embeddings` (`{"input":[...], "model":...}`, `text_embed/cpu_operator.py:47-48`) → matches a `vllm serve` embeddings endpoint.
- `query_command` reads `EMBED_INVOKE_URL` and `RERANKER_INVOKE_URL` from env (`main.py:641-643`).
- **Reranker mismatch:** `rerank.py` POSTs `{endpoint}/v1/ranking` with the NIM shape (`{"model","query":{"text"},"passages":[{"text"}]}`) — NOT vLLM's native `/score`/`/rerank`. So reranker-warm via plain `vllm serve` is unlikely; it's investigated in Task 3, not built here.
- Unknown to pin in the spike: exact `vllm serve` task flag for the embedder (`--task embed` vs `embedding` vs auto).

All commits `--no-gpg-sign`. Tasks 1, 3, 5 are GPU-heavy/live; Task 2/4 code+unit tests are no-GPU.

---

### Task 1: Embedder feasibility spike (GATE — no commit)

**Goal:** prove `vllm serve` can warm-serve the embedder in a shape `retriever query` consumes. If this fails, STOP and report — do not build on it.

- [ ] **Step 1: Launch a warm embedder via vLLM (background)**

Try the most likely invocation; if the task flag is wrong, read the error and adjust (record what works):
```bash
nohup /home/edwardk/git/nv-ingest/retriever/bin/vllm serve nvidia/llama-nemotron-embed-1b-v2 \
  --task embed --host 127.0.0.1 --port 8081 > /tmp/vllm_embed.log 2>&1 &
echo $! > /tmp/vllm_embed.pid
```
Poll readiness (up to ~10 min for first load): `for i in $(seq 1 120); do curl -sf http://127.0.0.1:8081/health && break; sleep 5; done`.
If `--task embed` is rejected, inspect `/tmp/vllm_embed.log`, try `--task embedding` or no task flag, and **record the working invocation** (Task 2 uses it).

- [ ] **Step 2: Confirm the endpoint speaks `/v1/embeddings`**

Run: `curl -sf http://127.0.0.1:8081/v1/embeddings -H 'Content-Type: application/json' -d '{"model":"nvidia/llama-nemotron-embed-1b-v2","input":["hello"]}' | head -c 300`
Expected: a JSON embeddings response (a `data[0].embedding` vector). If the path/shape differs, record the actual one.

- [ ] **Step 3: Confirm `retriever query` uses it WARM (the real gate)**

```bash
mkdir -p /tmp/sm && printf 'Widget XJ-4417 ships in March.\n' > /tmp/sm/d.txt
# NOTE: ingest still cold-loads its own embedder; that's fine — we are testing QUERY warmth.
/home/edwardk/git/nv-ingest/retriever/bin/retriever ingest /tmp/sm/ --table-name sm --lancedb-uri /tmp/sm_db --embed-model-name nvidia/llama-nemotron-embed-1b-v2 --quiet
# First warm query (point at the server):
time /home/edwardk/git/nv-ingest/retriever/bin/retriever query "widget" --top-k 2 --table-name sm --lancedb-uri /tmp/sm_db --embed-invoke-url http://127.0.0.1:8081/v1/embeddings | head -c 200
# Second query should be similarly fast (no per-query cold-load):
time /home/edwardk/git/nv-ingest/retriever/bin/retriever query "ship date" --top-k 2 --table-name sm --lancedb-uri /tmp/sm_db --embed-invoke-url http://127.0.0.1:8081/v1/embeddings | head -c 200
```
Expected: both queries return hits; the second is fast (a few seconds, no ~30–60s vLLM startup). This proves the warm-endpoint path works end-to-end.

- [ ] **Step 4: Tear down + record**

`kill "$(cat /tmp/vllm_embed.pid)" 2>/dev/null; rm -rf /tmp/sm /tmp/sm_db`.
**Record the exact working `vllm serve` invocation (esp. the task flag).** If Step 1-3 failed irrecoverably, STOP: report that `vllm serve` can't warm-serve this embedder and the mechanism needs rethinking (do not proceed to Task 2).

---

### Task 2: Build the `serve-models` command (embedder)

**Files:** create `nemo_retriever/src/nemo_retriever/adapters/cli/serve_models.py`; modify `adapters/cli/main.py`; create `nemo_retriever/tests/test_serve_models.py`

- [ ] **Step 1: Create `serve_models.py` (pure helpers + supervisor)**
```python
# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch a warm vLLM OpenAI-compatible embeddings server (kills per-query cold-load)."""

from __future__ import annotations

import shutil
import time
import urllib.request


def build_vllm_argv(model: str, host: str, port: int, task: str = "embed") -> list[str]:
    """argv for `vllm serve` of an embedding model. `task` is what the Task-1 spike confirmed."""
    vllm = shutil.which("vllm") or "vllm"
    return [vllm, "serve", model, "--task", task, "--host", host, "--port", str(port)]


def embeddings_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/v1/embeddings"


def export_line(host: str, port: int) -> str:
    return f"export EMBED_INVOKE_URL={embeddings_url(host, port)}"


def wait_ready(host: str, port: int, timeout: float = 600.0, interval: float = 3.0) -> bool:
    """Poll the vLLM /health endpoint until 200 or timeout."""
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310
                if resp.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(interval)
    return False
```
> If Task 1 found a task flag other than `embed` (e.g. `embedding`) or a different path, update the `task` default here and `embeddings_url` accordingly.

- [ ] **Step 2: Add the `serve-models` command to `main.py`** — insert after `mcp_command` (before `@app.callback()`):
```python
@app.command("serve-models")
def serve_models_command(
    embed_model_name: str = typer.Option(
        "nvidia/llama-nemotron-embed-1b-v2", "--embed-model-name", help="Embedding model to serve warm."
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind the vLLM server."),
    embed_port: int = typer.Option(8081, "--embed-port", help="Port for the embeddings server."),
    ready_timeout: float = typer.Option(600.0, "--ready-timeout", help="Seconds to wait for readiness."),
) -> None:
    """Serve a WARM embedder so `retriever query` avoids the per-query cold-load.

    Prints an `export EMBED_INVOKE_URL=...` line; query/verify/MCP honor that env var.
    """
    import subprocess

    from nemo_retriever.adapters.cli import serve_models as sm

    argv = sm.build_vllm_argv(embed_model_name, host, embed_port)
    proc = subprocess.Popen(argv)
    try:
        if not sm.wait_ready(host, embed_port, timeout=ready_timeout):
            typer.echo("Error: embedder server did not become ready in time.", err=True)
            raise typer.Exit(1)
        typer.echo(f"Embedder warm at {sm.embeddings_url(host, embed_port)}")
        typer.echo(sm.export_line(host, embed_port))
        typer.echo("Leave this running; in another shell paste the export line, then run `retriever query`.")
        proc.wait()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()
```

- [ ] **Step 3: Create `nemo_retriever/tests/test_serve_models.py` (no GPU)**
```python
# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import http.server
import threading

from nemo_retriever.adapters.cli import serve_models as sm


def test_build_vllm_argv_and_urls() -> None:
    argv = sm.build_vllm_argv("my/embed", "127.0.0.1", 8081)
    assert argv[1:] == ["serve", "my/embed", "--task", "embed", "--host", "127.0.0.1", "--port", "8081"]
    assert sm.embeddings_url("127.0.0.1", 8081) == "http://127.0.0.1:8081/v1/embeddings"
    assert sm.export_line("127.0.0.1", 8081) == "export EMBED_INVOKE_URL=http://127.0.0.1:8081/v1/embeddings"


def test_wait_ready_true_against_fake_health_server() -> None:
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):  # silence
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        assert sm.wait_ready("127.0.0.1", port, timeout=5, interval=0.2) is True
    finally:
        srv.shutdown()


def test_wait_ready_false_when_nothing_listening() -> None:
    # Port 1 is not listening; short timeout returns False fast.
    assert sm.wait_ready("127.0.0.1", 1, timeout=1.0, interval=0.2) is False
```

- [ ] **Step 4: Run unit tests + import check (no GPU)**

Run: `/home/edwardk/git/nv-ingest/retriever/bin/python -m pytest nemo_retriever/tests/test_serve_models.py -q && /home/edwardk/git/nv-ingest/retriever/bin/retriever serve-models --help >/dev/null && echo "CMD OK"`
Expected: 3 passed; `CMD OK` (command imports + help works without launching vLLM).

- [ ] **Step 5: Commit**
```bash
git add nemo_retriever/src/nemo_retriever/adapters/cli/serve_models.py nemo_retriever/src/nemo_retriever/adapters/cli/main.py nemo_retriever/tests/test_serve_models.py
git commit --no-gpg-sign -m "feat(retriever): add serve-models (warm vLLM embedder; kills per-query cold-load)"
```

---

### Task 3: Reranker feasibility spike (investigation — decide defer vs follow-up; no build)

**Goal:** determine whether `vllm serve` can satisfy `rerank.py`'s `/v1/ranking` NIM contract; record the outcome. Embedder-warm ships regardless.

- [ ] **Step 1: Probe the reranker server's protocol**

Launch `vllm serve nvidia/llama-nemotron-rerank-vl-1b-v2 --task score --host 127.0.0.1 --port 8082` (background, health-gate). Then probe whether it exposes `/v1/ranking` with the NIM payload:
```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8082/v1/ranking \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/llama-nemotron-rerank-vl-1b-v2","query":{"text":"q"},"passages":[{"text":"p"}]}'
```
Also list routes: `curl -s http://127.0.0.1:8082/openapi.json | /home/edwardk/git/nv-ingest/retriever/bin/python -c "import json,sys; print(sorted(json.load(sys.stdin).get('paths',{})))"`. Tear down after.

- [ ] **Step 2: Record the decision in the spec**

In `docs/superpowers/specs/2026-06-03-warm-model-server-3a-design.md`, append a "## Reranker spike result" section stating, factually, whether `vllm serve` exposes a `/v1/ranking`-compatible endpoint:
- If **yes**: note that a follow-up can extend `serve-models` with `--reranker-*` + `RERANKER_INVOKE_URL`.
- If **no** (expected): note reranker-warm needs a `/v1/ranking`→vLLM-`/score` shim (or a `rerank.py` protocol option) — a separate future slice. Reranking stays cold/off-by-default for now.

- [ ] **Step 3: Commit the recorded finding**
```bash
git add docs/superpowers/specs/2026-06-03-warm-model-server-3a-design.md
git commit --no-gpg-sign -m "docs(3a): record reranker vllm-serve spike result (reranker-warm deferred or viable)"
```

---

### Task 4: Contract 1.6.0 + skill doc

**Files:** `skills/nemo-retriever/contract/cli-contract.json`, `.../CONTRACT.md`, `skills/nemo-retriever/SKILL.md`, `.../references/setup.md`

- [ ] **Step 1: Add `serve-models` to subcommands + bump version** — in `cli-contract.json`, replace:
```json
  "contract_version": "1.5.0",
  "subcommands": ["ingest", "query", "verify", "mcp"],
```
with:
```json
  "contract_version": "1.6.0",
  "subcommands": ["ingest", "query", "verify", "mcp", "serve-models"],
```

- [ ] **Step 2: Changelog in `CONTRACT.md`** — add above the 1.5.0 entry:
```markdown
- **1.6.0** — `serve-models` subcommand added: launches a warm vLLM embeddings server and prints `export EMBED_INVOKE_URL=…`, so `query` avoids the per-query cold-load. (Reranker-warm deferred — see the 3a spec.)
```

- [ ] **Step 3: Bump declared version in `SKILL.md`** — replace `contract_version 1.5.0` with `contract_version 1.6.0` in the line that declares it.

- [ ] **Step 4: Document in `setup.md`** — add:
```markdown
**Warm querying (optional, avoids per-query cold-load):** in a separate shell run `<RETRIEVER_VENV>/bin/retriever serve-models` once; paste the printed `export EMBED_INVOKE_URL=…`, then run queries normally — the embedder stays resident instead of cold-loading (~30–60s) on every query.
```

- [ ] **Step 5: Validate JSON + commit**
```bash
/home/edwardk/git/nv-ingest/retriever/bin/python -c "import json; json.load(open('skills/nemo-retriever/contract/cli-contract.json')); print('JSON OK')"
git add skills/nemo-retriever/contract/ skills/nemo-retriever/SKILL.md skills/nemo-retriever/references/setup.md
git commit --no-gpg-sign -m "docs(skill): contract 1.6.0 + document serve-models warm querying"
```

---

### Task 5: Live end-to-end validation

- [ ] **Step 1: Unit + help (no GPU)** — `pytest nemo_retriever/tests/test_serve_models.py -q` (3 passed) and `retriever serve-models --help` exit 0.
- [ ] **Step 2: `doctor` asserts `serve-models` (GPU live probe)** — run full `doctor.py`; expect ``subcommand `serve-models` exists`` `[PASS]`, `N/N checks passed`, exit 0.
- [ ] **Step 3: Live warm round-trip (GPU)** — start `retriever serve-models` in the background; wait for the `export EMBED_INVOKE_URL=` line; set it; ingest a tiny corpus; run two `retriever query` calls and confirm both return hits and the **second is fast (no cold-load)**; then stop `serve-models` and confirm no orphaned vLLM process. (This is the headline proof.)
- [ ] **Step 4: No commit (validation only).** If green, slice 3a — and Objective 3 — is complete.

---

## Self-review

**Spec coverage (3a, embedder-first):**
- `serve-models` launches warm vLLM embeddings server + prints export → Task 2. ✓
- query consumes it unchanged (env/flag) → relies on verified `main.py:643`; Task 1 Step 3 proves it. ✓
- spike-first gate (embedder) → Task 1 (no build before it passes). ✓
- reranker via gating spike, deferred if mismatch → Task 3 (investigation + recorded decision; not built). ✓
- contract 1.6.0 + subcommand + doctor + setup.md → Task 4 + Task 5 Step 2. ✓
- pure helpers unit-tested without GPU; live launch is the gate → Task 2 Step 3 + Task 5 Step 3. ✓

**Placeholder scan:** No TBD/TODO. The one runtime-resolved value (the exact `vllm serve` task flag) is explicitly pinned by Task 1 and flagged where Task 2 uses it — spike-driven, not a placeholder. Task 3's recorded result is written during execution (its content is genuinely spike-dependent, which is the point).

**Type consistency:** `build_vllm_argv`/`embeddings_url`/`export_line`/`wait_ready` signatures match their unit-test calls and the `serve_models_command` usage. The export var name `EMBED_INVOKE_URL` matches what `query_command` reads (`main.py:643`). `serve-models` added to the contract `subcommands` list that `doctor` iterates; `--help` exits 0 without launching vLLM (Typer).
