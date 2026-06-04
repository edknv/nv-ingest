# Objective 3a — warm model server (`retriever serve-models`)

**Date:** 2026-06-03
**Type:** Engine addition to `nemo_retriever` (final slice of rebuild Objective 3)
**Parent:** `2026-06-03-skill-first-retriever-rebuild-design.md` (Objective 3 = agent-economics). Follows 3c (compact I/O) and 3b (MCP surface).

## Problem

Every `retriever query` cold-loads the embedder (and, with `--rerank`, the reranker) on GPU via vLLM (~30–60s). This per-query cold-load was the single biggest observed cost across this session's runs. Objective 3a kills it.

## Key finding (grounding)

The query path **already supports warm endpoints**; the cold-load only happens in *in-process* mode:
- Embedding: `_BatchEmbedActor` (`text_embed/operators.py:31`) uses `embed_invoke_url`/`embedding_endpoint` when set; otherwise loads the vLLM embedder in-process (`text_embed/vllm.py`).
- Reranking: `rerank/rerank.py` uses `--reranker-invoke-url` (documented as "vLLM ≥0.14 or NIM") when set; otherwise loads from HuggingFace in-process.
- The CLI already reads `RERANKER_INVOKE_URL` (and embed endpoint) from env in `query_command`.

So 3a is **not** a new query-serving daemon — it is "stand up a warm embedder + reranker server once and point the existing `--embed-invoke-url`/`--reranker-invoke-url` (or their env vars) at it." No query-side code change.

## Decisions (from brainstorming)

- **Mechanism:** wrap **vLLM's built-in OpenAI-compatible server** (`vllm serve`) rather than building a custom FastAPI server or growing the ingest-only `service` daemon.
- **Scope:** hold **both** the embedder and reranker resident (two warm models).

## Design

### The command: `retriever serve-models`
A new long-running CLI command that:
1. Launches **two** vLLM OpenAI-compatible server subprocesses — one for the embedder (embeddings endpoint), one for the reranker (score/rerank endpoint) — on configurable ports (defaults, e.g. embed 8081 / rerank 8082).
2. Waits until both report ready (health-gated) before announcing.
3. Prints the two endpoint URLs **and** the ready-to-paste env exports:
   ```
   export EMBED_INVOKE_URL=http://localhost:8081/v1/embeddings
   export RERANKER_INVOKE_URL=http://localhost:8082/...   # exact path per rerank.py's contract
   ```
4. Stays in the foreground supervising both subprocesses; on SIGINT/SIGTERM, tears both down cleanly.

Options: `--embed-model-name`, `--reranker-model-name`, `--embed-port`, `--reranker-port`, `--host`, plus a `--no-reranker` escape hatch (embedder-only).

### Usage
```
retriever serve-models &            # warm, once
eval "$(... the printed exports ...)"  # or copy the export lines
retriever query "…" --rerank        # transparently warm; no cold-load
```
Because `query`/`verify`/the MCP tools already consume `embed_invoke_url`/`reranker_invoke_url` (via flags or env), nothing downstream changes.

### Contract + skill
- `cli-contract.json`: add `serve-models` to `subcommands`; bump `contract_version` 1.5.0 → **1.6.0**.
- `CONTRACT.md`: changelog for 1.6.0.
- `doctor.py`: assert `serve-models` exists (static `--help`).
- `setup.md`: document "for repeated querying, run `retriever serve-models` once and export the printed URLs to avoid the per-query cold-load."

## Components & boundaries
- `serve-models` command (`adapters/cli/main.py` or a small `adapters/cli/serve_models.py` it delegates to) — process supervision + readiness + teardown + URL/export printing. One clear responsibility.
- vLLM servers — external processes the command launches and supervises; the retriever does not reimplement serving.
- No change to `query`/`verify`/`Retriever`/operators — they already consume the endpoints.

## Risks / unknowns (this is the riskiest slice; the plan front-loads a spike)

1. **Embedder via `vllm serve`:** can `nvidia/llama-nemotron-embed-1b-v2` be served as an OpenAI `/v1/embeddings` endpoint (embedding/pooling task) that `_BatchEmbedActor` accepts? **Make-or-break.**
2. **Reranker via `vllm serve`:** can the reranker be served so `rerank.py`'s endpoint payload (its "vLLM ≥0.14" contract) matches? **Make-or-break for the rerank half.**
3. **GPU memory:** two ~1B models resident at once must fit alongside query-time use.
4. **Lifecycle:** one foreground command supervising two subprocesses — readiness gating and clean teardown (no orphaned vLLM processes).

**Plan shape (spike-first):** Task 1 is a *feasibility spike* — manually launch a warm embedder via `vllm serve`, hit `/v1/embeddings`, and confirm `retriever query --embed-invoke-url <it>` returns hits with **no cold-load** — before building the supervised command. Then the reranker spike, then the `serve-models` command, then contract/docs. If a spike fails (#1/#2), stop and reconsider the mechanism rather than build on a broken assumption.

## Validation
- Spike (GPU): warm embedder serves `/v1/embeddings`; `query --embed-invoke-url` works and is warm on the 2nd call.
- Command: `retriever serve-models --help` (no GPU); a live start that prints URLs + exports and a warm `query` round-trip (GPU, slow).
- `doctor` asserts `serve-models` in `subcommands`.
- **Honest boundary:** full validation is GPU- and time-heavy; the spike is the gate that de-risks the build.

## Non-goals
- No custom embeddings/rerank server implementation (wrap vLLM).
- No new query-serving daemon or `/v1/query` endpoint (query stays client-side; only the models go warm).
- No auto-start of the server from `query` (the user runs `serve-models` explicitly; query just honors the endpoints/env).
- No change to ingest.

## Open questions (resolve during the plan)
- Exact vLLM serve flags/task for the embedder (`--task embed`?) and reranker (`--task score`/rerank?), and the exact reranker endpoint path `rerank.py` expects — pin in the spike.
- Port defaults and whether to also write the exports to a dotfile the CLI can source automatically (deferred; explicit export for v1).
