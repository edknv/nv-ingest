# retriever — skill-first design artifacts

These are the **design driver** for a from-scratch `retriever` skill. The skill is
the product; the library is its backend, specified to satisfy `contract.schema.json`.
Design rationale: `docs/superpowers/specs/2026-06-04-retriever-skill-first-design.md`.

**Not yet activated.** `SKILL.md` here is intentionally NOT in `skills/` — it depends
on a `retrieve`/`verify`/`index` backend that does not exist yet, so loading it now
would be a broken skill. SP-D moves it into `skills/retriever/` once `retrieve` is live.

## Files
- `SKILL.md` — the judgment-only skill (the irreducible retrieval wisdom).
- `contract.schema.json` — machine-readable `retrieve` / `verify` / `index` result shapes
  the library must satisfy.

## Skill ↔ library boundary
- **Library** (backend): retrieve, fuse strategies, tag `fidelity`, cite, `verify`,
  report `coverage`, serve warm. Mechanical, typed, testable.
- **Skill** (the model): choose the move, judge trust/sufficiency, decide when to
  verify, compose an honest answer, decide when to refuse. Judgment only.

## Library sub-projects (built to satisfy the contract; dependency order)
- **SP-A — ingest provenance → `fidelity`**: record extractor/OCR/ASR/caption provenance
  per chunk so a true `fidelity` exists. Foundational (today only `modality` is stored).
- **SP-B — `retrieve` primitive**: fuse strategies → attach `fidelity`+`citation` →
  compute `coverage` → return answer-ready evidence. Composes existing hybrid/content-type/
  verify work + SP-A's fidelity.
- **SP-C — serving + MCP**: expose `retrieve`/`verify`/`index` warm (build on `serve-models`)
  and as MCP tools (build on `retriever mcp`), re-pointed to `retrieve`.
- **SP-D — ship the skill**: move `SKILL.md` into `skills/retriever/`; retire the old
  CLI `nemo-retriever` skill.

## Warm MCP (today's wiring)
The library already supports the contract over MCP, warm:
1. `retriever serve-models` — warm embedder; export the printed `EMBED_INVOKE_URL`.
2. Point a harness at `retriever mcp` — its `retrieve`/`query`/`verify` tools then run
   warm (`retrieve` honors `EMBED_INVOKE_URL`), no per-call cold-load.
`index` remains a CLI/setup step (`retriever ingest`), not an MCP tool.
