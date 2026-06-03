# multimodal-sweep Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable, named Claude Code workflow (`.claude/workflows/multimodal-sweep.js`) that answers one hard question over a `nemo-retriever` corpus by sweeping multiple blind retrieval angles in parallel, merging/deduping the hits, and adversarially verifying any chart/image-only claims against prose.

**Architecture:** Pure-orchestration JS workflow script. Phase 0 setup agent resolves the `retriever` venv and ensures the LanceDB index exists. Phase 1 fans out one isolated subagent per retrieval angle (each does a single disciplined `retriever` query → structured output). Phase 2 merges/dedupes into a draft answer and flags chart/image-only claims. Phase 3 verifies flagged claims against pdfium prose extraction, then a finalize agent folds verdicts in and produces a markdown report string. The workflow returns the answer + `reportMarkdown`; the caller writes the report file.

**Tech Stack:** Claude Code Workflow JS runtime (`agent()`/`parallel()`/`phase()`/`log()`, JSON-Schema `agent({schema})`); the `nemo-retriever` `retriever` CLI + `scripts/grep_corpus.py`; LanceDB.

---

## Key constraints (read before starting)

- **Workflow scripts cannot run Bash or touch the filesystem** — only `agent()`s can. So *ingest* is a setup agent, *queries* are angle agents, and the *report file* is written by the caller from the returned `reportMarkdown` string.
- **The `nemo-retriever` skill bans subagents inside a query turn and caps it at ≤2 Bash calls.** Each angle agent must therefore do exactly one disciplined query. The *workflow* owns all fan-out.
- **The LanceDB index is built once and is concurrent-safe to read.** Never ingest in parallel; Phase 0 is a barrier before any query.
- **`retriever` `page_number` is 1-indexed** (first page = 1). Citations stay 1-indexed.
- **Validation note:** workflow scripts use top-level `await`/`return`, so plain `node --check` rejects them. Use the wrapper parse-check command given in Task 1 (it compiles the body inside an async function via `new Function`, catching syntax errors without executing).

## File structure

- **Create:** `.claude/workflows/multimodal-sweep.js` — the entire workflow (meta, config, schemas, angle-prompt builders, four phases, return). One file: it is a single orchestration unit and the named-workflow registry expects one script per name.
- **Modify:** `docs/superpowers/specs/2026-06-03-multimodal-sweep-workflow-design.md` — only to record the "caller writes the report" refinement (Task 6).

The script's internal boundaries: `cfg` (arg parsing), `*_SCHEMA` consts (inter-agent contracts), `ANGLE_SPECS` (per-angle prompt builders — swap an angle by editing one entry), and the four phase blocks (control flow). Each is independently readable.

---

### Task 1: Scaffold the workflow file — meta, config, schemas, angle builders

**Files:**
- Create: `.claude/workflows/multimodal-sweep.js`

- [ ] **Step 1: Create the directory and file with the static declarations**

Create `.claude/workflows/multimodal-sweep.js` with exactly this content:

```javascript
export const meta = {
  name: 'multimodal-sweep',
  description: 'Answer one question over a nemo-retriever corpus by sweeping multiple blind retrieval angles in parallel, then adversarially verifying chart/image-only claims against prose',
  phases: [
    { title: 'Setup', detail: 'resolve retriever venv + ensure LanceDB index exists' },
    { title: 'Sweep', detail: 'parallel angle agents, one disciplined query each' },
    { title: 'Merge', detail: 'dedupe hits, draft answer, flag chart/image-only claims' },
    { title: 'Verify', detail: 'adversarially check flagged claims against prose, then finalize' },
  ],
}

// ---------- config / args ----------
const cfg = {
  question:    args?.question,
  corpusDir:   args?.corpusDir   ?? './pdfs',
  indexDir:    args?.indexDir    ?? './lancedb',
  tableName:   args?.tableName   ?? 'nv-ingest',
  topK:        args?.topK        ?? 10,
  embedModel:  args?.embedModel  ?? 'nvidia/llama-nemotron-embed-1b-v2',
  angles:      args?.angles      ?? ['semantic', 'reformulated', 'keyword', 'visual', 'tabular'],
  verify:      args?.verify      ?? true,
  writeReport: args?.writeReport ?? true,
  grepScript:  args?.grepScript  ?? 'skills/nemo-retriever/scripts/grep_corpus.py',
  reportPath:  args?.reportPath  ?? './multimodal-sweep-report.md',
  repoRoot:    args?.repoRoot    ?? '/home/edwardk/git/nv-ingest',
}
if (!cfg.question) throw new Error('multimodal-sweep: args.question is required')

// ---------- schemas (inter-agent contracts) ----------
const HIT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['doc', 'page', 'type', 'snippet', 'rank'],
  properties: {
    doc: { type: 'string', description: 'pdf_basename without .pdf' },
    page: { type: 'number', description: '1-indexed page number' },
    type: { type: 'string', enum: ['text', 'table', 'chart', 'image'] },
    snippet: { type: 'string' },
    rank: { type: 'number' },
  },
}

const SETUP_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['retrieverVenv', 'indexReady', 'docCount', 'distinctDocs'],
  properties: {
    retrieverVenv: { type: 'string', description: 'absolute venv root containing bin/retriever' },
    indexReady: { type: 'boolean' },
    docCount: { type: 'number' },
    distinctDocs: { type: 'array', items: { type: 'string' } },
    note: { type: 'string' },
  },
}

const ANGLE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['angle', 'candidateAnswer', 'hits', 'confidence'],
  properties: {
    angle: { type: 'string' },
    candidateAnswer: { type: 'string' },
    hits: { type: 'array', items: HIT_SCHEMA },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
}

const CITATION_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['doc', 'page', 'type'],
  properties: { doc: { type: 'string' }, page: { type: 'number' }, type: { type: 'string' } },
}

const MERGE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['draftAnswer', 'citations', 'claims_to_verify', 'confidence'],
  properties: {
    draftAnswer: { type: 'string' },
    citations: { type: 'array', items: CITATION_SCHEMA },
    claims_to_verify: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['claim', 'doc', 'page'],
        properties: { claim: { type: 'string' }, doc: { type: 'string' }, page: { type: 'number' } },
      },
    },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
}

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['claim', 'verdict', 'evidence'],
  properties: {
    claim: { type: 'string' },
    verdict: { type: 'string', enum: ['confirmed', 'refuted', 'not_found'] },
    evidence: { type: 'string' },
  },
}

const FINAL_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['final_answer', 'citations', 'confidence', 'reportMarkdown'],
  properties: {
    final_answer: { type: 'string' },
    citations: { type: 'array', items: CITATION_SCHEMA },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    reportMarkdown: { type: 'string' },
  },
}

// ---------- angle prompt builders ----------
const baseContext = (venv) => `You are ONE retrieval angle in a multi-angle sweep answering this question over a PRE-BUILT nemo-retriever LanceDB corpus.

QUESTION: ${cfg.question}

retriever venv: ${venv} (use ${venv}/bin/retriever and ${venv}/bin/python). Run from ${cfg.repoRoot}.
Index: ${cfg.indexDir}, table "${cfg.tableName}". It is ALREADY BUILT — NEVER ingest or re-extract the whole corpus.
DISCIPLINE: at most 2 Bash calls; no narration between calls; do NOT spawn subagents; go straight from your command to returning structured output.
Each LanceDB hit has: text, pdf_basename, page_number (1-indexed int), metadata.type in {text,table,chart,image}, _distance.
Return your candidate answer plus the hits you used: doc = pdf_basename without ".pdf"; page = page_number as-is (1-indexed); rank starts at 1. confidence reflects how well the hits actually answer the question.`

const ANGLE_SPECS = {
  semantic: (venv) => `${baseContext(venv)}

YOUR ANGLE = "semantic": straight semantic search with reranking. Run exactly this single pipeline:
${venv}/bin/retriever query "${cfg.question}" --top-k ${cfg.topK} --rerank --embed-model-name ${cfg.embedModel} | tee /tmp/sweep_semantic.json | ${venv}/bin/python -c "import json,sys;[print(f'rank={h.get(\\"rank\\",0)} page={h[\\"page_number\\"]} pdf={h[\\"pdf_basename\\"]} type={h.get(\\"metadata\\",{}).get(\\"type\\",\\"?\\")}') for h in json.load(sys.stdin)]"
Then read the hit text you need from /tmp/sweep_semantic.json and synthesize.`,

  reformulated: (venv) => `${baseContext(venv)}

YOUR ANGLE = "reformulated": semantic search is phrasing-sensitive. Rephrase the question into 2-3 alternatives (one keyword-dense; one HyDE-style: a single hypothetical SENTENCE that, if present in a doc, would answer it). Run one query per phrasing (combine into a single Bash command to stay within budget), union the hits, dedupe by (pdf,page), report the best. Use:
${venv}/bin/retriever query "<phrasing>" --top-k ${cfg.topK} --rerank --embed-model-name ${cfg.embedModel}`,

  keyword: (venv) => `${baseContext(venv)}

YOUR ANGLE = "keyword": exact-term matches semantic search may miss. Extract the key identifiers/terms/numbers from the question, build a regex, and run:
${venv}/bin/python ${cfg.grepScript} "<regex>" --max-hits 50
Output is "<pdf>:p<page>:<type>:  ...snippet..." per line, or NO_MATCH. Map those lines to hits (rank by order). If NO_MATCH, return empty hits with confidence "low".`,

  visual: (venv) => `${baseContext(venv)}

YOUR ANGLE = "visual": facts hidden in figures. Run the semantic query, then KEEP ONLY chart/image-type hits:
${venv}/bin/retriever query "${cfg.question}" --top-k ${cfg.topK} --rerank --embed-model-name ${cfg.embedModel} | tee /tmp/sweep_visual.json
Filter /tmp/sweep_visual.json to hits with metadata.type in {chart,image}. Their text is a model caption that MAY misread numbers/directions — report it but set confidence "low" for any exact number unless corroborated elsewhere.`,

  tabular: (venv) => `${baseContext(venv)}

YOUR ANGLE = "tabular": facts in tables. Run the semantic query, then KEEP ONLY table-type hits:
${venv}/bin/retriever query "${cfg.question}" --top-k ${cfg.topK} --rerank --embed-model-name ${cfg.embedModel} | tee /tmp/sweep_tabular.json
Filter /tmp/sweep_tabular.json to hits with metadata.type == "table" and report the rows relevant to the question.`,
}
```

- [ ] **Step 2: Parse-check the file**

Run:
```bash
node -e 'const fs=require("fs");let s=fs.readFileSync(".claude/workflows/multimodal-sweep.js","utf8").replace(/^export\s+const\s+meta/m,"const meta");new Function("agent","parallel","pipeline","phase","log","args","budget","workflow","return (async()=>{"+s+"})()");console.log("PARSE OK")'
```
Expected: `PARSE OK` (no `SyntaxError`).

- [ ] **Step 3: Commit**

```bash
git add .claude/workflows/multimodal-sweep.js
git commit --no-gpg-sign -m "feat(workflow): scaffold multimodal-sweep meta, config, schemas, angles"
```
(`--no-gpg-sign` because GPG signing fails in this environment.)

---

### Task 2: Phase 0 — setup agent + barrier guard

**Files:**
- Modify: `.claude/workflows/multimodal-sweep.js` (append after the `ANGLE_SPECS` block)

- [ ] **Step 1: Append the Phase 0 block**

Append to the end of the file:

```javascript

// ---------- Phase 0: setup (barrier) ----------
phase('Setup')
const setup = await agent(
  `Prepare the nemo-retriever index for a multi-angle query sweep. Run from ${cfg.repoRoot}.
1. Resolve the retriever venv: run \`command -v retriever\`. If it prints .../bin/retriever, the venv ROOT is the directory two levels up (strip /bin/retriever). If retriever is NOT found, follow ${cfg.repoRoot}/skills/nemo-retriever/references/install.md to install it, then resolve the path.
2. Check whether the index exists at ${cfg.indexDir}/${cfg.tableName}.lance. If it does NOT exist, ingest the corpus: \`<venv>/bin/retriever ingest ${cfg.corpusDir}/ --embed-model-name ${cfg.embedModel}\`. Build an index even if the corpus is large (a text-only index beats none; see ${cfg.repoRoot}/skills/nemo-retriever/references/setup.md for the >800-page recipe).
3. Report: retrieverVenv (the venv root), indexReady (true once the .lance table exists/queryable), docCount, and distinctDocs. Get counts via:
   \`<venv>/bin/python -c "import lancedb; df=lancedb.connect('${cfg.indexDir}').open_table('${cfg.tableName}').to_pandas(); print(len(df)); import json; print(json.dumps(sorted(df.pdf_basename.unique().tolist())))"\``,
  { label: 'setup', phase: 'Setup', schema: SETUP_SCHEMA }
)

if (!setup || !setup.indexReady) {
  return { error: 'index not ready — cannot run sweep', setup: setup ?? null, final_answer: null }
}
log(`index ready: ${setup.docCount} chunks across ${setup.distinctDocs?.length ?? 0} docs`)
```

- [ ] **Step 2: Parse-check**

Run the same command as Task 1 Step 2. Expected: `PARSE OK`.

- [ ] **Step 3: Commit**

```bash
git add .claude/workflows/multimodal-sweep.js
git commit --no-gpg-sign -m "feat(workflow): add Phase 0 setup agent + index-ready guard"
```

---

### Task 3: Phase 1 — parallel angle sweep + zero-hit early return

**Files:**
- Modify: `.claude/workflows/multimodal-sweep.js` (append)

- [ ] **Step 1: Append the Phase 1 block**

```javascript

// ---------- Phase 1: sweep (barrier — merge needs all hits to dedupe) ----------
phase('Sweep')
const activeAngles = cfg.angles.filter(a => ANGLE_SPECS[a])
const sweep = (await parallel(
  activeAngles.map(a => () =>
    agent(ANGLE_SPECS[a](setup.retrieverVenv), { label: `angle:${a}`, phase: 'Sweep', schema: ANGLE_SCHEMA })
  )
)).filter(Boolean)

const totalHits = sweep.reduce((n, r) => n + (r.hits?.length ?? 0), 0)
log(`${sweep.length}/${activeAngles.length} angles returned; ${totalHits} hits total`)

if (totalHits === 0) {
  return {
    final_answer: `No hits across any retrieval angle for: "${cfg.question}". The corpus likely does not contain this information.`,
    citations: [],
    confidence: 'low',
    byAngle: sweep,
    verified: [],
    reportMarkdown: `# multimodal-sweep\n\n**Question:** ${cfg.question}\n\nNo hits across any of ${activeAngles.length} retrieval angles.\n`,
    reportPath: cfg.writeReport ? cfg.reportPath : null,
  }
}
```

- [ ] **Step 2: Parse-check** — same command. Expected: `PARSE OK`.

- [ ] **Step 3: Commit**

```bash
git add .claude/workflows/multimodal-sweep.js
git commit --no-gpg-sign -m "feat(workflow): add Phase 1 parallel angle sweep + zero-hit early return"
```

---

### Task 4: Phase 2 — merge / dedupe / flag chart-image-only claims

**Files:**
- Modify: `.claude/workflows/multimodal-sweep.js` (append)

- [ ] **Step 1: Append the Phase 2 block**

```javascript

// ---------- Phase 2: merge ----------
phase('Merge')
const merge = await agent(
  `Merge ${sweep.length} BLIND retrieval-angle results into one draft answer.

QUESTION: ${cfg.question}

ANGLE RESULTS (JSON):
${JSON.stringify(sweep, null, 2)}

Tasks:
1. Dedupe hits by (doc, page, type). For any exact number or directional claim, PREFER text/table hits over chart/image hits.
2. draftAnswer: one paragraph answering the question, citing sources inline as [doc p.N] (1-indexed). Address every entity/year/category the question names, even if some are "not provided".
3. claims_to_verify: every number OR directional claim in draftAnswer that is supported ONLY by a chart- or image-type hit, with NO corroborating text/table hit for the same fact. Each entry = {claim, doc, page}. If none qualify, return [].
4. citations: the (doc, page, type) hits the draft relies on.
5. confidence: overall.`,
  { label: 'merge', phase: 'Merge', schema: MERGE_SCHEMA }
)
log(`merge: ${merge.claims_to_verify?.length ?? 0} chart/image-only claim(s) to verify`)
```

- [ ] **Step 2: Parse-check** — same command. Expected: `PARSE OK`.

- [ ] **Step 3: Commit**

```bash
git add .claude/workflows/multimodal-sweep.js
git commit --no-gpg-sign -m "feat(workflow): add Phase 2 merge/dedupe + claim flagging"
```

---

### Task 5: Phase 3 — adversarial verify + finalize + return

**Files:**
- Modify: `.claude/workflows/multimodal-sweep.js` (append)

- [ ] **Step 1: Append the Phase 3 + return block**

```javascript

// ---------- Phase 3: adversarial verify (conditional) + finalize ----------
phase('Verify')
let verdicts = []
if (cfg.verify && merge.claims_to_verify?.length) {
  verdicts = (await parallel(
    merge.claims_to_verify.map(c => () =>
      agent(
        `Adversarially verify ONE claim against PROSE text. retriever venv: ${setup.retrieverVenv}. Run from ${cfg.repoRoot}.
CLAIM: "${c.claim}" (attributed to ${c.doc}, page ${c.page}, 1-indexed).
Run the targeted prose extract on that PDF, then read its page ${c.page}:
${setup.retrieverVenv}/bin/retriever pdf stage page-elements ${cfg.corpusDir} --method pdfium --json-output-dir /tmp/sweep_verify --compact-json
then inspect /tmp/sweep_verify/${c.doc}.pdf.pdf_extraction.json for page ${c.page}'s prose.
verdict: "confirmed" if prose states the same number/direction; "refuted" if prose contradicts it; "not_found" if prose doesn't mention it. evidence = the relevant verbatim prose snippet, or a note on why not found.`,
        { label: `verify:${c.doc}-p${c.page}`, phase: 'Verify', schema: VERDICT_SCHEMA }
      )
    )
  )).filter(Boolean)
  log(`${verdicts.filter(v => v.verdict === 'confirmed').length}/${verdicts.length} claim(s) confirmed against prose`)
}

const finalize = await agent(
  `Produce the FINAL answer and a markdown report.

QUESTION: ${cfg.question}

DRAFT (merge stage): ${JSON.stringify(merge, null, 2)}
VERIFICATION VERDICTS: ${JSON.stringify(verdicts, null, 2)}
PER-ANGLE RESULTS: ${JSON.stringify(sweep, null, 2)}

Rules:
- Fold verdicts into the answer: "confirmed" -> assert the number/direction confidently; "refuted" or "not_found" -> hedge by quoting the chart phrase verbatim and tagging "(chart-derived, not verified against prose)". NEVER restate a refuted/not_found chart number as fact.
- final_answer: one paragraph with [doc p.N] citations (1-indexed pages).
- citations: the (doc, page, type) hits the final answer relies on.
- confidence: overall ("high"/"medium"/"low").
- reportMarkdown: a complete GitHub-flavored markdown report containing, in order: an H1 title; the question; a "## Answer" section with final_answer; a "## By angle" section with one subsection per angle (its candidateAnswer + a hit table with columns doc | page | type | rank); a "## Verification" table (claim | verdict | evidence) IF any verdicts exist; and a "## Citations" list. Do NOT write any file — just return the markdown string.`,
  { label: 'finalize', phase: 'Verify', schema: FINAL_SCHEMA }
)

return {
  final_answer: finalize.final_answer,
  citations: finalize.citations,
  confidence: finalize.confidence,
  reportMarkdown: finalize.reportMarkdown,
  reportPath: cfg.writeReport ? cfg.reportPath : null,
  byAngle: sweep,
  verified: verdicts,
}
```

- [ ] **Step 2: Parse-check** — same command. Expected: `PARSE OK`.

- [ ] **Step 3: Commit**

```bash
git add .claude/workflows/multimodal-sweep.js
git commit --no-gpg-sign -m "feat(workflow): add Phase 3 verify + finalize + return shape"
```

---

### Task 6: End-to-end smoke run + caller-writes-report + spec note

This task validates the assembled workflow against a real (small) corpus and records the caller-writes-report contract. It requires the `retriever` CLI environment; if that environment is unavailable, do Step 1 (parse-check) and Step 4 (spec note) and mark the live run as blocked rather than skipping silently.

**Files:**
- Modify: `docs/superpowers/specs/2026-06-03-multimodal-sweep-workflow-design.md`

- [ ] **Step 1: Final parse-check of the whole file**

Run the Task 1 Step 2 command. Expected: `PARSE OK`.

- [ ] **Step 2: Pick a tiny corpus and run the workflow**

Identify a small folder of 1–2 PDFs already present in the repo (e.g. under `./data` or `./pdfs`; if none, copy one small PDF into a fresh `./tmp_sweep_pdfs/`). Then invoke the workflow via the **Workflow tool** (not Bash) with args, e.g.:

```
Workflow({ name: 'multimodal-sweep', args: {
  question: '<a question whose answer is in the chosen PDF>',
  corpusDir: './tmp_sweep_pdfs',
  indexDir: './tmp_sweep_lancedb'
}})
```

Expected: the run progresses Setup → Sweep → Merge → Verify and returns an object with non-null `final_answer`, a `citations` array, and a non-empty `reportMarkdown` string. Confirm in `/workflows` that angle agents ran in parallel under the "Sweep" group.

- [ ] **Step 3: Write the report from the returned markdown (caller's job)**

Because the workflow cannot touch the filesystem, after the run returns, the caller writes the report when `reportPath` is non-null:

Use the Write tool to write `result.reportMarkdown` to `result.reportPath` (here `./multimodal-sweep-report.md`). Confirm the file exists and renders the answer + per-angle tables.

- [ ] **Step 4: Record the caller-writes-report contract in the spec**

In `docs/superpowers/specs/2026-06-03-multimodal-sweep-workflow-design.md`, under the `### Output` section, replace the line:

```
If `writeReport`, also write a markdown report (`final_answer`, per-angle hit tables, verification verdicts) to the repo.
```

with:

```
The workflow returns `reportMarkdown` (a full report string) and `reportPath`
(the intended path, or null when `writeReport` is false). Workflow scripts have
no filesystem access, so the **caller writes** `reportMarkdown` to `reportPath`
after the run returns.
```

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-06-03-multimodal-sweep-workflow-design.md
git commit --no-gpg-sign -m "docs(workflow): record caller-writes-report contract; smoke-tested multimodal-sweep"
```

---

## Self-review

**Spec coverage:** Inputs/args → Task 1 `cfg`. Phase 0 setup + barrier → Task 2. Phase 1 five angles + barrier + zero-hit exit → Task 3 (+ angle prompts in Task 1). Phase 2 merge/dedupe/flag → Task 4. Phase 3 conditional verify + finalize → Task 5. Output shape + report-writing → Task 5 return + Task 6. Indexing 1-indexed note → carried in every prompt and the spec. Non-goals respected (no re-ingest logic beyond "ingest if missing"; no eval/scoring; angles don't spawn subagents).

**Placeholder scan:** No TBD/TODO. The only `<...>` tokens are inside agent *prompt* strings where the agent is instructed to fill them at runtime (the regex in `keyword`, the phrasing in `reformulated`, the smoke-test question in Task 6) — these are intentional runtime substitutions, not plan gaps.

**Type consistency:** `HIT_SCHEMA`/`CITATION_SCHEMA` are reused by reference in `ANGLE_SCHEMA`/`MERGE_SCHEMA`/`FINAL_SCHEMA`. `setup.retrieverVenv`, `setup.indexReady`, `setup.docCount`, `setup.distinctDocs` match `SETUP_SCHEMA`. `merge.claims_to_verify`/`merge.draftAnswer` match `MERGE_SCHEMA`. `finalize.final_answer`/`finalize.citations`/`finalize.confidence`/`finalize.reportMarkdown` match `FINAL_SCHEMA`. `ANGLE_SPECS` keys equal the default `cfg.angles` entries, and `activeAngles` filters by `ANGLE_SPECS[a]` so an unknown angle is dropped rather than crashing.
