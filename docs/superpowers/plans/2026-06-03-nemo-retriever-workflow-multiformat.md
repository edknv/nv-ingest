# Multi-format Support for nemo-retriever-workflow — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `workflows/nemo-retriever-workflow.js` so it ingests and verifies non-PDF corpora (image, Office, HTML/TXT, audio, video), not just PDFs.

**Architecture:** All changes are in the single orchestration file. Phase 0's setup agent becomes format-aware (inventory extensions → one ingest pass per bucket, first `--overwrite` then `--append`, skip+report missing-dep buckets). Phase 3's verify agent branches on source type (PDF → pdfium prose re-extract; non-PDF → index re-query for corroborating text/table). Schemas gain `ingestedTypes`/`skippedTypes` and an `unverifiable` verdict; prompts tolerate audio/video segment "pages".

**Tech Stack:** Claude Code Workflow JS runtime; `retriever ingest`/`query` CLI (`--input-type`, `--overwrite`/`--append`, `--content-types`); LanceDB.

---

## Validation note

Workflow scripts use top-level `await`/`return`, so plain `node --check` rejects them. After every edit, run this wrapper parse-check (compiles the body inside an async function without executing):

```bash
node -e 'const fs=require("fs");let s=fs.readFileSync("workflows/nemo-retriever-workflow.js","utf8").replace(/^export\s+const\s+meta/m,"const meta");new Function("agent","parallel","pipeline","phase","log","args","budget","workflow","return (async()=>{"+s+"})()");console.log("PARSE OK")'
```

All commits use `--no-gpg-sign` (GPG signing fails in this environment). The file is reachable both as `workflows/nemo-retriever-workflow.js` (canonical) and via the `.claude/workflows` symlink — always edit the canonical path.

## File structure

- **Modify only:** `workflows/nemo-retriever-workflow.js`. No new files — this is a focused extension of one orchestration unit.

---

### Task 1: Config + schema changes

**Files:**
- Modify: `workflows/nemo-retriever-workflow.js`

- [ ] **Step 1: Add `ocrLang` and `installExtras` to `cfg`**

Replace:
```javascript
  reportPath:  A.reportPath  ?? './nemo-retriever-workflow-report.md',
  repoRoot:    A.repoRoot    ?? '/home/edwardk/git/nv-ingest',
}
```
with:
```javascript
  reportPath:  A.reportPath  ?? './nemo-retriever-workflow-report.md',
  repoRoot:    A.repoRoot    ?? '/home/edwardk/git/nv-ingest',
  ocrLang:     A.ocrLang     ?? 'english',
  installExtras: A.installExtras ?? false,
}
```

- [ ] **Step 2: Extend `SETUP_SCHEMA` with `ingestedTypes` / `skippedTypes`**

Replace the whole `SETUP_SCHEMA` block:
```javascript
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
```
with:
```javascript
const SETUP_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['retrieverVenv', 'indexReady', 'docCount', 'distinctDocs', 'ingestedTypes', 'skippedTypes'],
  properties: {
    retrieverVenv: { type: 'string', description: 'absolute venv root containing bin/retriever' },
    indexReady: { type: 'boolean' },
    docCount: { type: 'number' },
    distinctDocs: { type: 'array', items: { type: 'string' } },
    ingestedTypes: { type: 'array', items: { type: 'string' }, description: 'format buckets ingested this run; [] if the index was reused' },
    skippedTypes: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['type', 'reason'],
        properties: { type: { type: 'string' }, reason: { type: 'string' } },
      },
      description: 'format buckets skipped because host deps were missing',
    },
    note: { type: 'string' },
  },
}
```

- [ ] **Step 3: Add `unverifiable` to the `VERDICT_SCHEMA` enum**

Replace:
```javascript
    verdict: { type: 'string', enum: ['confirmed', 'refuted', 'not_found'] },
```
with:
```javascript
    verdict: { type: 'string', enum: ['confirmed', 'refuted', 'not_found', 'unverifiable'] },
```

- [ ] **Step 4: Parse-check** — run the wrapper command from the Validation note. Expected: `PARSE OK`.

- [ ] **Step 5: Commit**
```bash
git add workflows/nemo-retriever-workflow.js
git commit --no-gpg-sign -m "feat(workflow): config + schema for multi-format (ocrLang, installExtras, ingested/skipped types, unverifiable verdict)"
```

---

### Task 2: Format-aware multi-pass Phase 0 ingest

> **SUPERSEDED during implementation (live finding):** the installed `retriever ingest` has **no `--input-type` flag** and auto-ingests a mixed-format folder in a **single pass** (verified: pdf+txt+png+docx+wav → one table). The multi-pass/`--append` approach below was replaced by a **single ingest pass** + `source`-JSON parsing for `distinctDocs` + an ingested-vs-skipped inventory diff. See the spec's Phase 0 section and commit "fix(workflow): single-pass ingest" for the shipped version. The steps below are kept for history.

**Files:**
- Modify: `workflows/nemo-retriever-workflow.js`

- [ ] **Step 1: Replace the Phase 0 setup agent prompt**

Replace the entire `const setup = await agent( ... )` call (the block starting `const setup = await agent(` and ending at the matching `)` before the `if (!setup ...` guard):
```javascript
const setup = await agent(
  `Prepare the nemo-retriever index for a multi-angle query sweep. Run from ${cfg.repoRoot}.
1. Resolve the retriever venv: run \`command -v retriever\`. If it prints .../bin/retriever, the venv ROOT is the directory two levels up (strip /bin/retriever). If retriever is NOT found, follow ${cfg.repoRoot}/skills/nemo-retriever/references/install.md to install it, then resolve the path.
2. Check whether the index exists at ${cfg.indexDir}/${cfg.tableName}.lance. If it does NOT exist, ingest the corpus: \`<venv>/bin/retriever ingest ${cfg.corpusDir}/ --embed-model-name ${cfg.embedModel}\`. Build an index even if the corpus is large (a text-only index beats none; see ${cfg.repoRoot}/skills/nemo-retriever/references/setup.md for the >800-page recipe).
3. Report: retrieverVenv (the venv root), indexReady (true once the .lance table exists/queryable), docCount, and distinctDocs. Get counts via:
   \`<venv>/bin/python -c "import lancedb; df=lancedb.connect('${cfg.indexDir}').open_table('${cfg.tableName}').to_pandas(); print(len(df)); import json; print(json.dumps(sorted(df.pdf_basename.unique().tolist())))"\``,
  { label: 'setup', phase: 'Setup', schema: SETUP_SCHEMA }
)
```
with:
```javascript
const setup = await agent(
  `Prepare the nemo-retriever index for a multi-angle query sweep. Run from ${cfg.repoRoot}.

1. Resolve the retriever venv: run \`command -v retriever\`. If it prints .../bin/retriever, the venv ROOT is the directory two levels up (strip /bin/retriever). If retriever is NOT found, follow ${cfg.repoRoot}/skills/nemo-retriever/references/install.md to install it, then resolve the path.

2. If the table ALREADY exists at ${cfg.indexDir}/${cfg.tableName}.lance, REUSE it — do NOT ingest, and set ingestedTypes=[]. Otherwise build it with FORMAT-AWARE MULTI-PASS ingest:
   a. Inventory file extensions in ${cfg.corpusDir}: \`find ${cfg.corpusDir} -type f -name '*.*' | sed 's/.*\\.//' | tr 'A-Z' 'a-z' | sort -u\`
   b. Group extensions into buckets (skip empty buckets):
      - default (.pdf .html .txt): NO --input-type flag
      - image (.jpg .jpeg .png .tiff .bmp): --input-type image --ocr-version v2 --ocr-lang ${cfg.ocrLang}
      - doc (.docx .pptx): --input-type doc          [needs libreoffice host pkg]
      - audio (.mp3 .wav .m4a): --input-type audio    [needs the [multimedia] extra + ffmpeg]
      - video (.mp4 .mov .mkv): --input-type video    [needs the [multimedia] extra + ffmpeg]
   c. Run ONE ingest pass per non-empty bucket, ALL into the same table. The FIRST pass uses --overwrite (the default; creates a fresh table). EVERY SUBSEQUENT pass MUST add --append (adds rows, no dup-check) so earlier passes are not clobbered. Base form:
      \`<venv>/bin/retriever ingest ${cfg.corpusDir}/ --table-name ${cfg.tableName} --lancedb-uri ${cfg.indexDir} --embed-model-name ${cfg.embedModel} <bucket flags> [--append for passes after the first]\`
      An explicit --input-type processes only that format's files in the folder, so all passes can point at the same ${cfg.corpusDir}.
   d. If a bucket needs a host dep that is missing: ${cfg.installExtras ? 'attempt to install it per references/install.md, then ingest that bucket' : 'do NOT install anything — skip that bucket'}. For every skipped bucket add {type, reason} to skippedTypes. NEVER silently drop a format. Record each successfully-ingested bucket in ingestedTypes.

3. Report retrieverVenv, indexReady (true once the table is queryable), docCount, distinctDocs, ingestedTypes, skippedTypes. Get counts SCHEMA-TOLERANTLY (the table may have a "source" column instead of "pdf_basename"):
   \`<venv>/bin/python -c "import lancedb,os,json; df=lancedb.connect('${cfg.indexDir}').open_table('${cfg.tableName}').to_pandas(); col='pdf_basename' if 'pdf_basename' in df.columns else 'source'; print(len(df)); print(json.dumps(sorted({os.path.basename(str(x)) for x in df[col].tolist()})))"\``,
  { label: 'setup', phase: 'Setup', schema: SETUP_SCHEMA }
)
```

- [ ] **Step 2: Parse-check** — run the wrapper command. Expected: `PARSE OK`.

- [ ] **Step 3: Verify the `installExtras` ternary renders both ways**

Run:
```bash
node -e '
const fs=require("fs");const src=fs.readFileSync("workflows/nemo-retriever-workflow.js","utf8");
const m=src.match(/2\. If the table ALREADY[\s\S]*?Record each successfully-ingested bucket in ingestedTypes\./);
const cfg={installExtras:false}; console.log("installExtras=false ->", /do NOT install anything/.test(eval("`"+m[0].replace(/`/g,"\\`")+"`"))?"skip text OK":"MISSING");
const cfg2={installExtras:true};
'
```
Expected: prints `installExtras=false -> skip text OK` (confirms the embedded `${cfg.installExtras ? ...}` ternary is syntactically valid and selects the skip branch). If it errors, the ternary text was mis-transcribed.

- [ ] **Step 4: Commit**
```bash
git add workflows/nemo-retriever-workflow.js
git commit --no-gpg-sign -m "feat(workflow): format-aware multi-pass ingest in Phase 0 (overwrite then append; skip+report missing deps; schema-tolerant counts)"
```

---

### Task 3: Audio/video citation tolerance + format-aware verify

**Files:**
- Modify: `workflows/nemo-retriever-workflow.js`

- [ ] **Step 1: Make the `baseContext` page bullet tolerate A/V segments**

Replace:
```javascript
- page = "page_number" as-is (1-indexed).
```
with:
```javascript
- page = "page_number" as-is (1-indexed; for audio/video this may instead be a segment index or timestamp — report it verbatim).
```

- [ ] **Step 2: Replace the Phase 3 verify agent prompt (format-aware branching)**

Replace:
```javascript
        `Adversarially verify ONE claim against PROSE text. retriever venv: ${setup.retrieverVenv}. Run from ${cfg.repoRoot}.
CLAIM: "${c.claim}" (attributed to ${c.doc}, page ${c.page}, 1-indexed).
Run the targeted prose extract on that PDF, then read its page ${c.page}:
${setup.retrieverVenv}/bin/retriever pdf stage page-elements ${cfg.corpusDir} --method pdfium --json-output-dir /tmp/sweep_verify --compact-json
then inspect /tmp/sweep_verify/${c.doc}.pdf.pdf_extraction.json for page ${c.page}'s prose.
verdict: "confirmed" if prose states the same number/direction; "refuted" if prose contradicts it; "not_found" if prose doesn't mention it. evidence = the relevant verbatim prose snippet, or a note on why not found.`,
```
with:
```javascript
        `Adversarially verify ONE claim against an INDEPENDENT modality. retriever venv: ${setup.retrieverVenv}. Run from ${cfg.repoRoot}.
CLAIM: "${c.claim}" (attributed to ${c.doc}, page/segment ${c.page}).
Pick the check by the source file type of "${c.doc}":
- PDF source: run the targeted prose re-extract, then read page ${c.page}:
  ${setup.retrieverVenv}/bin/retriever pdf stage page-elements ${cfg.corpusDir} --method pdfium --json-output-dir /tmp/sweep_verify --compact-json
  then inspect /tmp/sweep_verify/${c.doc}.pdf.pdf_extraction.json for page ${c.page}'s prose.
- Non-PDF source (image / office / audio / video): re-query the index for corroborating TEXT/TABLE content and check whether a hit whose "source" matches "${c.doc}" states the same number/direction in genuine (non-caption) text:
  ${setup.retrieverVenv}/bin/retriever query "${c.claim}" --top-k 10 --rerank --content-types text,table --table-name ${cfg.tableName} --lancedb-uri ${cfg.indexDir} --embed-model-name ${cfg.embedModel}
verdict: "confirmed" if an independent modality states the same number/direction; "refuted" if it contradicts; "not_found" if that modality is present but silent on the claim; "unverifiable" if NO independent modality exists for this source (e.g. a number only in an image with no transcript). evidence = the relevant verbatim snippet, or why none.`,
```

- [ ] **Step 3: Update the finalize prompt to fold in `unverifiable`**

Replace:
```javascript
- Fold verdicts into the answer: "confirmed" -> assert the number/direction confidently; "refuted" or "not_found" -> hedge by quoting the chart phrase verbatim and tagging "(chart-derived, not verified against prose)". NEVER restate a refuted/not_found chart number as fact.
```
with:
```javascript
- Fold verdicts into the answer: "confirmed" -> assert the number/direction confidently; "refuted" / "not_found" / "unverifiable" -> hedge by quoting the source phrase verbatim and tagging "(derived from chart/image, not confirmed against an independent modality)". NEVER restate a refuted/not_found/unverifiable number as a confident fact.
```

- [ ] **Step 4: Parse-check** — run the wrapper command. Expected: `PARSE OK`.

- [ ] **Step 5: Commit**
```bash
git add workflows/nemo-retriever-workflow.js
git commit --no-gpg-sign -m "feat(workflow): A/V segment citations + format-aware verify (pdfium for PDF, index re-query for non-PDF, unverifiable verdict)"
```

---

### Task 4: Regression smoke test + multi-format dep check

**Files:**
- Modify: none (validation only)

- [ ] **Step 1: Final parse-check** — run the wrapper command. Expected: `PARSE OK`.

- [ ] **Step 2: Confirm no PDF regression (live run)**

Recreate the single-PDF corpus and run the workflow against the existing index via the **Workflow tool** (the harness passes `args` as a JSON string; the script already parses it):
```bash
mkdir -p tmp_sweep_pdfs && cp data/multimodal_test.pdf tmp_sweep_pdfs/
```
Then `Workflow({ scriptPath: "/home/edwardk/git/nv-ingest/workflows/nemo-retriever-workflow.js", args: { question: "In Chart 1, which gadget is most expensive and its approximate cost?", corpusDir: "./tmp_sweep_pdfs", indexDir: "./lancedb", tableName: "nv-ingest" } })`.
Expected: completes with non-null `final_answer`, and `setup` now returns `ingestedTypes: []` (index reused) — confirming the new SETUP_SCHEMA fields don't break the reuse path. Write `reportMarkdown` to a file with the Write tool, then `rm -rf tmp_sweep_pdfs`.

- [ ] **Step 3: Record which non-PDF formats are live-testable here**

Run:
```bash
command -v libreoffice >/dev/null && echo "libreoffice: yes" || echo "libreoffice: NO (doc bucket not live-testable)"
command -v ffmpeg >/dev/null && echo "ffmpeg: yes" || echo "ffmpeg: NO (audio/video buckets not live-testable)"
/home/edwardk/git/nv-ingest/retriever/bin/python -c "import importlib.util as u; print('multimedia extra:', 'yes' if u.find_spec('nemo_retriever_extraction') else 'unknown')" 2>/dev/null || echo "multimedia extra: unknown"
ls data/*.png data/*.jpg data/*.docx data/*.pptx data/*.mp3 data/*.mp4 2>/dev/null || echo "(no non-PDF samples in ./data)"
```
Record the results. If a format's host deps are present AND a sample file exists, optionally run a fresh-index multi-format live test in a temp dir (`indexDir: ./tmp_mf_lancedb`, a temp corpus mixing a PDF + that format) and confirm `ingestedTypes` lists both buckets. If deps are absent, **state that the non-PDF live path is unverified here and rely on parse-check + the design** — do not claim it works without evidence.

- [ ] **Step 4: Commit any report artifact decision / cleanup** (no code commit if Tasks 1-3 already committed). Remove temp dirs:
```bash
rm -rf tmp_sweep_pdfs tmp_mf_lancedb tmp_mf_pdfs 2>/dev/null; true
```

---

## Self-review

**Spec coverage:**
- Inputs `ocrLang`/`installExtras` → Task 1 Step 1. ✓
- `SETUP_SCHEMA` ingestedTypes/skippedTypes → Task 1 Step 2. ✓
- `VERDICT` `unverifiable` → Task 1 Step 3. ✓
- Phase 0 format-aware multi-pass (inventory → buckets → overwrite-then-append → skip+report) → Task 2. ✓
- Schema-tolerant counts (the `pdf_basename`→`source` fix) → Task 2 Step 1. ✓ (also corrects a latent bug in the old prompt)
- A/V segment "page" → Task 3 Step 1. ✓
- Format-aware verify (PDF pdfium / non-PDF re-query / unverifiable) → Task 3 Step 2. ✓
- Finalize folds unverifiable → Task 3 Step 3. ✓
- Multi-format matrix is a documentation artifact in the spec; no code task needed.

**Placeholder scan:** No TBD/TODO. `<venv>`, `<bucket flags>`, `<regex>`, `<phrasing>` appear only inside agent *prompt* strings as runtime substitutions the agent fills — intentional, not plan gaps.

**Type consistency:** `ingestedTypes` (string[]) and `skippedTypes` ([{type,reason}]) names match between SETUP_SCHEMA (Task 1) and the Phase 0 prompt instructions (Task 2). `unverifiable` matches between VERDICT_SCHEMA (Task 1), the verify prompt (Task 3 Step 2), and the finalize prompt (Task 3 Step 3). `--overwrite`/`--append` and `--content-types text,table` match the verified CLI flags.
