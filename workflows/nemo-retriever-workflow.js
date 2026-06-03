export const meta = {
  name: 'nemo-retriever-workflow',
  description: 'Answer one question over a nemo-retriever corpus by sweeping multiple blind retrieval angles in parallel, then adversarially verifying chart/image-only claims against prose',
  phases: [
    { title: 'Setup', detail: 'resolve retriever venv + ensure LanceDB index exists' },
    { title: 'Sweep', detail: 'parallel angle agents, one disciplined query each' },
    { title: 'Merge', detail: 'dedupe hits, draft answer, flag chart/image-only claims' },
    { title: 'Verify', detail: 'adversarially check flagged claims against prose, then finalize' },
  ],
}

// ---------- config / args ----------
// args may arrive as an object OR as a JSON-encoded string (the harness can
// stringify the Workflow tool's `args` input) — normalize both into an object.
const A = typeof args === 'string' ? JSON.parse(args) : (args ?? {})
const cfg = {
  question:    A.question,
  corpusDir:   A.corpusDir   ?? './pdfs',
  indexDir:    A.indexDir    ?? './lancedb',
  tableName:   A.tableName   ?? 'nv-ingest',
  topK:        A.topK        ?? 10,
  embedModel:  A.embedModel  ?? 'nvidia/llama-nemotron-embed-1b-v2',
  angles:      A.angles      ?? ['semantic', 'reformulated', 'keyword', 'visual', 'tabular'],
  verify:      A.verify      ?? true,
  writeReport: A.writeReport ?? true,
  grepScript:  A.grepScript  ?? 'skills/nemo-retriever/scripts/grep_corpus.py',
  reportPath:  A.reportPath  ?? './nemo-retriever-workflow-report.md',
  repoRoot:    A.repoRoot    ?? '/home/edwardk/git/nv-ingest',
  ocrLang:     A.ocrLang     ?? 'english',
  installExtras: A.installExtras ?? false,
}
if (!cfg.question) throw new Error('nemo-retriever-workflow: args.question is required (pass {question: "..."} as the workflow args)')

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
    verdict: { type: 'string', enum: ['confirmed', 'refuted', 'not_found', 'unverifiable'] },
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
// Schema-tolerant: the installed retriever query CLI emits lean hits
// {page_number, source, text}; other versions add {pdf_basename, metadata.type,
// _distance}. Prompts derive doc from pdf_basename-or-source and infer type from
// the producing angle when the CLI omits it. All queries target the index
// explicitly (--table-name/--lancedb-uri) instead of relying on CLI defaults.
const baseContext = (venv) => `You are ONE retrieval angle in a multi-angle sweep answering this question over a PRE-BUILT nemo-retriever LanceDB corpus.

QUESTION: ${cfg.question}

retriever venv: ${venv} (use ${venv}/bin/retriever and ${venv}/bin/python). Run from ${cfg.repoRoot}.
ALWAYS target this index explicitly by passing --table-name ${cfg.tableName} --lancedb-uri ${cfg.indexDir} to retriever query (and grep_corpus.py). Do NOT rely on CLI defaults — the default table name differs. The index is ALREADY BUILT — NEVER ingest or re-extract.
DISCIPLINE: at most 2 Bash calls; no narration between calls; do NOT spawn subagents; go straight from your command to returning structured output.
The installed CLI emits lean hits: each hit is JSON with at least "page_number" (1-indexed int), "source" (a file path), and "text". It MAY also carry "pdf_basename" and "metadata.type" on other versions.
For each hit you return:
- doc = the "pdf_basename" field if present, else the basename of "source" with any ".pdf" suffix stripped (e.g. /a/b/foo.pdf -> foo).
- page = "page_number" as-is (1-indexed).
- rank = position in the returned list, starting at 1.
- type = the hit's metadata.type if present, otherwise infer from YOUR angle (stated below).
confidence reflects how well the hits actually answer the question.`

const ANGLE_SPECS = {
  semantic: (venv) => `${baseContext(venv)}

YOUR ANGLE = "semantic": straight semantic search with reranking. Tag every hit type="text". Run exactly this single pipeline:
${venv}/bin/retriever query "${cfg.question}" --top-k ${cfg.topK} --rerank --table-name ${cfg.tableName} --lancedb-uri ${cfg.indexDir} --embed-model-name ${cfg.embedModel} | tee /tmp/sweep_semantic.json | ${venv}/bin/python -c "import json,sys;[print(f'rank={i+1} page={h.get(\\"page_number\\")} src={h.get(\\"source\\",\\"\\")}') for i,h in enumerate(json.load(sys.stdin))]"
Then read the hit text you need from /tmp/sweep_semantic.json and synthesize.`,

  reformulated: (venv) => `${baseContext(venv)}

YOUR ANGLE = "reformulated": semantic search is phrasing-sensitive. Tag every hit type="text". Rephrase the question into 2-3 alternatives (one keyword-dense; one HyDE-style: a single hypothetical SENTENCE that, if present in a doc, would answer it). Run one query per phrasing (combine into a single Bash command to stay within budget), union the hits, dedupe by (source,page), report the best. Use:
${venv}/bin/retriever query "<phrasing>" --top-k ${cfg.topK} --rerank --table-name ${cfg.tableName} --lancedb-uri ${cfg.indexDir} --embed-model-name ${cfg.embedModel}`,

  keyword: (venv) => `${baseContext(venv)}

YOUR ANGLE = "keyword": exact-term matches semantic search may miss. Extract the key identifiers/terms/numbers from the question, build a regex, and run:
${venv}/bin/python ${cfg.grepScript} "<regex>" --lancedb-uri ${cfg.indexDir} --table-name ${cfg.tableName} --max-hits 50
Output is "<source>:p<page>:<type>:  ...snippet..." per line, or NO_MATCH. Map those lines to hits, ranked by order; use the printed <type> for each hit's type. If NO_MATCH, return empty hits with confidence "low".`,

  visual: (venv) => `${baseContext(venv)}

YOUR ANGLE = "visual": facts hidden in figures (charts/images). Filter to chart/image content SERVER-SIDE with --content-types:
${venv}/bin/retriever query "${cfg.question}" --top-k ${cfg.topK} --rerank --content-types chart,image --table-name ${cfg.tableName} --lancedb-uri ${cfg.indexDir} --embed-model-name ${cfg.embedModel} | tee /tmp/sweep_visual.json
Every returned hit is a chart or image: tag type="chart" (or "image" if its text reads like a photo/picture caption rather than a chart). Their text is a model caption that MAY misread numbers/directions — report it but set confidence "low" for any exact number. If the CLI rejects --content-types (older build), rerun without it and keep only hits whose metadata.type is chart or image.`,

  tabular: (venv) => `${baseContext(venv)}

YOUR ANGLE = "tabular": facts in tables. Filter to table content SERVER-SIDE with --content-types:
${venv}/bin/retriever query "${cfg.question}" --top-k ${cfg.topK} --rerank --content-types table --table-name ${cfg.tableName} --lancedb-uri ${cfg.indexDir} --embed-model-name ${cfg.embedModel} | tee /tmp/sweep_tabular.json
Every returned hit is a table: tag type="table" and report the rows relevant to the question. If the CLI rejects --content-types (older build), rerun without it and keep only hits whose metadata.type == "table".`,
}

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
    reportMarkdown: `# nemo-retriever-workflow\n\n**Question:** ${cfg.question}\n\nNo hits across any of ${activeAngles.length} retrieval angles.\n`,
    reportPath: cfg.writeReport ? cfg.reportPath : null,
  }
}

// ---------- Phase 2: merge ----------
phase('Merge')
const merge = await agent(
  `Merge ${sweep.length} BLIND retrieval-angle results into one draft answer.

QUESTION: ${cfg.question}

ANGLE RESULTS (JSON):
${JSON.stringify(sweep, null, 2)}

Each result carries its "angle": chart/image evidence comes from the "visual" angle and tables from "tabular"; "semantic"/"reformulated"/"keyword" return prose-or-mixed hits. Note: some corpora store a figure's caption AS a text chunk too, so the SAME chart number may also surface as a text-tagged hit — that is the chart's own caption, NOT independent corroboration.

Tasks:
1. Dedupe hits by (doc, page, type). For any exact number or directional claim, PREFER genuine prose/table evidence over chart/image captions.
2. draftAnswer: one paragraph answering the question, citing sources inline as [doc p.N] (1-indexed). Address every entity/year/category the question names, even if some are "not provided".
3. claims_to_verify: every number OR directional claim in draftAnswer whose evidence traces to a chart- or image-type hit (typically the "visual" angle), UNLESS a DISTINCT prose passage — not the same figure's caption restated as text — independently states the same fact. Each entry = {claim, doc, page}. If none qualify, return [].
4. citations: the (doc, page, type) hits the draft relies on.
5. confidence: overall.`,
  { label: 'merge', phase: 'Merge', schema: MERGE_SCHEMA }
)
log(`merge: ${merge.claims_to_verify?.length ?? 0} chart/image-only claim(s) to verify`)

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
