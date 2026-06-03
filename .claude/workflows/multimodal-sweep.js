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
${venv}/bin/retriever query "${cfg.question}" --top-k ${cfg.topK} --rerank --embed-model-name ${cfg.embedModel} | tee /tmp/sweep_semantic.json | ${venv}/bin/python -c "import json,sys;[print(f'rank={h.get(\"rank\",0)} page={h[\"page_number\"]} pdf={h[\"pdf_basename\"]} type={h.get(\"metadata\",{}).get(\"type\",\"?\")}') for h in json.load(sys.stdin)]"
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
    reportMarkdown: `# multimodal-sweep\n\n**Question:** ${cfg.question}\n\nNo hits across any of ${activeAngles.length} retrieval angles.\n`,
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

Tasks:
1. Dedupe hits by (doc, page, type). For any exact number or directional claim, PREFER text/table hits over chart/image hits.
2. draftAnswer: one paragraph answering the question, citing sources inline as [doc p.N] (1-indexed). Address every entity/year/category the question names, even if some are "not provided".
3. claims_to_verify: every number OR directional claim in draftAnswer that is supported ONLY by a chart- or image-type hit, with NO corroborating text/table hit for the same fact. Each entry = {claim, doc, page}. If none qualify, return [].
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
