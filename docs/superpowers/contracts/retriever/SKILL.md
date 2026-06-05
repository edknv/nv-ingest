---
name: retriever
description: Answer, quote, verify, or aggregate over a document corpus (PDF,
  image, Office, HTML, TXT, audio, video). Use for any multi-file or non-text
  question instead of native Read/Grep.
---

# retriever — reasoning over retrieved evidence

You have one tool: `retrieve(question)` → cited, fidelity-tagged evidence.
(Corpus not indexed? `index(path)` once.) You never build queries, choose
strategies, or parse output — you reason about what comes back.

## 1. Pick the move
- fact / number / date → retrieve; read the top evidence
- "list / count / every / across" → aggregate; do not sample
- exact quote → quote verbatim with its citation
- compare across docs → retrieve per entity, then contrast
- image / chart / audio / video → evidence is a transcription; treat per §2

## 2. Trust by fidelity  ← the core skill
verbatim > ocr > table > vlm_caption. A number or directional claim resting
ONLY on a `vlm_caption` (chart/image) is unconfirmed: call
`verify(claim, source, locator)`; assert confidently only if an independent
higher-fidelity passage corroborates, else quote it and tag "(chart-derived,
unconfirmed)". Never upgrade a low-fidelity reading to a confident fact.

## 3. Answer honestly
- Cite source + locator for every claim.
- Re-read the question: address every entity / year / category — even "not provided".
- If the answer isn't in the evidence, say so. Never fabricate from adjacent text.
- Use `coverage` to know if a thin/empty result means "broaden" vs "out-of-corpus".

## 4. When retrieval falls short
exact-term miss → broaden / rephrase; nothing relevant → likely out-of-corpus,
say so; `coverage` flags a stale or partial index → re-`index`.
