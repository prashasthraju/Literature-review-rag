# Ingestion and chunking improvement plan

## Goal

Improve corpus completeness, evidence traceability, and retrieval-ready chunks without silently changing source papers or treating provisional evaluation labels as gold truth.

## Baseline findings

- The original snapshot had 17 PDFs, 16 parsed manifests, and 973 chunks; the missing `Instructavatar.pdf` manifest was not recorded as a failure.
- The original chunks had no page provenance and used a character limit. That snapshot is historical; the current implementation and counts are recorded below.
- Docling exports figures and tables, but this pipeline uses their extracted Markdown text rather than visual OCR or image descriptions.
- The 26-query retrieval development set has provisional judgments. Scores are directional, not gold-standard quality claims.

## Phases

1. **Corpus coverage and ingestion QA** — compare source PDFs, manifests, parsed Markdown, chunks, and indexed documents; report missing, duplicate, empty, or malformed artifacts. Re-parse missing sources and make future runs fail visibly when inputs are skipped.
2. **Page provenance** — preserve Docling page provenance through Markdown extraction and chunk formation; verify each chunk’s page references against the source document. Keep page references optional for genuinely multi-page chunks.
3. **Chunk quality** — replace character-only limits with a model-token budget; retain section hierarchy and useful overlap; keep table captions, headers, and associated rows together; filter or de-prioritize low-value references and boilerplate only after review.
4. **Evaluation and selection** — review/update affected relevance labels, compare the existing baseline against one change at a time, and retain the versioned reports. Do not select a strategy based on the provisional development set alone.
5. **Figure/table enrichment** — only if representative questions require evidence not present in text, add caption-aware table records and OCR or image descriptions, then evaluate their effect separately.

## Acceptance checks

- Every PDF under `papers_avatar/` is accounted for by a successful manifest or an explicit failure record.
- Parsed documents are non-empty and chunk counts are explainable; every evidence anchor in the relevance labels resolves to a current chunk (`python -m src.evaluation.audit_ground_truth`). Chunk-ID labels are kept for history only.
- Page citations are populated and traceable wherever Docling provides page provenance.
- Chunk lengths stay within the configured model token budget, aside from explicitly handled atomic structures such as a single oversized table row.
- Each retrieval change is compared on the same corpus snapshot, candidate depth, and reviewed evaluation labels; scores and known limitations are recorded.

## Completed work and results (2026-10-19, before the retrieval quality pass)

This section records the state after token-aware chunking. The corpus counts, label check and scores below are historical; the current state is in the follow-up section and in `docs/retrieval_results.md`.

- Added single-PDF parsing in `src/ingestion/docparse.py` and parsed the missing `text_driven_generation/Instructavatar.pdf`; all 17 source PDFs now have manifests and Markdown.
- Added page-start/page-break markers from Docling provenance and propagated `page_number`/`page_numbers` into chunks and the Qdrant payload.
- Replaced character-only splitting with a 480-token budget (including the section prefix), table caption/header context, and row splitting. The chunker run reported zero chunks above the budget.
- Corpus audit at that point: 17 PDFs, 17 manifests, 1,304 chunks across 17 documents; zero empty chunks, zero chunks missing page numbers, zero multi-page chunks, and zero label IDs missing from the corpus. There are 38 chunks shorter than 200 characters; these were not removed automatically.
- Rebuilt the local index into ignored `qdrant_db_token_chunks/` rather than reusing the stale local store. Verified exactly 1,304 unique indexed points and page metadata on every point. The original `qdrant_db/` remains untouched.
- Final hybrid retrieval run (`top_k=5`, candidate depth 20, 26 queries): Recall@5 0.5833, Precision@5 0.1692, MRR 0.5397, nDCG@5 0.5175. Saved in `reports/retrieval_eval_hybrid_token_chunks.json`.
- Table lookups retrieval-015 and retrieval-026 each returned their labeled relevant chunk at rank 1. This supports deferring visual table/image enrichment for now; the current benchmark does not establish figure-only coverage.
- Earlier complete-corpus metrics (Recall@5 0.7500, Precision@5 0.2231, MRR 0.5910, nDCG@5 0.5936) are not a clean before/after comparison because the chunk IDs and relevance labels were remapped during chunking. The current judgments also remain provisional.

## Remaining follow-up

- Manually review the provisional judgments, especially exact table-value questions, against source PDFs before using these metrics for strategy selection.
- Add representative figure-dependent questions before deciding whether OCR/image descriptions are worthwhile; otherwise keep visual enrichment deferred.
- Keep corpus artifacts and local databases out of Git; no commit or push was made as part of this work.

## Follow-up: retrieval quality pass (2026-10-19)

- Hybrid Recall@5 had dropped to 0.5833 (chunk-ID labels) after token-aware chunking. Diagnosis: one-chunk-per-table-row output (32% of chunks, caption repeated in each) flooded the top 5; reference lists (163 chunks) and image-only chunks added noise; chunks never named their paper; sections were cut at every page break.
- The chunker now packs table rows, drops reference/acknowledgement sections and content-free chunks, continues paragraphs across page breaks (multi-page chunks list every page in `page_numbers`), records `document_title`, splits oversized text at character offsets, and resolves Windows manifest paths on Linux. The corpus is 863 chunks across 17 documents, none above the 480-token budget.
- The 7 papers whose parsed Markdown was not in the repository were re-parsed with `python -m src.ingestion.docparse --no-ocr --pdf ...`; the previous chunker reproduces 1,246 of the original 1,304 chunks from the new parse.
- Relevance labels are now evidence anchors that survive re-chunking (see `docs/evaluation.md`). Base hybrid Recall@5 went from 0.6154 to 0.7244 (evidence-level metrics), and 0.7628 with reranking; MRR from 0.5686 to 0.5782, and 0.7353 with reranking. Most of that gain comes from the paper-title header, the BM25 cleanup and the reranker; the chunker changes are correctness fixes and score level with the original chunking once reranking is on (details in `docs/evaluation.md`).
- Full results, timeline and reproduction commands: `docs/retrieval_results.md`.

## Follow-up: front-matter filter (2026-10-20)

- The chunker now drops publisher and author front matter:
  - permission and copyright notices;
  - ACM ISBN and DOI lines;
  - author contact, funding and e-mail blocks;
  - unpunctuated author and affiliation lines under the title.
- It also skips author names that Docling parsed as headings (a person-name heading followed by an e-mail), which had split MotionDiffuse's abstract into sections titled with author names.
- The corpus is now 847 chunks.
- No evidence anchor was lost.
- The effect on retrieval is in `docs/retrieval_results.md` (phase 2).
