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
- Parsed documents are non-empty and chunk counts are explainable; chunk IDs in relevance labels resolve to current chunks.
- Page citations are populated and traceable wherever Docling provides page provenance.
- Chunk lengths stay within the configured model token budget, aside from explicitly handled atomic structures such as a single oversized table row.
- Each retrieval change is compared on the same corpus snapshot, candidate depth, and reviewed evaluation labels; scores and known limitations are recorded.

## Completed work and results

- Added single-PDF parsing in `src/ingestion/docparse.py` and parsed the missing `text_driven_generation/Instructavatar.pdf`; all 17 source PDFs now have manifests and Markdown.
- Added page-start/page-break markers from Docling provenance and propagated `page_number`/`page_numbers` into chunks and the Qdrant payload.
- Replaced character-only splitting with a 480-token budget (including the section prefix), table caption/header context, and row splitting. The chunker run reported zero chunks above the budget.
- Current corpus audit: 17 PDFs, 17 manifests, 1,304 chunks across 17 documents; zero empty chunks, zero chunks missing page numbers, zero multi-page chunks, and zero label IDs missing from the corpus. There are 38 chunks shorter than 200 characters; these were not removed automatically.
- Rebuilt the local index into ignored `qdrant_db_token_chunks/` rather than reusing the stale local store. Verified exactly 1,304 unique indexed points and page metadata on every point. The original `qdrant_db/` remains untouched.
- Final hybrid retrieval run (`top_k=5`, candidate depth 20, 26 queries): Recall@5 0.5833, Precision@5 0.1692, MRR 0.5397, nDCG@5 0.5175. Saved in `reports/retrieval_eval_hybrid_token_chunks.json`.
- Table lookups retrieval-015 and retrieval-026 each returned their labeled relevant chunk at rank 1. This supports deferring visual table/image enrichment for now; the current benchmark does not establish figure-only coverage.
- Earlier complete-corpus metrics (Recall@5 0.7500, Precision@5 0.2231, MRR 0.5910, nDCG@5 0.5936) are not a clean before/after comparison because the chunk IDs and relevance labels were remapped during chunking. The current judgments also remain provisional.

## Remaining follow-up

- Manually review the provisional judgments, especially exact table-value questions, against source PDFs before using these metrics for strategy selection.
- Add representative figure-dependent questions before deciding whether OCR/image descriptions are worthwhile; otherwise keep visual enrichment deferred.
- Keep corpus artifacts and local databases out of Git; no commit or push was made as part of this work.
