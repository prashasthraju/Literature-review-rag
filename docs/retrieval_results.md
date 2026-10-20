# Retrieval results: base pipeline, phase 1

This is the record of everything measured for the retrieval pipeline up to 2026-10-19, at the end of the retrieval phase and before answer generation starts. `docs/evaluation.md` holds the methodology and the current numbers; this document holds the history, the experiments behind each decision, and the commands that reproduce them.

## Summary

- The **current base pipeline** (hybrid BGE + BM25, no reranker) scores Recall@5 **0.724** and MRR **0.578**, against **0.615** and **0.569** for the pipeline it replaces.
- With the **`bge-reranker-base` cross-encoder**, which interactive search uses by default, it scores Recall@5 **0.763** and MRR **0.735**.
- The gains come from three changes:
  - the paper title in every passage;
  - BM25 over cleaned text with stopwords;
  - the reranker.
- The chunker rewrite fixes real defects, but on its own it does not raise the scores.
- Scores in this document use **evidence-level labels**, which survive re-chunking. Scores before 2026-10-19 used chunk-ID labels and are not comparable; see [History](#history).
- **Dense retrieval got worse** on the current chunks: Recall@5 0.635 → 0.558, MRR 0.426 → 0.389. The hybrid gain is carried by BM25. This is the first item for the next phase.
- The dev set has **26 questions and known label gaps**. Differences of a few points are noise.

## Corpus

17 papers on avatar generation, talking heads, deepfake detection and avatar fingerprinting, under `papers_avatar/`. They are parsed with Docling into `parsed/` and chunked into `chunks.jsonl`.

| Paper (document_id) | Chunks |
|---|---:|
| DF40 (`df 40`) | 206 |
| AVAPrintDB (`avaprintdb`) | 70 |
| Real3D-Portrait | 60 |
| Multi-Track Timeline Control / STMC (`2401.08559v2`) | 53 |
| InstructAvatar | 53 |
| Avatar Forcing | 52 |
| MotionDiffuse | 52 |
| ARTalk | 46 |
| FFP-ChT deepfake detection | 46 |
| SadTalker | 45 |
| E3 Gen | 33 |
| Micro-expression avatar fingerprinting | 33 |
| FaceCraft4D | 32 |
| Biometric verification in talking-head avatars | 29 |
| GaussianAvatar | 26 |
| Text2Avatar | 18 |
| AniPortrait | 9 |
| **Total** | **863** |

## Final configuration

| Component | Setting |
|---|---|
| Parsing | Docling, OCR off (born-digital PDFs), page provenance kept |
| Chunking | 480-token budget (BGE tokenizer) including the header. One block of overlap. Table rows packed with the caption once. Reference and acknowledgement sections dropped. Paragraphs continue across page breaks. |
| Passage text, all channels | `<paper title>. <section title>. <cleaned text>` (`passage_for_record`) |
| Dense | `BAAI/bge-base-en-v1.5`, cosine, local Qdrant `qdrant_db_token_chunks` / `research_papers_v1` |
| Lexical | BM25 (k1 = 1.5, b = 0.75), stopwords removed |
| Fusion | Reciprocal rank fusion, k = 60, 20 candidates per channel |
| Reranker | `BAAI/bge-reranker-base` over the 20 fused candidates |
| Output | top 5 |

The reranker is on by default in `python -m src.search`; pass `--reranker-model none` to turn it off. The evaluator defaults to **no** reranker, so `reports/retrieval_eval_latest.json` is the base pipeline. Pass `--reranker-model BAAI/bge-reranker-base` to score what search returns.

## How it is measured

- The dataset is `data/evaluation/retrieval_dev.jsonl`: 26 questions over 17 categories, with 42 graded evidence units.
- Each unit is a short quote, or a set of quotes for a table row, that must appear in a chunk of the named paper. Matching ignores case, whitespace, punctuation and accents.
- Recall@5 is the share of evidence units found in the top 5.
- Precision@5 is the share of the top 5 that supports any unit.
- MRR is the reciprocal rank of the first supporting result.
- nDCG@5 rewards each newly covered unit once.
- Details are in `docs/evaluation.md`.

Every score uses the same questions, top 5, 20 candidates and `bge-base-en-v1.5` unless the row says otherwise.

## History

| Date | Commit | Change | Chunks | Score as reported then (chunk-ID labels) | Re-scored on evidence labels |
|---|---|---|---:|---|---|
| 2026-10-11 | `70b8b72` | Docling ingestion (text and multimodal), structure-aware chunking | | | |
| 2026-10-14 | `a0a7f75` | Character-based chunks, Qdrant index, dense search | 973 | | |
| 2026-10-14 | `1dab303` | Evaluation pipeline and 26-question dev set | 973 | | |
| 2026-10-17 | `2921d65` | BM25, RRF hybrid, cross-encoder reranker | 973 | hybrid R 0.750 / MRR 0.591 | hybrid R **0.673** / MRR **0.559** |
| 2026-10-19 | `a0735cf` | Missing paper parsed, page provenance, 480-token chunker | 1,304 | hybrid R 0.583 / MRR 0.540 | hybrid R **0.615** / MRR **0.569** (the baseline below) |
| 2026-10-19 | uncommitted | Retrieval quality pass (this document) | 863 | | hybrid R **0.724** / MRR **0.578**; reranked R **0.763** / MRR **0.735** |

The drop from 0.750 to 0.583 that started this work was mostly an artefact of the labels. Chunk-ID labels were remapped by hand at each re-chunking, so the two numbers were measured against different labels. Scored against the same evidence labels, token chunking moved hybrid recall from 0.673 to 0.615 and MRR from 0.559 to 0.569: a smaller loss in recall and no loss in ranking. The 973-chunk corpus also lacks InstructAvatar, but no question targets that paper.

Reports from before evidence labels, kept as history and **not comparable** with anything above:

| Report | Labels / chunks | Recall@5 | Precision@5 | MRR | nDCG@5 |
|---|---|---:|---:|---:|---:|
| `retrieval_eval_dense.json` | chunk IDs / 973 | 0.6603 | 0.1923 | 0.4391 | 0.4676 |
| `retrieval_eval_bm25.json` | chunk IDs / 973 | 0.6346 | 0.1846 | 0.4731 | 0.4831 |
| `retrieval_eval_hybrid.json` | chunk IDs / 973 | 0.7500 | 0.2231 | 0.5910 | 0.5943 |
| `retrieval_eval_hybrid_complete_corpus.json` | chunk IDs / not recorded (17 papers, with InstructAvatar) | 0.7500 | 0.2231 | 0.5910 | 0.5936 |
| `retrieval_eval_hybrid_reranked.json` | chunk IDs / 973 | 0.7115 | 0.2154 | 0.6538 | 0.5996 |
| `retrieval_eval_hybrid_token_chunks.json` | chunk IDs / 1,304 | 0.5833 | 0.1692 | 0.5397 | 0.5175 |

## Why the old base pipeline scored low

1. **Table rows flooded the results.**
   - The chunker emitted one chunk per table row, with the full caption repeated in each: 421 chunks, 32% of the corpus.
   - All five results for the DF40 question (retrieval-003) were rows of the same table.
2. **Reference lists and empty chunks.**
   - There were 163 reference and acknowledgement chunks.
   - Reference lists are dense with other papers' method names, so they outranked the papers' own text. Chunks that held only an image link added more noise.
3. **Chunks never named their paper.** "The proposed framework comprises two modules…" cannot match a question about AniPortrait (retrieval-009).
4. **BM25 indexed raw text.** This included `file:///` image paths and words such as "what" and "does".
5. **Sections were cut at every page break.** This left 49 chunks ending mid-sentence.
6. **Oversized text was split by decoding uncased tokens,** which lowercased and stripped accents from the stored text.
7. **`search.py` carried its own copy of the pipeline** and read `./qdrant_db` while the indexer wrote `./qdrant_db_token_chunks`.
8. **Chunk-ID labels broke on every re-chunking,** so before and after scores were not comparable.

## Ablation ladder

Each row adds one change to the row above.

| Configuration | Chunks | Recall@5 | Precision@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| Baseline: token chunks, hybrid, raw-text BM25 | 1,304 | 0.6154 | 0.1923 | 0.5686 | 0.5390 |
| + chunker fixes, no title | 863 | 0.6090 | 0.1692 | 0.4994 | 0.4831 |
| + BM25 on cleaned passage text with stopwords | 863 | 0.6474 | 0.1846 | 0.4897 | 0.4995 |
| + paper title in every passage (**current base**) | 863 | **0.7244** | **0.2154** | **0.5782** | **0.5608** |
| + `bge-reranker-base` (**search default**) | 863 | **0.7628** | **0.2462** | **0.7353** | **0.6817** |

## Single channels, before and after

| Channel | Baseline (1,304 chunks, raw BM25) | Current (863 chunks, titled, BM25 with stopwords) |
|---|---|---|
| Dense | R 0.6346 / P 0.1923 / MRR 0.4263 / nDCG 0.4582 | R 0.5577 / P 0.1538 / MRR 0.3891 / nDCG 0.3914 |
| BM25 | R 0.5833 / P 0.1769 / MRR 0.4609 / nDCG 0.4592 | R 0.6923 / P 0.2000 / MRR 0.5442 / nDCG 0.5446 |
| Hybrid | R 0.6154 / P 0.1923 / MRR 0.5686 / nDCG 0.5390 | R 0.7244 / P 0.2154 / MRR 0.5782 / nDCG 0.5608 |

BM25 gained more than ten points of recall. Dense lost about eight points of recall and four of MRR, and is now the weakest channel. The hybrid gain therefore comes from BM25.

Both suspected causes contribute, and neither has been fixed yet:
- **The new chunks.** With no title, dense on the current chunks scores R 0.5962 / MRR 0.3526, below the baseline's 0.6346 / 0.4263. Packed 480-token chunks dilute the embedding of any single fact. With one chunk per table row restored (and titles on), dense recovers to R 0.5962 / MRR 0.4660 ([chunker reverts](#chunker-changes-are-correctness-fixes-not-a-metric-gain)).
- **The title prefix.** Adding the title moves dense from R 0.5962 / MRR 0.3526 to R 0.5577 / MRR 0.3891: it helps ranking but costs recall, likely by pulling every chunk of a paper toward the same vector. The same title is what lifts BM25 and the hybrid, so it should not simply be removed.

## Variants tried and not adopted

| Variant (current chunks) | Recall@5 | Precision@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| `bge-reranker-base`, 50 candidates | 0.7628 | 0.2462 | 0.7276 | 0.6895 |
| `bge-reranker-v2-m3`, 20 candidates | 0.8013 | 0.2615 | 0.6833 | 0.6562 |
| `bge-reranker-v2-m3`, 50 candidates | 0.8077 | 0.2615 | 0.6833 | 0.6616 |
| `bge-large-en-v1.5` embeddings, no reranker | 0.6667 | 0.2077 | 0.5256 | 0.5136 |
| `bge-large-en-v1.5` + `bge-reranker-base` | 0.7500 | 0.2385 | 0.7276 | 0.6807 |

- **`bge-reranker-v2-m3`** gains about four points of recall but loses five of MRR. That is within the noise of 26 questions. `bge-reranker-base` is kept because it is about half the size and faster on CPU.
- **More reranker candidates** change nothing material.
- **`bge-large`** is worse than `bge-base` without a reranker and no better with one.

## Chunker changes are correctness fixes, not a metric gain

Scoring the original 1,304 chunks with the same title header and BM25 isolates the chunker:

| Chunks | Reranker | Recall@5 | Precision@5 | MRR | nDCG@5 |
|---|---|---:|---:|---:|---:|
| Original (1,304) + title | none | 0.7628 | 0.2462 | 0.6429 | 0.6137 |
| Current (863) | none | 0.7244 | 0.2154 | 0.5782 | 0.5608 |
| Original (1,304) + title | `bge-reranker-base` | 0.7756 | 0.2615 | 0.7449 | 0.6898 |
| Current (863) | `bge-reranker-base` | 0.7628 | 0.2462 | 0.7353 | 0.6817 |

- **Without the reranker,** the original chunks score higher. Most of the gap comes from four questions:
  - Exact table lookups 015 and 026 fall from rank 1 to rank 3, because a packed table chunk is less specific than a single row. The reranker restores both.
  - In 019 and 022, the current chunks surface relevant passages that are not labelled.
- **Going the other way,** questions 003, 007, 008 and 012 rank higher on the current chunks.
- **With the reranker,** the two chunkings are level.

Reverting each chunker change on its own did not recover the difference. These runs used a throwaway variant of the chunker, titled passages and the hybrid pipeline without a reranker:

| Revert | Chunks | Recall@5 | Precision@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| None (current) | 863 | 0.7244 | 0.2154 | 0.5782 | 0.5608 |
| Keep content-free chunks | 868 | 0.7244 | 0.2154 | 0.5782 | 0.5608 |
| Keep reference sections | 1,025 | 0.6859 | 0.2077 | 0.5526 | 0.5319 |
| Cut sections at page breaks | 927 | 0.7051 | 0.2077 | 0.5250 | 0.5243 |
| One chunk per table row | 1,092 | 0.6154 | 0.2000 | 0.5545 | 0.5263 |
| All of the above | 1,325 | 0.7500 | 0.2385 | 0.6205 | 0.5929 |

Every single revert scores at or below the current chunks, yet reverting everything scores above them. That points to noise and interaction effects, not one harmful change.

The rewrite is kept for the defects it fixes:
- table flooding;
- lowercased text;
- sentences cut at page breaks;
- reference-list noise;
- Windows-only paths.

It is also kept because search reranks by default. A control run of the old chunker over the re-parsed Markdown scored 0.5962 / 0.1923 / 0.5609 / 0.5298, close to the baseline. So re-parsing seven papers did not move the scores.

## Per question

Recall@5 and MRR for the baseline, the current base pipeline and the reranked pipeline.

| Query | Category | R base­line | R base | R +rerank | MRR base­line | MRR base | MRR +rerank |
|---|---|---:|---:|---:|---:|---:|---:|
| 001 | semantic | 0.50 | 0.50 | 0.50 | 1.00 | 0.50 | 0.33 |
| 002 | paraphrase | 1.00 | 1.00 | 1.00 | 1.00 | 0.33 | 0.50 |
| 003 | exact_dataset_fact | 0.00 | 1.00 | 1.00 | 0.00 | 0.50 | 1.00 |
| 004 | method_comparison | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 |
| 005 | security_risk | 0.67 | 0.67 | 0.67 | 1.00 | 1.00 | 1.00 |
| 006 | table_lookup | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 007 | audio_driven_generation | 1.00 | 0.50 | 0.50 | 0.50 | 0.33 | 1.00 |
| 008 | method_lookup | 1.00 | 1.00 | 1.00 | 0.25 | 0.50 | 0.50 |
| 009 | pipeline_components | 0.00 | 0.50 | 0.50 | 0.00 | 1.00 | 1.00 |
| 010 | method_comparison | 0.50 | 0.50 | 1.00 | 0.50 | 1.00 | 1.00 |
| 011 | exact_dataset_fact | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 012 | method_detail | 0.00 | 1.00 | 1.00 | 0.00 | 0.50 | 0.50 |
| 013 | security_method | 0.00 | 0.00 | 0.50 | 0.00 | 0.00 | 0.20 |
| 014 | method_detail | 0.00 | 0.00 | 0.50 | 0.00 | 0.00 | 1.00 |
| 015 | table_lookup | 1.00 | 1.00 | 1.00 | 1.00 | 0.33 | 1.00 |
| 016 | method_lookup | 1.00 | 1.00 | 1.00 | 1.00 | 0.20 | 0.33 |
| 017 | system_design | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 018 | method_lookup | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 019 | method_detail | 1.00 | 0.00 | 0.00 | 0.20 | 0.00 | 0.00 |
| 020 | security_definition | 0.33 | 0.67 | 0.67 | 0.50 | 0.50 | 0.25 |
| 021 | benchmark_fact | 0.00 | 0.50 | 0.50 | 0.00 | 1.00 | 1.00 |
| 022 | method_detail | 1.00 | 1.00 | 0.00 | 1.00 | 0.50 | 0.00 |
| 023 | exact_dataset_fact | 1.00 | 1.00 | 1.00 | 0.33 | 1.00 | 1.00 |
| 024 | architecture_detail | 0.50 | 1.00 | 0.50 | 1.00 | 0.50 | 0.50 |
| 025 | corpus_fact | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| 026 | exact_table_value | 1.00 | 1.00 | 1.00 | 1.00 | 0.33 | 1.00 |
| **Mean** | | **0.6154** | **0.7244** | **0.7628** | **0.5686** | **0.5782** | **0.7353** |

**Base pipeline compared with the baseline**
- Recall is better on 7 questions and worse on 2 (007 and 019).
- MRR is better on 8 and worse on 9.
- Questions with nothing relevant in the top 5 fall from 7 to 4: 013, 014, 019 and 025.

**Reranker compared with the base pipeline**
- It fixes 013, 014 and 025, and moves 003, 007, 015 and 026 to rank 1.
- It also makes some questions worse:
  - 022 falls from recall 1 to 0;
  - 024 falls from 1 to 0.5;
  - MRR drops on 001 (0.50 → 0.33) and 020 (0.50 → 0.25).
- Questions with nothing relevant in the top 5 fall to 2: 019 and 022, both affected by the label gaps below.

## Known label gaps

The 26-question set is a provisional development seed. Some correct passages are not labelled, so the scores understate quality and can penalise a retriever for finding them:

- **retrieval-003:** DF40's introduction also states the 40-technique count.
- **retrieval-019:** E3 Gen's abstract and conclusion also describe the UV feature plane.
- **retrieval-022:** STMC's method section (3.3) explains the stitching in more detail than the labelled introduction.

The labels have not been edited. Review them against the papers before using the scores to choose between models.

## Reproduce

```bash
pip install -r requirements.txt            # Python 3.12
# parsed/ Markdown, manifests and tables for all 17 papers are in git (images and Docling JSON
# for 7 of them are not). To re-parse a paper: python -m src.ingestion.docparse --no-ocr --pdf <path under papers_avatar/>
python -m src.ingestion.chunker            # writes chunks.jsonl (863 chunks)
python -m src.index                        # rebuilds qdrant_db_token_chunks
python -m src.evaluation.audit_ground_truth
python -m src.evaluation.evaluate_retrieval --retriever dense  --output reports/v2_retrieval_eval_dense.json
python -m src.evaluation.evaluate_retrieval --retriever bm25   --output reports/v2_retrieval_eval_bm25.json
python -m src.evaluation.evaluate_retrieval                    --output reports/v2_retrieval_eval_hybrid.json
python -m src.evaluation.evaluate_retrieval --reranker-model BAAI/bge-reranker-base --output reports/v2_retrieval_eval_hybrid_reranked.json
python -m pytest tests
```

The baseline and ablation rows come from `scripts/ablate_retrieval.py`. It runs the same `Retriever` with an in-memory dense index, which gives the same rankings as Qdrant, and can swap the BM25 input back to the old raw-text configuration. Results are in `reports/ablations/`; the commands are in the script's docstring. Older chunkings come from git:

| Result file | Chunks |
|---|---|
| `baseline_token_chunks_a0735cf.json` | `git show a0735cf:chunks.jsonl`, raw-text BM25 |
| `char_chunks_2921d65.json` | `git show 2921d65:chunks.jsonl`, raw-text BM25 |
| `original_chunks_titled*.json` | `a0735cf` chunks with each paper's `document_title` added (title taken from the current `chunks.jsonl` by `document_id`) |
| `control_old_chunker_new_parse.json` | `a0735cf` chunker run over the current `parsed/` |
| `current_*.json` | current `chunks.jsonl` |
| `chunker_reverts/*.json` | throwaway chunker variants, not kept in the repository |

`scripts/add_evidence_anchors.py` is the one-off migration that turned chunk-ID labels into evidence anchors. It has already been applied.

## Open items for the next phase

1. **Dense regression.** Find out why dense lost recall on the current chunks: smaller chunks or table-aware splitting, a title only in the BM25 passage, or a different embedder. A stronger dense channel is the most likely next retrieval gain.
2. **Label review.** Fix the gaps above, check every exact-value question against the PDFs, and add questions. Twenty-six questions cannot separate changes of a few points.
3. **Figure-dependent questions.** None exist yet, so there is no evidence either way for adding OCR or image descriptions.
4. **Answer generation.** It is not built yet, so there are no measures of answer quality, citations or abstention. Retrieval returns page numbers and paper titles, ready for citations.
