# Retrieval results

This is the record of everything measured for the retrieval pipeline. Phase 1 (2026-10-19, tag `retrieval-v1`) fixed the base pipeline on a 26-question dev set; phase 2 (2026-10-20, [below](#phase-2-benchmark-v2-and-retrieval-v2)) built a larger benchmark with a held-out test set and produced `retrieval-v2`. `docs/evaluation.md` holds the methodology and the current numbers; this document holds the history, the experiments behind each decision, and the commands that reproduce them.

## Phase 1 summary

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

## Phase 1 configuration

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

The phase 1 numbers are reproduced exactly by `scripts/run_experiments.py`, which runs the same `Retriever` with an in-memory dense index and named configs (`configs/retrieval/baseline-v0.json` reproduces the raw-text BM25 of the old pipeline). Phase 1 used the v1.1 dataset and the 863-chunk corpus, both at the `retrieval-v1` tag:

```bash
git show retrieval-v1:data/evaluation/retrieval_dev.jsonl > /tmp/dev_v1.1.jsonl
git show retrieval-v1:chunks.jsonl > /tmp/chunks_v1.jsonl
git show a0735cf:chunks.jsonl > /tmp/chunks_a0735cf.jsonl
python scripts/run_experiments.py baseline-v0 --dataset /tmp/dev_v1.1.jsonl --chunks /tmp/chunks_a0735cf.jsonl   # 0.6154 / 0.5686
python scripts/run_experiments.py retrieval-v1-base retrieval-v1 --dataset /tmp/dev_v1.1.jsonl --chunks /tmp/chunks_v1.jsonl  # 0.7244 / 0.5782 and 0.7628 / 0.7353
```

Phase 1 ablation outputs are kept in `reports/ablations/`:

| Result file | Chunks |
|---|---|
| `baseline_token_chunks_a0735cf.json` | `git show a0735cf:chunks.jsonl`, raw-text BM25 |
| `char_chunks_2921d65.json` | `git show 2921d65:chunks.jsonl`, raw-text BM25 |
| `original_chunks_titled*.json` | `a0735cf` chunks with each paper's `document_title` added (title taken from the current `chunks.jsonl` by `document_id`) |
| `control_old_chunker_new_parse.json` | `a0735cf` chunker run over the current `parsed/` |
| `current_*.json` | the 863-chunk corpus of phase 1 |
| `chunker_reverts/*.json` | throwaway chunker variants, not kept in the repository |

`scripts/add_evidence_anchors.py` is the one-off migration that turned chunk-ID labels into evidence anchors. It has already been applied.

## Phase 1 open items, and what phase 2 did with them

1. **Dense regression:** fixed. A different embedder (`bge-m3`) over 64-word windows takes dense-only Recall@5 from 0.667 to 0.818 on the larger dev set (phase 2 below).
2. **Label review:** partly done. The three known gaps were fixed, and a retriever-independent search for restated facts added alternatives to 23 units. The questions are still model-drafted and need a human review.
3. **Figure-dependent questions:** still open; benchmark v2 has none.
4. **Answer generation:** built in phase 2; see `docs/generation.md`.

## Phase 2: benchmark v2 and retrieval-v2

Phase 2 (2026-10-20) replaced the 26-question set with a benchmark large enough to separate configurations, kept a held-out test set aside, fixed the dense regression and the front-matter noise, and settled the configuration with a pre-registered decision rule.

### Benchmark v2

| Split | File | Questions | Answerable | Evidence units | Unanswerable |
|---|---|---:|---:|---:|---:|
| Dev | `data/evaluation/retrieval_dev.jsonl` | 96 | 89 | 135 | 7 |
| Test (held out) | `data/evaluation/retrieval_test.jsonl` | 61 | 54 | 80 | 7 |

- **The 26 phase 1 questions** stay in dev.
- **117 new questions** were written from the parsed papers (not from chunks or retrieval output), 4–9 per paper, covering 17 categories. They include table and exact-value questions and 7 cross-paper questions.
  - The wording avoids the anchor sentences: the build script rejected any question sharing 5 or more consecutive words with its anchor.
  - Within each paper, questions alternate between dev and test.
- **14 unanswerable questions** were checked absent from the corpus with a regular expression; they are split 7/7.
- **Label corrections.**
  - **Alternatives.** A unit can list `alternatives`: other passages, in the same paper or another, that state the same fact. Any one of them covers the unit, without adding a unit.
  - **Known gaps.** The phase 1 gaps (003, 019, 022) were fixed this way.
  - **Search for restatements.** A search found passages that contain an anchor's numbers or at least 70% of its content words. It did not use any retriever. Reviewing its output added alternatives to 23 units in dev and test. Two items whose units restated one fact (020, 021) were merged into one unit each.
- **Who wrote it.** Every new question is marked `model-drafted`: the same assistant that tuned the pipeline wrote both splits. The labels need a human review before the scores are treated as more than directional.
- **Test file.** SHA-256 at the single test run: `da01cba608afd89c345931883b7e00982e37e2712f0adfa9975e1cb10801d03b`.

### Decision rule (fixed before the experiments)

- Configurations are compared per question with a paired bootstrap (10,000 resamples) on Recall@k, MRR and nDCG@k (`scripts/compare_runs.py`).
- A change is adopted only if the 95% interval for Recall or MRR excludes zero and the other metric does not get significantly worse. Otherwise the simpler or cheaper option stays.
- Tuning used dev only. Test was run once, at the end, for the reference and final configurations together.

### Front-matter filter (ingestion fix)

- **The problem.** Front matter took result slots in the error analysis:
  - copyright and ACM permission notices;
  - author lists, affiliations and e-mail blocks;
  - funding notes;
  - MotionDiffuse's author names, parsed as headings that split its abstract into sections.
- **The fix.** The chunker now drops these blocks, and skips author-name headings that are followed by an e-mail. The corpus goes from 863 to 847 chunks, and no anchor was lost.
- **The effect.** On dev, base MRR rose 0.627 → 0.645 and the bge-m3 windows hybrid's recall 0.863 → 0.875; with the reranker there was no change. It is kept as a correctness fix.

### Dense channel and embedders (screening on dev)

Screening runs used the dev set before the label corrections and front-matter filter, so compare rows within this table only.

| Dense channel alone | Recall@5 | MRR |
|---|---:|---:|
| `bge-base-en-v1.5`, whole chunks, titled (v1) | 0.620 | 0.466 |
| `bge-base`, whole chunks, no title | 0.556 | 0.354 |
| `bge-base`, 96-word windows | 0.626 | 0.478 |
| `bge-base`, 64-word windows | 0.702 | 0.564 |
| `bge-base`, 160-word windows | 0.633 | 0.469 |
| `bge-m3`, whole chunks | 0.738 | 0.632 |
| `bge-m3`, 64-word windows | 0.764 | 0.622 |
| `snowflake-arctic-embed-m-v1.5`, whole chunks | 0.760 | 0.579 |
| `snowflake-arctic-embed-m-v1.5`, 64-word windows | 0.768 | 0.590 |

- **Windows.** Each chunk is split into overlapping 64-word windows, with table rows kept whole. Each window is embedded with the paper and section header, and a chunk scores as its best window. BM25, the reranker and the generator still see whole chunks.
- **The title helps dense on this set** (0.620 vs 0.556). The phase 1 finding that it hurt recall came from 26 questions and does not hold up.
- **Embedders.** `bge-m3` and `arctic` are both far stronger than `bge-base`.

| Hybrid (screening) | Recall@5 | MRR | p50 latency |
|---|---:|---:|---:|
| v1-base (`bge-base` chunks) | 0.832 | 0.627 | 6 ms |
| `bge-base` 64-word windows | 0.811 | 0.662 | 5 ms |
| `bge-m3` chunks | 0.817 | 0.707 | 11 ms |
| `bge-m3` 64-word windows | 0.863 | 0.705 | 12 ms |
| `arctic` 64-word windows | 0.876 | 0.674 | 6 ms |
| `bge-m3` windows, dense weight 1.5 in RRF | 0.852 | 0.710 | 11 ms |
| v1 (`bge-base` + `bge-reranker-base`) | 0.848 | 0.743 | 305 ms |
| `bge-m3` windows + `bge-reranker-base` | 0.860 | 0.737 | 316 ms |
| `bge-m3` windows + `bge-reranker-v2-m3` | 0.882 | 0.720 | 1,124 ms |
| `arctic` windows + `bge-reranker-base` | 0.854 | 0.739 | 310 ms |
| `arctic` windows + `bge-reranker-v2-m3` | 0.873 | 0.716 | 1,131 ms |

- **The reranker evens out first stages.** Once results are reranked, every first stage lands within a point of v1 at top 5.
- **`bge-reranker-v2-m3`** trades MRR for recall again, at almost four times the latency. It is not adopted.

### Final dev results (corrected labels, 847-chunk corpus)

| Config | Recall@5 | MRR | nDCG@5 | Recall@8 | p50 latency (Qdrant) |
|---|---:|---:|---:|---:|---:|
| BM25 alone | 0.841 | 0.607 | 0.650 | | |
| Dense alone, v1 (`bge-base`) | 0.667 | 0.490 | 0.510 | | |
| Dense alone, v2 (`bge-m3` windows) | 0.818 | 0.659 | 0.678 | | |
| `retrieval-v1-base` | 0.863 | 0.654 | 0.685 | 0.920 | 8 ms |
| **`retrieval-v2-base`** | **0.918** | **0.732** | **0.760** | 0.942 | 28 ms |
| `retrieval-v1` | 0.880 | 0.751 | 0.761 | 0.938 | 310 ms |
| **`retrieval-v2`** | **0.886** | **0.750** | **0.759** | **0.961** | 330 ms |

Paired bootstrap (dev, 89 questions):
- **`retrieval-v2-base` vs `retrieval-v1-base`:**
  - Recall@5 +0.054 [+0.006, +0.107]
  - MRR +0.072 [+0.005, +0.138]
  - nDCG@5 +0.071 [+0.022, +0.121]
  - All three are significant.
- **`retrieval-v2` vs `retrieval-v1`:**
  - Recall@5 and MRR are tied (±0.01).
  - Recall@8 is +0.022 [0.000, +0.051], with 3 questions better and 0 worse.

The Qdrant collection reproduces the in-memory scores exactly for v1. For v2-base it gives MRR 0.732 against 0.726 in memory, because batch order changes the bge-m3 encodings slightly and near-ties flip.

### Held-out test (single run)

| Config | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|
| `retrieval-v1-base` | 0.667 | 0.519 | 0.559 |
| **`retrieval-v2-base`** | **0.769** | **0.607** | **0.644** |
| `retrieval-v1` | 0.759 | 0.638 | 0.658 |
| **`retrieval-v2`** | **0.769** | **0.639** | **0.661** |

- **The test questions are harder than dev.** Most are detail and exact-value questions that alternate with the dev questions within each paper.
- **The ordering is the same as on dev:**
  - v2-base beats v1-base significantly: Recall +0.102 [+0.009, +0.204], MRR +0.088 [+0.006, +0.174].
  - v2 and v1 tie.
  - v2-base is roughly level with v1 at a tenth of the latency.

### Decision

- **`retrieval-v2` is the new default** for search and for the generator's context. It uses:
  - `bge-m3` over 64-word windows;
  - titled BM25 with stopwords;
  - RRF with k = 60 and 20 candidates;
  - `bge-reranker-base`.
- **Why adopt it when it only ties v1 after reranking.** Its first stage is significantly better on both dev and test. That matters in fast mode (`retrieval-v2-base`, about 28 ms). It also matters for the 8 chunks the generator reads, where v2 finds 96% of the evidence against 94%. With reranking it is no worse than v1 anywhere.
- **The cost:** a larger embedder (568M parameters, against 110M) and about 5.6 times as many vectors (4,760 window points for 847 chunks).
- **Rollback:** `retrieval-v1` remains a config with its own Qdrant collection.

### Reproduce phase 2

```bash
python -m src.ingestion.chunker                       # 847 chunks
python -m src.index --config retrieval-v1             # collection research_papers_v1
python -m src.index --config retrieval-v2             # collection research_papers_v2 (window points)
python -m src.evaluation.audit_ground_truth --dataset data/evaluation/retrieval_dev.jsonl
python -m src.evaluation.audit_ground_truth --dataset data/evaluation/retrieval_test.jsonl
python -m src.evaluation.evaluate_retrieval --config retrieval-v2 --output reports/v3_retrieval_eval_dev_retrieval-v2.json
python -m src.evaluation.evaluate_retrieval --config retrieval-v2 --dataset data/evaluation/retrieval_test.jsonl \
    --output reports/v3_retrieval_eval_test_retrieval-v2.json
# experiments: any config with inline overrides, in-memory dense index, encodings cached in .cache/
python scripts/run_experiments.py retrieval-v1 "retrieval-v1-base:embedding_model=BAAI/bge-m3,query_prefix=,dense_units=window,window_words=64,window_stride=32"
python scripts/compare_runs.py reports/experiments/retrieval_dev/retrieval-v1.json reports/experiments/retrieval_dev/<candidate>.json --categories
```

Per-query experiment outputs are in `reports/experiments/retrieval_dev/`. Full ranked outputs for the four reference configurations are in `reports/v3_retrieval_eval_{dev,test}_*.json`.

### Remaining retrieval misses

On dev, `retrieval-v2` misses some evidence at 8 for 6 questions (retrieval-001, 007, 014, 024, v2-microexp-03, v2-df40-01). Five are partial misses, where a secondary unit of a two-part question is not found. The full miss is a table caption that uses the abbreviations FS/FR/EFS/FE while the question spells them out (v2-df40-01). None points to another systematic fault like the front matter.
