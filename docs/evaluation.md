# Retrieval evaluation

The evaluation pipeline measures retrieval against evidence-level relevance labels. The default retriever is hybrid: BGE dense search fused with BM25 via reciprocal rank fusion (RRF), optionally reranked by a cross-encoder. Pass `--retriever dense` or `--retriever bm25` to score a single channel. It does not evaluate generated answers, citations, abstention, or end-to-end RAG quality because the repository does not yet implement answer generation.

## Run

Set up the environment (Python 3.12) and rebuild chunks and the index whenever parsing or chunking changes:

```bash
pip install -r requirements.txt
python -m src.ingestion.chunker
python -m src.index
python -m src.evaluation.evaluate_retrieval
python -m src.evaluation.audit_ground_truth
```

`src/retrieval.py` holds the only implementation of the retrieval pipeline; `src/search.py` and the evaluator both call it, so the scores describe exactly what interactive search does. Every channel sees the same passage text for a chunk (`passage_for_record` in `src/embed_text.py`): the paper title, the section title, and the cleaned chunk text with image paths and other parser artifacts removed.

Options include `--dataset`, `--chunks`, `--db-path`, `--collection`, `--model`, `--retriever`, `--top-k`, `--candidate-k`, `--reranker-model`, `--dense-backend`, and `--output`. `--dense-backend memory` encodes `--chunks` on the fly instead of reading Qdrant, which lets you score a candidate chunking without rebuilding the index; it produces the same rankings as the Qdrant backend. Keep the dataset, model, retriever, and `top_k` fixed when comparing retrieval changes, and save separately named reports for experiments you want to retain.

To evaluate the cross-encoder reranker over the hybrid candidate pool:

```bash
python -m src.evaluation.evaluate_retrieval --reranker-model BAAI/bge-reranker-base --output reports/v2_retrieval_eval_hybrid_reranked.json
```

Interactive search (`python -m src.search`) reranks with `BAAI/bge-reranker-base` by default; pass `--reranker-model none` to disable it.

## Dataset labels

Each JSONL item has a stable `query_id`, `split`, `category`, `question`, answer points for future answer evaluation, and `evidence`: a list of graded evidence units. Each unit names a `document_id`, a `grade` (higher is more relevant), and an `anchor`: a short quote from the paper, or a list of quotes that must all occur in the same chunk (used for table rows). A chunk supports a unit when it is from that document and contains every anchor after normalization (case, whitespace, punctuation and accents are ignored).

Evidence anchors replace chunk-ID labels as the source of truth. Chunk IDs change whenever the chunker changes, so every chunking experiment previously required a manual label remap, and before/after scores were not comparable. Anchors survive re-chunking; the evaluator and `audit_ground_truth` report any anchor that no longer matches a chunk. Anchors were written from the 44 chunk-ID labels before any chunking change. On the original chunking, every anchor matches its source chunk; five (retrieval-005, 013, 015, 020, 022) also match a neighbouring chunk that repeats the same paragraph through block overlap, which is correct because that chunk contains the evidence. Two pairs of labels that were the same passage split across overlapping chunks (retrieval-006 and 019) became one unit each, so the 44 labels are 42 evidence units. `relevant_chunk_ids` is kept for history; the evaluator reports legacy chunk-ID metrics only when all of those IDs exist in the current corpus.

The 26-question set is a **provisional development seed**, not a validated gold set or held-out test set. Known gaps: several queries have relevant passages that are not labeled. For example, DF40's introduction also states the 40-technique count (retrieval-003), E3 Gen's abstract and conclusion describe the UV feature plane (retrieval-019), and STMC's method section (3.3) explains the stitching in more detail than the labeled introduction (retrieval-022). Scores therefore underestimate retrieval quality, and they penalize a retriever for finding unlabeled evidence. Review labels against the source papers and expand coverage before using the scores for model or product decisions. Add an unanswerable question only when answer-generation and abstention metrics are available.

## Metrics

Metrics are computed per evidence unit, so splitting a passage into two chunks, or the same paragraph appearing in two overlapping chunks, does not change the denominator.

- `Recall@k`: fraction of evidence units covered by at least one of the first `k` results.
- `Precision@k`: fraction of the first `k` results that support any evidence unit, with denominator `k`. Most queries have only one or two evidence units, usually in one or two chunks each, so precision@5 stays low even for a perfect ranking; recall, MRR and nDCG are the informative numbers.
- `MRR`: reciprocal rank of the first result that supports any evidence unit, averaged across queries.
- `nDCG@k`: graded gain for each newly covered unit; re-retrieving an already covered unit earns nothing, so near-duplicate chunks are not rewarded.

Aggregate values are macro-averages over the labeled queries. No target thresholds are set until a reviewed baseline exists.

## Retrieval comparison (2026-10-19)

The complete record of results, including the history before evidence labels, per-query tables and reproduction commands, is in [`retrieval_results.md`](retrieval_results.md).

Evidence-level metrics, 26 queries / 42 evidence units, `top_k=5`, candidate depth 20, `BAAI/bge-base-en-v1.5`. Each row adds one change to the row above.

| Configuration | Chunks | Recall@5 | Precision@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| Baseline: token chunks, hybrid (raw-text BM25) | 1,304 | 0.6154 | 0.1923 | 0.5686 | 0.5390 |
| + chunker fixes (see below), no title header | 863 | 0.6090 | 0.1692 | 0.4994 | 0.4831 |
| + BM25 on cleaned passage text with stopwords | 863 | 0.6474 | 0.1846 | 0.4897 | 0.4995 |
| + paper title in every passage (**current base pipeline**) | 863 | **0.7244** | **0.2154** | **0.5782** | **0.5608** |
| + `bge-reranker-base` reranking (**interactive search default**) | 863 | **0.7628** | **0.2462** | **0.7353** | **0.6817** |

Single channels on the current chunks: dense 0.5577 / 0.1538 / 0.3891 / 0.3914; BM25 0.6923 / 0.2000 / 0.5442 / 0.5446 (same column order). Dense retrieval is the weakest channel on this set.

Other variants tried on the current chunks, not adopted:

| Variant | Recall@5 | Precision@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| `bge-reranker-base`, candidate depth 50 | 0.7628 | 0.2462 | 0.7276 | 0.6895 |
| `bge-reranker-v2-m3`, candidate depth 20 | 0.8013 | 0.2615 | 0.6833 | 0.6562 |
| `bge-reranker-v2-m3`, candidate depth 50 | 0.8077 | 0.2615 | 0.6833 | 0.6616 |
| `bge-large-en-v1.5` embeddings, no reranker | 0.6667 | 0.2077 | 0.5256 | 0.5136 |
| `bge-large-en-v1.5` embeddings + `bge-reranker-base` | 0.7500 | 0.2385 | 0.7276 | 0.6807 |

**Chunker changes are correctness fixes, not a metric gain.** The original 1,304 chunks with the same title header and BM25 score 0.7628 / 0.2462 / 0.6429 / 0.6137 without reranking, above the current chunks, and 0.7756 / 0.2615 / 0.7449 / 0.6898 with `bge-reranker-base`, level with them. Per query, the base-pipeline gap comes from exact table lookups (retrieval-015 and 026 drop from rank 1 to rank 3 because a packed table chunk is less specific than a single row; the reranker restores both to rank 1) and from retrieval-019 and 022, where the current chunks surface unlabeled relevant passages. In the other direction, retrieval-003, 007, 008 and 012 rank higher on the current chunks. Reverting each chunker change on its own did not recover the difference, which indicates noise and interactions rather than one harmful change. The rewrite is kept for what it fixes (table flooding, lowercased text, sentences cut at page breaks, reference-list noise, Linux paths), and because reranking is on by default in search.

The two rerankers trade recall against MRR by amounts that are within the noise of 26 queries; `bge-reranker-base` is kept because it is half the size. A control that ran the previous chunker over the same re-parsed Markdown scored 0.5962 / 0.1923 / 0.5609 / 0.5298, close to the baseline, so the gains come from the pipeline changes rather than from re-parsing.

Per-query against the baseline: the base pipeline improves Recall@5 on 7 queries and lowers it on 2, with MRR better on 8 and worse on 9; the reranked pipeline improves Recall@5 on 10 (worse on 3) and MRR on 12 (worse on 7). Queries with zero recall fell from 7 (retrieval-003, 009, 012, 013, 014, 021, 025) to 4 for the base pipeline and 2 for the reranked pipeline (retrieval-019 and 022, both affected by the label gaps above). Treat differences of a few points as noise on a set this small, and do not tune RRF `k` or channel weights against it.

Full ranked outputs are saved in `reports/v2_retrieval_eval_{dense,bm25,hybrid,hybrid_reranked}.json`; `reports/retrieval_eval_latest.json` is the base hybrid run. Reports without the `v2_` prefix predate evidence-level labels and use chunk-ID metrics on earlier chunkings.

### What changed in the pipeline

- **Table rows are packed into chunks.** Each table row used to be its own chunk with the full caption repeated, which made up 32% of the corpus; one table could fill the whole top 5 (all five results for retrieval-003 were rows of DF40's Table 1). Rows are now packed up to the token budget with the caption once per chunk.
- **Reference lists and acknowledgements are dropped** (163 chunks), along with chunks that hold only an image reference. Reference lists are dense with other papers' method names and outranked the papers' own content.
- **Paragraphs continue across page breaks.** Sections were previously cut at every page break, leaving 49 chunks ending mid-sentence; chunks now record every page they span in `page_numbers` (`page_number` is set only for single-page chunks).
- **Every passage carries the paper title.** Chunks such as AniPortrait's "The proposed framework comprises two modules..." never name the paper, so queries naming the method could not match them (retrieval-009).
- **BM25 indexes the same cleaned, titled passage as the embedder** and ignores stopwords; it previously indexed raw text including image file paths, and question words such as "what" and "does" added score.
- **Oversized text is split at character offsets** instead of decoding uncased BGE tokens, which had lowercased and de-accented the stored text.
- **One retrieval implementation.** `search.py` had its own copy of the pipeline and read `./qdrant_db` while the indexer wrote `./qdrant_db_token_chunks`.
