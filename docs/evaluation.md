# Retrieval evaluation

The current evaluation pipeline measures retrieval against chunk-level relevance labels. The default retriever is hybrid: BGE dense search fused with BM25 via reciprocal rank fusion. Pass `--retriever dense` or `--retriever bm25` to score a single channel. It does not evaluate generated answers, citations, abstention, or end-to-end RAG quality because the repository does not yet implement answer generation.

## Run

Build or refresh the local index first when the corpus or chunking changes:

```powershell
python -m src.index
python -m src.evaluation.evaluate_retrieval
python .\src\evaluation\audit_ground_truth.py
```

The evaluator uses `data/evaluation/retrieval_dev.jsonl`, the `research_papers_v1` collection, `chunks.jsonl` for BM25, and the same embedding model and query prefix as `src/search.py`. Document embeddings use cleaned passage text with the section title prepended; image paths and other parser artifacts are stripped before encoding so they do not consume BGE's 512-token window. The evaluator writes the aggregate scores and full ranked passages to `reports/retrieval_eval_latest.json`. The audit command checks that all labeled chunk IDs exist and have positive grades, then prints the associated passages for manual relevance review.

Options include `--dataset`, `--db-path`, `--collection`, `--model`, `--retriever`, `--top-k`, `--candidate-k`, and `--output`. Keep the dataset, model, collection configuration, retriever, and `top_k` fixed when comparing retrieval changes. Save separately named reports for experiments you want to retain.

To evaluate a cross-encoder reranker over the hybrid candidate pool, use:

```powershell
python -m src.evaluation.evaluate_retrieval --retriever hybrid --candidate-k 20 --top-k 5 --reranker-model BAAI/bge-reranker-base --output reports/retrieval_eval_hybrid_reranked.json
```

The first run downloads the reranker model if it is not cached. The report records the reranker model and its scores. The interactive search supports the same optional model with `python -m src.search --reranker-model BAAI/bge-reranker-base`. Compare reranked and non-reranked reports on the same benchmark; keep in mind the current relevance labels are provisional and the reranker may overfit them if repeatedly tuned against this small set.

## Dataset labels

Each JSONL item has a stable `query_id`, `split`, `category`, `question`, answer points for future answer evaluation, and a `relevant_chunk_ids` object mapping chunk IDs to graded relevance. Grade 0 means not relevant; higher positive grades indicate stronger relevance. Chunk-level relevance measures whether the required evidence was retrieved, which is stricter than checking only whether the right paper appeared.

The included 26-question set is a **provisional development seed**, not a validated gold set or held-out test set. It now spans motion generation, audio-driven portraits, avatar fingerprinting, deepfake detection, dataset facts, method details, and table lookups. Review each question and its labels against the source paper, add relevant chunks missed by the initial pass, and expand coverage before using the scores to make model or product decisions. Add an unanswerable question only when answer-generation and abstention metrics are available; retrieval-only scores cannot determine whether an answer should be refused.

## Current retrieval comparison

On 2026-10-17, the 26-query development set had 44 positive chunk-level relevance labels. The local Qdrant collection was rebuilt using the current cleaned passage embeddings before running each retriever at `k=5` with a candidate depth of 20.

| Retriever | Recall@5 | Precision@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| Dense | 0.6603 | 0.1923 | 0.4391 | 0.4676 |
| BM25 | 0.6346 | 0.1846 | 0.4731 | 0.4831 |
| Hybrid (RRF) | **0.7500** | **0.2231** | **0.5910** | **0.5943** |

Hybrid is the strongest of these initial runs on all four aggregate metrics. It still retrieves none of the judged chunks for `retrieval-009` (AniPortrait pipeline components) and `retrieval-014` (ARTalk low-latency method). The labels remain provisional, so inspect misses and relevance judgments before tuning on this development set. Full ranked outputs are saved in `reports/retrieval_eval_dense.json`, `reports/retrieval_eval_bm25.json`, `reports/retrieval_eval_hybrid.json`, and the latest default run in `reports/retrieval_eval_latest.json`.

## Metrics

- `Recall@k`: fraction of judged relevant chunks present in the first `k` results. It measures evidence coverage.
- `Precision@k`: fraction of the first `k` results judged relevant, with denominator `k` even when fewer than `k` results are returned.
- `MRR`: reciprocal rank of the first relevant chunk, averaged across queries. It rewards placing at least one useful passage near the top.
- `nDCG@k`: rank-sensitive gain using graded chunk relevance, normalized by the ideal ordering for that query.

Aggregate values are macro-averages over answerable labeled queries. Queries with no positive relevance labels are undefined for recall and nDCG and must be handled by task-specific abstention evaluation. No target thresholds are set until a reviewed baseline exists.
