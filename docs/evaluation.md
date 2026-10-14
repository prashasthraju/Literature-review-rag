# Retrieval evaluation

The current evaluation pipeline measures dense retrieval against chunk-level relevance labels. It does not evaluate generated answers, citations, abstention, or end-to-end RAG quality because the repository does not yet implement answer generation.

## Run

Build or refresh the local index first when the corpus or chunking changes:

```powershell
python -m src.index
python -m src.evaluation.evaluate_retrieval
python .\src\evaluation\audit_ground_truth.py
```

The evaluator uses `data/evaluation/retrieval_dev.jsonl`, the `research_papers_v1` collection, and the same embedding model and query prefix as `src/search.py`. It writes the aggregate scores and full ranked passages to `reports/retrieval_eval_latest.json`. The audit command checks that all labeled chunk IDs exist and have positive grades, then prints the associated passages for manual relevance review.

Options include `--dataset`, `--db-path`, `--collection`, `--model`, `--top-k`, and `--output`. Keep the dataset, model, collection configuration, and `top_k` fixed when comparing retrieval changes. Save separately named reports for experiments you want to retain.

## Dataset labels

Each JSONL item has a stable `query_id`, `split`, `category`, `question`, answer points for future answer evaluation, and a `relevant_chunk_ids` object mapping chunk IDs to graded relevance. Grade 0 means not relevant; higher positive grades indicate stronger relevance. Chunk-level relevance measures whether the required evidence was retrieved, which is stricter than checking only whether the right paper appeared.

The included 26-question set is a **provisional development seed**, not a validated gold set or held-out test set. It now spans motion generation, audio-driven portraits, avatar fingerprinting, deepfake detection, dataset facts, method details, and table lookups. Review each question and its labels against the source paper, add relevant chunks missed by the initial pass, and expand coverage before using the scores to make model or product decisions. Add an unanswerable question only when answer-generation and abstention metrics are available; retrieval-only scores cannot determine whether an answer should be refused.

## Metrics

- `Recall@k`: fraction of judged relevant chunks present in the first `k` results. It measures evidence coverage.
- `Precision@k`: fraction of the first `k` results judged relevant, with denominator `k` even when fewer than `k` results are returned.
- `MRR`: reciprocal rank of the first relevant chunk, averaged across queries. It rewards placing at least one useful passage near the top.
- `nDCG@k`: rank-sensitive gain using graded chunk relevance, normalized by the ideal ordering for that query.

Aggregate values are macro-averages over answerable labeled queries. Queries with no positive relevance labels are undefined for recall and nDCG and must be handled by task-specific abstention evaluation. No target thresholds are set until a reviewed baseline exists.
