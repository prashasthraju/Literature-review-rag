# Evaluation

The evaluation measures retrieval against evidence-level relevance labels, context assembly by evidence coverage, and generated answers by abstention, citation validity and citation coverage. Results and the experiments behind each decision are in [`retrieval_results.md`](retrieval_results.md) (retrieval) and [`generation.md`](generation.md) (context and answers).

## Run

Set up the environment (Python 3.12), rebuild chunks and the index whenever parsing or chunking changes, then evaluate:

```bash
pip install -r requirements.txt
python -m src.ingestion.chunker
python -m src.index --config retrieval-v2
python -m src.evaluation.audit_ground_truth --dataset data/evaluation/retrieval_dev.jsonl
python -m src.evaluation.evaluate_retrieval --config retrieval-v2
python scripts/eval_generation.py --generate          # answers on the dev set (local GPU model)
python -m pytest tests
```

- **Configs.** Retrieval settings live in named configs under `configs/retrieval/`:
  - `retrieval-v2` is the default; `retrieval-v2-base` is the same without the reranker;
  - `retrieval-v1` and `retrieval-v1-base` are the phase 1 pipeline;
  - `baseline-v0` is the pipeline before 2026-10-19.
- **One implementation.** `src/retrieval.py` is the only implementation; search, the indexer, the evaluator, the experiment runner and the generator all use it. Every channel sees the same passage text for a chunk (`passage_for_record`): the paper title, the section title, and the cleaned chunk text. Under `retrieval-v2` the dense channel embeds 64-word windows of that text, and a chunk scores as its best window.
- **Evaluator options.** `src.evaluation.evaluate_retrieval` takes `--config`, `--dataset`, `--chunks`, `--retriever` (dense or bm25 alone), `--top-k`, `--candidate-k`, `--reranker-model` (`none` disables reranking), `--dense-backend memory` (scores a chunks file without rebuilding Qdrant) and `--output`.
- **Report contents.** The report holds:
  - the config;
  - the summary, a per-category summary and latency;
  - the full ranked results per question.
- **Comparing configs.** Use `scripts/run_experiments.py`, which accepts inline overrides such as `retrieval-v2:reranker=null`, then `scripts/compare_runs.py` for a paired bootstrap.

## Datasets

| Split | File | Questions | Answerable | Evidence units |
|---|---|---:|---:|---:|
| Dev | `data/evaluation/retrieval_dev.jsonl` | 96 | 89 | 135 |
| Test (held out) | `data/evaluation/retrieval_test.jsonl` | 61 | 54 | 80 |

Each JSONL item has:
- a stable `query_id`, a `split`, a `category` and the `question`;
- `answerable` and `expected_answer_points`, for answer evaluation;
- `annotation_status` and `annotation_notes`;
- `evidence`, a list of graded evidence units.

**Evidence units.** Each unit names:
- a `document_id`;
- a `grade` (2 is direct evidence, 1 is supporting);
- an `anchor`: a short quote from the paper, or a list of quotes that must all occur in the same chunk (used for table rows);
- optionally `alternatives`: other places that state the same fact. Each is an anchor in the same paper, or `{"document_id": ..., "anchor": ...}` in another paper.

**When a chunk supports a unit.** A chunk supports a unit when it is from the named paper and contains every anchor (or every anchor of one alternative) after normalization: case, whitespace, punctuation and accents are ignored. Anchors survive re-chunking. `audit_ground_truth` reports any anchor or alternative that no longer matches a chunk, and exits non-zero if there is a problem.

**Unanswerable items** (`answerable: false`) have no evidence. Their absence from the corpus was checked with a regular expression, recorded in `annotation_notes`. They are searched but excluded from retrieval metrics, and are used to score abstention.

**Status of the labels.**
- The 26 phase 1 questions were written by hand from chunk-ID labels.
- The 131 phase 2 questions are `model-drafted`: written from the papers by the same assistant that tuned the pipeline, and not yet reviewed by a person.
- Review the labels against the PDFs before treating scores as more than directional, and do not tune against the test split. It was run once for the phase 2 decision; see `retrieval_results.md` for its hash.
- `relevant_chunk_ids` on the phase 1 items is kept for history only. The evaluator reports legacy chunk-ID metrics only when all of those IDs exist in the current corpus.

## Metrics

Retrieval metrics are computed per evidence unit, so splitting a passage into two chunks, or the same paragraph appearing in two overlapping chunks, does not change the denominator.

- `Recall@k`: share of evidence units covered by at least one of the first `k` results.
- `Precision@k`: share of the first `k` results that support any evidence unit, with denominator `k`.
  - Most questions have one or two units, so precision stays low even for a perfect ranking.
  - Recall, MRR and nDCG are the informative numbers.
- `MRR`: reciprocal rank of the first result that supports any unit, averaged over questions.
- `nDCG@k`: graded gain for each newly covered unit. Re-retrieving an already covered unit earns nothing, so near-duplicate chunks are not rewarded.
- `Context evidence coverage`: share of units whose anchors appear in one passage of the assembled context. Context assembly merges neighbouring chunks and enforces a token budget, so this can differ from Recall@k.

Aggregates are macro-averages over answerable questions, reported overall and per category. Generation metrics are defined in `generation.md`. No target thresholds are set; differences are judged with the paired bootstrap rule in `retrieval_results.md`.
