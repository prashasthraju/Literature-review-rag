# Evals, guardrails and observability

## Guardrails (`src/generation/guardrails.py`, `src/generation/followup.py`)

Already in place: citations are checked against the passages of the same request, unsupported answers are withheld, the question's named paper must be cited, and retrieved text is treated as data (adversarial cases below).

Added:

- **Number check.** Every number in an answer sentence must occur in a passage that sentence cites (digits only; `25M` matches "$25 million", `.003` matches `0.003`, single digits and `3D`-style tokens are ignored). A miss adds the caveat "Not found in the cited passages, please check: …" and turns `answered` into `partial`. On the 86 dev answers it flags 1 (a derived "10×" from 1000 → 100); checking against the quote only would flag 13, mostly correct numbers outside the quoted fragment, so the passage is the unit. It does not catch a wrong number that also appears elsewhere in the same passage.
- **Named paper skips the rewrite.** A follow-up that already names a paper ("Which datasets are used in DF40?") is used as is. The model had turned it into a question about the previous turn's paper and, once, appended the previous topic. Messages without a name are still resolved from the conversation.

## Evals

| What | Command | Needs GPU |
|---|---|---|
| Generation on the dev set | `python scripts/eval_generation.py --generate` | yes |
| Re-score a saved run with the current validator | `python scripts/eval_generation.py --revalidate <report>` | no |
| **Regression gate** | `python scripts/eval_gate.py <report> [--baseline <report>]` or `--run` | `--run` only |
| Behaviour under attack (6 cases) | `python scripts/eval_adversarial.py` | yes |
| Follow-up rewrite quality (24 cases) | `python scripts/eval_followup.py` | yes |
| Rated answers → labelling queue | `python scripts/export_review_set.py` | no |

The gate (`configs/evals/gates.json`) holds floors and ceilings about 3 points under the answer_v2 dev run, because one of 96 questions moves a rate by about 1 point. It exits 1 on a breach, or when a metric drops more than 3 points against `--baseline`. The old answer_v1 run fails it as it should (citation coverage 0.63). The gate uses the dev set only; the held-out test set stays untouched.

Adversarial cases: injected instruction in a passage, forged citation requested by a passage, request to print the system prompt, irrelevant context, conflicting sources, answer in one of several passages. 6/6 pass.

Follow-up rewrites: 24 cases (pronouns, ellipsis, topic switches, standalone messages, two-turn references). Before the named-paper rule 22 of 24 passed; two failures were real (a DF40 question rewritten to MotionDiffuse, and `ARTalk … instead` carrying SadTalker along). The rule was written after seeing these, so the 24/24 now is not an independent estimate; the cases are a regression check, not a benchmark.

Review loop: thumbs-down answers are exported to `data/app/review_queue.jsonl` (kept out of git, it holds user questions). Label each as retrieval, generation or question fault, write the expected evidence, and add it to the dev set.

## Observability

Each request is traced (`queue_wait`, `rewrite_question`, `embed_query`, `dense_search`, `bm25_search`, `fusion`, `rerank`, `context`, `generation`, `validation`) with counts, status and model, never document text. Metrics come from those traces:

- `GET /api/metrics?hours=24` returns request counts, status mix, error and abstain rate, service-time p50/p95/max overall and per stage, and feedback.
- `python scripts/report_metrics.py` prints the same and checks `configs/evals/slo.json`; exit 1 on a violated SLO. It reads the SQLite file, so it works while the server runs.

Service time excludes `queue_wait`, so a burst of queued questions is not read as a slow model. SLOs (24 h, judged only with at least 20 requests): p95 ≤ 15 s, p50 ≤ 9 s, errors ≤ 2%, invalid output ≤ 3%, thumbs-down ≤ 20%. A typical answer takes 6 s.

## Not done

- No dashboard page in the UI yet; metrics are JSON and the CLI.
- The SLO numbers are targets from one-off timings, not from production traffic; re-fit them after real use.
- The number check does not cover names, dates in words, or claims that paraphrase the passage wrongly. That needs a second model or human review.
- Input limits are still message length 1–2000 and 50 papers; no rate limiting or authentication.
