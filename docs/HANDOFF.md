# Handoff

State of the project as of 2026-10-24, for whoever picks it up next. Details live in the linked documents; this page says where things are and what to do next.

## What exists

A literature-review RAG over 20-odd avatar and deepfake papers (`papers_avatar/`), in four layers:

| Layer | Where | Document |
|---|---|---|
| Ingestion and chunking | `src/ingestion`, `chunks.jsonl` | `docs/ingestion_chunking_plan.md` |
| Retrieval (retrieval-v2: bge-m3 windows + BM25, RRF, reranker) | `src/retrieval.py`, `configs/retrieval` | `docs/retrieval_results.md`, `docs/evaluation.md` |
| Cited answers (Qwen3-4B, prompt answer_v2, citation validator, number check) | `src/rag.py`, `src/generation` | `docs/generation.md`, `docs/evals_guardrails_observability.md` |
| Chat app (FastAPI + React, SQLite history, traces, metrics) | `src/api`, `frontend/` | `docs/chatbot.md` |

## Branches and tags (origin)

- `main`: the original baseline. Not merged into; still the default branch.
- `pipeline-v2`: retrieval and generation work (`retrieval-v1` tag marks the first retrieval pass).
- `chatbot`: everything above plus the chat app, guardrails, evals and observability. **This is the branch to continue from.** Merge to `main` when you are ready; nothing has been merged yet.

## Run it

```bash
cd frontend && npm install && npm run build && cd ..
.venv/bin/python -m src.api            # http://127.0.0.1:8000, models take ~1 min to load
.venv/bin/python -m pytest -q tests    # 53 tests, no GPU needed
```

Use `.venv/bin/python` (Python 3.12, no pip; install with `uv pip install --python .venv/bin/python …`). A plain `python` on the author's machine resolves to another project. Only one process may hold the GPU and the local Qdrant store, so stop the server before any `scripts/eval_*.py`.

## Quality numbers to keep in mind

- Retrieval, dev set: Recall@5 0.886, MRR 0.750, nDCG@5 0.759 (retrieval-v2).
- Generation, dev set: abstention accuracy 0.969, citation validity 0.903, citation coverage 0.765, cited-evidence rate 0.930, invalid output 0, unsupported-number rate 0.012.
- Regression gate: `python scripts/eval_gate.py <report>` against `configs/evals/gates.json`.
- Typical answer: about 6 s on the 8 GB GPU, 5.5 s of it generation.
- The test sets (`data/evaluation/retrieval_test.jsonl`) were not used for tuning. Keep it that way; add new cases to the dev set.

## Decisions worth knowing

- Only validated answers are shown; raw tokens are never streamed, because citation checks run after generation.
- Follow-ups change only the question (`prompts/condense_v1.md`); a message naming a paper is not rewritten.
- Each chat turn is its own task holding the GPU lock until saved, so Stop never overlaps two generations.
- Traces hold timings and counts, never document text. Service-time latency excludes queue wait.
- The paper filter is applied inside retrieval (BM25 and Qdrant), not after.

## Known gaps, in suggested order

1. **Dashboard page** in the UI over `/api/metrics` (the endpoint and `scripts/report_metrics.py` exist).
2. **Authentication and rate limiting** before the server is exposed beyond 127.0.0.1 (binds locally by default).
3. **Answers are short** (at most 5 sentences from the top 8 passages); broad survey questions get narrow answers. A multi-step or longer-answer mode needs its own prompt and evaluation.
4. **Secondary sources** can be cited (e.g. DF40's appendix describing SadTalker). Consider preferring the paper that is the subject of the question.
5. **Review loop is empty:** no thumbs ratings yet. Use `scripts/export_review_set.py` after real use and add labelled failures to the dev set.
6. **SLO numbers are guesses** from one-off timings; refit after real traffic (`configs/evals/slo.json`).
7. **Number check** covers digits only, not names or wrong paraphrases.
8. **Follow-up eval** (24 cases) was written alongside its fix, so it is a regression check, not a benchmark.

## Housekeeping

- The author's other project environment (`/home/hp/btp_avatar/stu/.venv`) has leftover packages from an early mistaken install: starlette, httptools, uvloop, watchfiles, websockets, python-dotenv, annotated-doc, typing-inspection. Remove only the ones that project does not use.
- `data/app/` (chat database, review queue) and `frontend/node_modules`, `frontend/dist` are git-ignored.
- `.claude/settings.json` enables the frontend-design plugin for this repository.
