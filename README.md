# Literature-review RAG

Question answering with citations over a collection of 17 research papers on avatar generation, talking heads, deepfake detection and avatar fingerprinting. Built lab by lab following `enterprise_rag_lab_workbook.md`.

```
PDFs ─ Docling ─> parsed/ ─ chunker ─> chunks.jsonl ─ indexer ─> Qdrant (bge-m3 windows)
                                                   └─ BM25 (titled passages)
question ─> hybrid retrieval (RRF) ─> bge-reranker-base ─> context assembly (cited, budgeted)
        ─> local LLM (Qwen3-4B, 4-bit) ─> answer contract ─> citation validator ─> answer + sources
browser (React) ─> FastAPI (/api, SSE) ─> the same pipeline; conversations and traces in SQLite
```

## Chat app

```bash
cd frontend && npm install && npm run build && cd ..
.venv/bin/python -m src.api  # open http://127.0.0.1:8000
```

Ask questions, follow up, restrict a question to chosen papers, open each cited passage or the PDF at the cited page, and export a conversation as Markdown notes. See `docs/chatbot.md`.

## Quick start

Use the project's own virtual environment (`.venv`, Python 3.12). It has no pip, so install with uv; on this machine a plain `python` may resolve to another project's environment.

```bash
source .venv/bin/activate                          # or call .venv/bin/python directly
uv pip install --python .venv/bin/python -r requirements.txt   # CUDA GPU with 8 GB recommended
python -m src.ingestion.chunker                    # parsed/ -> chunks.jsonl (847 chunks)
python -m src.index --config retrieval-v2          # builds the Qdrant collection
python -m src.search                               # interactive retrieval only
python -m src.rag "What does SadTalker predict from audio?"   # cited answer
python -m src.rag --json "..."                     # full record: citations, provenance, context log
```

`python -m src.rag --generator anthropic:<model>` uses the Anthropic API instead of the local model. It needs the `anthropic` package and `ANTHROPIC_API_KEY`.

## Layout

| Path | What it is |
|---|---|
| `src/ingestion/` | Docling parsing (`docparse.py`) and structure-aware, token-budgeted chunking (`chunker.py`) |
| `src/retrieval.py` | The only retrieval implementation: configs, dense (chunks or windows), BM25, fusion, reranking |
| `src/index.py`, `src/search.py` | Build a config's Qdrant collection; interactive search |
| `src/context.py` | Context assembly: merge neighbouring chunks, token budget, provenance, exclusion log |
| `src/generation/` | Answer contract, prompt construction, citation validation (`answer.py`); model backends |
| `src/rag.py` | End-to-end question answering, as traced stages (rewrite, search, context, generate) |
| `src/api/` | FastAPI chat server: streaming chat, papers and PDFs, conversations, feedback, traces |
| `src/tracing.py` | Per-request spans (timings and counts, no document text) |
| `frontend/` | React + TypeScript + Vite + Tailwind chat UI |
| `src/evaluation/` | Evidence-anchor labels, metrics, evaluator, label audit |
| `configs/retrieval/`, `configs/generation/` | Named, versioned configurations (`retrieval-v2` is the default) |
| `prompts/` | Versioned prompts (`answer_v1.md`, `answer_v2.md`, `condense_v1.md` for follow-ups) |
| `data/evaluation/` | Benchmark v2: `retrieval_dev.jsonl` (96 questions) and held-out `retrieval_test.jsonl` (61) |
| `scripts/` | Experiment runner, paired bootstrap comparison, generation evaluation |
| `reports/` | Saved evaluation outputs |
| `docs/` | Evaluation method, retrieval results and history, generation design and results, chatbot |

## Current results

| | Dev | Held-out test |
|---|---|---|
| Retrieval, `retrieval-v2` (Recall@5 / MRR) | 0.886 / 0.750 | 0.769 / 0.639 |
| Retrieval, fast mode `retrieval-v2-base` | 0.918 / 0.732 | 0.769 / 0.607 |

| Answers (Qwen3-4B, prompt v2) | Dev | Held-out test |
|---|---|---|
| Unanswerable questions refused | 7/7 | 7/7 |
| Answerable questions answered | 96.6% | 88.9% |
| Citations verified against the cited source | 90.3% | 91.6% |
| Answers citing the labelled evidence | 93.0% | 89.6% |

Details are in `docs/retrieval_results.md` and `docs/generation.md`. The phase 2 questions were drafted by the assistant that built the pipeline and have not been reviewed by a person, so treat all scores as directional. See `docs/evaluation.md`.

## Tests

```bash
python -m pytest tests
```
