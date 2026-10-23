# Chatbot: Paper Desk

A chat interface for literature review over the paper collection. It uses the same pipeline that is evaluated in `generation.md`: retrieval-v2, context assembly, Qwen3-4B with prompt v2, and the citation validator. The chat layer adds conversations, follow-up questions, a paper filter and a source viewer.

```
browser (React SPA) ──/api──> FastAPI (one process, one worker)
                                ├─ RagPipeline (embedder, reranker, generator on the GPU; behind a lock)
                                ├─ PaperCatalog (from chunks.jsonl; PDFs from papers_avatar/)
                                └─ ChatStore (SQLite: conversations, messages, feedback, traces)
```

## Running it

```bash
# once
cd frontend && npm install && npm run build && cd ..

source .venv/bin/activate              # the project's environment (Python 3.12)
python -m src.api                      # http://127.0.0.1:8000 serves the app and the API
python -m src.api --generator anthropic:<model>   # use the Anthropic API instead of the local model
```

The models take about a minute to load. Until they are ready, the page shows "Loading models…" and `/api/health` reports `loading`.

For frontend development, run `python -m src.api` and, in a second terminal, `cd frontend && npm run dev`, then open http://localhost:5173. Vite proxies `/api` to port 8000.

Only one server can run at a time. The local Qdrant store allows a single client, and the models fill the 8 GB GPU. Stop the server before running `scripts/eval_generation.py` or `src.evaluation.evaluate_retrieval`.

## What the user sees

- **Answers with checked citations.**
  - `[S1]` markers are clickable. Each verified source is shown with its paper, section, page and the quote that was verified.
  - A marker whose quote failed the check is struck through.
  - The number of removed citations is stated.
- **Statuses.** Each answer is answered, a partial answer with what is missing, or "Not found in the papers". There is no guess.
- **Source panel.** It shows the passage exactly as the model saw it, with the quotes highlighted, plus the PDF opened at the cited page.
- **Follow-ups.** "What are its limitations?" is rewritten into a standalone question, and the UI shows it as "Searched for …".
- **Paper filter.** Restrict a question to chosen papers, grouped by topic. Each paper has quick questions: contribution, method, evaluation, limitations.
- **History and export.** Conversations are kept. Each answer can be copied with its sources as Markdown, and a whole conversation can be exported as a `.md` file of notes.
- **Feedback.** A thumbs up or down is stored per answer, to feed the evaluation set later.
- **Progress.** Each turn shows its stage (rewriting, searching, reading N passages, writing) and how long it took.

## Design decisions

- **Only the validated answer is sent.** The model's output is JSON that has not yet been checked. The citation check runs after generation, so streaming raw tokens would show text that might be withheld. Instead the stream sends stage events and the retrieved sources while the answer is generated, about 6 seconds per question.
- **Follow-ups change only the question.** The conversation is used solely to rewrite the follow-up (`prompts/condense_v1.md`, up to 64 tokens, about 0.4 s). The answer step still sees only the standalone question and the retrieved passages. A chat turn therefore behaves like the single-turn setting measured in `generation.md`, and the named-paper check runs on the rewritten question.
- **Paper filter.** It is applied inside retrieval, in both channels: BM25 results are filtered before the cut-off, and Qdrant queries carry a `chunk_id` filter. It is not applied to the final results, so a filtered question still gets its full top 8.
- **One request at a time.** GPU work runs in a worker thread behind an `asyncio.Lock`. A second request waits, and its client is told it is queued.
- **Stopping is safe.** Each turn runs as its own task that holds the lock until the answer and trace are saved. The HTTP response only reads that task's events. Stop, a closed tab or a dropped connection cancels the reader, not the GPU work, so two generations never overlap. The stopped answer is still saved and appears when the conversation is reopened. This was verified with the real models: the aborted turn was saved, and the next question waited for it instead of running alongside it.
- **PDF access.** Paths come only from the catalog built from `chunks.jsonl` and must resolve inside `papers_avatar/`. Request input is never used as a path.

## API

| Method and path | Purpose |
|---|---|
| `GET /api/health` | Liveness, always 200: `loading` / `ready` / `error`, the generator, configs and paper count |
| `GET /api/ready` | Readiness: 503 until the models can answer |
| `GET /api/papers` | Papers with topic, page and passage counts, section outline |
| `GET /api/papers/{id}/pdf` | The PDF, served inline; append `#page=N` to open it at a page |
| `POST /api/chat/stream` | One turn as server-sent events (below) |
| `POST /api/chat` | The same turn as one JSON `AssistantMessage` |
| `GET /api/conversations`, `GET` / `DELETE /api/conversations/{id}` | History |
| `POST /api/messages/{id}/feedback` | `{"rating": 1 \| -1, "comment"?: str}` |
| `GET /api/traces/{request_id}` | Stage timings and attributes for one request |

**Chat request:**

```json
{"message": "...", "conversation_id": null, "paper_ids": [], "debug": false}
```

- `message`: 1–2,000 characters.
- `paper_ids`: up to 50 IDs from `/api/papers`.
- `debug: true`: also returns the validator's problems, the raw model output, the context log and the trace.

**Errors.**
- These are checked before streaming starts, so they come back as plain HTTP errors:
  - an unknown paper ID or an invalid message: 422;
  - an unknown conversation: 404;
  - models not ready: 503.
- A failure during a turn becomes an `error` event, or a 500 for `/api/chat`, carrying a generic message and the request ID. The exception goes to the server log and the trace, not to the client.

**Stream events, in order:**

| Event | Contents |
|---|---|
| `meta` | `request_id`, `conversation_id`, `user_message_id`, `message_id` |
| `stage` | `queued` (only when waiting), `rewriting` (only for follow-ups), `retrieving`, `generating` |
| `rewritten` | The standalone question, when it differs from the message |
| `sources` | The passages given to the model, labelled `[S1]`… (no citation status yet) |
| `answer` | The `AssistantMessage`: status, answer, verified citations, sources with a `cited` flag, caveats, `unverified_citations`, `timings_ms` |
| `error` | `message`, `request_id` |

The schemas are in `src/api/schemas.py`. FastAPI also serves them at `/docs`.

## Traces

Every request records spans:
- `rewrite_question`
- `embed_query`, `dense_search`, `bm25_search`, `fusion`, `rerank`
- `context`, `generation`, `validation`

They are stored with:
- the duration of each span;
- counts (candidates, sources, tokens, citations, invalid citations), the status, model and prompt version;
- the config in effect.

They contain no document text and no prompts (workbook Task 9.1).

A typical answered question takes about 6 s:

| Stage | Time |
|---|---:|
| Generation | 5.5 s |
| Reranking | 0.1–0.2 s |
| Query embedding | 20–250 ms |
| Dense search | 20 ms |
| BM25 | under 1 ms |

The traces are the starting point for the observability work: metrics, dashboards and SLOs.

## Tests

`tests/api_test.py` runs the API against a fake retriever and generator, so it needs no GPU. It covers:
- the SSE event order, and that only verified citations are sent;
- follow-up rewriting and conversation storage;
- the paper filter in the API and in both retrieval channels;
- abstention;
- request validation;
- PDF path safety;
- feedback, deletion and safe error responses.

The UI was checked by hand in Chrome at desktop and phone widths, in light and dark mode, with the real models.

## Known limitations

- **Single user.** There is no authentication. The server binds to 127.0.0.1 by default. Add authentication before exposing it (workbook Lab 10.1).
- **Answers are short.** Prompt v2 allows at most 5 sentences, and the model cites passages rather than writing a survey. Broad questions ("summarize the field") get a narrow answer drawn from the top 8 passages.
- **Secondary sources can be cited.** An answer about one paper may cite another paper that describes it, for example DF40's appendix describing SadTalker. The quote is genuine, but it is a secondary source.
- **Follow-up rewriting is not evaluated.** Rewriting quality has no benchmark yet. The rewritten question is shown so that the user can spot a bad rewrite.
