"""Chat API over the RAG pipeline.

  python -m src.api                      # http://127.0.0.1:8000 (serves frontend/dist if built)

One process holds the models: the embedder, reranker and generator share one
GPU and the local Qdrant store allows a single client, so requests run one at
a time behind a lock (queued requests are told so). Models load in a
background thread; /api/health reports "loading" until they are ready.

POST /api/chat/stream answers with server-sent events:
  meta       request, conversation and message IDs
  stage      queued | rewriting | retrieving | generating
  rewritten  the standalone question used for a follow-up
  sources    the passages given to the model, labelled [S1], [S2], ...
  answer     the validated answer (AssistantMessage)
  error      a safe error message with the request ID
Only the validated answer is sent; raw model output is never streamed as an
answer, because the citation check runs after generation.

Each turn runs as its own task that holds the lock until the turn is saved;
the HTTP response only reads that task's events. A client that disconnects
(the Stop button) therefore cancels only its reader: the GPU work finishes
under the lock, and the answer and trace are still stored.
"""

import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.api.catalog import PaperCatalog
from src.api.schemas import (
    AssistantMessage,
    ChatRequest,
    CitationView,
    ConversationDetail,
    ConversationSummary,
    FeedbackRequest,
    Health,
    MessageView,
    PaperView,
    SourceView,
)
from src.api.store import ChatStore, new_id, now
from src.tracing import Trace

ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = ROOT / "frontend" / "dist"
DEFAULT_DB = ROOT / "data" / "app" / "chat.sqlite3"
SAFE_ERROR = "Something went wrong while answering. The request ID identifies it in the server log."

log = logging.getLogger("rag.api")


class PipelineState:
    def __init__(self):
        self.status = "loading"
        self.detail = None
        self.pipeline = None

    def load(self, factory):
        try:
            self.pipeline = factory()
            self.status = "ready"
            log.info("pipeline ready: %s", self.pipeline.generator.name)
        except Exception as error:  # surfaced through /api/health
            log.exception("pipeline failed to load")
            self.status, self.detail = "error", f"{type(error).__name__}: {error}"


def source_view(passage, cited):
    return SourceView(
        source_id=passage.source_id,
        document_id=passage.document_id,
        document_title=passage.document_title,
        section_title=passage.section_title,
        pages=passage.pages,
        text=passage.text,
        cited=cited,
    )


def assistant_message(answer, bundle, trace, *, message_id, conversation_id, question, standalone, paper_ids, debug):
    valid = [citation for citation in answer.citations if citation.valid]
    cited = {citation.source_id for citation in valid}
    timings = trace.timings()
    timings["total"] = trace.to_dict()["total_ms"]
    return AssistantMessage(
        message_id=message_id,
        conversation_id=conversation_id,
        request_id=trace.request_id,
        question=question,
        standalone_question=standalone,
        paper_ids=paper_ids,
        status=answer.status,
        answer=answer.answer,
        citations=[
            CitationView(**citation.model_dump(include={
                "source_id", "quote", "match", "document_id", "document_title", "section_title", "pages",
            }))
            for citation in valid
        ],
        sources=[source_view(passage, passage.source_id in cited) for passage in bundle.passages],
        caveats=answer.caveats,
        unverified_citations=len(answer.citations) - len(valid),
        citation_coverage=answer.citation_coverage,
        model=answer.model,
        prompt_version=answer.prompt_version,
        timings_ms=timings,
        created_at=now(),
        debug={
            "problems": answer.problems,
            "raw_output": answer.raw_output,
            "context": answer.context,
            "trace": trace.to_dict(),
        } if debug else None,
    )


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(pipeline=None, pipeline_factory=None, records=None, db_path=DEFAULT_DB, frontend_dir=FRONTEND_DIST):
    """``pipeline``: a ready RagPipeline (tests); otherwise ``pipeline_factory(records)`` runs in the background."""
    if records is None:
        from src.retrieval import load_chunks

        records = load_chunks()
    catalog = PaperCatalog(records)
    store = ChatStore(db_path)
    state = PipelineState()
    lock = asyncio.Lock()
    turns = set()  # running turn tasks, kept referenced until they finish
    if pipeline is not None:
        state.pipeline, state.status = pipeline, "ready"

    @asynccontextmanager
    async def lifespan(_app):
        if state.pipeline is None:
            if pipeline_factory is None:
                from src.rag import RagPipeline

                factory = lambda: RagPipeline(records=records)  # noqa: E731
            else:
                factory = lambda: pipeline_factory(records)  # noqa: E731
            threading.Thread(target=state.load, args=(factory,), daemon=True).start()
        yield
        if state.pipeline is not None:
            state.pipeline.close()
        store.close()

    app = FastAPI(title="Literature-review RAG", version="1.0.0", lifespan=lifespan)
    app.state.store = store
    app.state.pipeline_state = state

    def config_view():
        if state.pipeline is None:
            from src.rag import GenerationConfig

            return asdict(GenerationConfig.load())
        return {**asdict(state.pipeline.config), "generator": state.pipeline.generator.name}

    @app.get("/api/health", response_model=Health)
    def health():
        """Liveness: always 200, with the model loading status."""
        config = config_view()
        return Health(
            status=state.status, detail=state.detail, generator=config["generator"],
            retrieval_config=config["retrieval_config"], prompt_version=config["prompt_version"], papers=len(catalog),
        )

    @app.get("/api/ready")
    def ready():
        """Readiness: 200 only once the models can answer."""
        if state.status != "ready":
            raise HTTPException(503, detail=state.detail or state.status)
        return {"status": "ready"}

    @app.get("/api/papers", response_model=list[PaperView])
    def papers():
        return catalog.views()

    @app.get("/api/papers/{paper_id}/pdf")
    def paper_pdf(paper_id: str):
        path = catalog.pdf_path(paper_id)
        if path is None:
            raise HTTPException(404, detail="paper not found")
        return FileResponse(path, media_type="application/pdf", content_disposition_type="inline", filename=path.name)

    @app.get("/api/conversations", response_model=list[ConversationSummary])
    def conversations():
        return store.conversations()

    @app.get("/api/conversations/{conversation_id}", response_model=ConversationDetail)
    def conversation(conversation_id: str):
        found = store.conversation(conversation_id)
        if not found:
            raise HTTPException(404, detail="conversation not found")
        messages = [
            MessageView(
                id=message["id"], role=message["role"], content=message["content"], created_at=message["created_at"],
                answer=AssistantMessage(**message["payload"]) if message["payload"] else None,
            )
            for message in store.messages(conversation_id)
        ]
        return ConversationDetail(**found, messages=messages)

    @app.delete("/api/conversations/{conversation_id}", status_code=204)
    def delete_conversation(conversation_id: str):
        if not store.delete_conversation(conversation_id):
            raise HTTPException(404, detail="conversation not found")

    @app.post("/api/messages/{message_id}/feedback", status_code=204)
    def feedback(message_id: str, body: FeedbackRequest):
        if not store.message_exists(message_id, role="assistant"):
            raise HTTPException(404, detail="message not found")
        store.set_feedback(message_id, body.rating, body.comment)

    @app.get("/api/traces/{request_id}")
    def trace(request_id: str):
        found = store.trace(request_id)
        if not found:
            raise HTTPException(404, detail="trace not found")
        return found

    def prepare(request: ChatRequest):
        """Checks that must fail with an HTTP status before any streaming starts."""
        if state.status != "ready":
            raise HTTPException(503, detail=f"models are {state.status}" + (f": {state.detail}" if state.detail else ""))
        unknown = [paper_id for paper_id in request.paper_ids if paper_id not in catalog]
        if unknown:
            raise HTTPException(422, detail=f"unknown paper ids: {unknown}")
        if request.conversation_id and not store.conversation(request.conversation_id):
            raise HTTPException(404, detail="conversation not found")

    async def execute_turn(request: ChatRequest, emit):
        """Run one chat turn, passing (event, data) pairs to ``emit``; the only place a turn is executed."""
        pipeline = state.pipeline
        conversation_id = request.conversation_id or store.create_conversation(request.message)
        history = store.history(conversation_id)
        trace = Trace()
        user_message_id, _ = store.add_message(conversation_id, "user", request.message, request_id=trace.request_id)
        message_id = new_id()
        emit("meta", {
            "request_id": trace.request_id, "conversation_id": conversation_id,
            "user_message_id": user_message_id, "message_id": message_id,
        })
        try:
            if lock.locked():
                emit("stage", {"stage": "queued"})
            async with lock:
                question = request.message
                if history:
                    emit("stage", {"stage": "rewriting"})
                    question = await asyncio.to_thread(pipeline.standalone_question, history, request.message, trace)
                    if question != request.message:
                        emit("rewritten", {"standalone_question": question})
                emit("stage", {"stage": "retrieving"})
                hits = await asyncio.to_thread(pipeline.search, question, request.paper_ids or None, trace)
                bundle = await asyncio.to_thread(pipeline.build_context, hits, trace)
                emit("sources", {"sources": [source_view(p, cited=False).model_dump() for p in bundle.passages]})
                emit("stage", {"stage": "generating"})
                answer = await asyncio.to_thread(pipeline.generate, question, bundle, trace)
            message = assistant_message(
                answer, bundle, trace, message_id=message_id, conversation_id=conversation_id,
                question=request.message, standalone=question, paper_ids=request.paper_ids, debug=request.debug,
            )
            store.add_message(
                conversation_id, "assistant", message.answer, payload=message.model_dump(exclude={"debug"}),
                request_id=trace.request_id, message_id=message_id,
            )
            store.save_trace(trace, conversation_id, answer.status, config_view())
            log.info("request %s %s %.0f ms %s", trace.request_id, answer.status, message.timings_ms["total"],
                     json.dumps(trace.timings()))
            emit("answer", message.model_dump())
        except Exception as error:
            log.exception("request %s failed", trace.request_id)
            store.save_trace(trace, conversation_id, "error", config_view(), error=type(error).__name__)
            emit("error", {"message": SAFE_ERROR, "request_id": trace.request_id})

    async def run_chat(request: ChatRequest):
        """Start a turn task and yield its events until it finishes."""
        queue = asyncio.Queue()
        done = object()

        async def produce():
            try:
                await execute_turn(request, lambda event, data: queue.put_nowait((event, data)))
            finally:
                queue.put_nowait((done, None))

        task = asyncio.create_task(produce())
        turns.add(task)
        task.add_done_callback(turns.discard)
        while (item := await queue.get())[0] is not done:
            yield item

    @app.post("/api/chat/stream")
    async def chat_stream(request: ChatRequest):
        prepare(request)

        async def events():
            async for event, data in run_chat(request):
                yield sse(event, data)

        return StreamingResponse(events(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
        })

    @app.post("/api/chat", response_model=AssistantMessage)
    async def chat(request: ChatRequest):
        prepare(request)
        async for event, data in run_chat(request):
            if event == "answer":
                return data
            if event == "error":
                raise HTTPException(500, detail=data)
        raise HTTPException(500, detail={"message": SAFE_ERROR})

    if Path(frontend_dir).is_dir():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    return app
