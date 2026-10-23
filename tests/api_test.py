import json

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.rag import GenerationConfig, RagPipeline

RECORDS = [
    {
        "chunk_id": "s1", "document_id": "sadtalker", "document_title": "SadTalker", "source_file": "missing.pdf",
        "section_title": "Abstract", "section_path": ["SadTalker", "Abstract"], "chunk_index": 1, "page_numbers": [1],
        "text": "SadTalker generates 3D motion coefficients (head pose, expression) of the 3DMM from audio.",
    },
    {
        "chunk_id": "d1", "document_id": "df 40", "document_title": "DF40", "source_file": "df 40.pdf",
        "section_title": "Abstract", "section_path": ["DF40", "Abstract"], "chunk_index": 1, "page_numbers": [1],
        "text": "DF40 comprises 40 distinct deepfake techniques.",
    },
]


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, question, top_k=None, document_ids=None, trace=None):
        self.calls.append({"question": question, "document_ids": document_ids})
        return [
            {"chunk_id": r["chunk_id"], "record": r}
            for r in RECORDS
            if not document_ids or r["document_id"] in document_ids
        ]

    def close(self):
        pass


class FakeGenerator:
    name = "fake"

    def __init__(self):
        self.condense_calls = 0

    def generate(self, messages, prefill="", max_new_tokens=None):
        user = messages[-1]["content"]
        if "Standalone question:" in user:
            self.condense_calls += 1
            return "What are the limitations of SadTalker?"
        if "SadTalker generates" not in user:
            return prefill + 'insufficient_evidence", "answer": "The sources do not say.", "citations": []}'
        return prefill + json.dumps({
            "status": "answered",
            "answer": "It predicts 3DMM motion coefficients from audio [S1].",
            "citations": [{"source_id": "S1", "quote": "generates 3D motion coefficients"},
                          {"source_id": "S2", "quote": "an invented quote"}],
        })[len('{"status": "'):]


@pytest.fixture
def client(tmp_path):
    retriever, generator = FakeRetriever(), FakeGenerator()
    pipeline = RagPipeline(GenerationConfig(), retriever=retriever, generator=generator)
    app = create_app(pipeline=pipeline, records=RECORDS, db_path=tmp_path / "chat.sqlite3", frontend_dir=tmp_path / "none")
    with TestClient(app) as test_client:
        test_client.retriever, test_client.generator = retriever, generator
        yield test_client


def events(response):
    parsed = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        parsed.append((lines["event"], json.loads(lines["data"])))
    return parsed


def test_health_and_papers(client):
    assert client.get("/api/health").json()["status"] == "ready"
    assert client.get("/api/ready").status_code == 200
    papers = {paper["id"]: paper for paper in client.get("/api/papers").json()}
    assert set(papers) == {"sadtalker", "df 40"}
    assert papers["sadtalker"]["has_pdf"] is False


def test_stream_sends_stages_sources_then_only_verified_citations(client):
    response = client.post("/api/chat/stream", json={"message": "What does SadTalker predict?"})
    names = [name for name, _data in events(response)]
    assert names == ["meta", "stage", "sources", "stage", "answer"]
    answer = events(response)[-1][1]
    assert answer["status"] == "answered"
    assert [c["source_id"] for c in answer["citations"]] == ["S1"]
    assert answer["unverified_citations"] == 1
    assert [s["cited"] for s in answer["sources"]] == [True, False]
    assert answer["debug"] is None and "raw_output" not in answer
    assert {"generation", "validation", "context", "total"} <= set(answer["timings_ms"])


def test_follow_up_is_rewritten_and_conversation_is_stored(client):
    first = client.post("/api/chat", json={"message": "What does SadTalker predict?"}).json()
    second = client.post("/api/chat", json={
        "message": "What are its limitations?", "conversation_id": first["conversation_id"],
    }).json()
    assert client.generator.condense_calls == 1
    assert second["standalone_question"] == "What are the limitations of SadTalker?"
    assert client.retriever.calls[-1]["question"] == "What are the limitations of SadTalker?"
    detail = client.get(f"/api/conversations/{first['conversation_id']}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant", "user", "assistant"]
    assert detail["messages"][1]["answer"]["citations"][0]["quote"] == "generates 3D motion coefficients"
    trace = client.get(f"/api/traces/{second['request_id']}").json()
    assert "rewrite_question" in [span["name"] for span in trace["spans"]]


def test_paper_filter_reaches_retrieval_and_abstention_is_returned(client):
    answer = client.post("/api/chat", json={"message": "What does SadTalker predict?", "paper_ids": ["df 40"]}).json()
    assert client.retriever.calls[-1]["document_ids"] == ["df 40"]
    assert answer["status"] == "insufficient_evidence" and answer["citations"] == []


def test_invalid_requests_fail_before_streaming(client):
    assert client.post("/api/chat/stream", json={"message": "  "}).status_code == 422
    assert client.post("/api/chat/stream", json={"message": "x" * 2001}).status_code == 422
    assert client.post("/api/chat/stream", json={"message": "q", "paper_ids": ["nope"]}).status_code == 422
    assert client.post("/api/chat/stream", json={"message": "q", "conversation_id": "nope"}).status_code == 404
    assert client.get("/api/papers/sadtalker/pdf").status_code == 404
    assert client.get("/api/papers/..%2F..%2Fetc%2Fpasswd/pdf").status_code == 404


def test_feedback_and_delete(client):
    answer = client.post("/api/chat", json={"message": "What does SadTalker predict?"}).json()
    assert client.post(f"/api/messages/{answer['message_id']}/feedback", json={"rating": 1}).status_code == 204
    detail = client.get(f"/api/conversations/{answer['conversation_id']}").json()
    assert detail["messages"][1]["answer"]["feedback"] == 1
    assert client.post("/api/messages/nope/feedback", json={"rating": 1}).status_code == 404
    assert client.delete(f"/api/conversations/{answer['conversation_id']}").status_code == 204
    assert client.get("/api/conversations").json() == []


def test_pipeline_error_returns_safe_message(client):
    def broken(*_args, **_kwargs):
        raise RuntimeError("CUDA out of memory: secret details")

    client.generator.generate = broken
    response = client.post("/api/chat/stream", json={"message": "What does SadTalker predict?"})
    name, data = events(response)[-1]
    assert name == "error" and "CUDA" not in data["message"]
    assert client.get(f"/api/traces/{data['request_id']}").json()["error"] == "RuntimeError"


def test_retriever_paper_filter_restricts_both_channels():
    from src.retrieval import MemoryDenseIndex, RetrievalConfig, Retriever

    import numpy as np

    class Model:
        def encode(self, texts, **_kwargs):
            vector = lambda text: np.array([1.0, 0.0]) if "SadTalker" in text else np.array([0.6, 0.8])  # noqa: E731
            return np.stack([vector(t) for t in texts]) if isinstance(texts, list) else vector(texts)

    config = RetrievalConfig(retriever="hybrid", query_prefix="")
    dense = MemoryDenseIndex(Model(), RECORDS, config)
    retriever = Retriever(RECORDS, config, model=Model(), dense_index=dense)
    assert retriever.search("SadTalker deepfake", top_k=2)[0]["chunk_id"] == "s1"
    hits = retriever.search("SadTalker deepfake", top_k=2, document_ids=["df 40"])
    assert [hit["chunk_id"] for hit in hits] == ["d1"]
    assert retriever.search("SadTalker", document_ids=["unknown"]) == []
