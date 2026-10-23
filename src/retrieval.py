"""Hybrid retrieval shared by interactive search, indexing and evaluation.

Both entry points previously carried their own copy of the dense + BM25 + RRF
logic and had drifted apart (search.py read a different Qdrant path than the
indexer wrote). Keep retrieval behaviour here only.

Every setting lives in a RetrievalConfig loaded from configs/retrieval/*.json,
so each experiment is a named, versioned configuration and the previous best
pipeline can be restored by name (docs/retrieval_results.md).
"""

import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from src.embed_text import (
    QUERY_PREFIX,
    clean_passage,
    context_header,
    passage_for_record,
)
from src.lexical import STOPWORDS, BM25Index, reciprocal_rank_fusion
from src.reranking import rerank_candidates
from src.tracing import Trace

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = Path("chunks.jsonl")
DB_PATH = Path("qdrant_db_token_chunks")
CONFIG_DIR = ROOT / "configs" / "retrieval"
DEFAULT_CONFIG = CONFIG_DIR / "retrieval-v2.json"
RETRIEVERS = ("dense", "bm25", "hybrid")

# Kept for callers that only need the v1 defaults.
COLLECTION = "research_papers_v1"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
DEFAULT_RERANKER = "BAAI/bge-reranker-base"
TOP_K = 5
CANDIDATE_K = 20

TABLE_ROW_RE = re.compile(r"^[^\n:]{1,80}: [^\n]*;")


@dataclass(frozen=True)
class RetrievalConfig:
    name: str = "retrieval-v1"
    retriever: str = "hybrid"
    collection: str = COLLECTION
    embedding_model: str = MODEL_NAME
    query_prefix: str = QUERY_PREFIX
    # Dense passages: whole chunks, or overlapping word windows inside each
    # chunk (scored by their best window) so one fact is not diluted by a
    # 480-token chunk. BM25 and the reranker always see whole chunks.
    dense_units: str = "chunk"
    dense_title: bool = True
    window_words: int = 96
    window_stride: int = 48
    bm25_title: bool = True
    bm25_clean: bool = True
    bm25_stopwords: bool = True
    fusion: str = "rrf"
    rrf_k: int = 60
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    candidate_k: int = CANDIDATE_K
    reranker: str | None = None
    top_k: int = TOP_K
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path=DEFAULT_CONFIG, **overrides):
        path = Path(path)
        if not path.is_absolute() and not path.exists():
            path = CONFIG_DIR / (path.name if path.suffix else f"{path}.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        return replace(cls(**data), **overrides)

    def to_dict(self):
        return asdict(self)


def load_chunks(path=CHUNKS_FILE):
    records = []
    with Path(path).open(encoding="utf-8") as chunk_file:
        for line in chunk_file:
            if line.strip():
                item = json.loads(line)
                if item.get("chunk_id"):
                    records.append(item)
    return records


def bm25_passage(record, config):
    if not config.bm25_clean:
        return record.get("text", "")
    if config.bm25_title:
        return passage_for_record(record)
    return passage_for_record({**record, "document_title": None})


def dense_header(record, config):
    title = record.get("document_title") if config.dense_title else None
    return context_header(record.get("section_title"), title)


def split_windows(text, window_words, stride):
    """Word windows over a chunk; table rows are kept whole as their own units."""
    units = []
    paragraph = []
    for line in (text or "").splitlines():
        if TABLE_ROW_RE.match(line.strip()):
            if paragraph:
                units.append(" ".join(paragraph))
                paragraph = []
            units.append(line.strip())
        elif line.strip():
            paragraph.append(line.strip())
    if paragraph:
        units.append(" ".join(paragraph))

    windows = []
    for unit in units:
        words = clean_passage(unit).split()
        if not words:
            continue
        if len(words) <= window_words:
            windows.append(" ".join(words))
            continue
        for start in range(0, len(words), stride):
            windows.append(" ".join(words[start:start + window_words]))
            if start + window_words >= len(words):
                break
    return windows


def dense_units(records, config):
    """(chunk_id, passage) pairs that are embedded for the dense channel."""
    for record in records:
        header = dense_header(record, config)
        if config.dense_units == "chunk":
            text = clean_passage(record.get("text"))
            yield record["chunk_id"], f"{header}. {text}" if header and text else header or text
            continue
        windows = split_windows(record.get("text"), config.window_words, config.window_stride)
        for window in windows or [header]:
            yield record["chunk_id"], f"{header}. {window}" if header else window


def best_per_chunk(scored, limit):
    """Max-pool window scores to their parent chunk."""
    best = {}
    for chunk_id, score in scored:
        if chunk_id not in best or score > best[chunk_id]:
            best[chunk_id] = score
    return sorted(best.items(), key=lambda item: item[1], reverse=True)[:limit]


class QdrantDenseIndex:
    def __init__(self, db_path=DB_PATH, collection=COLLECTION, oversample=1):
        from qdrant_client import QdrantClient

        self.collection = collection
        self.oversample = oversample
        self.client = QdrantClient(path=str(db_path))
        if not self.client.collection_exists(collection):
            self.client.close()
            raise RuntimeError(
                f"Qdrant collection {collection!r} not found in {db_path}; "
                "run `python -m src.index --config <config>` first."
            )

    def search(self, vector, limit, allowed=None):
        query_filter = None
        if allowed is not None:
            from qdrant_client import models

            # Window points carry only their chunk_id, so a paper filter is a chunk-id filter.
            query_filter = models.Filter(must=[
                models.FieldCondition(key="chunk_id", match=models.MatchAny(any=sorted(allowed)))
            ])
        points = self.client.query_points(
            collection_name=self.collection,
            query=vector.tolist(),
            limit=limit * self.oversample,
            with_payload=["chunk_id"],
            query_filter=query_filter,
        ).points
        return best_per_chunk(
            (
                (point.payload["chunk_id"], point.score)
                for point in points
                if point.payload and point.payload.get("chunk_id")
            ),
            limit,
        )

    def close(self):
        self.client.close()


class MemoryDenseIndex:
    """Exact cosine search over freshly encoded units; used to score a chunks
    file or a configuration without rebuilding the Qdrant store."""

    def __init__(self, model, records, config=None, batch_size=32, vectors=None):
        config = config or RetrievalConfig()
        units = list(dense_units(records, config))
        self.chunk_ids = [chunk_id for chunk_id, _text in units]
        self.vectors = vectors if vectors is not None else model.encode(
            [text for _chunk_id, text in units],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def search(self, vector, limit, allowed=None):
        scores = self.vectors @ vector
        if allowed is not None:
            scores = np.where([chunk_id in allowed for chunk_id in self.chunk_ids], scores, -np.inf)
        top = [index for index in np.argsort(-scores)[: limit * 8] if np.isfinite(scores[index])]
        return best_per_chunk(((self.chunk_ids[index], float(scores[index])) for index in top), limit)

    def close(self):
        pass


def fuse(config, dense_hits, lexical_hits, limit):
    dense_ids = [chunk_id for chunk_id, _score in dense_hits]
    lexical_ids = [chunk_id for chunk_id, _score in lexical_hits]
    if config.fusion == "rrf" and config.dense_weight == config.bm25_weight:
        return reciprocal_rank_fusion(dense_ids, lexical_ids, k=config.rrf_k, limit=limit)
    fused = {}
    for weight, ranked in ((config.dense_weight, dense_ids), (config.bm25_weight, lexical_ids)):
        for rank, chunk_id in enumerate(ranked, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (config.rrf_k + rank)
    return [chunk_id for chunk_id, _score in sorted(fused.items(), key=lambda item: item[1], reverse=True)[:limit]]


class Retriever:
    def __init__(
        self,
        records,
        config=None,
        dense_backend="qdrant",
        db_path=DB_PATH,
        model=None,
        reranker=None,
        dense_index=None,
        **overrides,
    ):
        config = replace(config or RetrievalConfig(), **{key: value for key, value in overrides.items() if value is not None})
        if config.retriever not in RETRIEVERS:
            raise ValueError(f"Unknown retriever {config.retriever!r}; expected one of {RETRIEVERS}")
        self.config = config
        self.records = {record["chunk_id"]: record for record in records}

        self.bm25 = None
        if config.retriever in {"bm25", "hybrid"}:
            self.bm25 = BM25Index(
                ((record["chunk_id"], bm25_passage(record, config)) for record in records),
                stopwords=STOPWORDS if config.bm25_stopwords else frozenset(),
            )

        self.model = model
        self.dense = dense_index
        if config.retriever in {"dense", "hybrid"} and self.dense is None:
            if self.model is None:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(config.embedding_model)
            if dense_backend == "memory":
                self.dense = MemoryDenseIndex(self.model, records, config)
            else:
                oversample = 1 if config.dense_units == "chunk" else 8
                self.dense = QdrantDenseIndex(db_path, config.collection, oversample)

        self.reranker = reranker
        if config.reranker and self.reranker is None:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(config.reranker)

    def search(self, question, top_k=None, document_ids=None, trace=None):
        """Return ranked hits: dicts with chunk_id, record and per-channel scores.

        ``document_ids`` restricts every channel to those papers; ``trace``
        receives one span per stage.
        """
        config = self.config
        trace = trace or Trace()
        top_k = top_k or config.top_k
        candidate_k = max(top_k, config.candidate_k)
        allowed = None
        if document_ids:
            wanted = set(document_ids)
            allowed = {chunk_id for chunk_id, record in self.records.items() if record.get("document_id") in wanted}
            if not allowed:
                return []

        dense_hits = []
        if self.dense is not None and config.retriever in {"dense", "hybrid"}:
            with trace.span("embed_query", model=config.embedding_model):
                vector = self.model.encode(
                    config.query_prefix + (question or "").strip(),
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
            with trace.span("dense_search", filtered=allowed is not None) as span:
                dense_hits = self.dense.search(vector, candidate_k, allowed)
                span["candidates"] = len(dense_hits)
        lexical_hits = []
        if self.bm25 is not None and config.retriever in {"bm25", "hybrid"}:
            with trace.span("bm25_search", filtered=allowed is not None) as span:
                lexical_hits = self.bm25.search(question, limit=candidate_k, allowed=allowed)
                span["candidates"] = len(lexical_hits)

        if config.retriever == "dense":
            candidate_ids = [chunk_id for chunk_id, _score in dense_hits]
        elif config.retriever == "bm25":
            candidate_ids = [chunk_id for chunk_id, _score in lexical_hits]
        else:
            with trace.span("fusion", method=config.fusion) as span:
                candidate_ids = fuse(config, dense_hits, lexical_hits, candidate_k)
                span["candidates"] = len(candidate_ids)

        rerank_scores = {}
        if self.reranker is not None:
            with trace.span("rerank", model=config.reranker, candidates=len(candidate_ids)):
                ranked = rerank_candidates(self.reranker, question, candidate_ids, self.records, top_k)
            ranked_ids = [chunk_id for chunk_id, _score in ranked]
            rerank_scores = dict(ranked)
        else:
            ranked_ids = candidate_ids[:top_k]

        dense_scores = dict(dense_hits)
        lexical_scores = dict(lexical_hits)
        hits = []
        for chunk_id in ranked_ids:
            scores = {
                "dense_score": dense_scores.get(chunk_id),
                "bm25_score": lexical_scores.get(chunk_id),
                "rerank_score": rerank_scores.get(chunk_id),
            }
            primary = next((value for value in (
                scores["rerank_score"], scores["dense_score"], scores["bm25_score"]
            ) if value is not None), None)
            hits.append({
                "chunk_id": chunk_id,
                "record": self.records.get(chunk_id, {}),
                "score": primary,
                **scores,
            })
        return hits

    def close(self):
        if self.dense is not None:
            self.dense.close()
