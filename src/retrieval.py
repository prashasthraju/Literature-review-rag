"""Hybrid retrieval shared by interactive search and evaluation.

Both entry points previously carried their own copy of the dense + BM25 + RRF
logic and had drifted apart (search.py read a different Qdrant path than the
indexer wrote). Keep retrieval behaviour here only.
"""

import json
from pathlib import Path

import numpy as np

from src.embed_text import passage_for_record, query_for_embedding
from src.lexical import BM25Index, reciprocal_rank_fusion
from src.reranking import rerank_candidates

CHUNKS_FILE = Path("chunks.jsonl")
DB_PATH = Path("qdrant_db_token_chunks")
COLLECTION = "research_papers_v1"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
# Reranking the hybrid candidates was the largest single gain on the dev set
# (see docs/evaluation.md), so interactive search uses it unless disabled.
DEFAULT_RERANKER = "BAAI/bge-reranker-base"
TOP_K = 5
CANDIDATE_K = 20
RETRIEVERS = ("dense", "bm25", "hybrid")


def load_chunks(path=CHUNKS_FILE):
    records = []
    with Path(path).open(encoding="utf-8") as chunk_file:
        for line in chunk_file:
            if line.strip():
                item = json.loads(line)
                if item.get("chunk_id"):
                    records.append(item)
    return records


class QdrantDenseIndex:
    def __init__(self, db_path=DB_PATH, collection=COLLECTION):
        from qdrant_client import QdrantClient

        self.collection = collection
        self.client = QdrantClient(path=str(db_path))
        if not self.client.collection_exists(collection):
            self.client.close()
            raise RuntimeError(
                f"Qdrant collection {collection!r} not found in {db_path}; run `python -m src.index` first."
            )

    def search(self, vector, limit):
        points = self.client.query_points(
            collection_name=self.collection,
            query=vector.tolist(),
            limit=limit,
            with_payload=["chunk_id"],
        ).points
        return [
            (point.payload["chunk_id"], point.score)
            for point in points
            if point.payload and point.payload.get("chunk_id")
        ]

    def close(self):
        self.client.close()


class MemoryDenseIndex:
    """Exact cosine search over freshly encoded chunks; used to score a chunks
    file without rebuilding the Qdrant store."""

    def __init__(self, model, records, batch_size=32):
        self.chunk_ids = [record["chunk_id"] for record in records]
        self.vectors = model.encode(
            [passage_for_record(record) for record in records],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def search(self, vector, limit):
        scores = self.vectors @ vector
        top = np.argsort(-scores)[:limit]
        return [(self.chunk_ids[index], float(scores[index])) for index in top]

    def close(self):
        pass


class Retriever:
    def __init__(
        self,
        records,
        mode="hybrid",
        model_name=MODEL_NAME,
        reranker_model=None,
        candidate_k=CANDIDATE_K,
        dense_backend="qdrant",
        db_path=DB_PATH,
        collection=COLLECTION,
    ):
        if mode not in RETRIEVERS:
            raise ValueError(f"Unknown retriever {mode!r}; expected one of {RETRIEVERS}")
        self.mode = mode
        self.candidate_k = candidate_k
        self.records = {record["chunk_id"]: record for record in records}

        self.bm25 = None
        if mode in {"bm25", "hybrid"}:
            self.bm25 = BM25Index(
                (record["chunk_id"], passage_for_record(record)) for record in records
            )

        self.model = None
        self.dense = None
        if mode in {"dense", "hybrid"}:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            if dense_backend == "memory":
                self.dense = MemoryDenseIndex(self.model, records)
            else:
                self.dense = QdrantDenseIndex(db_path, collection)

        self.reranker = None
        if reranker_model:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(reranker_model)

    def search(self, question, top_k=TOP_K):
        """Return ranked hits: dicts with chunk_id, record and per-channel scores."""
        candidate_k = max(top_k, self.candidate_k)

        dense_hits = []
        if self.dense is not None:
            vector = self.model.encode(
                query_for_embedding(question),
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            dense_hits = self.dense.search(vector, candidate_k)
        lexical_hits = self.bm25.search(question, limit=candidate_k) if self.bm25 else []

        dense_ids = [chunk_id for chunk_id, _score in dense_hits]
        lexical_ids = [chunk_id for chunk_id, _score in lexical_hits]
        if self.mode == "dense":
            candidate_ids = dense_ids
        elif self.mode == "bm25":
            candidate_ids = lexical_ids
        else:
            candidate_ids = reciprocal_rank_fusion(dense_ids, lexical_ids, limit=candidate_k)

        rerank_scores = {}
        if self.reranker is not None:
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
