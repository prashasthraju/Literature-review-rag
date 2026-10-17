import json
import argparse
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer
from qdrant_client import QdrantClient

from src.embed_text import query_for_embedding
from src.lexical import BM25Index, reciprocal_rank_fusion
from src.reranking import rerank_candidates

MODEL_NAME = "BAAI/bge-base-en-v1.5"
COLLECTION = "research_papers_v1"
DB_PATH = "./qdrant_db"
CHUNKS_FILE = Path("chunks.jsonl")
TOP_K = 5
CANDIDATE_K = 20


def load_bm25(path):
    documents = []
    records = {}
    with path.open(encoding="utf-8") as chunk_file:
        for line in chunk_file:
            if not line.strip():
                continue
            item = json.loads(line)
            chunk_id = item.get("chunk_id")
            if not chunk_id:
                continue
            documents.append((chunk_id, item.get("text", "")))
            records[chunk_id] = item
    return BM25Index(documents), records


def main():
    parser = argparse.ArgumentParser(description="Search the local research-paper index.")
    parser.add_argument(
        "--reranker-model",
        help="Optional sentence-transformers cross-encoder used to rerank hybrid candidates",
    )
    args = parser.parse_args()

    model = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(args.reranker_model) if args.reranker_model else None
    client = QdrantClient(path=DB_PATH)
    bm25, chunk_records = load_bm25(CHUNKS_FILE)

    try:
        mode = "hybrid dense + BM25 + cross-encoder reranking" if reranker else "hybrid dense + BM25"
        print(f"Research-paper search ({mode}). Type 'exit' to stop.")

        while True:
            question = input("\nQuestion: ").strip()

            if question.lower() in {"exit", "quit"}:
                break

            if not question:
                continue

            vector = model.encode(
                query_for_embedding(question),
                normalize_embeddings=True,
            ).tolist()

            points = client.query_points(
                collection_name=COLLECTION,
                query=vector,
                limit=CANDIDATE_K,
                with_payload=True,
            ).points

            dense_ids = []
            dense_points = {}
            for point in points:
                payload = point.payload or {}
                chunk_id = payload.get("chunk_id")
                if not chunk_id:
                    continue
                dense_ids.append(chunk_id)
                dense_points[chunk_id] = point

            lexical_ids = [chunk_id for chunk_id, _score in bm25.search(question, limit=CANDIDATE_K)]
            candidate_ids = reciprocal_rank_fusion(
                dense_ids,
                lexical_ids,
                limit=max(TOP_K, CANDIDATE_K),
            )
            reranker_scores = {}
            if reranker is not None:
                ranked_candidates = rerank_candidates(
                    reranker,
                    question,
                    candidate_ids,
                    chunk_records,
                    TOP_K,
                )
                ranked_ids = [chunk_id for chunk_id, _score in ranked_candidates]
                reranker_scores = dict(ranked_candidates)
            else:
                ranked_ids = candidate_ids[:TOP_K]

            for rank, chunk_id in enumerate(ranked_ids, start=1):
                point = dense_points.get(chunk_id)
                payload = (point.payload if point else None) or chunk_records.get(chunk_id) or {}
                score = point.score if point is not None else None

                print(f"\n--- Result {rank} ---")
                if score is not None:
                    print(f"Dense score: {score:.4f}")
                if chunk_id in reranker_scores:
                    print(f"Reranker score: {reranker_scores[chunk_id]:.4f}")
                print(f"Paper: {payload.get('source_file')}")
                print(f"Section: {payload.get('section_title')}")
                print(f"Page: {payload.get('page_number')}")
                print((payload.get("text") or "")[:1500])

    finally:
        client.close()


if __name__ == "__main__":
    main()
