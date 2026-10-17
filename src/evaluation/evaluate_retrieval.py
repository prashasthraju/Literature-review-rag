import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embed_text import query_for_embedding
from src.lexical import BM25Index, reciprocal_rank_fusion
from src.reranking import rerank_candidates

if __package__:
    from .metrics import evaluate_query, mean_metric
else:
    from metrics import evaluate_query, mean_metric

DEFAULT_DATASET = Path("data/evaluation/retrieval_dev.jsonl")
DEFAULT_DB_PATH = Path("qdrant_db")
DEFAULT_COLLECTION = "research_papers_v1"
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as dataset_file:
        for line in dataset_file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_dataset(path):
    examples = []
    with path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            example = json.loads(line)
            if "query_id" not in example or "question" not in example:
                raise ValueError(f"Missing query_id or question at {path}:{line_number}")
            if "relevant_chunk_ids" not in example:
                raise ValueError(f"Missing relevant_chunk_ids at {path}:{line_number}")
            examples.append(example)
    if not examples:
        raise ValueError(f"No evaluation examples found in {path}")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Evaluate dense retrieval against chunk-level judgments.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--retriever",
        choices=("dense", "bm25", "hybrid"),
        default="hybrid",
        help="dense = Qdrant embeddings, bm25 = lexical, hybrid = RRF of both",
    )
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument(
        "--reranker-model",
        help="Optional sentence-transformers cross-encoder used to rerank the candidate pool",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/retrieval_eval_latest.json"))
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    examples = read_dataset(args.dataset)
    chunks_path = ROOT / "chunks.jsonl"
    chunk_records = read_jsonl(chunks_path) if chunks_path.exists() else []
    chunk_by_id = {item["chunk_id"]: item for item in chunk_records if item.get("chunk_id")}
    bm25 = None
    if args.retriever in {"bm25", "hybrid"}:
        if not chunk_records:
            raise RuntimeError("chunks.jsonl is required for bm25/hybrid retrieval")
        bm25 = BM25Index(
            (item["chunk_id"], item.get("text", ""))
            for item in chunk_records
        )

    model = None
    reranker = None
    client = None
    if args.retriever in {"dense", "hybrid"}:
        print(f"Loading embedding model: {args.model}")
        model = SentenceTransformer(args.model)
        client = QdrantClient(path=str(args.db_path))
    if args.reranker_model:
        print(f"Loading reranker: {args.reranker_model}")
        reranker = CrossEncoder(args.reranker_model)

    query_results = []
    candidate_k = max(args.top_k, args.candidate_k)

    try:
        if client is not None and not client.collection_exists(args.collection):
            raise RuntimeError(f"Qdrant collection not found: {args.collection}")

        for example in examples:
            dense_ids = []
            dense_points = {}
            if client is not None:
                vector = model.encode(
                    query_for_embedding(example["question"]),
                    normalize_embeddings=True,
                ).tolist()
                points = client.query_points(
                    collection_name=args.collection,
                    query=vector,
                    limit=candidate_k,
                    with_payload=True,
                ).points
                for point in points:
                    if not point.payload:
                        continue
                    chunk_id = point.payload.get("chunk_id")
                    if not chunk_id:
                        continue
                    dense_ids.append(chunk_id)
                    dense_points[chunk_id] = point

            lexical_ids = []
            lexical_scores = {}
            if bm25 is not None:
                lexical_hits = bm25.search(example["question"], limit=candidate_k)
                lexical_ids = [chunk_id for chunk_id, _score in lexical_hits]
                lexical_scores = dict(lexical_hits)

            if args.retriever == "dense":
                candidate_ids = dense_ids[:candidate_k]
            elif args.retriever == "bm25":
                candidate_ids = lexical_ids[:candidate_k]
            else:
                candidate_ids = reciprocal_rank_fusion(
                    dense_ids,
                    lexical_ids,
                    limit=candidate_k,
                )

            reranker_scores = {}
            if reranker is not None:
                ranked_candidates = rerank_candidates(
                    reranker,
                    example["question"],
                    candidate_ids,
                    chunk_by_id,
                    args.top_k,
                )
                ranked_ids = [chunk_id for chunk_id, _score in ranked_candidates]
                reranker_scores = dict(ranked_candidates)
            else:
                ranked_ids = candidate_ids[: args.top_k]

            relevance = example["relevant_chunk_ids"]
            metrics = evaluate_query(relevance, ranked_ids, args.top_k)
            results = []
            for rank, chunk_id in enumerate(ranked_ids, start=1):
                point = dense_points.get(chunk_id)
                payload = (point.payload if point else None) or chunk_by_id.get(chunk_id) or {}
                score = reranker_scores.get(chunk_id)
                if score is None and point is not None:
                    score = point.score
                elif score is None and chunk_id in lexical_scores:
                    score = lexical_scores[chunk_id]
                results.append({
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "relevance": relevance.get(chunk_id, 0),
                    "score": score,
                    "document_id": payload.get("document_id"),
                    "source_file": payload.get("source_file"),
                    "section_title": payload.get("section_title"),
                    "text": payload.get("text", ""),
                })
            query_results.append({
                "query_id": example["query_id"],
                "question": example["question"],
                "category": example.get("category", "unspecified"),
                "metrics": metrics,
                "relevant_chunk_ids": relevance,
                "results": results,
            })
    finally:
        if client is not None:
            client.close()

    metric_names = query_results[0]["metrics"].keys()
    summary = {name: mean_metric([item["metrics"] for item in query_results], name) for name in metric_names}
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "dataset_version": "1.0.0",
        "retriever": args.retriever,
        "reranker_model": args.reranker_model,
        "collection": args.collection,
        "embedding_model": args.model,
        "top_k": args.top_k,
        "candidate_k": candidate_k,
        "query_count": len(query_results),
        "summary": summary,
        "queries": query_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Evaluated queries: {len(query_results)}")
    for name, value in summary.items():
        rendered = "n/a" if value is None else f"{value:.4f}"
        print(f"{name}: {rendered}")
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
