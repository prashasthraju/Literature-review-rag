import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

if __package__:
    from .metrics import evaluate_query, mean_metric
else:
    from metrics import evaluate_query, mean_metric

DEFAULT_DATASET = Path("data/evaluation/retrieval_dev.jsonl")
DEFAULT_DB_PATH = Path("qdrant_db")
DEFAULT_COLLECTION = "research_papers_v1"
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"


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
    parser.add_argument("--output", type=Path, default=Path("reports/retrieval_eval_latest.json"))
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    examples = read_dataset(args.dataset)
    print(f"Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)
    client = QdrantClient(path=str(args.db_path))
    query_results = []

    try:
        if not client.collection_exists(args.collection):
            raise RuntimeError(f"Qdrant collection not found: {args.collection}")

        for example in examples:
            query = "Represent this sentence for searching relevant passages: " + example["question"]
            vector = model.encode(query, normalize_embeddings=True).tolist()
            points = client.query_points(
                collection_name=args.collection,
                query=vector,
                limit=args.top_k,
                with_payload=True,
            ).points

            ranked_ids = [point.payload.get("chunk_id") for point in points if point.payload]
            relevance = example["relevant_chunk_ids"]
            metrics = evaluate_query(relevance, ranked_ids, args.top_k)
            query_results.append({
                "query_id": example["query_id"],
                "question": example["question"],
                "category": example.get("category", "unspecified"),
                "metrics": metrics,
                "relevant_chunk_ids": relevance,
                "results": [
                    {
                        "rank": rank,
                        "chunk_id": point.payload.get("chunk_id"),
                        "relevance": relevance.get(point.payload.get("chunk_id"), 0),
                        "score": point.score,
                        "document_id": point.payload.get("document_id"),
                        "source_file": point.payload.get("source_file"),
                        "section_title": point.payload.get("section_title"),
                        "text": point.payload.get("text", ""),
                    }
                    for rank, point in enumerate(points, start=1)
                    if point.payload
                ],
            })
    finally:
        client.close()

    metric_names = query_results[0]["metrics"].keys()
    summary = {name: mean_metric([item["metrics"] for item in query_results], name) for name in metric_names}
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "dataset_version": "1.0.0",
        "retriever": "qdrant_dense",
        "collection": args.collection,
        "embedding_model": args.model,
        "top_k": args.top_k,
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
