import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evidence import EvidenceResolver, evaluate_evidence
from src.evaluation.metrics import evaluate_query, mean_metric
from src.retrieval import (
    CANDIDATE_K,
    COLLECTION,
    DB_PATH,
    MODEL_NAME,
    RETRIEVERS,
    Retriever,
    load_chunks,
)

DEFAULT_DATASET = Path("data/evaluation/retrieval_dev.jsonl")
DATASET_VERSION = "1.1.0"


def read_dataset(path):
    examples = []
    with path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            example = json.loads(line)
            if "query_id" not in example or "question" not in example:
                raise ValueError(f"Missing query_id or question at {path}:{line_number}")
            if "evidence" not in example and "relevant_chunk_ids" not in example:
                raise ValueError(f"Missing evidence or relevant_chunk_ids at {path}:{line_number}")
            examples.append(example)
    if not examples:
        raise ValueError(f"No evaluation examples found in {path}")
    return examples


def summarize(rows):
    names = rows[0].keys()
    return {name: mean_metric(rows, name) for name in names}


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval against evidence-level relevance judgments.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--chunks", type=Path, default=ROOT / "chunks.jsonl")
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--retriever",
        choices=RETRIEVERS,
        default="hybrid",
        help="dense = embeddings, bm25 = lexical, hybrid = RRF of both",
    )
    parser.add_argument(
        "--dense-backend",
        choices=("qdrant", "memory"),
        default="qdrant",
        help="memory encodes --chunks on the fly, to score a chunking without rebuilding Qdrant",
    )
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    parser.add_argument(
        "--reranker-model",
        help="Optional sentence-transformers cross-encoder used to rerank the candidate pool",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/retrieval_eval_latest.json"))
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    examples = read_dataset(args.dataset)
    records = load_chunks(args.chunks)
    resolver = EvidenceResolver(records)
    chunk_ids = {record["chunk_id"] for record in records}

    unresolved = [
        (example["query_id"], unit["anchor"])
        for example in examples
        for unit in example.get("evidence", [])
        if unit["grade"] > 0 and not resolver.resolve_unit(unit)
    ]
    if unresolved:
        print(f"WARNING: {len(unresolved)} evidence anchors match no chunk in {args.chunks}:")
        for query_id, anchor in unresolved:
            print(f"  {query_id}: {anchor}")

    # Legacy chunk-ID labels are only meaningful for the chunking they were made on.
    legacy_labels_valid = all(
        chunk_id in chunk_ids
        for example in examples
        for chunk_id in example.get("relevant_chunk_ids", {})
    )

    print(f"Loading {args.retriever} retriever ({args.dense_backend} dense index)")
    retriever = Retriever(
        records,
        mode=args.retriever,
        model_name=args.model,
        reranker_model=args.reranker_model,
        candidate_k=args.candidate_k,
        dense_backend=args.dense_backend,
        db_path=args.db_path,
        collection=args.collection,
    )

    query_results = []
    try:
        for example in examples:
            hits = retriever.search(example["question"], top_k=args.top_k)
            ranked_ids = [hit["chunk_id"] for hit in hits]
            units = resolver.resolve(example)
            metrics = evaluate_evidence(units, ranked_ids, args.top_k)
            legacy = example.get("relevant_chunk_ids", {})
            legacy_metrics = (
                evaluate_query(legacy, ranked_ids, args.top_k) if legacy_labels_valid and legacy else None
            )

            results = []
            for rank, hit in enumerate(hits, start=1):
                record = hit["record"]
                results.append({
                    "rank": rank,
                    "chunk_id": hit["chunk_id"],
                    "supports_evidence": [
                        index for index, (_grade, ids) in enumerate(units) if hit["chunk_id"] in ids
                    ],
                    "relevance": max(
                        (grade for grade, ids in units if hit["chunk_id"] in ids), default=0
                    ),
                    "score": hit["score"],
                    "dense_score": hit["dense_score"],
                    "bm25_score": hit["bm25_score"],
                    "rerank_score": hit["rerank_score"],
                    "document_id": record.get("document_id"),
                    "source_file": record.get("source_file"),
                    "section_title": record.get("section_title"),
                    "page_numbers": record.get("page_numbers"),
                    "text": record.get("text", ""),
                })
            query_results.append({
                "query_id": example["query_id"],
                "question": example["question"],
                "category": example.get("category", "unspecified"),
                "metrics": metrics,
                "legacy_chunk_metrics": legacy_metrics,
                "evidence": example.get("evidence", []),
                "results": results,
            })
    finally:
        retriever.close()

    summary = summarize([item["metrics"] for item in query_results])
    legacy_summary = (
        summarize([item["legacy_chunk_metrics"] for item in query_results if item["legacy_chunk_metrics"]])
        if legacy_labels_valid
        else None
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset.as_posix(),
        "dataset_version": DATASET_VERSION,
        "chunks": args.chunks.as_posix(),
        "chunk_count": len(records),
        "retriever": args.retriever,
        "dense_backend": args.dense_backend,
        "reranker_model": args.reranker_model,
        "collection": args.collection,
        "embedding_model": args.model,
        "top_k": args.top_k,
        "candidate_k": max(args.top_k, args.candidate_k),
        "query_count": len(query_results),
        "evidence_unit_count": sum(len(resolver.resolve(example)) for example in examples),
        "unresolved_evidence": [{"query_id": qid, "anchor": anchor} for qid, anchor in unresolved],
        "summary": summary,
        "legacy_chunk_label_summary": legacy_summary,
        "queries": query_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Evaluated queries: {len(query_results)} ({report['evidence_unit_count']} evidence units)")
    for name, value in summary.items():
        rendered = "n/a" if value is None else f"{value:.4f}"
        print(f"{name}: {rendered}")
    if legacy_summary:
        rendered = ", ".join(f"{name} {value:.4f}" for name, value in legacy_summary.items())
        print(f"legacy chunk-ID labels: {rendered}")
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
