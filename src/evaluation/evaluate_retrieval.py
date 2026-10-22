import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evidence import EvidenceResolver, is_answerable
from src.evaluation.metrics import evaluate_query
from src.evaluation.runner import aggregate, read_dataset, run_queries, summarize
from src.retrieval import DB_PATH, DEFAULT_CONFIG, RETRIEVERS, RetrievalConfig, Retriever, load_chunks

DEFAULT_DATASET = Path("data/evaluation/retrieval_dev.jsonl")
DATASET_VERSION = "2.0.0"


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval against evidence-level relevance judgments.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Retrieval config name or path (configs/retrieval/)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--chunks", type=Path, default=ROOT / "chunks.jsonl")
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--collection", help="Override the config's Qdrant collection")
    parser.add_argument("--model", help="Override the config's embedding model")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--retriever", choices=RETRIEVERS, help="Score a single channel (dense or bm25)")
    parser.add_argument(
        "--dense-backend",
        choices=("qdrant", "memory"),
        default="qdrant",
        help="memory encodes --chunks on the fly, to score a chunking without rebuilding Qdrant",
    )
    parser.add_argument("--candidate-k", type=int)
    parser.add_argument("--reranker-model", help="Override the config's cross-encoder; 'none' disables reranking")
    parser.add_argument("--output", type=Path, default=Path("reports/retrieval_eval_latest.json"))
    args = parser.parse_args()

    overrides = {
        "collection": args.collection,
        "embedding_model": args.model,
        "top_k": args.top_k,
        "retriever": args.retriever,
        "candidate_k": args.candidate_k,
    }
    if args.reranker_model:
        overrides["reranker"] = None if args.reranker_model.lower() == "none" else args.reranker_model
    overrides = {key: value for key, value in overrides.items() if key == "reranker" or value is not None}
    config = replace(RetrievalConfig.load(args.config), **overrides)
    if config.top_k <= 0:
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

    print(f"Loading {config.name} ({config.retriever}, {args.dense_backend} dense index)")
    retriever = Retriever(records, config, dense_backend=args.dense_backend, db_path=args.db_path)
    try:
        results = run_queries(retriever, examples, resolver, config.top_k)
    finally:
        retriever.close()

    query_results = []
    legacy_rows = []
    for item in results:
        example, units = item["example"], item["units"]
        legacy = example.get("relevant_chunk_ids", {})
        legacy_metrics = None
        if legacy_labels_valid and legacy:
            legacy_metrics = evaluate_query(legacy, [hit["chunk_id"] for hit in item["hits"]], config.top_k)
            legacy_rows.append(legacy_metrics)
        query_results.append({
            "query_id": example["query_id"],
            "question": example["question"],
            "category": example.get("category", "unspecified"),
            "answerable": is_answerable(example),
            "metrics": item["metrics"],
            "legacy_chunk_metrics": legacy_metrics,
            "latency_ms": item["latency_ms"],
            "evidence": example.get("evidence", []),
            "results": [
                {
                    "rank": rank,
                    "chunk_id": hit["chunk_id"],
                    "supports_evidence": [
                        index for index, (_grade, ids) in enumerate(units) if hit["chunk_id"] in ids
                    ],
                    "relevance": max((grade for grade, ids in units if hit["chunk_id"] in ids), default=0),
                    "score": hit["score"],
                    "dense_score": hit["dense_score"],
                    "bm25_score": hit["bm25_score"],
                    "rerank_score": hit["rerank_score"],
                    "document_id": hit["record"].get("document_id"),
                    "source_file": hit["record"].get("source_file"),
                    "section_title": hit["record"].get("section_title"),
                    "page_numbers": hit["record"].get("page_numbers"),
                    "text": hit["record"].get("text", ""),
                }
                for rank, hit in enumerate(item["hits"], start=1)
            ],
        })

    aggregated = aggregate(results)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset.as_posix(),
        "dataset_version": DATASET_VERSION,
        "chunks": args.chunks.name if args.chunks.parent == ROOT else args.chunks.as_posix(),
        "chunk_count": len(records),
        "config": config.to_dict(),
        "dense_backend": args.dense_backend,
        "query_count": len(query_results),
        "evidence_unit_count": sum(len(item["units"]) for item in results),
        "unresolved_evidence": [{"query_id": qid, "anchor": anchor} for qid, anchor in unresolved],
        **aggregated,
        "legacy_chunk_label_summary": summarize(legacy_rows) if legacy_rows else None,
        "queries": query_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Evaluated {aggregated['answerable_count']} answerable questions "
        f"({report['evidence_unit_count']} evidence units); "
        f"{aggregated['unanswerable_count']} unanswerable questions excluded from metrics"
    )
    for name, value in aggregated["summary"].items():
        rendered = "n/a" if value is None else f"{value:.4f}"
        print(f"{name}: {rendered}")
    print(f"latency p50/p95 ms: {aggregated['latency_ms']['p50']:.1f} / {aggregated['latency_ms']['p95']:.1f}")
    if report["legacy_chunk_label_summary"]:
        rendered = ", ".join(f"{name} {value:.4f}" for name, value in report["legacy_chunk_label_summary"].items())
        print(f"legacy chunk-ID labels: {rendered}")
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
