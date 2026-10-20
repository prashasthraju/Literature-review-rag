"""Ablation harness behind docs/retrieval_results.md.

Runs the shared Retriever with an in-memory dense index (identical rankings to
Qdrant, no re-indexing) over any chunks file, and swaps the BM25 input so the
baseline configuration can be reproduced next to the current one:

  raw_nostop  BM25 over raw chunk text, no stopwords (pipeline before 2026-10-19)
  ctx_stop    BM25 over passage_for_record (title + section + cleaned text), stopwords

Examples:
  # baseline: original token chunks, old BM25
  git show a0735cf:chunks.jsonl > /tmp/chunks_a0735cf.jsonl
  python scripts/ablate_retrieval.py --chunks /tmp/chunks_a0735cf.jsonl --variants raw_nostop --output reports/ablations/baseline.json
  # current base pipeline, then with the reranker
  python scripts/ablate_retrieval.py --variants ctx_stop --output reports/ablations/current.json
  python scripts/ablate_retrieval.py --variants ctx_stop --reranker BAAI/bge-reranker-base --output reports/ablations/current_reranked.json

The output maps "<bm25 variant>/<mode>" to per-query evidence metrics.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embed_text import passage_for_record
from src.evaluation.evidence import EvidenceResolver, evaluate_evidence
from src.evaluation.metrics import mean_metric
from src.lexical import BM25Index
from src.retrieval import CANDIDATE_K, MODEL_NAME, TOP_K, Retriever, load_chunks

DATASET = ROOT / "data" / "evaluation" / "retrieval_dev.jsonl"
BM25_VARIANTS = {
    "raw_nostop": lambda records: BM25Index(
        ((record["chunk_id"], record.get("text", "")) for record in records), stopwords=frozenset()
    ),
    "ctx_stop": lambda records: BM25Index(
        (record["chunk_id"], passage_for_record(record)) for record in records
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunks", default=str(ROOT / "chunks.jsonl"))
    parser.add_argument("--variants", default=",".join(BM25_VARIANTS))
    parser.add_argument("--modes", default="dense,bm25,hybrid")
    parser.add_argument("--reranker", help="Cross-encoder model; implies --modes hybrid")
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--no-title", action="store_true", help="Drop document_title from every chunk")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    examples = [json.loads(line) for line in DATASET.open(encoding="utf-8") if line.strip()]
    records = load_chunks(args.chunks)
    if args.no_title:
        for record in records:
            record.pop("document_title", None)
    resolver = EvidenceResolver(records)
    unresolved = [
        (example["query_id"], unit["anchor"])
        for example in examples
        for unit in example["evidence"]
        if not resolver.resolve_unit(unit)
    ]
    print(f"chunks={len(records)} unresolved_anchors={len(unresolved)}")
    for query_id, anchor in unresolved:
        print(f"  unresolved {query_id}: {anchor}")

    modes = ["hybrid"] if args.reranker else args.modes.split(",")
    retriever = Retriever(
        records,
        mode="hybrid",
        model_name=args.model,
        reranker_model=args.reranker,
        candidate_k=args.candidate_k,
        dense_backend="memory",
    )
    results = {}
    for variant in args.variants.split(","):
        retriever.bm25 = BM25_VARIANTS[variant](records)
        for mode in modes:
            retriever.mode = mode
            per_query = {}
            for example in examples:
                ranked = [hit["chunk_id"] for hit in retriever.search(example["question"], top_k=TOP_K)]
                per_query[example["query_id"]] = evaluate_evidence(resolver.resolve(example), ranked, TOP_K)
            rows = list(per_query.values())
            summary = {metric: round(mean_metric(rows, metric), 4) for metric in rows[0]}
            print(f"{variant:11s} {mode:6s} {summary}")
            results[f"{variant}/{mode}"] = per_query

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "chunks": str(args.chunks),
        "chunk_count": len(records),
        "unresolved_anchors": len(unresolved),
        "model": args.model,
        "reranker": args.reranker,
        "candidate_k": args.candidate_k,
        "no_title": args.no_title,
        "results": results,
    }, indent=1), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
