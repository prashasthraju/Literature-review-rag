"""Run a retriever over labeled questions and aggregate evidence-level metrics.

Shared by the evaluator (one configuration, full ranked output) and the
experiment runner (many configurations, metrics only).
"""

import json
import time
from collections import defaultdict
from pathlib import Path

from src.evaluation.evidence import evaluate_evidence, is_answerable
from src.evaluation.metrics import mean_metric


def read_dataset(path):
    examples = []
    with Path(path).open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            example = json.loads(line)
            if "query_id" not in example or "question" not in example:
                raise ValueError(f"Missing query_id or question at {path}:{line_number}")
            if is_answerable(example) and not example.get("evidence"):
                raise ValueError(f"Answerable item without evidence at {path}:{line_number}")
            examples.append(example)
    if not examples:
        raise ValueError(f"No evaluation examples found in {path}")
    return examples


def summarize(rows):
    if not rows:
        return {}
    return {name: mean_metric(rows, name) for name in rows[0]}


def run_queries(retriever, examples, resolver, top_k):
    """Search every question; answerable ones get evidence metrics.

    Unanswerable items are searched too (their top results are what a
    generator would see) but are excluded from retrieval metrics.
    """
    results = []
    for example in examples:
        start = time.perf_counter()
        hits = retriever.search(example["question"], top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        answerable = is_answerable(example)
        units = resolver.resolve(example) if answerable else []
        ranked_ids = [hit["chunk_id"] for hit in hits]
        results.append({
            "example": example,
            "hits": hits,
            "units": units,
            "answerable": answerable,
            "latency_ms": latency_ms,
            "metrics": evaluate_evidence(units, ranked_ids, top_k) if answerable else None,
        })
    return results


def aggregate(results):
    scored = [item for item in results if item["answerable"]]
    by_category = defaultdict(list)
    for item in scored:
        by_category[item["example"].get("category", "unspecified")].append(item["metrics"])
    latencies = sorted(item["latency_ms"] for item in results)
    return {
        "summary": summarize([item["metrics"] for item in scored]),
        "by_category": {
            category: {"count": len(rows), **summarize(rows)}
            for category, rows in sorted(by_category.items())
        },
        "answerable_count": len(scored),
        "unanswerable_count": len(results) - len(scored),
        "latency_ms": {
            "mean": sum(latencies) / len(latencies),
            "p50": latencies[len(latencies) // 2],
            "p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
        },
    }
