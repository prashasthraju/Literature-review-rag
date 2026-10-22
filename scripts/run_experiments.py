"""Score retrieval configurations on a labeled dataset with an in-memory dense index.

Each configuration in configs/retrieval/ is run through the shared Retriever,
so results describe exactly what search and the evaluator do. Embedding
models, encoded passages and rerankers are reused across configurations.
Output: one JSON per configuration with per-query metrics, a per-category
summary and latency, under reports/experiments/<dataset>/.

Examples:
  python scripts/run_experiments.py retrieval-v1-base retrieval-v1
  # baseline before 2026-10-19 (needs the v1.1 dataset and the original chunks)
  git show 7166cf3:data/evaluation/retrieval_dev.jsonl > /tmp/dev_v1.1.jsonl
  git show a0735cf:chunks.jsonl > /tmp/chunks_a0735cf.jsonl
  python scripts/run_experiments.py baseline-v0 --dataset /tmp/dev_v1.1.jsonl --chunks /tmp/chunks_a0735cf.jsonl

A config may carry inline overrides, e.g. retrieval-v1-base:dense_units=window,retriever=dense
(values are parsed as JSON when possible). Compare two runs with scripts/compare_runs.py.
"""

import argparse
import hashlib
import json
import re
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.evaluation.evidence import EvidenceResolver
from src.evaluation.runner import aggregate, read_dataset, run_queries
from src.retrieval import MemoryDenseIndex, RetrievalConfig, Retriever, dense_units, load_chunks

CACHE_DIR = ROOT / ".cache" / "dense"


def load_config(spec):
    """Config name or path, optionally followed by :key=value,key=value overrides."""
    name, _, raw = spec.partition(":")
    overrides = {}
    for item in filter(None, raw.split(",")):
        key, _, value = item.partition("=")
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value
    config = RetrievalConfig.load(name, **overrides)
    if overrides:
        suffix = "+".join(f"{key}={value}" for key, value in overrides.items())
        config = replace(config, name=re.sub(r"[^A-Za-z0-9_.=+-]+", "_", f"{config.name}+{suffix}"))
    return config


class Resources:
    def __init__(self, records, chunks_digest):
        self.records = records
        self.chunks_digest = chunks_digest
        self.models = {}
        self.rerankers = {}
        self.indexes = {}

    def model(self, name):
        if name not in self.models:
            from sentence_transformers import SentenceTransformer

            self.models[name] = SentenceTransformer(name, trust_remote_code=True)
        return self.models[name]

    def reranker(self, name):
        if name and name not in self.rerankers:
            from sentence_transformers import CrossEncoder

            self.rerankers[name] = CrossEncoder(name)
        return self.rerankers.get(name)

    def dense_index(self, config):
        if config.retriever == "bm25":
            return None
        key = json.dumps([
            self.chunks_digest, config.embedding_model, config.dense_units, config.dense_title,
            config.window_words, config.window_stride,
        ])
        if key not in self.indexes:
            digest = hashlib.sha256(key.encode()).hexdigest()[:16]
            path = CACHE_DIR / f"{digest}.npy"
            model = self.model(config.embedding_model)
            if path.exists():
                vectors = np.load(path)
            else:
                texts = [text for _chunk_id, text in dense_units(self.records, config)]
                vectors = model.encode(
                    texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, vectors)
            self.indexes[key] = MemoryDenseIndex(model, self.records, config, vectors=vectors)
        return self.indexes[key]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("configs", nargs="+", help="Config names or paths")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/evaluation/retrieval_dev.jsonl")
    parser.add_argument("--chunks", type=Path, default=ROOT / "chunks.jsonl")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tag", default="", help="Suffix for output file names")
    args = parser.parse_args()

    examples = read_dataset(args.dataset)
    records = load_chunks(args.chunks)
    resolver = EvidenceResolver(records)
    unresolved = [
        (example["query_id"], unit["anchor"])
        for example in examples
        for unit in example.get("evidence", [])
        if not resolver.resolve_unit(unit)
    ]
    print(f"{len(examples)} questions, {len(records)} chunks, {len(unresolved)} unresolved anchors")
    for query_id, anchor in unresolved:
        print(f"  unresolved {query_id}: {anchor}")

    resources = Resources(records, hashlib.sha256(args.chunks.read_bytes()).hexdigest()[:16])
    output_dir = args.output_dir or ROOT / "reports" / "experiments" / args.dataset.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in args.configs:
        config = load_config(name)
        dense = resources.dense_index(config)
        retriever = Retriever(
            records,
            config,
            dense_backend="memory",
            model=resources.model(config.embedding_model) if dense else None,
            dense_index=dense,
            reranker=resources.reranker(config.reranker),
        )
        results = run_queries(retriever, examples, resolver, config.top_k)
        aggregated = aggregate(results)
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": args.dataset.name,
            "chunks": args.chunks.name,
            "chunk_count": len(records),
            "unresolved_anchors": len(unresolved),
            "config": config.to_dict(),
            **aggregated,
            "per_query": {
                item["example"]["query_id"]: {
                    "category": item["example"].get("category"),
                    "latency_ms": round(item["latency_ms"], 2),
                    **(item["metrics"] or {}),
                }
                for item in results
                if item["answerable"]
            },
        }
        path = output_dir / f"{config.name}{args.tag}.json"
        path.write_text(json.dumps(report, indent=1), encoding="utf-8")
        summary = {key: round(value, 4) for key, value in aggregated["summary"].items()}
        print(f"{config.name:32s} {summary} p50 {aggregated['latency_ms']['p50']:.0f} ms -> {path}")


if __name__ == "__main__":
    main()
