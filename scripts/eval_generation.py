"""Evaluate context assembly and, optionally, generated answers on a labeled dataset.

Context (no model needed):
  evidence coverage   share of evidence units present in the assembled context
Generation (--generate):
  abstention accuracy answerable items answered/partial, unanswerable items insufficient_evidence
  citation validity   share of returned citations whose source exists and quote is found in it
  citation coverage   share of answer sentences carrying a verified citation
  cited-evidence rate answered items whose verified citations include a passage holding labeled evidence
Every answer is saved for manual review; automatic scores are not a substitute for it.
--revalidate re-scores a saved report with the current validator (no model needed).

  python scripts/eval_generation.py                      # context only, dev set
  python scripts/eval_generation.py --generate --limit 30
  python scripts/eval_generation.py --revalidate reports/generation/retrieval_dev/<run>.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.context import assemble_context, evidence_coverage
from src.evaluation.evidence import EvidenceResolver, is_answerable
from src.evaluation.runner import read_dataset
from src.generation.answer import invalid_output, load_aliases, named_documents, parse_draft, validate
from src.rag import GENERATION_CONFIG, GenerationConfig, RagPipeline
from src.retrieval import RetrievalConfig, Retriever, load_chunks


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def score_answer(row, answer, example, resolver):
    answerable = is_answerable(example)
    evidence_chunks = set().union(*(ids for _grade, ids in resolver.resolve(example))) if answerable else set()
    valid = [c for c in answer.citations if c.valid]
    row.update({
        "status": answer.status,
        "answer": answer.answer,
        "citations": [c.model_dump() for c in answer.citations],
        "caveats": answer.caveats,
        "problems": answer.problems,
        "unsupported_numbers": answer.unsupported_numbers,
        "raw_output": answer.raw_output,
        "citation_coverage": answer.citation_coverage,
        "abstention_correct": (answer.status == "insufficient_evidence") != answerable,
        "citation_validity": len(valid) / len(answer.citations) if answer.citations else None,
        "cites_evidence": bool(evidence_chunks & {cid for c in valid for cid in c.chunk_ids})
        if answerable and answer.status in {"answered", "partial"} else None,
    })
    return row


def summarize(rows, generated):
    answerable_rows = [row for row in rows if row["answerable"]]
    summary = {
        "questions": len(rows),
        "answerable": len(answerable_rows),
        "context_evidence_coverage": mean(row["context_coverage"] for row in answerable_rows),
        "context_full_coverage_rate": mean(float(row["context_coverage"] == 1.0) for row in answerable_rows),
        "mean_context_tokens": mean(row["context_tokens"] for row in rows),
    }
    if generated:
        summary.update({
            "abstention_accuracy": mean(float(row["abstention_correct"]) for row in rows),
            "answered_rate_on_answerable": mean(float(row["status"] in {"answered", "partial"}) for row in answerable_rows),
            "abstain_rate_on_unanswerable": mean(
                float(row["status"] == "insufficient_evidence") for row in rows if not row["answerable"]
            ),
            "citation_validity": mean(row["citation_validity"] for row in rows),
            "approximate_quote_rate": mean(
                float(c["match"] == "approximate") for row in rows for c in row["citations"] if c["valid"]
            ),
            "citation_coverage": mean(row["citation_coverage"] for row in rows),
            "cited_evidence_rate": mean(float(row["cites_evidence"]) for row in rows if row["cites_evidence"] is not None),
            "invalid_output_rate": mean(float(row["status"] == "invalid_output") for row in rows),
            "unsupported_number_rate": mean(
                float(bool(row["unsupported_numbers"])) for row in rows if row["status"] in {"answered", "partial"}
            ),
        })
    return summary


def revalidate(report_path, dataset, records, resolver):
    """Re-run parsing and validation on saved raw outputs, rebuilding each context from its log."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    by_id = {record["chunk_id"]: record for record in records}
    examples = {example["query_id"]: example for example in read_dataset(dataset)}
    config = GenerationConfig(**report["generation_config"])
    aliases = load_aliases()
    for row in report["rows"]:
        example = examples[row["query_id"]]
        ranked = row["context"].get("ranked_chunk_ids") or [
            chunk_id for source in row["context"]["included"] for chunk_id in source["chunk_ids"]
        ]
        hits = [{"chunk_id": chunk_id, "record": by_id[chunk_id]} for chunk_id in ranked]
        bundle = assemble_context(hits, budget=config.context_budget)
        try:
            answer = validate(
                example["question"], parse_draft(row["raw_output"]), bundle, config.prompt_version,
                config.generator, row["raw_output"],
                required_documents=named_documents(example["question"], aliases),
            )
        except ValueError as error:
            answer = invalid_output(example["question"], row["raw_output"], error, bundle, config.prompt_version, config.generator)
        score_answer(row, answer, example, resolver)
    report["summary"] = summarize(report["rows"], generated=True)
    report["revalidated_at"] = datetime.now(timezone.utc).isoformat()
    Path(report_path).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=1))
    print(f"Revalidated {report_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/evaluation/retrieval_dev.jsonl")
    parser.add_argument("--config", default=str(GENERATION_CONFIG))
    parser.add_argument("--retrieval-config", help="Override the generation config's retrieval config")
    parser.add_argument("--generator")
    parser.add_argument("--prompt-version")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--revalidate", type=Path, help="Re-score a saved report's raw outputs with the current validator")
    args = parser.parse_args()

    records = load_chunks()
    resolver = EvidenceResolver(records)
    if args.revalidate:
        revalidate(args.revalidate, args.dataset, records, resolver)
        return

    config = GenerationConfig.load(
        args.config, generator=args.generator, retrieval_config=args.retrieval_config, prompt_version=args.prompt_version
    )
    examples = read_dataset(args.dataset)[: args.limit]
    if args.generate:
        pipeline = RagPipeline(config, records=records)
        retriever = pipeline.retriever
    else:
        pipeline = None
        retriever = Retriever(records, RetrievalConfig.load(config.retrieval_config))

    rows = []
    try:
        for example in examples:
            answerable = is_answerable(example)
            hits = retriever.search(example["question"], top_k=config.context_hits)
            bundle = assemble_context(hits, budget=config.context_budget)
            row = {
                "query_id": example["query_id"],
                "category": example.get("category"),
                "answerable": answerable,
                "question": example["question"],
                "expected_answer_points": example.get("expected_answer_points"),
                "context_coverage": evidence_coverage(bundle, example["evidence"]) if answerable else None,
                "context_tokens": bundle.token_count,
                "context": bundle.log(),
            }
            if pipeline:
                score_answer(row, pipeline.answer_from_hits(example["question"], hits), example, resolver)
            rows.append(row)
            print(f"{example['query_id']:28s} coverage {row['context_coverage']}  {row.get('status', '')}")
    finally:
        retriever.close()

    summary = summarize(rows, generated=pipeline is not None)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset.name,
        "generation_config": config.__dict__,
        "summary": summary,
        "rows": rows,
    }
    name = f"{config.retrieval_config}" + (
        f"__{config.generator.replace('/', '_').replace(':', '-')}__{config.prompt_version}" if pipeline else "__context"
    )
    output = args.output or ROOT / "reports" / "generation" / args.dataset.stem / f"{name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
