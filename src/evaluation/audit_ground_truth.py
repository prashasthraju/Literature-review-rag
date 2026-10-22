import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evidence import EvidenceResolver, is_answerable, unit_alternatives, unit_supports

QUESTIONS_FILE = ROOT / "data" / "evaluation" / "retrieval_dev.jsonl"
CHUNKS_FILE = ROOT / "chunks.jsonl"


def load_jsonl(path):
    with path.open(encoding="utf-8") as data_file:
        return [json.loads(line) for line in data_file if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Check that every evidence anchor resolves to a chunk.")
    parser.add_argument("--dataset", type=Path, default=QUESTIONS_FILE)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_FILE)
    parser.add_argument("--quiet", action="store_true", help="Only print problems and totals")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    questions = load_jsonl(args.dataset)
    chunks = load_jsonl(args.chunks)
    chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks if chunk.get("chunk_id")}
    resolver = EvidenceResolver(chunks)

    print(f"Questions: {len(questions)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Duplicate chunk IDs: {len(chunks) - len(chunk_map)}")

    problems = []
    total_units = 0
    legacy_total = 0
    legacy_present = 0

    unanswerable = 0
    for index, item in enumerate(questions, start=1):
        query_id = item.get("query_id", f"row-{index}")
        if not args.quiet:
            print(f"\n{'=' * 80}")
            print(f"{query_id}: {item.get('question', '<missing question>')}")

        legacy = item.get("relevant_chunk_ids", {})
        legacy_total += len(legacy)
        legacy_present += sum(chunk_id in chunk_map for chunk_id in legacy)

        evidence = item.get("evidence")
        if not is_answerable(item):
            unanswerable += 1
            if evidence:
                problems.append((query_id, "unanswerable item has evidence units"))
            continue
        if not evidence:
            problems.append((query_id, "no evidence units"))
            continue

        for unit in evidence:
            total_units += 1
            grade = unit.get("grade")
            if not args.quiet:
                print(f"\nEvidence (grade {grade}, {unit.get('document_id')}): {unit_alternatives(unit)}")
            if not isinstance(grade, (int, float)) or grade <= 0:
                problems.append((query_id, f"invalid grade {grade!r}"))
            matches = sorted(resolver.resolve_unit(unit))
            if not matches:
                problems.append((query_id, f"unresolved anchor {unit.get('anchor')!r}"))
                continue
            for document_id, anchors in unit_supports(unit)[1:]:
                if not resolver.resolve_unit({"document_id": document_id, "anchor": anchors}):
                    problems.append((query_id, f"unresolved alternative {anchors!r} in {document_id}"))
            if args.quiet:
                continue
            for chunk_id in matches:
                chunk = chunk_map[chunk_id]
                pages = chunk.get("page_numbers") or []
                print(f"  -> {chunk_id} | {chunk.get('section_title')} | pages {pages}")
                print("    ", (chunk.get("text") or "")[:400].replace("\n", " "))

    print(f"\n{'=' * 80}")
    print(f"Questions: {len(questions)} ({unanswerable} unanswerable)")
    print(f"Evidence units: {total_units}")
    print(
        f"Legacy chunk-ID labels present in the current corpus: {legacy_present}/{legacy_total}"
        " (informational; evidence anchors are the source of truth)"
    )
    print(f"Problems: {len(problems)}")

    if problems or len(chunks) != len(chunk_map):
        print("\nFix these before trusting the metrics:")
        for query_id, problem in problems:
            print(f"{query_id}: {problem}")
        raise SystemExit(1)
    print("\nAll evidence anchors resolve. Now manually verify that each matched chunk answers its question.")


if __name__ == "__main__":
    main()
