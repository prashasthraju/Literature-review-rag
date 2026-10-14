import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_FILE = ROOT / "data" / "evaluation" / "retrieval_dev.jsonl"
CHUNKS_FILE = ROOT / "chunks.jsonl"


def load_jsonl(path):
    with path.open(encoding="utf-8") as data_file:
        return [json.loads(line) for line in data_file if line.strip()]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    questions = load_jsonl(QUESTIONS_FILE)
    chunks = load_jsonl(CHUNKS_FILE)

    chunk_counts = Counter(
        chunk.get("chunk_id") for chunk in chunks if chunk.get("chunk_id")
    )
    chunk_map = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
        if chunk.get("chunk_id")
    }

    print(f"Questions: {len(questions)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Unique chunk IDs: {len(chunk_map)}")
    print(f"Duplicate chunk IDs: {sum(count > 1 for count in chunk_counts.values())}")

    invalid_labels = []
    total_labels = 0

    for index, item in enumerate(questions, start=1):
        query_id = item.get("query_id", f"row-{index}")
        question = item.get("question", "<missing question>")
        relevance = item.get("relevant_chunk_ids")

        print(f"\n{'=' * 80}")
        print(f"{query_id}: {question}")

        if not isinstance(relevance, dict):
            print("ERROR: relevant_chunk_ids must be an object mapping chunk IDs to grades.")
            invalid_labels.append((query_id, question, "invalid relevance format"))
            continue

        total_labels += len(relevance)
        print(f"Labeled relevant chunks: {len(relevance)}")

        for chunk_id, grade in relevance.items():
            if not isinstance(grade, (int, float)) or grade <= 0:
                print(f"ERROR: invalid relevance grade for {chunk_id}: {grade!r}")
                invalid_labels.append((query_id, question, chunk_id))

            print(f"\nChunk ID: {chunk_id} (grade {grade})")
            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                print("ERROR: ID DOES NOT EXIST IN chunks.jsonl")
                invalid_labels.append((query_id, question, chunk_id))
                continue

            print("Source:", chunk.get("source_file", "unknown"))
            print("Section:", chunk.get("section_path", "unknown"))
            print("Text:", (chunk.get("text") or "")[:700].replace("\n", " "))

    print(f"\n{'=' * 80}")
    print(f"Total relevance labels: {total_labels}")
    duplicate_count = sum(count > 1 for count in chunk_counts.values())
    print(f"Invalid labels: {len(invalid_labels)}")

    if invalid_labels or duplicate_count:
        print("\nINVALID LABELS — fix these before trusting the metrics:")
        for query_id, question, chunk_id in invalid_labels:
            print(f"{query_id}: {chunk_id} | {question}")
        if duplicate_count:
            print(f"Duplicate chunk IDs in corpus: {duplicate_count}")
        raise SystemExit(1)
    else:
        print("\nAll labeled chunk IDs exist and have positive relevance grades.")
        print("Now manually verify that each labeled chunk answers its question.")


if __name__ == "__main__":
    main()
