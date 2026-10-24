"""Export rated answers from the chat database as a queue for manual labelling.

Thumbs-down answers are the candidates for new evaluation questions: for each one, decide whether
retrieval, generation or the question was at fault, write the expected evidence, and add it to the
dev set (never the held-out test set). Output holds user questions, so it stays under data/app/.

  python scripts/export_review_set.py [--include-up] [--output data/app/review_queue.jsonl]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.app import DEFAULT_DB
from src.api.store import ChatStore


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "app" / "review_queue.jsonl")
    parser.add_argument("--include-up", action="store_true")
    args = parser.parse_args()
    store = ChatStore(args.db)
    ratings = (-1, 1) if args.include_up else (-1,)
    rows = store._read(
        "SELECT m.payload, m.request_id, f.rating, f.comment FROM feedback f JOIN messages m ON m.id = f.message_id"
        f" WHERE f.rating IN ({','.join('?' * len(ratings))}) ORDER BY f.created_at", ratings,
    )
    queue = []
    for row in rows:
        answer = json.loads(row["payload"])
        queue.append({
            "request_id": row["request_id"],
            "rating": row["rating"],
            "comment": row["comment"],
            "question": answer["question"],
            "standalone_question": answer["standalone_question"],
            "paper_ids": answer["paper_ids"],
            "status": answer["status"],
            "answer": answer["answer"],
            "citations": [{"source_id": c["source_id"], "document_id": c["document_id"], "quote": c["quote"]} for c in answer["citations"]],
            "sources": [{"source_id": s["source_id"], "document_id": s["document_id"], "section": s["section_title"]} for s in answer["sources"]],
            "label": None,  # retrieval | generation | question | ok, filled in by hand
        })
    store.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in queue), encoding="utf-8")
    print(f"{len(queue)} answers written to {args.output}")


if __name__ == "__main__":
    main()
