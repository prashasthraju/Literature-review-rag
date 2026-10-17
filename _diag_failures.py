import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

chunks = {}
with Path("chunks.jsonl").open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            c = json.loads(line)
            chunks[c["chunk_id"]] = c

report = json.loads(Path("reports/retrieval_eval_latest.json").read_text(encoding="utf-8"))
failed = [q for q in report["queries"] if q["metrics"]["recall@5"] == 0.0]

for q in failed:
    print("=" * 80)
    print(q["query_id"], q["question"])
    print("Retrieved:")
    for r in q["results"]:
        print(f"  #{r['rank']} {r['document_id'][:40]!r} {r['section_title']!r} score={r['score']:.3f}")
        print("   ", (r["text"] or "").replace("\n", " ")[:180])
    print("Gold:")
    for cid, grade in q["relevant_chunk_ids"].items():
        c = chunks.get(cid)
        if not c:
            print("  MISSING", cid)
            continue
        print(f"  grade={grade} {c['document_id']!r} {c['section_title']!r} chars={len(c['text'])}")
        print("   ", c["text"].replace("\n", " ")[:220])
    print()
