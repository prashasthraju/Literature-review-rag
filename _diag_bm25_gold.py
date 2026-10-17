import json
from pathlib import Path
from src.lexical import BM25Index

questions = []
with Path("data/evaluation/retrieval_dev.jsonl").open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            questions.append(json.loads(line))

docs = []
with Path("chunks.jsonl").open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            item = json.loads(line)
            docs.append((item["chunk_id"], item.get("text", "")))

bm25 = BM25Index(docs)
print("docs", len(docs))
for q in questions:
    gold = set(q["relevant_chunk_ids"])
    hits = [cid for cid, _s in bm25.search(q["question"], limit=5)]
    found = gold.intersection(hits)
    print(
        q["query_id"],
        f"hits={len(found)}/{len(gold)}",
        "RANK_OK" if found else "MISS",
        q["question"][:70],
    )
