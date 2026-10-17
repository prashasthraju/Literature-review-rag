import json
import re
from pathlib import Path

report = json.loads(Path("reports/retrieval_eval_latest.json").read_text(encoding="utf-8"))
print("SUMMARY", json.dumps(report["summary"], indent=2))
print()
for q in report["queries"]:
    m = q["metrics"]
    nrel = sum(1 for g in q["relevant_chunk_ids"].values() if g > 0)
    hits = sum(1 for r in q["results"] if r["relevance"] > 0)
    print(
        f"{q['query_id']}  rec={m['recall@5']:.2f} p={m['precision@5']:.2f} "
        f"mrr={m['mrr']:.2f} ndcg={m['ndcg@5']:.2f}  labeled={nrel} hits={hits}  {q['category']}"
    )

n = 0
img = 0
fileurl = 0
start_img = 0
longish = 0
lens = []
img_chars = 0
total_chars = 0
img_re = re.compile(r"!\[[^\]]*\]\([^)]+\)")
with Path("chunks.jsonl").open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        c = json.loads(line)
        t = c.get("text") or ""
        n += 1
        lens.append(len(t))
        total_chars += len(t)
        imgs = img_re.findall(t)
        if imgs:
            img += 1
            img_chars += sum(len(x) for x in imgs)
        if t.lstrip().startswith("!["):
            start_img += 1
        if "file:///" in t:
            fileurl += 1
        if len(t) > 1200:
            longish += 1
print("\nCHUNKS", n)
print("with image markdown", img, f"{img / n:.1%}")
print("start with image", start_img, f"{start_img / n:.1%}")
print("with file://", fileurl, f"{fileurl / n:.1%}")
print("len>1200", longish, f"{longish / n:.1%}")
print("avg chars", sum(lens) / n)
print("median chars", sorted(lens)[n // 2])
print("image markdown char fraction", img_chars / total_chars)
