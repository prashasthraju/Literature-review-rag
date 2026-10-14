
import json
from collections import Counter
from pathlib import Path

path = Path("chunks.jsonl")

counts = Counter()
missing_text = 0
missing_source = 0
missing_page = 0
lengths = []

with path.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        chunk = json.loads(line)
        counts[chunk.get("document_id", "MISSING_ID")] += 1

        text = chunk.get("text", "").strip()
        if not text:
            missing_text += 1
        else:
            lengths.append(len(text))

        if not chunk.get("source_file"):
            missing_source += 1

        if chunk.get("page_number") is None:
            missing_page += 1

print("Total chunks:", sum(counts.values()))
print("Unique documents:", len(counts))
print("Chunks per document:")
for doc_id, count in counts.most_common():
    print(f"  {doc_id}: {count}")

print("\nEmpty chunks:", missing_text)
print("Missing source paths:", missing_source)
print("Missing page numbers:", missing_page)

if lengths:
    lengths.sort()
    print("Min characters:", lengths[0])
    print("Median characters:", lengths[len(lengths) // 2])
    print("Max characters:", lengths[-1])
