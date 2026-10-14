
from pathlib import Path
import json
from collections import Counter

PDF_DIR = Path("papers_avatar")
PARSED_DIR = Path("parsed")
CHUNKS_FILE = Path("chunks.jsonl")

# All PDFs, including nested folders
pdfs = list(PDF_DIR.rglob("*.pdf"))

# All successfully created parser manifests
manifests = list(PARSED_DIR.rglob("manifest.json"))

parsed_names = set()
for manifest_path in manifests:
    try:
        data = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        parsed_names.add(
            Path(data["source_file"]).stem.lower()
        )
    except Exception as exc:
        print("Bad manifest:", manifest_path, exc)

chunk_counts = Counter()
if CHUNKS_FILE.exists():
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunk = json.loads(line)
                chunk_counts[chunk["document_id"]] += 1

print("PDFs found:", len(pdfs))
print("Manifests found:", len(manifests))
print("Documents with chunks:", len(chunk_counts))

print("\nPDFs with no matching manifest:")
for pdf in pdfs:
    if pdf.stem.lower() not in parsed_names:
        print("  ", pdf)

print("\nManifests with no chunks:")
for manifest_path in manifests:
    data = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    doc_id = data.get("document_id")
    if doc_id not in chunk_counts:
        print("  ", doc_id, manifest_path)
