import argparse
import json
from collections import Counter
from pathlib import Path


def normalized_path(value):
    return str(value).replace("\\", "/").casefold()


def read_jsonl(path):
    records = []
    errors = []
    if not path.exists():
        return records, [f"Missing file: {path}"]

    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_number}: {exc}")
    return records, errors


def audit_corpus(input_dir, parsed_dir, chunks_file):
    issues = []
    pdfs = sorted(input_dir.rglob("*.pdf"))
    manifests = []

    for manifest_path in sorted(parsed_dir.rglob("manifest.json")):
        try:
            manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"Unreadable manifest {manifest_path}: {exc}")

    source_paths = {normalized_path(path.relative_to(input_dir)) for path in pdfs}
    manifest_paths = {
        normalized_path(manifest.get("source_file", ""))
        for manifest in manifests
        if manifest.get("source_file")
    }
    missing_manifests = sorted(source_paths - manifest_paths)
    orphan_manifests = sorted(manifest_paths - source_paths)

    for source in missing_manifests:
        issues.append(f"Source PDF has no manifest: {source}")
    for source in orphan_manifests:
        issues.append(f"Manifest has no matching source PDF: {source}")

    document_ids = [manifest.get("document_id") for manifest in manifests]
    duplicate_documents = [
        document_id
        for document_id, count in Counter(document_ids).items()
        if document_id and count > 1
    ]
    for document_id in duplicate_documents:
        issues.append(f"Duplicate manifest document_id: {document_id}")

    for manifest in manifests:
        markdown_path = Path(manifest.get("markdown_path", ""))
        if not markdown_path.is_absolute():
            markdown_path = Path.cwd() / markdown_path
        if not markdown_path.is_file():
            issues.append(f"Manifest Markdown file is missing: {markdown_path}")
        elif markdown_path.stat().st_size == 0:
            issues.append(f"Manifest Markdown file is empty: {markdown_path}")

    chunks, chunk_errors = read_jsonl(chunks_file)
    issues.extend(chunk_errors)
    chunk_ids = [chunk.get("chunk_id") for chunk in chunks]
    duplicate_chunks = [
        chunk_id
        for chunk_id, count in Counter(chunk_ids).items()
        if chunk_id and count > 1
    ]
    for chunk_id in duplicate_chunks:
        issues.append(f"Duplicate chunk_id: {chunk_id}")

    manifest_documents = {doc_id for doc_id in document_ids if doc_id}
    chunk_documents = {chunk.get("document_id") for chunk in chunks if chunk.get("document_id")}
    for document_id in sorted(manifest_documents - chunk_documents):
        issues.append(f"Manifest has no chunks: {document_id}")
    for document_id in sorted(chunk_documents - manifest_documents):
        issues.append(f"Chunks have no manifest: {document_id}")

    empty_chunks = sum(not (chunk.get("text") or "").strip() for chunk in chunks)
    missing_pages = sum(not chunk.get("page_numbers") for chunk in chunks)
    multi_page_chunks = sum(len(chunk.get("page_numbers") or []) > 1 for chunk in chunks)
    lengths = [len(chunk.get("text") or "") for chunk in chunks]
    short_chunks = sum(length < 200 for length in lengths)

    print(f"Source PDFs: {len(pdfs)}")
    print(f"Manifests: {len(manifests)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Unique chunk documents: {len(chunk_documents)}")
    print(f"Chunks without page numbers: {missing_pages}")
    print(f"Chunks spanning multiple pages: {multi_page_chunks}")
    print(f"Empty chunks: {empty_chunks}")
    print(f"Chunks shorter than 200 characters: {short_chunks}")
    if lengths:
        print(f"Chunk character range: {min(lengths)}-{max(lengths)}")

    if issues:
        print(f"\nAudit issues: {len(issues)}")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("\nAudit passed: all source PDFs, manifests, Markdown files, and chunks are accounted for.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Read-only consistency audit for the PDF ingestion and chunk pipeline."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("papers_avatar"))
    parser.add_argument("--parsed-dir", type=Path, default=Path("parsed"))
    parser.add_argument("--chunks-file", type=Path, default=Path("chunks.jsonl"))
    args = parser.parse_args()
    return audit_corpus(args.input_dir, args.parsed_dir, args.chunks_file)


if __name__ == "__main__":
    raise SystemExit(main())
