
from pathlib import Path
import hashlib
import json
import re

PARSED_DIR = Path("parsed")
OUTPUT_FILE = Path("chunks.jsonl")

MAX_CHARS = 1800
OVERLAP_BLOCKS = 1


def split_large_block(block):
    """Split oversized paragraphs or tables without losing table headers."""
    if len(block) <= MAX_CHARS:
        return [block]

    lines = block.splitlines()

    # Preserve a Markdown table's header in each resulting chunk.
    if lines and all(
        line.lstrip().startswith("|")
        for line in lines if line.strip()
    ):
        header = lines[:2] if len(lines) >= 2 else lines[:1]
        data_rows = lines[len(header):]
        parts = []
        current = header.copy()

        for line in data_rows:
            if len("\n".join(current + [line])) > MAX_CHARS and len(current) > len(header):
                parts.append("\n".join(current))
                current = header.copy()

            current.append(line)

        if current:
            parts.append("\n".join(current))

        return parts

    # Split long prose into sentences, then hard-split if necessary.
    sentences = re.split(r"(?<=[.!?])\s+", block)
    parts = []
    current = ""

    for sentence in sentences:
        if len(sentence) > MAX_CHARS:
            if current:
                parts.append(current)
                current = ""

            parts.extend(
                sentence[i:i + MAX_CHARS]
                for i in range(0, len(sentence), MAX_CHARS)
            )
            continue

        candidate = f"{current} {sentence}".strip()

        if current and len(candidate) > MAX_CHARS:
            parts.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        parts.append(current)

    return parts


def parse_sections(markdown):
    """Split Markdown by headings and retain the heading hierarchy."""
    sections = []
    heading_stack = []
    blocks = []

    def flush():
        if blocks:
            sections.append({
                "section_path": heading_stack.copy() or ["Document"],
                "blocks": blocks.copy(),
            })
            blocks.clear()

    for block in re.split(r"\n\s*\n", markdown):
        block = block.strip()

        if not block:
            continue

        match = re.match(r"^(#{1,6})\s+(.+)$", block)

        if match:
            flush()

            level = len(match.group(1))
            title = match.group(2).strip()

            heading_stack[:] = heading_stack[:level - 1]
            heading_stack.append(title)
        else:
            blocks.append(block)

    flush()
    return sections


def chunk_document(markdown, document_id, source_file):
    chunks = []
    chunk_index = 0

    for section in parse_sections(markdown):
        blocks = []

        for block in section["blocks"]:
            blocks.extend(split_large_block(block))

        current = []

        def emit(items):
            nonlocal chunk_index

            text = "\n\n".join(items).strip()
            if not text:
                return

            chunk_index += 1
            section_path = section["section_path"]

            chunk_id = hashlib.sha256(
                f"{document_id}|{chunk_index}|{text}".encode("utf-8")
            ).hexdigest()[:32]

            chunks.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "source_file": source_file,
                "section_path": section_path,
                "section_title": section_path[-1],
                "chunk_index": chunk_index,
                "text": text,
                "page_number": None,
                "contains_table": bool(
                    re.search(r"(?m)^\s*\|.*\|", text)
                ),
                "contains_image_ref": bool(
                    re.search(r"!\[[^\]]*\]\([^)]+\)", text)
                ),
            })

        for block in blocks:
            candidate = "\n\n".join(current + [block])

            if current and len(candidate) > MAX_CHARS:
                emit(current)
                current = current[-OVERLAP_BLOCKS:]

                while current and len(
                    "\n\n".join(current + [block])
                ) > MAX_CHARS:
                    current.pop(0)

            current.append(block)

        if current:
            emit(current)

    return chunks


def main():
    manifests = list(PARSED_DIR.rglob("manifest.json"))

    if not manifests:
        raise RuntimeError(
            "No manifests found. Check the parsed/ directory."
        )

    total = 0
    failures = []

    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for manifest_path in manifests:
            try:
                manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )

                markdown_path = Path(manifest["markdown_path"])
                if not markdown_path.is_absolute():
                    markdown_path = Path.cwd() / markdown_path

                if not markdown_path.exists():
                    raise FileNotFoundError(str(markdown_path))

                markdown = markdown_path.read_text(encoding="utf-8")

                chunks = chunk_document(
                    markdown,
                    manifest["document_id"],
                    manifest["source_file"],
                )

                for chunk in chunks:
                    out.write(
                        json.dumps(chunk, ensure_ascii=False) + "\n"
                    )

                total += len(chunks)
                print(
                    f"{manifest['document_id']}: "
                    f"{len(chunks)} chunks"
                )

            except Exception as exc:
                failures.append({
                    "manifest": str(manifest_path),
                    "error": str(exc),
                })
                print(f"Failed: {manifest_path}: {exc}")

    Path("chunk_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nTotal chunks: {total}")
    print(f"Failed documents: {len(failures)}")
    print(f"Output: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
