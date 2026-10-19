import argparse
from collections import defaultdict, deque
from pathlib import Path
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embed_text import passage_for_embedding

PARSED_DIR = Path("parsed")
OUTPUT_FILE = Path("chunks.jsonl")
MODEL_NAME = "BAAI/bge-base-en-v1.5"
MAX_TOKENS = 480
OVERLAP_BLOCKS = 1
PAGE_MARKER_RE = re.compile(r"<!-- page-(start|break):(\d+) -->")
CAPTION_RE = re.compile(r"^\s*(?:table|tab\.|figure|fig\.)\s*(?:\d+|[ivxlcdm]+)\b", re.IGNORECASE)


def token_count(text, section_title, tokenizer):
    passage = passage_for_embedding(text, section_title)
    return len(tokenizer.encode(passage, add_special_tokens=True))


def split_by_tokens(text, tokenizer, section_title, max_tokens):
    prefix_tokens = token_count("", section_title, tokenizer)
    content_budget = max(1, max_tokens - prefix_tokens)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    pieces = []

    for offset in range(0, len(token_ids), content_budget):
        piece = tokenizer.decode(
            token_ids[offset:offset + content_budget],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if piece:
            pieces.append(piece)

    return pieces or [text]


def table_parts(block):
    lines = block.splitlines()
    table_start = next(
        (index for index, line in enumerate(lines) if line.lstrip().startswith("|")),
        None,
    )
    if table_start is None:
        return None

    table_lines = [line for line in lines[table_start:] if line.strip()]
    if len(table_lines) < 2 or any(not line.lstrip().startswith("|") for line in table_lines):
        return None

    caption = "\n".join(lines[:table_start]).strip()

    def cells(line):
        return [
            value.strip().replace("\\|", "|")
            for value in re.split(r"(?<!\\)\|", line.strip().strip("|"))
        ]

    header = cells(table_lines[0])
    data_lines = table_lines[1:]
    if data_lines and re.fullmatch(r"[|\s:=-]+", data_lines[0]):
        data_lines = data_lines[1:]
    rows = [cells(line) for line in data_lines]
    return caption, header, rows


def render_table_fields(caption, fields, section_title):
    label = f"Table: {caption}" if caption else "Table evidence"
    return f"{label}\n" + "; ".join(
        f"{header}: {value}" for header, value in fields if header or value
    )


def split_table_row(caption, headers, row, tokenizer, section_title, max_tokens):
    fields = [
        (headers[index] if index < len(headers) and headers[index] else f"Column {index + 1}", value)
        for index, value in enumerate(row)
        if value
    ]
    if not fields:
        return []

    key_field = fields[0]
    current_fields = []
    parts = []

    def render(fields_to_render):
        return render_table_fields(caption, [key_field] + fields_to_render, section_title)

    for field in fields[1:]:
        candidate = render(current_fields + [field])
        if current_fields and token_count(candidate, section_title, tokenizer) > max_tokens:
            parts.append(render(current_fields))
            current_fields = []
            candidate = render([field])

        if token_count(candidate, section_title, tokenizer) <= max_tokens:
            current_fields.append(field)
            continue

        prefix = render([])
        available_tokens = max_tokens - token_count(prefix, section_title, tokenizer)
        value_parts = split_by_tokens(
            field[1],
            tokenizer,
            section_title,
            max(1, available_tokens),
        )
        for value_part in value_parts:
            parts.append(render([(field[0], value_part)]))

    if current_fields or not parts:
        parts.append(render(current_fields))

    key_only = render([])
    if token_count(key_only, section_title, tokenizer) > max_tokens:
        prefix = f"Table: {caption}\n{key_field[0]}: " if caption else f"Table evidence\n{key_field[0]}: "
        available_tokens = max_tokens - token_count(prefix, section_title, tokenizer)
        values = split_by_tokens(
            key_field[1], tokenizer, section_title, max(1, available_tokens)
        )
        parts = [
            render_table_fields(caption, [(key_field[0], value)], section_title)
            for value in values
        ] + parts

    return parts


def split_large_block(block, tokenizer, section_title, max_tokens):
    table = table_parts(block)
    if table:
        caption, header, rows = table
        compact_rows = []
        for row in rows:
            compact_rows.extend(
                split_table_row(
                    caption,
                    header,
                    row,
                    tokenizer,
                    section_title,
                    max_tokens,
                )
            )
        return compact_rows or [render_table_fields(caption, list(zip(header, header)), section_title)]

    if token_count(block, section_title, tokenizer) <= max_tokens:
        return [block]

    sentences = re.split(r"(?<=[.!?])\s+", block)
    parts = []
    current = ""

    for sentence in sentences:
        if token_count(sentence, section_title, tokenizer) > max_tokens:
            if current:
                parts.append(current)
                current = ""
            parts.extend(
                split_by_tokens(sentence, tokenizer, section_title, max_tokens)
            )
            continue

        candidate = f"{current} {sentence}".strip()
        if current and token_count(candidate, section_title, tokenizer) > max_tokens:
            parts.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        parts.append(current)
    return parts


def parse_sections(markdown):
    sections = []
    heading_stack = []
    blocks = []
    page_number = 1
    marker_block = "__DOCLING_PAGE_MARKER__"
    markdown = PAGE_MARKER_RE.sub(
        lambda match: f"\n\n{marker_block}:{match.group(1)}:{match.group(2)}\n\n",
        markdown,
    )

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

        marker_match = re.fullmatch(
            re.escape(marker_block) + r":(start|break):(\d+)",
            block,
        )
        if marker_match:
            if marker_match.group(1) == "start":
                page_number = int(marker_match.group(2))
            else:
                flush()
                page_number = int(marker_match.group(2))
            continue

        match = re.match(r"^(#{1,6})\s+(.+)$", block)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack[:] = heading_stack[:level - 1]
            heading_stack.append(title)
            continue

        table = table_parts(block)
        if table and blocks and blocks[-1][1] == page_number:
            previous_text, previous_page = blocks[-1]
            if CAPTION_RE.match(previous_text):
                blocks[-1] = (f"{previous_text}\n\n{block}", previous_page)
                continue
        blocks.append((block, page_number))

    flush()
    return sections


def load_existing_ids(path):
    existing = defaultdict(deque)
    if not path.exists():
        return existing

    with path.open(encoding="utf-8") as chunk_file:
        for line in chunk_file:
            if not line.strip():
                continue
            item = json.loads(line)
            key = (
                item.get("document_id"),
                tuple(item.get("section_path") or []),
                item.get("text", ""),
            )
            if item.get("chunk_id"):
                existing[key].append(item["chunk_id"])
    return existing


def chunk_document(
    markdown,
    document_id,
    source_file,
    tokenizer,
    max_tokens=MAX_TOKENS,
    existing_ids=None,
):
    chunks = []
    chunk_index = 0
    existing_ids = existing_ids if existing_ids is not None else defaultdict(deque)

    for section in parse_sections(markdown):
        section_path = section["section_path"]
        section_title = section_path[-1]
        blocks = []
        for block, page_number in section["blocks"]:
            blocks.extend(
                (part, page_number)
                for part in split_large_block(
                    block,
                    tokenizer,
                    section_title,
                    max_tokens,
                )
            )

        current = []

        def emit(items):
            nonlocal chunk_index
            text = "\n\n".join(block for block, _page in items).strip()
            if not text:
                return

            chunk_index += 1
            page_numbers = sorted({page for _block, page in items if page is not None})
            identity = (document_id, tuple(section_path), text)
            if existing_ids[identity]:
                chunk_id = existing_ids[identity].popleft()
            else:
                chunk_id = hashlib.sha256(
                    f"{document_id}|{chunk_index}|{text}".encode("utf-8")
                ).hexdigest()[:32]

            chunks.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "source_file": source_file,
                "section_path": section_path,
                "section_title": section_title,
                "chunk_index": chunk_index,
                "text": text,
                "page_number": page_numbers[0] if len(page_numbers) == 1 else None,
                "page_numbers": page_numbers,
                "contains_table": bool(
                    re.search(r"(?m)^\s*\|.*\|", text)
                    or re.search(r"(?im)^\s*Table:", text)
                ),
                "contains_image_ref": bool(re.search(r"!\[[^\]]*\]\([^)]+\)", text)),
            })

        for block in blocks:
            candidate = "\n\n".join([text for text, _page in current] + [block[0]])
            if current and token_count(candidate, section_title, tokenizer) > max_tokens:
                emit(current)
                current = current[-OVERLAP_BLOCKS:]
                while current and token_count(
                    "\n\n".join([text for text, _page in current] + [block[0]]),
                    section_title,
                    tokenizer,
                ) > max_tokens:
                    current.pop(0)
            current.append(block)

        if current:
            emit(current)

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Create page-aware, token-limited chunks from parsed Markdown.")
    parser.add_argument("--parsed-dir", type=Path, default=PARSED_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args()

    if args.max_tokens <= 0 or args.max_tokens > 512:
        parser.error("--max-tokens must be between 1 and 512")

    manifests = list(args.parsed_dir.rglob("manifest.json"))
    if not manifests:
        raise RuntimeError(f"No manifests found under {args.parsed_dir}.")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.model_max_length = 100000
    existing_ids = load_existing_ids(args.output)
    total = 0
    failures = []
    max_chunk_tokens = 0
    oversized_chunks = []

    with args.output.open("w", encoding="utf-8") as out:
        for manifest_path in manifests:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                markdown_path = Path(manifest["markdown_path"])
                if not markdown_path.is_absolute():
                    markdown_path = Path.cwd() / markdown_path
                if not markdown_path.is_file():
                    raise FileNotFoundError(str(markdown_path))

                markdown = markdown_path.read_text(encoding="utf-8")
                chunks = chunk_document(
                    markdown,
                    manifest["document_id"],
                    manifest["source_file"],
                    tokenizer,
                    max_tokens=args.max_tokens,
                    existing_ids=existing_ids,
                )

                for chunk in chunks:
                    actual_tokens = token_count(
                        chunk["text"],
                        chunk["section_title"],
                        tokenizer,
                    )
                    max_chunk_tokens = max(max_chunk_tokens, actual_tokens)
                    if actual_tokens > args.max_tokens:
                        oversized_chunks.append((chunk["chunk_id"], actual_tokens))
                    out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

                total += len(chunks)
                print(f"{manifest['document_id']}: {len(chunks)} chunks")
            except Exception as exc:
                failures.append({"manifest": str(manifest_path), "error": str(exc)})
                print(f"Failed: {manifest_path}: {exc}")

    Path("chunk_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nTotal chunks: {total}")
    print(f"Failed documents: {len(failures)}")
    print(f"Maximum chunk tokens (including section title): {max_chunk_tokens}")
    print(f"Chunks above token budget: {len(oversized_chunks)}")
    if oversized_chunks:
        print(f"Oversized examples: {oversized_chunks[:5]}")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
