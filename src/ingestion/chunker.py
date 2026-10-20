import argparse
from collections import defaultdict, deque
from pathlib import Path, PureWindowsPath
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embed_text import IMAGE_RE, clean_passage, passage_for_embedding

PARSED_DIR = Path("parsed")
OUTPUT_FILE = Path("chunks.jsonl")
MODEL_NAME = "BAAI/bge-base-en-v1.5"
MAX_TOKENS = 480
OVERLAP_BLOCKS = 1
PAGE_MARKER_RE = re.compile(r"<!-- page-(start|break):(\d+) -->")
CAPTION_RE = re.compile(r"^\s*(?:table|tab\.|figure|fig\.)\s*(?:\d+|[ivxlcdm]+)\b", re.IGNORECASE)
# Reference lists and acknowledgements are dense with other papers' titles and
# method names, so they outrank the papers' own content for many queries.
BOILERPLATE_SECTION_RE = re.compile(
    r"^(?:\d+\.?\s*)?(?:references|bibliography|acknowledge?ments?\b.*|acm reference format:?)$",
    re.IGNORECASE,
)
# Chunks with less real text than this (e.g. a lone figure reference) are dropped.
MIN_CONTENT_CHARS = 40


class TokenBudget:
    """Counts tokens of a chunk exactly as it will be embedded, header included."""

    def __init__(self, tokenizer, section_title, document_title, max_tokens):
        self.tokenizer = tokenizer
        self.section_title = section_title
        self.document_title = document_title
        self.max_tokens = max_tokens

    def count(self, text):
        passage = passage_for_embedding(text, self.section_title, self.document_title)
        return len(self.tokenizer.encode(passage, add_special_tokens=True))

    def fits(self, text):
        return self.count(text) <= self.max_tokens


def split_by_tokens(text, budget, reserved_text=""):
    """Split text so that reserved_text + each piece fits the budget."""
    content_budget = max(1, budget.max_tokens - budget.count(reserved_text))
    # Slice the original string at token offsets; decoding the uncased BGE
    # tokens would lowercase and strip accents from the stored text.
    offsets = budget.tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True
    )["offset_mapping"]
    pieces = []

    for start in range(0, len(offsets), content_budget):
        end_index = start + content_budget
        end = offsets[end_index][0] if end_index < len(offsets) else len(text)
        piece = text[offsets[start][0]:end].strip()
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


def row_fields(headers, row):
    return [
        (headers[index] if index < len(headers) and headers[index] else f"Column {index + 1}", value)
        for index, value in enumerate(row)
        if value
    ]


def render_fields(fields):
    return "; ".join(f"{header}: {value}" for header, value in fields if header or value)


def split_table(caption, headers, rows, budget):
    """Pack rendered table rows into as few chunks as fit, caption once per chunk.

    Emitting one chunk per row repeated the caption in every chunk, so a single
    large table could fill the whole top-k for any query matching its caption.
    """
    label = f"Table: {caption}" if caption else "Table evidence"
    parts = []
    current = []

    def flush():
        if current:
            parts.append("\n".join([label] + current))
            current.clear()

    for row in rows:
        fields = row_fields(headers, row)
        if not fields:
            continue
        line = render_fields(fields)
        if not budget.fits(f"{label}\n{line}"):
            # A single row too wide for one chunk: repeat its key field on each piece.
            flush()
            key = render_fields(fields[:1])
            reserved = f"{label}\n{key}; "
            for piece in split_by_tokens(render_fields(fields[1:]) or key, budget, reserved):
                parts.append(reserved + piece)
            continue
        if current and not budget.fits("\n".join([label] + current + [line])):
            flush()
        current.append(line)

    flush()
    return parts or [f"{label}\n{'; '.join(headers)}"]


def split_large_block(block, budget):
    table = table_parts(block)
    if table:
        caption, header, rows = table
        return split_table(caption, header, rows, budget)

    if budget.fits(block):
        return [block]

    sentences = re.split(r"(?<=[.!?])\s+", block)
    parts = []
    current = ""

    for sentence in sentences:
        if not budget.fits(sentence):
            if current:
                parts.append(current)
                current = ""
            parts.extend(split_by_tokens(sentence, budget))
            continue

        candidate = f"{current} {sentence}".strip()
        if current and not budget.fits(candidate):
            parts.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        parts.append(current)
    return parts


def parse_sections(markdown):
    """Return (document_title, sections); the title is the first heading."""
    sections = []
    heading_stack = []
    blocks = []
    page_number = 1
    document_title = None
    skip_section = False
    marker_block = "__DOCLING_PAGE_MARKER__"
    markdown = PAGE_MARKER_RE.sub(
        lambda match: f"\n\n{marker_block}:{match.group(1)}:{match.group(2)}\n\n",
        markdown,
    )

    def flush():
        if blocks and not skip_section:
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
            # Page breaks do not end a section: paragraphs continue across
            # pages, and each block keeps its own page for provenance.
            page_number = int(marker_match.group(2))
            continue

        match = re.match(r"^(#{1,6})\s+(.+)$", block)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            document_title = document_title or title
            skip_section = bool(BOILERPLATE_SECTION_RE.match(title))
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
    return document_title, sections


def has_content(text):
    content = clean_passage(text).replace("[figure]", " ")
    return len(re.sub(r"[^A-Za-z0-9]", "", content)) >= MIN_CONTENT_CHARS


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
    document_title, sections = parse_sections(markdown)
    document_title = document_title or document_id

    for section in sections:
        section_path = section["section_path"]
        section_title = section_path[-1]
        budget = TokenBudget(tokenizer, section_title, document_title, max_tokens)
        blocks = []
        for block, page_number in section["blocks"]:
            blocks.extend(
                (part, page_number)
                for part in split_large_block(block, budget)
            )

        current = []

        def emit(items):
            nonlocal chunk_index
            text = "\n\n".join(block for block, _page in items).strip()
            if not text or not has_content(text):
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
                "document_title": document_title,
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
                "contains_image_ref": bool(IMAGE_RE.search(text)),
            })

        for block in blocks:
            candidate = "\n\n".join([text for text, _page in current] + [block[0]])
            if current and not budget.fits(candidate):
                emit(current)
                current = current[-OVERLAP_BLOCKS:]
                while current and not budget.fits(
                    "\n\n".join([text for text, _page in current] + [block[0]])
                ):
                    current.pop(0)
            current.append(block)

        if current:
            emit(current)

    return chunks


def markdown_path_for(manifest, manifest_path):
    # Manifests written on Windows store backslash paths; the Markdown always
    # sits next to its manifest, so fall back to that location.
    recorded = Path(PureWindowsPath(manifest["markdown_path"]).as_posix())
    if not recorded.is_absolute():
        recorded = Path.cwd() / recorded
    if recorded.is_file():
        return recorded
    return manifest_path.parent / "document.md"


def main():
    parser = argparse.ArgumentParser(description="Create page-aware, token-limited chunks from parsed Markdown.")
    parser.add_argument("--parsed-dir", type=Path, default=PARSED_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args()

    if args.max_tokens <= 0 or args.max_tokens > 512:
        parser.error("--max-tokens must be between 1 and 512")

    manifests = sorted(args.parsed_dir.rglob("manifest.json"))
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
                markdown_path = markdown_path_for(manifest, manifest_path)
                if not markdown_path.is_file():
                    raise FileNotFoundError(str(markdown_path))

                markdown = markdown_path.read_text(encoding="utf-8")
                chunks = chunk_document(
                    markdown,
                    manifest["document_id"],
                    PureWindowsPath(manifest["source_file"]).as_posix(),
                    tokenizer,
                    max_tokens=args.max_tokens,
                    existing_ids=existing_ids,
                )

                for chunk in chunks:
                    actual_tokens = TokenBudget(
                        tokenizer,
                        chunk["section_title"],
                        chunk["document_title"],
                        args.max_tokens,
                    ).count(chunk["text"])
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
    print(f"Maximum chunk tokens (including title and section header): {max_chunk_tokens}")
    print(f"Chunks above token budget: {len(oversized_chunks)}")
    if oversized_chunks:
        print(f"Oversized examples: {oversized_chunks[:5]}")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
