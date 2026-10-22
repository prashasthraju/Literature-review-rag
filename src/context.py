"""Context assembly: turn ranked chunks into cited evidence under a token budget.

Neighbouring chunks of the same section share one block of overlap, so when
both are retrieved they are merged and the repeated block is kept once.
Every passage keeps its provenance (paper, section, pages, chunk IDs) and
gets a source label [S1], [S2], ... that the generator cites. Chunks left out
are logged with the reason, so the final context is inspectable.
"""

import math
import re
from dataclasses import dataclass, field

from src.embed_text import FILE_URL_RE, HTML_COMMENT_RE, IMAGE_RE, MARKDOWN_LINK_RE
from src.evaluation.evidence import normalize, unit_supports

SPACES_RE = re.compile(r"[ \t]+")
DEFAULT_BUDGET = 2000


def approx_tokens(text):
    """Rough English subword count; good enough to enforce a budget."""
    return math.ceil(len(text.split()) * 1.35)


def context_text(text):
    """Readable source text: drop image links and paths, keep line breaks (table rows)."""
    text = IMAGE_RE.sub("", text or "")
    text = HTML_COMMENT_RE.sub(" ", text)
    text = FILE_URL_RE.sub(" ", text)
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    lines = [SPACES_RE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def merge_overlap(first, second):
    """Join two consecutive chunk texts, keeping their shared lines once."""
    a, b = first.splitlines(), second.splitlines()
    for size in range(min(len(a), len(b)), 0, -1):
        if a[-size:] == b[:size]:
            return "\n".join(a + b[size:])
    return f"{first}\n{second}"


@dataclass
class Passage:
    source_id: str
    document_id: str
    document_title: str
    section_title: str
    pages: list
    chunk_ids: list
    chunk_indexes: list
    best_rank: int
    text: str

    def header(self):
        pages = self.pages
        if not pages:
            where = ""
        elif len(pages) == 1:
            where = f", p. {pages[0]}"
        else:
            where = f", pp. {pages[0]}-{pages[-1]}"
        return f"[{self.source_id}] {self.document_title} — {self.section_title}{where}"


@dataclass
class ContextBundle:
    passages: list
    excluded: list = field(default_factory=list)
    token_count: int = 0
    budget: int = DEFAULT_BUDGET
    ranked_chunk_ids: list = field(default_factory=list)

    def render(self):
        return "\n\n".join(f"{passage.header()}\n{passage.text}" for passage in self.passages)

    def source(self, source_id):
        return next((passage for passage in self.passages if passage.source_id == source_id), None)

    def log(self):
        return {
            "token_count": self.token_count,
            "budget": self.budget,
            "ranked_chunk_ids": self.ranked_chunk_ids,
            "included": [
                {"source_id": p.source_id, "chunk_ids": p.chunk_ids, "document_id": p.document_id, "pages": p.pages}
                for p in self.passages
            ],
            "excluded": self.excluded,
        }


def assemble_context(hits, budget=DEFAULT_BUDGET, count_tokens=approx_tokens, max_chunks=8):
    """Build a cited context from ranked hits (dicts with 'chunk_id' and 'record')."""
    groups = []  # each: {"key", "chunks": [(chunk_index, chunk_id, record, rank)]}
    excluded = []
    seen_text = set()

    for rank, hit in enumerate(hits, start=1):
        record = hit["record"]
        if rank > max_chunks:
            excluded.append({"chunk_id": hit["chunk_id"], "rank": rank, "reason": "below max_chunks"})
            continue
        text = context_text(record.get("text"))
        if not text or normalize(text) in seen_text:
            excluded.append({"chunk_id": hit["chunk_id"], "rank": rank, "reason": "duplicate or empty"})
            continue
        seen_text.add(normalize(text))
        key = (record.get("document_id"), tuple(record.get("section_path") or ()))
        index = record.get("chunk_index")
        group = next(
            (
                g for g in groups
                if g["key"] == key and index is not None
                and any(abs(index - other) == 1 for other, *_ in g["chunks"])
            ),
            None,
        )
        candidate = (group["chunks"] if group else []) + [(index, hit["chunk_id"], record, rank)]
        new_text = _join(candidate)
        current = sum(count_tokens(g["text"]) for g in groups if g is not group)
        if current + count_tokens(new_text) > budget:
            excluded.append({"chunk_id": hit["chunk_id"], "rank": rank, "reason": "token budget"})
            continue
        if group:
            group["chunks"], group["text"] = candidate, new_text
        else:
            groups.append({"key": key, "chunks": candidate, "text": new_text})

    groups.sort(key=lambda g: min(rank for *_, rank in g["chunks"]))
    passages = []
    for number, group in enumerate(groups, start=1):
        chunks = sorted(group["chunks"], key=lambda item: item[0] if item[0] is not None else 0)
        first = chunks[0][2]
        pages = sorted({page for _i, _c, record, _r in chunks for page in record.get("page_numbers") or []})
        passages.append(Passage(
            source_id=f"S{number}",
            document_id=first.get("document_id"),
            document_title=first.get("document_title") or first.get("source_file") or "",
            section_title=first.get("section_title") or "",
            pages=pages,
            chunk_ids=[chunk_id for _i, chunk_id, _r, _k in chunks],
            chunk_indexes=[index for index, *_ in chunks],
            best_rank=min(rank for *_, rank in chunks),
            text=group["text"],
        ))
    return ContextBundle(
        passages=passages,
        excluded=excluded,
        token_count=sum(count_tokens(p.text) for p in passages),
        budget=budget,
        ranked_chunk_ids=[hit["chunk_id"] for hit in hits],
    )


def _join(chunks):
    ordered = sorted(chunks, key=lambda item: item[0] if item[0] is not None else 0)
    text = context_text(ordered[0][2].get("text"))
    for _index, _chunk_id, record, _rank in ordered[1:]:
        text = merge_overlap(text, context_text(record.get("text")))
    return text


def evidence_coverage(bundle, evidence):
    """Share of evidence units whose anchors all appear in one passage of the
    unit's paper (the context-level counterpart of Recall@k)."""
    units = [unit for unit in evidence if unit.get("grade", 0) > 0]
    if not units:
        return None
    normalized = [(p.document_id, normalize(p.text)) for p in bundle.passages]
    covered = 0
    for unit in units:
        supports = [(doc, [normalize(anchor) for anchor in anchors]) for doc, anchors in unit_supports(unit)]
        if any(
            document_id == doc and all(a in text for a in anchors)
            for document_id, text in normalized
            for doc, anchors in supports
        ):
            covered += 1
    return covered / len(units)
