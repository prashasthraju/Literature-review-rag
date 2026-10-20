import re

IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
FILE_URL_RE = re.compile(r"file:///\S+")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
WHITESPACE_RE = re.compile(r"\s+")

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
GENERIC_SECTION_TITLES = {"document", "untitled"}


def clean_passage(text):
    """Strip parser artifacts that waste BGE's 512-token window."""
    cleaned = IMAGE_RE.sub("[figure]", text or "")
    cleaned = HTML_COMMENT_RE.sub(" ", cleaned)
    cleaned = FILE_URL_RE.sub(" ", cleaned)
    cleaned = MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    return WHITESPACE_RE.sub(" ", cleaned).strip()


def context_header(section_title=None, document_title=None):
    """Paper title and section heading, so a chunk is retrievable by the paper's name.

    Chunks such as "The proposed framework comprises two modules..." never
    mention the paper they come from, so queries naming the method miss them.
    """
    parts = []
    document_title = (document_title or "").strip()
    section_title = (section_title or "").strip()
    if document_title:
        parts.append(document_title)
    if (
        section_title
        and section_title.lower() not in GENERIC_SECTION_TITLES
        and section_title != document_title
    ):
        parts.append(section_title)
    return ". ".join(parts)


def passage_for_embedding(text, section_title=None, document_title=None):
    cleaned = clean_passage(text)
    header = context_header(section_title, document_title)
    if header and cleaned:
        return f"{header}. {cleaned}"
    return header or cleaned


def passage_for_record(record):
    """The text every retrieval channel (dense, BM25, reranker) sees for a chunk."""
    return passage_for_embedding(
        record.get("text"),
        record.get("section_title"),
        record.get("document_title"),
    )


def query_for_embedding(question):
    return QUERY_PREFIX + (question or "").strip()
