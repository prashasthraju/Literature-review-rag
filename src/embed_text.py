import re

IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
FILE_URL_RE = re.compile(r"file:///\S+")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
WHITESPACE_RE = re.compile(r"\s+")

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def clean_passage(text):
    """Strip parser artifacts that waste BGE's 512-token window."""
    cleaned = IMAGE_RE.sub("[figure]", text or "")
    cleaned = HTML_COMMENT_RE.sub(" ", cleaned)
    cleaned = FILE_URL_RE.sub(" ", cleaned)
    cleaned = MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    return WHITESPACE_RE.sub(" ", cleaned).strip()


def passage_for_embedding(text, section_title=None):
    cleaned = clean_passage(text)
    title = (section_title or "").strip()
    if title and title.lower() not in {"document", "untitled"}:
        if cleaned:
            return f"{title}. {cleaned}"
        return title
    return cleaned


def query_for_embedding(question):
    return QUERY_PREFIX + (question or "").strip()
