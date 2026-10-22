import pytest

transformers = pytest.importorskip("transformers")

from src.ingestion.chunker import MODEL_NAME, chunk_document, parse_sections


@pytest.fixture(scope="module")
def tokenizer():
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.model_max_length = 100000
    return tok


MARKDOWN = """<!-- page-start:1 -->

## Paper Title: A Method

Abstract. We propose a method that spans

<!-- page-break:2 -->

two pages in a single paragraph.

## 3 Results

Table 1. Accuracy of every method.

| Method | ACC |
|--------|-----|
| A | 90.1 |
| B | 91.2 |
| Ours | 99.7 |

## References

- [1] Some Other Paper. Proceedings of a Conference.

## Acknowledgments

We thank our funders.

## A Appendix

Extra details about the Method that are long enough to keep.
"""


def test_parse_sections_keeps_paragraph_across_page_break():
    title, sections = parse_sections(MARKDOWN)
    assert title == "Paper Title: A Method"
    first = sections[0]["blocks"]
    assert [page for _text, page in first] == [1, 2]


def test_parse_sections_drops_reference_and_acknowledgement_sections():
    _title, sections = parse_sections(MARKDOWN)
    titles = [section["section_path"][-1] for section in sections]
    assert "References" not in titles
    assert "Acknowledgments" not in titles
    assert "A Appendix" in titles


def test_chunks_carry_title_and_pack_table_rows(tokenizer):
    chunks = chunk_document(MARKDOWN, "doc", "doc.pdf", tokenizer)
    assert all(chunk["document_title"] == "Paper Title: A Method" for chunk in chunks)
    table_chunks = [chunk for chunk in chunks if "Table:" in chunk["text"]]
    assert len(table_chunks) == 1
    text = table_chunks[0]["text"]
    assert text.count("Table: Table 1.") == 1
    assert "Method: Ours; ACC: 99.7" in text
    assert table_chunks[0]["contains_table"]


def test_multi_page_chunk_records_both_pages(tokenizer):
    chunks = chunk_document(MARKDOWN, "doc", "doc.pdf", tokenizer)
    assert chunks[0]["page_numbers"] == [1, 2]
    assert chunks[0]["page_number"] is None


def test_oversized_text_is_split_without_lowercasing(tokenizer):
    sentence = "Ünïque CamelCase Token " * 400
    markdown = f"## Title\n\n{sentence}"
    chunks = chunk_document(markdown, "doc", "doc.pdf", tokenizer, max_tokens=128)
    assert len(chunks) > 1
    assert all("CamelCase" in chunk["text"] for chunk in chunks)
    assert "Ünïque" in chunks[0]["text"]


def test_front_matter_and_author_headings_are_dropped():
    markdown = """## A Paper Title

Jane Doe1, John Roe2

1 University of Somewhere

## Liang Pan

S-Lab, Some University E-mail: liang.pan@example.edu

Abstract. Our method does useful things across many words of real content here.

Permission to make digital or hard copies of all or part of this work is granted.

* Equal contribution. The abstract continues here with more real content.

## 1 Introduction

Body text that is long enough to be kept as content in the chunk.
"""
    _title, sections = parse_sections(markdown)
    texts = [text for section in sections for text, _page in section["blocks"]]
    assert texts == [
        "Abstract. Our method does useful things across many words of real content here.",
        "The abstract continues here with more real content.",
        "Body text that is long enough to be kept as content in the chunk.",
    ]
    assert [section["section_path"][-1] for section in sections] == ["A Paper Title", "1 Introduction"]
