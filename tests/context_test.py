from src.context import assemble_context, context_text, evidence_coverage, merge_overlap


def record(chunk_id, index, text, section="2 Method", document="paper", pages=(3,)):
    return {
        "chunk_id": chunk_id,
        "record": {
            "chunk_id": chunk_id,
            "document_id": document,
            "document_title": "A Paper",
            "section_title": section,
            "section_path": ["A Paper", section],
            "chunk_index": index,
            "page_numbers": list(pages),
            "text": text,
        },
    }


def test_context_text_drops_image_links_but_keeps_rows():
    text = "![Image](file:///C:/x/fig.png)\nMethods: A; ACC: 1\nMethods: B; ACC: 2"
    assert context_text(text) == "Methods: A; ACC: 1\nMethods: B; ACC: 2"


def test_merge_overlap_keeps_shared_block_once():
    assert merge_overlap("one\ntwo\nshared", "shared\nthree") == "one\ntwo\nshared\nthree"
    assert merge_overlap("one", "two") == "one\ntwo"


def test_adjacent_chunks_merge_into_one_cited_passage():
    hits = [
        record("b", 5, "shared block\nsecond part", pages=(4,)),
        record("x", 9, "other section text", section="4 Results"),
        record("a", 4, "first part\nshared block", pages=(3,)),
    ]
    bundle = assemble_context(hits)
    assert [p.chunk_ids for p in bundle.passages] == [["a", "b"], ["x"]]
    first = bundle.passages[0]
    assert first.text == "first part\nshared block\nsecond part"
    assert first.pages == [3, 4] and first.source_id == "S1"
    assert "[S1] A Paper — 2 Method, pp. 3-4" in bundle.render()


def test_budget_excludes_whole_chunks_and_logs_reason():
    hits = [record("a", 1, "word " * 100), record("b", 7, "other " * 100, section="5 X")]
    bundle = assemble_context(hits, budget=150)
    assert [p.chunk_ids for p in bundle.passages] == [["a"]]
    assert bundle.excluded == [{"chunk_id": "b", "rank": 2, "reason": "token budget"}]


def test_evidence_coverage_counts_units_found_in_context():
    bundle = assemble_context([record("a", 1, "The model uses 468 facial points.")])
    evidence = [
        {"grade": 2, "document_id": "paper", "anchor": "468 facial points"},
        {"grade": 1, "document_id": "paper", "anchor": "missing", "alternatives": ["uses 468"]},
        {"grade": 1, "document_id": "other", "anchor": "468 facial points"},
    ]
    assert evidence_coverage(bundle, evidence) == 2 / 3
