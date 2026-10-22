import pytest

from src.embed_text import passage_for_embedding
from src.evaluation.evidence import EvidenceResolver, evaluate_evidence, normalize
from src.lexical import BM25Index, tokenize


def test_passage_for_embedding_adds_paper_title():
    text = passage_for_embedding(
        "The proposed framework comprises two modules.",
        "2 Method",
        "AniPortrait: Audio-Driven Synthesis",
    )
    assert text.startswith("AniPortrait: Audio-Driven Synthesis. 2 Method. ")


def test_passage_for_embedding_skips_title_repeated_as_section():
    title = "DF40: Toward Next-Generation Deepfake Detection"
    assert passage_for_embedding("Abstract text.", title, title) == f"{title}. Abstract text."


def test_tokenize_drops_question_words():
    assert tokenize("What does each module produce?") == ["module", "produce"]


def test_bm25_ignores_stopword_mass():
    index = BM25Index([
        ("long", "what does each one of the things do and what does it have " * 5),
        ("gold", "Audio2Lmk produces landmarks"),
    ])
    assert index.search("What does Audio2Lmk produce?", limit=1)[0][0] == "gold"


def test_normalize_is_robust_to_spacing_and_hyphenation():
    assert normalize("micro-expression  aware\nF5C") == normalize("microexpression aware F5C")


def evidence_records():
    return [
        {"chunk_id": "a", "document_id": "df40", "text": "DF40 comprises 40 distinct deepfake techniques."},
        {"chunk_id": "a2", "document_id": "df40", "text": "Overlap: DF40 comprises 40 distinct deepfake techniques."},
        {"chunk_id": "b", "document_id": "df40", "text": "We run 2,000 evaluations with 8 detectors."},
        {"chunk_id": "c", "document_id": "other", "text": "DF40 comprises 40 distinct deepfake techniques."},
        {"chunk_id": "n", "document_id": "df40", "text": "Unrelated."},
    ]


def test_evidence_resolves_within_document_only():
    resolver = EvidenceResolver(evidence_records())
    unit = {"grade": 2, "document_id": "df40", "anchor": "comprises 40 distinct deepfake techniques"}
    assert resolver.resolve_unit(unit) == {"a", "a2"}


def test_evidence_anchor_list_requires_all_parts():
    resolver = EvidenceResolver(evidence_records())
    unit = {"grade": 2, "document_id": "df40", "anchor": ["2000 evaluations", "8 detectors"]}
    assert resolver.resolve_unit(unit) == {"b"}
    unit["anchor"] = ["2000 evaluations", "40 distinct"]
    assert resolver.resolve_unit(unit) == set()


def test_duplicate_chunks_do_not_inflate_evidence_metrics():
    units = [(2, {"a", "a2"}), (1, {"b"})]
    metrics = evaluate_evidence(units, ["a", "a2", "n"], 3)
    assert metrics["recall@3"] == 0.5
    assert metrics["mrr"] == 1.0
    # The repeated unit earns no second gain.
    assert metrics["ndcg@3"] == pytest.approx(2 / (2 + 1 / 1.584962500721156))


def test_evidence_metrics_perfect_ranking():
    units = [(2, {"a"}), (1, {"b"})]
    metrics = evaluate_evidence(units, ["a", "b", "n"], 3)
    assert metrics["recall@3"] == 1.0
    assert metrics["precision@3"] == pytest.approx(2 / 3)
    assert metrics["ndcg@3"] == pytest.approx(1.0)


def test_evidence_alternatives_add_supporting_chunks_without_new_units():
    resolver = EvidenceResolver(evidence_records())
    unit = {
        "grade": 2,
        "document_id": "df40",
        "anchor": "2000 evaluations",
        "alternatives": ["comprises 40 distinct"],
    }
    assert resolver.resolve_unit(unit) == {"a", "a2", "b"}
    assert len(resolver.resolve({"evidence": [unit]})) == 1


def test_evidence_alternative_can_point_to_another_paper():
    resolver = EvidenceResolver(evidence_records())
    unit = {
        "grade": 2,
        "document_id": "df40",
        "anchor": "2000 evaluations",
        "alternatives": [{"document_id": "other", "anchor": "40 distinct"}],
    }
    assert resolver.resolve_unit(unit) == {"b", "c"}


def test_split_windows_keeps_table_rows_whole_and_overlaps_prose():
    from src.retrieval import split_windows

    prose = " ".join(f"w{i}" for i in range(100))
    text = f"Table: Table 1. Results\nMethod: A; ACC: 1\nMethod: B; ACC: 2\n{prose}"
    windows = split_windows(text, window_words=64, stride=32)
    assert "Method: A; ACC: 1" in windows and "Method: B; ACC: 2" in windows
    prose_windows = [w for w in windows if w.startswith("w")]
    assert prose_windows[0].split()[0] == "w0" and prose_windows[1].split()[0] == "w32"
    assert prose_windows[-1].split()[-1] == "w99"


def test_best_per_chunk_max_pools_window_scores():
    from src.retrieval import best_per_chunk

    scored = [("a", 0.2), ("b", 0.9), ("a", 0.95), ("c", 0.1)]
    assert best_per_chunk(scored, 2) == [("a", 0.95), ("b", 0.9)]


def test_config_load_applies_overrides_including_null():
    from src.retrieval import RetrievalConfig

    config = RetrievalConfig.load("retrieval-v2", reranker=None, top_k=8)
    assert config.name == "retrieval-v2" and config.reranker is None and config.top_k == 8
    assert config.dense_units == "window" and config.collection == "research_papers_v2"


def test_weighted_fusion_prefers_heavier_channel():
    from src.retrieval import RetrievalConfig, fuse

    config = RetrievalConfig(dense_weight=2.0, bm25_weight=1.0)
    assert fuse(config, [("d", 0.9)], [("b", 9.0)], limit=2) == ["d", "b"]
    config = RetrievalConfig(dense_weight=1.0, bm25_weight=2.0)
    assert fuse(config, [("d", 0.9)], [("b", 9.0)], limit=2) == ["b", "d"]
