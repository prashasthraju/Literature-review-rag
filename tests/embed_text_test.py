from src.embed_text import clean_passage, passage_for_embedding, query_for_embedding
from src.lexical import BM25Index, reciprocal_rank_fusion
from src.reranking import rerank_candidates


def test_clean_passage_strips_image_paths():
    text = (
        "Hello\n\n"
        "![Image](file:///C:/Users/prash/parsed/paper/images/image_000001_abc.png)\n\n"
        "Useful sentence about HTB-SR."
    )
    cleaned = clean_passage(text)
    assert "file:///" not in cleaned
    assert "image_000001" not in cleaned
    assert "[figure]" in cleaned
    assert "HTB-SR" in cleaned


def test_passage_for_embedding_prefixes_section():
    text = passage_for_embedding("Audio2Lmk predicts landmarks.", "2 Method")
    assert text.startswith("2 Method.")
    assert "Audio2Lmk" in text


def test_query_prefix():
    query = query_for_embedding("How many techniques are in DF40?")
    assert query.startswith("Represent this sentence for searching relevant passages: ")


def test_bm25_ranks_exact_dataset_fact_first():
    docs = [
        ("noise", "This paper introduces a talking-head generation framework."),
        ("gold", "We use the TFHP dataset with 1052 video clips from 588 subjects."),
        ("other", "HumanML3D provides 44970 annotations on 14616 motion sequences."),
    ]
    index = BM25Index(docs)
    ranked = [doc_id for doc_id, _score in index.search(
        "What dataset does ARTalk use, and how many clips and subjects?",
        limit=3,
    )]
    assert ranked[0] == "gold"


def test_rrf_prefers_items_high_in_both_lists():
    fused = reciprocal_rank_fusion(
        ["a", "b", "c"],
        ["c", "a", "d"],
        limit=3,
    )
    assert fused[0] == "a"


def test_cross_encoder_reranks_candidate_passages():
    class FakeCrossEncoder:
        def predict(self, pairs, show_progress_bar=False):
            return [0.2, 0.9]

    records = {
        "first": {"text": "first passage"},
        "second": {"section_title": "Methods", "text": "second passage"},
    }
    ranked = rerank_candidates(
        FakeCrossEncoder(),
        "question",
        ["first", "second"],
        records,
        limit=2,
    )
    assert ranked == [("second", 0.9), ("first", 0.2)]
