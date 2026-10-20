"""Chunking-independent relevance labels.

Chunk IDs change whenever the chunker changes, which made every chunking
experiment require a manual label remap. Each labeled example therefore also
carries ``evidence`` units: a graded fact plus one or more short anchor strings
quoted from the source paper. A chunk supports a unit when it belongs to the
same document and contains every anchor after normalization. Metrics are then
computed over evidence units, so splitting or merging chunks does not change
the denominator.
"""

import re
import unicodedata
from math import log2

NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize(text):
    """Lowercase alphanumerics only, robust to spacing, hyphenation and accents."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(char for char in text if not unicodedata.combining(char))
    return NON_ALNUM_RE.sub("", text.lower())


def unit_anchors(unit):
    anchors = unit["anchor"]
    return [anchors] if isinstance(anchors, str) else list(anchors)


class EvidenceResolver:
    def __init__(self, chunk_records):
        self.by_document = {}
        for record in chunk_records:
            self.by_document.setdefault(record.get("document_id"), []).append(
                (record["chunk_id"], normalize(record.get("text", "")))
            )

    def resolve_unit(self, unit):
        anchors = [normalize(anchor) for anchor in unit_anchors(unit)]
        return {
            chunk_id
            for chunk_id, text in self.by_document.get(unit["document_id"], [])
            if all(anchor in text for anchor in anchors)
        }

    def resolve(self, example):
        """Return [(grade, supporting_chunk_ids), ...] for an example's evidence."""
        return [
            (unit["grade"], self.resolve_unit(unit))
            for unit in example.get("evidence", [])
            if unit["grade"] > 0
        ]


def evaluate_evidence(units, ranked_ids, k):
    """Evidence-level retrieval metrics.

    ``units`` is the output of ``EvidenceResolver.resolve``. A unit is covered
    once any of its supporting chunks is retrieved; repeated coverage of the
    same unit earns no extra gain, so near-duplicate chunks are not rewarded.
    """
    if not units:
        return {f"recall@{k}": None, f"precision@{k}": 0.0, "mrr": 0.0, f"ndcg@{k}": None}

    covered = set()
    gains = []
    first_hit = None
    useful = 0
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        hit_units = [index for index, (_grade, ids) in enumerate(units) if chunk_id in ids]
        if hit_units and first_hit is None:
            first_hit = rank
        if rank > k:
            continue
        if hit_units:
            useful += 1
        new_units = [index for index in hit_units if index not in covered]
        covered.update(new_units)
        gains.append(sum(units[index][0] for index in new_units))

    def dcg(values):
        return sum(value / log2(rank + 1) for rank, value in enumerate(values, start=1))

    # A single chunk may cover several units at once, so cap at a perfect score.
    ideal = dcg(sorted((grade for grade, _ids in units), reverse=True)[:k])
    return {
        f"recall@{k}": len(covered) / len(units),
        f"precision@{k}": useful / k,
        "mrr": 1 / first_hit if first_hit else 0.0,
        f"ndcg@{k}": min(1.0, dcg(gains) / ideal) if ideal else None,
    }
