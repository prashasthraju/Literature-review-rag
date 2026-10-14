from math import log2


def recall_at_k(relevant_ids, ranked_ids, k):
    relevant = set(relevant_ids)
    if not relevant:
        return None
    return len(relevant.intersection(ranked_ids[:k])) / len(relevant)


def precision_at_k(relevant_ids, ranked_ids, k):
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant_ids)
    return sum(item in relevant for item in ranked_ids[:k]) / k


def reciprocal_rank(relevant_ids, ranked_ids):
    relevant = set(relevant_ids)
    for rank, item in enumerate(ranked_ids, start=1):
        if item in relevant:
            return 1 / rank
    return 0.0


def ndcg_at_k(relevance_by_id, ranked_ids, k):
    def dcg(gains):
        return sum(gain / log2(rank + 1) for rank, gain in enumerate(gains, start=1))

    actual = [relevance_by_id.get(item, 0) for item in ranked_ids[:k]]
    ideal = sorted(relevance_by_id.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return None
    return dcg(actual) / ideal_dcg


def evaluate_query(relevance_by_id, ranked_ids, k):
    relevant_ids = [item for item, grade in relevance_by_id.items() if grade > 0]
    return {
        f"recall@{k}": recall_at_k(relevant_ids, ranked_ids, k),
        f"precision@{k}": precision_at_k(relevant_ids, ranked_ids, k),
        "mrr": reciprocal_rank(relevant_ids, ranked_ids),
        f"ndcg@{k}": ndcg_at_k(relevance_by_id, ranked_ids, k),
    }


def mean_metric(rows, name):
    values = [row[name] for row in rows if row[name] is not None]
    return sum(values) / len(values) if values else None
