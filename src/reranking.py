from src.embed_text import passage_for_record


def rerank_candidates(model, question, chunk_ids, records, limit):
    candidates = [
        chunk_id
        for chunk_id in chunk_ids
        if chunk_id in records and passage_for_record(records[chunk_id])
    ]
    if not candidates:
        return []

    pairs = [
        (question, passage_for_record(records[chunk_id]))
        for chunk_id in candidates
    ]
    scores = model.predict(pairs, show_progress_bar=False)
    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [(chunk_id, float(score)) for chunk_id, score in ranked[:limit]]
