def passage_for_reranking(record):
    section_title = (record.get("section_title") or "").strip()
    text = (record.get("text") or "").strip()
    if section_title and not text.startswith(section_title):
        return f"{section_title}\n{text}"
    return text


def rerank_candidates(model, question, chunk_ids, records, limit):
    candidates = [
        chunk_id
        for chunk_id in chunk_ids
        if chunk_id in records and passage_for_reranking(records[chunk_id])
    ]
    if not candidates:
        return []

    pairs = [
        (question, passage_for_reranking(records[chunk_id]))
        for chunk_id in candidates
    ]
    scores = model.predict(pairs, show_progress_bar=False)
    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [(chunk_id, float(score)) for chunk_id, score in ranked[:limit]]
