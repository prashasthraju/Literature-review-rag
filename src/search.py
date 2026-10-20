import argparse

from src.retrieval import CANDIDATE_K, DEFAULT_RERANKER, RETRIEVERS, TOP_K, Retriever, load_chunks


def main():
    parser = argparse.ArgumentParser(description="Search the local research-paper index.")
    parser.add_argument("--retriever", choices=RETRIEVERS, default="hybrid")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER,
        help="Cross-encoder used to rerank candidates; pass 'none' to disable",
    )
    args = parser.parse_args()
    if args.reranker_model.lower() == "none":
        args.reranker_model = None

    retriever = Retriever(
        load_chunks(),
        mode=args.retriever,
        reranker_model=args.reranker_model,
        candidate_k=args.candidate_k,
    )

    try:
        mode = args.retriever + (" + cross-encoder reranking" if args.reranker_model else "")
        print(f"Research-paper search ({mode}). Type 'exit' to stop.")

        while True:
            question = input("\nQuestion: ").strip()

            if question.lower() in {"exit", "quit"}:
                break

            if not question:
                continue

            for rank, hit in enumerate(retriever.search(question, top_k=args.top_k), start=1):
                record = hit["record"]
                pages = ", ".join(str(page) for page in record.get("page_numbers") or []) or "?"

                print(f"\n--- Result {rank} ---")
                if hit["dense_score"] is not None:
                    print(f"Dense score: {hit['dense_score']:.4f}")
                if hit["rerank_score"] is not None:
                    print(f"Reranker score: {hit['rerank_score']:.4f}")
                print(f"Paper: {record.get('document_title') or record.get('source_file')}")
                print(f"Section: {record.get('section_title')}")
                print(f"Page: {pages}")
                print((record.get("text") or "")[:1500])

    finally:
        retriever.close()


if __name__ == "__main__":
    main()
