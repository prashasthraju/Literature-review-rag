import argparse
from dataclasses import replace

from src.retrieval import DEFAULT_CONFIG, RETRIEVERS, RetrievalConfig, Retriever, load_chunks


def main():
    parser = argparse.ArgumentParser(description="Search the local research-paper index.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Retrieval config name or path")
    parser.add_argument("--retriever", choices=RETRIEVERS)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--candidate-k", type=int)
    parser.add_argument(
        "--reranker-model",
        help="Override the config's cross-encoder; pass 'none' to disable",
    )
    args = parser.parse_args()

    overrides = {"retriever": args.retriever, "top_k": args.top_k, "candidate_k": args.candidate_k}
    if args.reranker_model:
        overrides["reranker"] = None if args.reranker_model.lower() == "none" else args.reranker_model
    overrides = {key: value for key, value in overrides.items() if key == "reranker" or value is not None}
    config = replace(RetrievalConfig.load(args.config), **overrides)
    retriever = Retriever(load_chunks(), config)

    try:
        mode = config.retriever + (f" + {config.reranker} reranking" if config.reranker else "")
        print(f"Research-paper search ({config.name}: {mode}). Type 'exit' to stop.")

        while True:
            question = input("\nQuestion: ").strip()

            if question.lower() in {"exit", "quit"}:
                break

            if not question:
                continue

            for rank, hit in enumerate(retriever.search(question), start=1):
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
