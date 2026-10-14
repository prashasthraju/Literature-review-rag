
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

MODEL_NAME = "BAAI/bge-base-en-v1.5"
COLLECTION = "research_papers_v1"
DB_PATH = "./qdrant_db"


def main():
    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(path=DB_PATH)

    try:
        print("Research-paper search. Type 'exit' to stop.")

        while True:
            question = input("\nQuestion: ").strip()

            if question.lower() in {"exit", "quit"}:
                break

            if not question:
                continue

            query = (
                "Represent this sentence for searching relevant passages: "
                + question
            )

            vector = model.encode(
                query,
                normalize_embeddings=True,
            ).tolist()

            results = client.query_points(
                collection_name=COLLECTION,
                query=vector,
                limit=5,
                with_payload=True,
            ).points

            for rank, point in enumerate(results, start=1):
                payload = point.payload or {}

                print(f"\n--- Result {rank} ---")
                print(f"Score: {point.score:.4f}")
                print(f"Paper: {payload.get('source_file')}")
                print(f"Section: {payload.get('section_title')}")
                print(f"Page: {payload.get('page_number')}")
                print(payload.get("text", "")[:1500])

    finally:
        client.close()


if __name__ == "__main__":
    main()
