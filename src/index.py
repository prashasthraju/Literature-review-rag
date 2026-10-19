
from pathlib import Path
import json
import shutil
import uuid

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

from src.embed_text import passage_for_embedding

CHUNKS_FILE = Path("chunks.jsonl")
DB_PATH = "./qdrant_db_token_chunks"
COLLECTION = "research_papers_v1"

MODEL_NAME = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 32

# This script rebuilds the collection on every full indexing run.
REBUILD_COLLECTION = True


def read_batches(path, batch_size):
    batch = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            batch.append(json.loads(line))

            if len(batch) >= batch_size:
                yield batch
                batch = []

    if batch:
        yield batch


def main():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            "chunks.jsonl not found. Run chunk_documents.py first."
        )

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    vector_size = model.get_embedding_dimension()

    # Persistent local database. Close it before opening another
    # client on the same path.
    client = QdrantClient(path=DB_PATH)

    try:
        exists = client.collection_exists(COLLECTION)

        if exists and REBUILD_COLLECTION:
            collection_names = [item.name for item in client.get_collections().collections]
            if collection_names != [COLLECTION]:
                raise RuntimeError(
                    f"Refusing to reset {DB_PATH}; expected only {COLLECTION}, "
                    f"found {collection_names}."
                )
            client.close()
            database_path = Path(DB_PATH).resolve()
            if database_path.name != Path(DB_PATH).name:
                raise RuntimeError(f"Refusing to reset unexpected database path: {database_path}")
            if database_path.exists():
                shutil.rmtree(database_path)
            client = QdrantClient(path=DB_PATH)
            exists = client.collection_exists(COLLECTION)

        if not exists:
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

        total = 0

        for batch in read_batches(CHUNKS_FILE, BATCH_SIZE):
            texts = [
                passage_for_embedding(item.get("text"), item.get("section_title"))
                for item in batch
            ]

            vectors = model.encode(
                texts,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            points = []

            for item, vector in zip(batch, vectors):
                # Qdrant point IDs must be integers or UUIDs.
                point_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        item["chunk_id"],
                    )
                )

                payload = {
                    "chunk_id": item["chunk_id"],
                    "document_id": item["document_id"],
                    "source_file": item["source_file"],
                    "section_path": item["section_path"],
                    "section_title": item["section_title"],
                    "chunk_index": item["chunk_index"],
                    "text": item["text"],
                    "page_number": item["page_number"],
                    "page_numbers": item["page_numbers"],
                    "contains_table": item["contains_table"],
                    "contains_image_ref": item["contains_image_ref"],
                }

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload=payload,
                    )
                )

            client.upsert(
                collection_name=COLLECTION,
                points=points,
                wait=True,
            )

            total += len(points)
            print(f"Indexed {total} chunks")

        print("\nIndexing complete.")
        stored_count = client.count(collection_name=COLLECTION, exact=True).count
        if stored_count != total:
            raise RuntimeError(
                f"Indexed {total} chunks but Qdrant contains {stored_count} points."
            )
        print(f"Collection: {COLLECTION}")
        print(f"Stored chunks: {stored_count}")
        print(f"Vector dimension: {vector_size}")
        print(f"Database path: {Path(DB_PATH).resolve()}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
