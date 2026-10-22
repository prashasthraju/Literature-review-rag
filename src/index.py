import argparse
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from src.retrieval import CHUNKS_FILE, DB_PATH, DEFAULT_CONFIG, RetrievalConfig, dense_units, load_chunks

BATCH_SIZE = 32
PAYLOAD_FIELDS = (
    "document_id", "document_title", "source_file", "section_path", "section_title",
    "chunk_index", "page_number", "page_numbers", "contains_table", "contains_image_ref",
)


def main():
    parser = argparse.ArgumentParser(description="Embed chunks into the Qdrant collection a retrieval config names.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--chunks", type=Path, default=CHUNKS_FILE)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    args = parser.parse_args()

    if not args.chunks.exists():
        raise FileNotFoundError(f"{args.chunks} not found. Run `python -m src.ingestion.chunker` first.")
    config = RetrievalConfig.load(args.config)
    records = {record["chunk_id"]: record for record in load_chunks(args.chunks)}
    units = list(dense_units(records.values(), config))

    print(f"Config {config.name}: {config.embedding_model}, {config.dense_units} units -> {config.collection}")
    model = SentenceTransformer(config.embedding_model, trust_remote_code=True)
    client = QdrantClient(path=str(args.db_path))
    try:
        # Rebuild only this config's collection; other configs' collections stay.
        if client.collection_exists(config.collection):
            client.delete_collection(config.collection)
        client.create_collection(
            collection_name=config.collection,
            vectors_config=models.VectorParams(
                size=model.get_embedding_dimension(),
                distance=models.Distance.COSINE,
            ),
        )

        position = {}
        for start in range(0, len(units), BATCH_SIZE):
            batch = units[start:start + BATCH_SIZE]
            vectors = model.encode(
                [text for _chunk_id, text in batch],
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            points = []
            for (chunk_id, text), vector in zip(batch, vectors):
                record = records[chunk_id]
                unit_index = position.get(chunk_id, 0)
                position[chunk_id] = unit_index + 1
                points.append(models.PointStruct(
                    # Qdrant point IDs must be integers or UUIDs.
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk_id}#{unit_index}")),
                    vector=vector.tolist(),
                    payload={
                        "chunk_id": chunk_id,
                        "unit_index": unit_index,
                        "unit_text": text,
                        "text": record["text"],
                        **{key: record.get(key) for key in PAYLOAD_FIELDS},
                    },
                ))
            client.upsert(collection_name=config.collection, points=points, wait=True)
            print(f"Indexed {start + len(batch)} / {len(units)} units")

        stored = client.count(collection_name=config.collection, exact=True).count
        if stored != len(units):
            raise RuntimeError(f"Indexed {len(units)} units but Qdrant contains {stored} points.")
        print(f"\nCollection {config.collection}: {stored} points for {len(records)} chunks")
        print(f"Database path: {args.db_path.resolve()}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
