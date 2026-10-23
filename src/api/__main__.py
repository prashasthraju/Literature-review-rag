"""Run the chat server: python -m src.api [--host 127.0.0.1] [--port 8000] [--generator anthropic:<model>]"""

import argparse
import logging

import uvicorn

from src.api.app import DEFAULT_DB, create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", help="Generation config JSON (default configs/generation/default.json)")
    parser.add_argument("--generator", help="hf:<model id> or anthropic:<model>")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite file for conversations and traces")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    def factory(records):
        from src.rag import GENERATION_CONFIG, GenerationConfig, RagPipeline

        config = GenerationConfig.load(args.config or GENERATION_CONFIG, generator=args.generator)
        return RagPipeline(config, records=records)

    # One worker: the models and the local Qdrant store cannot be shared between processes.
    uvicorn.run(create_app(pipeline_factory=factory, db_path=args.db), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
