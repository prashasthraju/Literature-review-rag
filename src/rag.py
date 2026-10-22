"""Question answering over the paper collection: retrieve, assemble context, generate, validate.

  python -m src.rag "What does SadTalker predict from audio?"
  python -m src.rag            # interactive
  python -m src.rag --json "..."  # full answer record with provenance and context log
"""

import argparse
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from src.context import assemble_context
from src.generation.answer import build_messages, invalid_output, load_aliases, named_documents, parse_draft, validate
from src.generation.backends import make_generator
from src.retrieval import RetrievalConfig, Retriever, load_chunks

ROOT = Path(__file__).resolve().parents[1]
GENERATION_CONFIG = ROOT / "configs" / "generation" / "default.json"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Starting the reply inside the JSON object keeps small local models in the answer format.
ANSWER_PREFILL = '{"status": "'


@dataclass(frozen=True)
class GenerationConfig:
    generator: str = "hf:Qwen/Qwen3-4B-Instruct-2507"
    prompt_version: str = "answer_v2"
    retrieval_config: str = "retrieval-v2"
    context_hits: int = 8
    context_budget: int = 2000
    max_new_tokens: int = 512
    quantize: str | None = "4bit"

    @classmethod
    def load(cls, path=GENERATION_CONFIG, **overrides):
        data = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
        return replace(cls(**data), **{key: value for key, value in overrides.items() if value is not None})


def half_precision_models(config):
    """Embedder and reranker in fp16, so they fit on one 8 GB GPU next to the generator."""
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    if not torch.cuda.is_available():
        return {}
    kwargs = {"model_kwargs": {"dtype": torch.float16}}
    models = {}
    if config.retriever in {"dense", "hybrid"}:
        models["model"] = SentenceTransformer(config.embedding_model, **kwargs)
    if config.reranker:
        models["reranker"] = CrossEncoder(config.reranker, **kwargs)
    return models


class RagPipeline:
    def __init__(self, config=None, records=None, retriever=None, generator=None):
        self.config = config or GenerationConfig.load()
        if retriever is None:
            retrieval = RetrievalConfig.load(self.config.retrieval_config)
            retriever = Retriever(records or load_chunks(), retrieval, **half_precision_models(retrieval))
        self.retriever = retriever
        self.aliases = load_aliases()
        self.generator = generator or make_generator(self.config.generator, **(
            {"max_new_tokens": self.config.max_new_tokens, "quantize": self.config.quantize}
            if self.config.generator.startswith("hf:")
            else {}
        ))

    def answer(self, question):
        return self.answer_from_hits(question, self.retriever.search(question, top_k=self.config.context_hits))

    def answer_from_hits(self, question, hits):
        bundle = assemble_context(hits, budget=self.config.context_budget)
        messages = build_messages(question, bundle, self.config.prompt_version)
        raw = self.generator.generate(messages, prefill=ANSWER_PREFILL)
        try:
            draft = parse_draft(raw)
        except ValueError as error:
            return invalid_output(question, raw, error, bundle, self.config.prompt_version, self.generator.name)
        return validate(
            question, draft, bundle, self.config.prompt_version, self.generator.name, raw,
            required_documents=named_documents(question, self.aliases),
        )

    def close(self):
        self.retriever.close()


def render(answer):
    lines = [f"[{answer.status}] {answer.answer}"]
    for citation in answer.citations:
        if citation.valid:
            pages = ", ".join(map(str, citation.pages)) or "?"
            lines.append(f"  [{citation.source_id}] {citation.document_title} — {citation.section_title}, p. {pages}")
            lines.append(f"       \"{citation.quote}\"")
    for caveat in answer.caveats:
        lines.append(f"  note: {caveat}")
    for problem in answer.problems:
        lines.append(f"  check: {problem}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Answer questions about the paper collection with citations.")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--config", default=str(GENERATION_CONFIG))
    parser.add_argument("--generator", help="hf:<model id> or anthropic:<model>")
    parser.add_argument("--json", action="store_true", help="Print the full answer record")
    args = parser.parse_args()

    pipeline = RagPipeline(GenerationConfig.load(args.config, generator=args.generator))
    try:
        questions = [args.question] if args.question else iter(lambda: input("\nQuestion: ").strip(), "exit")
        for question in questions:
            if not question:
                continue
            answer = pipeline.answer(question)
            print(answer.model_dump_json(indent=2) if args.json else render(answer))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
