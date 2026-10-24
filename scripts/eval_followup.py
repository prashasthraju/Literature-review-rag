"""Follow-up rewrite quality (needs the GPU generator; stop the API server first).

Each case in data/evaluation/followup_dev.jsonl has a short conversation, a follow-up message and
what the standalone question must mention (groups of alternatives, all groups required) and must
not mention (a paper from an earlier turn that the user moved away from). Scoring is by string
match, so it is strict about names and loose about wording; the output lists every rewrite for reading.

  python scripts/eval_followup.py [--generator hf:Qwen/Qwen3-4B-Instruct-2507]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.backends import make_generator
from src.generation.answer import load_aliases
from src.generation.followup import rewrite
from src.rag import GenerationConfig


def passes(question, case):
    text = question.lower()
    missing = [group for group in case["must_include"] if not any(option.lower() in text for option in group)]
    leaked = [word for word in case["must_exclude"] if word.lower() in text]
    return not missing and not leaked, missing, leaked


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/evaluation/followup_dev.jsonl")
    parser.add_argument("--generator")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/generation/followup_dev.json")
    args = parser.parse_args()
    config = GenerationConfig.load(generator=args.generator)
    generator = make_generator(config.generator)
    aliases = load_aliases()
    cases = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]

    rows = []
    for case in cases:
        question = rewrite(generator, case["history"], case["message"], config.condense_max_new_tokens, aliases)
        ok, missing, leaked = passes(question, case)
        rows.append({**case, "rewritten": question, "ok": ok, "missing": missing, "leaked": leaked})
        print(f"{'ok  ' if ok else 'FAIL'} {case['id']} [{case['kind']}] {case['message']!r} -> {question!r}")

    by_kind = {}
    for row in rows:
        by_kind.setdefault(row["kind"], []).append(row["ok"])
    summary = {"cases": len(rows), "pass_rate": sum(r["ok"] for r in rows) / len(rows),
               "by_kind": {kind: sum(v) / len(v) for kind, v in by_kind.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"generator": config.generator, "summary": summary, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
