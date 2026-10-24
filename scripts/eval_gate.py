"""Regression gate: fail (exit 1) if a generation report is worse than configs/evals/gates.json.

  python scripts/eval_gate.py reports/generation/retrieval_dev/<run>.json
  python scripts/eval_gate.py --run        # generate on the dev set first (needs the GPU; stop the server)

Compare with a baseline by passing --baseline <report>: any metric that drops
by more than --tolerance (default 0.03) against it also fails.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "configs" / "evals" / "gates.json"
LOWER_IS_BETTER = {"invalid_output_rate", "unsupported_number_rate", "approximate_quote_rate"}


def check(summary, gates, baseline=None, tolerance=0.03):
    """Return a list of (metric, value, limit, ok) rows."""
    rows = []
    for metric, floor in gates["min"].items():
        value = summary.get(metric)
        rows.append((metric, value, f">= {floor}", value is not None and value >= floor))
    for metric, ceiling in gates["max"].items():
        value = summary.get(metric)
        rows.append((metric, value, f"<= {ceiling}", value is not None and value <= ceiling))
    if baseline:
        for metric, value in summary.items():
            before = baseline.get(metric)
            if not isinstance(value, float) or not isinstance(before, float) or metric == "mean_context_tokens":
                continue
            worse = value - before if metric in LOWER_IS_BETTER else before - value
            if worse > tolerance:
                rows.append((metric, value, f"baseline {before:.3f} (-{worse:.3f})", False))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", nargs="?", type=Path)
    parser.add_argument("--run", action="store_true", help="Run scripts/eval_generation.py --generate on the dev set first")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.03)
    args = parser.parse_args()
    gates = json.loads(GATES.read_text(encoding="utf-8"))

    report_path = args.report
    if args.run:
        report_path = ROOT / "reports" / "generation" / "gate_latest.json"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_generation.py"), "--generate",
                        "--dataset", str(ROOT / "data" / "evaluation" / gates["dataset"]),
                        "--output", str(report_path)], check=True)
    if not report_path:
        parser.error("give a report path or --run")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("dataset") != gates["dataset"]:
        sys.exit(f"gate is defined for {gates['dataset']}, report is for {report.get('dataset')}")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))["summary"] if args.baseline else None

    rows = check(report["summary"], gates, baseline, args.tolerance)
    for metric, value, limit, ok in rows:
        shown = "missing" if value is None else f"{value:.3f}"
        print(f"{'PASS' if ok else 'FAIL'}  {metric:32s} {shown:>8s}  {limit}")
    failed = [row for row in rows if not row[3]]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
