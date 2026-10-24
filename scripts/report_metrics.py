"""Print service metrics and SLO status from the chat database (works while the server runs).

  python scripts/report_metrics.py [--hours 24] [--db data/app/chat.sqlite3]

Exit 1 if an SLO with enough traffic is violated.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.app import DEFAULT_DB
from src.api.metrics import check_slo, summarize
from src.api.store import ChatStore


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    store = ChatStore(args.db)
    metrics = summarize(store.trace_rows(args.hours), store.feedback_counts(args.hours))
    store.close()

    print(f"last {args.hours:g} h: {metrics['requests']} requests, status {metrics['status']}")
    print(f"service time  p50 {metrics['service_ms']['p50']} ms   p95 {metrics['service_ms']['p95']} ms")
    for name, stat in metrics["stage_ms"].items():
        print(f"  {name:16s} p50 {stat['p50']:>8}  p95 {stat['p95']:>8}")
    print(f"feedback {metrics['feedback']}")
    failed = False
    for name, value, limit, ok in check_slo(metrics):
        label = "n/a " if ok is None else "PASS" if ok else "FAIL"
        failed |= ok is False
        print(f"{label}  {name:22s} {value if value is not None else '-':>10}  limit {limit}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
