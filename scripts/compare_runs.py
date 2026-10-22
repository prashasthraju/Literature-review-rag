"""Paired comparison of two experiment runs on the same questions.

Reports the mean difference (candidate - reference) per metric with a 95%
paired bootstrap interval, how many questions improved or got worse, and the
per-category difference. Decision rule used in docs/retrieval_results.md: a
change is adopted only if the interval for MRR or Recall@5 excludes zero and
the other metric does not get significantly worse; otherwise keep the simpler
or cheaper configuration.

  python scripts/compare_runs.py reports/experiments/retrieval_dev/retrieval-v1.json \
      reports/experiments/retrieval_dev/<candidate>.json
"""

import argparse
import json
import random
from collections import defaultdict


def paired_bootstrap(diffs, samples=10000, seed=13):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(samples))
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def compare(reference, candidate):
    ref, cand = reference["per_query"], candidate["per_query"]
    shared = sorted(set(ref) & set(cand))
    if len(shared) != len(ref) or len(shared) != len(cand):
        raise SystemExit("Runs were made on different question sets")
    metrics = [m for m in next(iter(ref.values())) if m.startswith(("recall@", "ndcg@")) or m == "mrr"]
    rows = {}
    for metric in metrics:
        diffs = [cand[q][metric] - ref[q][metric] for q in shared if ref[q][metric] is not None and cand[q][metric] is not None]
        low, high = paired_bootstrap(diffs)
        rows[metric] = {
            "reference": sum(ref[q][metric] for q in shared) / len(shared),
            "candidate": sum(cand[q][metric] for q in shared) / len(shared),
            "diff": sum(diffs) / len(diffs),
            "ci95": (low, high),
            "better": sum(d > 1e-9 for d in diffs),
            "worse": sum(d < -1e-9 for d in diffs),
            "significant": low > 0 or high < 0,
        }
    by_category = defaultdict(list)
    for q in shared:
        by_category[ref[q]["category"]].append(q)
    categories = {
        category: {
            "count": len(qs),
            **{metric: sum(cand[q][metric] - ref[q][metric] for q in qs) / len(qs) for metric in metrics[:2]},
        }
        for category, qs in sorted(by_category.items())
    }
    return len(shared), rows, categories


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference")
    parser.add_argument("candidates", nargs="+")
    parser.add_argument("--categories", action="store_true", help="Also print per-category differences")
    args = parser.parse_args()

    reference = json.load(open(args.reference, encoding="utf-8"))
    print(f"reference: {reference['config']['name']}")
    for path in args.candidates:
        candidate = json.load(open(path, encoding="utf-8"))
        n, rows, categories = compare(reference, candidate)
        print(f"\n{candidate['config']['name']} vs {reference['config']['name']} ({n} questions)")
        for metric, row in rows.items():
            flag = " *" if row["significant"] else ""
            print(
                f"  {metric:9s} {row['reference']:.4f} -> {row['candidate']:.4f}  "
                f"diff {row['diff']:+.4f} [{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}]{flag}  "
                f"better {row['better']} / worse {row['worse']}"
            )
        ref_lat = reference["latency_ms"]["p50"]
        cand_lat = candidate["latency_ms"]["p50"]
        print(f"  latency p50 {ref_lat:.0f} ms -> {cand_lat:.0f} ms")
        if args.categories:
            for category, row in categories.items():
                rendered = "  ".join(f"{k} {v:+.3f}" for k, v in row.items() if k != "count")
                print(f"    {category:24s} n={row['count']:3d}  {rendered}")


if __name__ == "__main__":
    main()
