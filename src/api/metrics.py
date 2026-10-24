"""Service metrics computed from stored traces and feedback.

Latency is service time: the total minus the wait for the GPU lock, so a burst of
queued requests does not look like a slow model. Nothing here reads message text.
"""

import json
from collections import Counter
from pathlib import Path

SLO_PATH = Path(__file__).resolve().parents[2] / "configs" / "evals" / "slo.json"


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 1)


def stats(values):
    return {"p50": percentile(values, 0.5), "p95": percentile(values, 0.95), "max": percentile(values, 1.0)}


def summarize(traces, feedback):
    """``traces``: rows from ChatStore.trace_rows; ``feedback``: {rating: count}."""
    statuses = Counter(trace["status"] for trace in traces)
    service, stages = [], {}
    for trace in traces:
        waited = sum(span["ms"] for span in trace["spans"] if span["name"] == "queue_wait")
        if trace["status"] != "error" and trace["total_ms"] is not None:
            service.append(trace["total_ms"] - waited)
        for span in trace["spans"]:
            stages.setdefault(span["name"], []).append(span["ms"])
    requests = len(traces)
    rated = sum(feedback.values())
    return {
        "requests": requests,
        "status": dict(statuses),
        "error_rate": statuses["error"] / requests if requests else None,
        "invalid_output_rate": statuses["invalid_output"] / requests if requests else None,
        "abstain_rate": statuses["insufficient_evidence"] / requests if requests else None,
        "service_ms": stats(service),
        "stage_ms": {name: stats(values) for name, values in sorted(stages.items())},
        "feedback": {"up": feedback.get(1, 0), "down": feedback.get(-1, 0)},
        "thumbs_down_rate": feedback.get(-1, 0) / rated if rated else None,
    }


def check_slo(metrics, slo=None):
    """(name, value, limit, ok) rows; 'ok' is None when there is too little traffic to judge."""
    slo = slo or json.loads(SLO_PATH.read_text(encoding="utf-8"))
    enough = metrics["requests"] >= slo["min_requests"]
    checks = [
        ("service_p95_ms", metrics["service_ms"]["p95"], slo["service_p95_ms"]),
        ("service_p50_ms", metrics["service_ms"]["p50"], slo["service_p50_ms"]),
        ("error_rate", metrics["error_rate"], slo["error_rate_max"]),
        ("invalid_output_rate", metrics["invalid_output_rate"], slo["invalid_output_rate_max"]),
        ("thumbs_down_rate", metrics["thumbs_down_rate"], slo["thumbs_down_rate_max"]),
    ]
    return [(name, value, limit, (value <= limit) if enough and value is not None else None) for name, value, limit in checks]
