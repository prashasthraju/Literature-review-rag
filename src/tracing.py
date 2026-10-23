"""Per-request trace: named spans with timings and small attributes.

Spans carry IDs, counts, versions and statuses, never document text or
prompts, so a trace can be stored and inspected without copying content
(workbook Task 9.1). Every pipeline stage takes an optional Trace; without
one, a throwaway Trace is used so callers need no special cases.
"""

import time
import uuid
from contextlib import contextmanager


class Trace:
    def __init__(self, request_id=None):
        self.request_id = request_id or uuid.uuid4().hex
        self.spans = []
        self._start = time.perf_counter()

    @contextmanager
    def span(self, name, **attributes):
        """Time a stage; the yielded dict collects attributes set inside the block."""
        span = {"name": name, "start_ms": self._elapsed_ms(), "attributes": dict(attributes)}
        started = time.perf_counter()
        try:
            yield span["attributes"]
        except Exception as error:
            span["error"] = type(error).__name__
            raise
        finally:
            span["ms"] = round((time.perf_counter() - started) * 1000, 1)
            self.spans.append(span)

    def _elapsed_ms(self):
        return round((time.perf_counter() - self._start) * 1000, 1)

    def timings(self):
        """Stage name -> milliseconds, summed if a stage ran more than once."""
        timings = {}
        for span in self.spans:
            timings[span["name"]] = round(timings.get(span["name"], 0.0) + span["ms"], 1)
        return timings

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "total_ms": self._elapsed_ms(),
            "spans": sorted(self.spans, key=lambda span: span["start_ms"]),
        }
