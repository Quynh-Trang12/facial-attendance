"""Optional, aggregate-only diagnostics. No pixels or identity data are logged."""

from collections import defaultdict
import json
import sys
from threading import Lock
from time import monotonic


class Diagnostics:
    def __init__(self, enabled=False, interval=5.0, clock=monotonic):
        self.enabled = enabled
        self.interval = interval
        self.clock = clock
        self._lock = Lock()
        self._started = clock()
        self._counts = defaultdict(int)
        self._timings = {}

    def count(self, name, amount=1):
        if self.enabled:
            with self._lock:
                self._counts[name] += amount

    def duration(self, name, seconds):
        if self.enabled:
            with self._lock:
                count, total, maximum = self._timings.get(name, (0, 0.0, 0.0))
                self._timings[name] = count + 1, total + seconds, max(maximum, seconds)

    def snapshot(self, *, force=False):
        if not self.enabled:
            return None
        with self._lock:
            now = self.clock()
            elapsed = now - self._started
            if elapsed <= 0 or (not force and elapsed < self.interval):
                return None
            counts = dict(self._counts)
            result = {
                "window_s": round(elapsed, 2),
                "capture_fps": round(counts.get("captured", 0) / elapsed, 2),
                "preview_fps": round(counts.get("previewed", 0) / elapsed, 2),
                "counts": counts,
                "timings_ms": {
                    name: {"mean": round(total / count * 1000, 3),
                           "max": round(maximum * 1000, 3), "n": count}
                    for name, (count, total, maximum) in self._timings.items()
                },
            }
            self._started = now
            self._counts.clear()
            self._timings.clear()
            return result

    def report(self, *, force=False):
        result = self.snapshot(force=force)
        if result is not None:
            print("SYS05_PERF " + json.dumps(result, sort_keys=True), file=sys.stderr)
        return result
