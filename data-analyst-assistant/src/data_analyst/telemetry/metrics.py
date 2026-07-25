"""In-memory metrics recorder. Stands in for a StatsD/Azure Monitor metrics
client: same shape (`increment`, `observe`), swap the body for a real backend
without touching call sites."""
from __future__ import annotations

import threading
from collections import defaultdict


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._observations: dict[str, list[float]] = defaultdict(list)

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] += value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._observations[name].append(value)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "observations": {k: list(v) for k, v in self._observations.items()},
            }


metrics = Metrics()
