"""Lightweight tracing. Stands in for an OpenTelemetry tracer wired to Azure
Monitor / Application Insights: same call shape (`with trace_span("name"):`),
so swapping the implementation later doesn't touch call sites.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from data_analyst.telemetry.logging import get_logger

_logger = get_logger("telemetry.tracing")


@contextmanager
def trace_span(name: str, **attributes: object) -> Iterator[None]:
    start = time.perf_counter()
    _logger.debug("span start: %s %s", name, attributes)
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        _logger.debug("span end: %s (%.1fms)", name, duration_ms)
