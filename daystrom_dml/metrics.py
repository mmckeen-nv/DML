"""Prometheus metrics instrumentation for the DML."""
from __future__ import annotations

from typing import Iterable, Optional

try:  # pragma: no cover - optional dependency for lean environments
    from prometheus_client import (  # type: ignore
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
except Exception:  # pragma: no cover - fallback when prometheus_client missing
    CollectorRegistry = None  # type: ignore[assignment]
    Counter = Gauge = Histogram = None  # type: ignore[assignment]
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"  # type: ignore[assignment]

    def generate_latest(_: Optional[CollectorRegistry] = None) -> bytes:  # type: ignore[misc]
        return b""


class _NoOpMetric:
    def __init__(self, *_, **__):
        pass

    def inc(self, *_: float, **__: float) -> None:
        return

    def observe(self, *_: float, **__: float) -> None:
        return

    def set(self, *_: float, **__: float) -> None:
        return

    def labels(self, **__: str) -> "_NoOpMetric":
        return self


if CollectorRegistry is not None:  # pragma: no cover - executed when dependency present
    REGISTRY = CollectorRegistry()
else:  # pragma: no cover - fallback when dependency absent
    REGISTRY = None  # type: ignore[assignment]


def _build_counter(name: str, documentation: str, labels: Optional[Iterable[str]] = None):
    if CollectorRegistry is None:
        return _NoOpMetric()
    return Counter(name, documentation, labelnames=tuple(labels or ()), registry=REGISTRY)


def _build_histogram(name: str, documentation: str, buckets: Optional[Iterable[float]] = None):
    if CollectorRegistry is None:
        return _NoOpMetric()
    return Histogram(name, documentation, buckets=tuple(buckets or ()), registry=REGISTRY)


def _build_gauge(name: str, documentation: str, labels: Optional[Iterable[str]] = None):
    if CollectorRegistry is None:
        return _NoOpMetric()
    return Gauge(name, documentation, labelnames=tuple(labels or ()), registry=REGISTRY)


INGEST_COUNTER = _build_counter("dml_ingest_total", "Number of ingested memory fragments")
RETRIEVAL_COUNTER = _build_counter(
    "dml_retrieval_total", "Number of query_database calls", labels=["mode"]
)
RETRIEVAL_LATENCY = _build_histogram(
    "dml_retrieval_latency_seconds",
    "Latency of retrieval operations",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)
ACTIVE_MEMORIES = _build_gauge("dml_memory_active", "Number of active memory items")


def record_ingest() -> None:
    """Increment the ingest counter."""

    INGEST_COUNTER.inc()


def record_retrieval(mode: str, latency_seconds: float) -> None:
    """Record a retrieval event for the given ``mode``."""

    RETRIEVAL_COUNTER.labels(mode=mode).inc()
    RETRIEVAL_LATENCY.observe(max(latency_seconds, 0.0))


def update_memory_gauge(count: int) -> None:
    """Update the active memory gauge."""

    ACTIVE_MEMORIES.set(max(0, int(count)))


def latest_metrics() -> tuple[bytes, str]:
    """Return the latest metrics payload and content type."""

    if CollectorRegistry is None:
        return b"", CONTENT_TYPE_LATEST
    payload = generate_latest(REGISTRY)
    return payload, CONTENT_TYPE_LATEST
