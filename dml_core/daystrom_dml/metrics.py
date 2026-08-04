"""Prometheus metrics instrumentation for the Daystrom Memory Lattice."""
from __future__ import annotations

from typing import Iterable, Optional

try:  # pragma: no cover - optional dependency for lean environments
    from prometheus_client import (  # type: ignore
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        CONTENT_TYPE_LATEST,
        generate_latest,
    )
except Exception:  # pragma: no cover - graceful degradation when dependency absent
    CollectorRegistry = None  # type: ignore[assignment]
    Counter = Gauge = Histogram = None  # type: ignore[assignment]
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"  # type: ignore[assignment]

    def generate_latest(_: Optional[CollectorRegistry] = None) -> bytes:  # type: ignore[misc]
        return b""


class _NoOpMetric:
    """Fallback metric implementation used when prometheus_client is unavailable."""

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


def _build_counter(
    name: str, documentation: str, labels: Optional[Iterable[str]] = None
):
    if CollectorRegistry is None:
        return _NoOpMetric()
    return Counter(name, documentation, labelnames=tuple(labels or ()), registry=REGISTRY)


def _build_histogram(
    name: str,
    documentation: str,
    buckets: Optional[Iterable[float]] = None,
    labels: Optional[Iterable[str]] = None,
):
    if CollectorRegistry is None:
        return _NoOpMetric()
    return Histogram(
        name,
        documentation,
        labelnames=tuple(labels or ()),
        buckets=tuple(buckets or ()),
        registry=REGISTRY,
    )


def _build_gauge(
    name: str,
    documentation: str,
    labels: Optional[Iterable[str]] = None,
):
    if CollectorRegistry is None:
        return _NoOpMetric()
    return Gauge(name, documentation, labelnames=tuple(labels or ()), registry=REGISTRY)


RETRIEVAL_LATENCY = _build_histogram(
    "dml_retrieval_latency_ms",
    "Latency of retrieval operations in milliseconds.",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
)
MODE_COUNTER = _build_counter(
    "dml_mode_count",
    "Number of retrieval queries handled per mode.",
    labels=["mode"],
)
TOKENS_CONSUMED = _build_counter(
    "dml_tokens_consumed",
    "Total tokens consumed when serving queries.",
)
TOKENS_SAVED = _build_counter(
    "dml_tokens_saved",
    "Estimated tokens saved due to Daystrom retrieval.",
)
DML_ITEMS = _build_gauge(
    "dml_items",
    "Number of items currently stored in the lattice.",
)
OPERATION_LATENCY = _build_histogram(
    "dml_operation_latency_ms",
    "Latency of named DML hot-path operations in milliseconds.",
    buckets=[0.1, 0.5, 1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000, 30000],
    labels=["operation"],
)
OPERATION_COUNTER = _build_counter(
    "dml_operation_count",
    "Count of named DML operations and policy outcomes.",
    labels=["operation"],
)
SOURCE_COUNT = _build_histogram(
    "dml_selected_source_count",
    "Number of source records selected for a retrieval response.",
    buckets=[0, 1, 2, 3, 4, 6, 8, 10, 20],
)
EXPANDED_CONTEXT_SIZE = _build_histogram(
    "dml_expanded_context_chars",
    "Expanded procedural context size in characters.",
    buckets=[0, 128, 512, 1024, 2048, 4096, 8000, 12000, 16000],
)


def record_retrieval(mode: str, latency_ms: float) -> None:
    """Record latency and mode information for a retrieval."""

    MODE_COUNTER.labels(mode=mode).inc()
    RETRIEVAL_LATENCY.observe(max(float(latency_ms), 0.0))


def record_tokens(consumed: int, saved: int) -> None:
    """Increment token consumption and savings counters."""

    if consumed > 0:
        TOKENS_CONSUMED.inc(consumed)
    if saved > 0:
        TOKENS_SAVED.inc(saved)


def update_memory_gauge(count: int) -> None:
    """Update the gauge tracking the number of stored items."""

    DML_ITEMS.set(max(0, int(count)))


def record_operation(operation: str, *, latency_ms: float | None = None, count: int = 1) -> None:
    """Record a structured hot-path operation and optional latency."""

    OPERATION_COUNTER.labels(operation=operation).inc(max(0, int(count)))
    if latency_ms is not None:
        OPERATION_LATENCY.labels(operation=operation).observe(max(float(latency_ms), 0.0))


def record_source_expansion(selected_sources: int, expanded_chars: int) -> None:
    """Record bounded source-expansion cardinality and output size."""

    SOURCE_COUNT.observe(max(0, int(selected_sources)))
    EXPANDED_CONTEXT_SIZE.observe(max(0, int(expanded_chars)))


def latest_metrics() -> tuple[bytes, str]:
    """Return the latest metrics payload and content type."""

    if CollectorRegistry is None:
        return b"", CONTENT_TYPE_LATEST
    payload = generate_latest(REGISTRY)
    return payload, CONTENT_TYPE_LATEST

