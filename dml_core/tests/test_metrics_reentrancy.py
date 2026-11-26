"""Telemetry must not deadlock cleanup invoked by an allocation-time finalizer."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from daystrom_dml import metrics


# A subprocess bounds the actual deadlock on the unguarded implementation. This
# uses Prometheus's real labels lock and DMLAdapter's real destructor/close path.
_FINALIZER_REENTRY = r'''
import gc
import tempfile
import weakref
from prometheus_client import Histogram
from daystrom_dml import metrics
from daystrom_dml.dml_adapter import DMLAdapter

with tempfile.TemporaryDirectory() as directory:
    adapter = DMLAdapter(config_overrides={
        "model_name": "dummy", "embedding_model": None,
        "storage_dir": directory, "persistence": {"enable": False},
        "metrics_enabled": True,
    }, start_aging_loop=False)
    # Force the cleanup transaction to exercise its ordinary state-reload metric.
    adapter.persistence_coordinator.refresh = lambda **kwargs: True
    closed = []
    original_close = type(adapter.store).close
    def track_close(self):
        original_close(self)
        closed.append(True)
    type(adapter.store).close = track_close
    reference = weakref.ref(adapter)
    retained = [adapter]
    del adapter
    original_init = Histogram._metric_init
    invoked = []
    def init_and_collect(self):
        if not invoked:
            invoked.append(True)
            assert metrics.OPERATION_LATENCY._lock.locked()
            retained.clear()
            gc.collect()
            assert reference() is None
            assert closed == [True], "business cleanup must finish"
        return original_init(self)
    Histogram._metric_init = init_and_collect
    metrics.record_operation("finalizer_outer", latency_ms=1)
    Histogram._metric_init = original_init
    assert invoked == [True]
    metrics.record_operation("after_finalizer", latency_ms=2)
    payload, _ = metrics.latest_metrics()
    assert b'dml_operation_count_total{operation="after_finalizer"} 1.0' in payload
    assert b'dml_operation_latency_ms_count{operation="finalizer_outer"} 1.0' in payload
    print("finalizer cleanup and subsequent metrics completed", flush=True)
'''


def test_adapter_finalizer_inside_histogram_labels_completes() -> None:
    environment = dict(os.environ)
    source_root = str(Path(metrics.__file__).resolve().parent.parent)
    environment["PYTHONPATH"] = os.pathsep.join([source_root, environment.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-c", _FINALIZER_REENTRY], env=environment,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "finalizer cleanup and subsequent metrics completed" in result.stdout


def test_metric_exception_is_propagated_and_guard_is_reset(monkeypatch) -> None:
    class BrokenCounter:
        def labels(self, **kwargs):
            raise ValueError("metric failed")

    counter = metrics.OPERATION_COUNTER
    monkeypatch.setattr(metrics, "OPERATION_COUNTER", BrokenCounter())
    with pytest.raises(ValueError, match="metric failed"):
        metrics.record_operation("failed_metric")
    monkeypatch.setattr(metrics, "OPERATION_COUNTER", counter)
    metrics.record_operation("after_metric_exception", count=3)
    payload, _ = metrics.latest_metrics()
    assert b'dml_operation_count_total{operation="after_metric_exception"} 3.0' in payload


def test_guard_is_thread_local_and_covers_cross_metric_reentry(monkeypatch) -> None:
    entered = threading.Event()
    release = threading.Event()
    normal = metrics.OPERATION_COUNTER

    class BlockingCounter:
        def labels(self, **kwargs):
            if kwargs["operation"] == "blocked_outer":
                # A different telemetry entry point must not reenter Prometheus.
                metrics.record_retrieval("nested_finalizer", 1)
                entered.set()
                assert release.wait(5)
            return normal.labels(**kwargs)

    monkeypatch.setattr(metrics, "OPERATION_COUNTER", BlockingCounter())
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(metrics.record_operation, "blocked_outer")
        try:
            assert entered.wait(5)
            pool.submit(metrics.record_operation, "independent_thread", count=2).result(timeout=5)
        finally:
            release.set()
        first.result(timeout=5)
    payload, _ = metrics.latest_metrics()
    assert b'dml_operation_count_total{operation="independent_thread"} 2.0' in payload
    assert b'dml_operation_count_total{operation="blocked_outer"} 1.0' in payload
    assert b'mode="nested_finalizer"' not in payload


def test_scrape_guards_finalizer_metrics_and_resets_after_error(monkeypatch) -> None:
    def collect(_registry):
        metrics.record_operation("nested_scrape_finalizer")
        assert metrics.latest_metrics()[0] == b""
        raise RuntimeError("scrape failed")

    original = metrics.generate_latest
    monkeypatch.setattr(metrics, "generate_latest", collect)
    with pytest.raises(RuntimeError, match="scrape failed"):
        metrics.latest_metrics()
    monkeypatch.setattr(metrics, "generate_latest", original)
    metrics.record_operation("after_scrape_exception")
    payload, _ = metrics.latest_metrics()
    assert b'operation="nested_scrape_finalizer"' not in payload
    assert b'dml_operation_count_total{operation="after_scrape_exception"} 1.0' in payload
