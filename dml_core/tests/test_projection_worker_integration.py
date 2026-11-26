"""Adapter-owned projection workers: authority isolation and bounded shutdown."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from daystrom_dml.services.projection import SQLiteProjection
from test_receipt_adapter import append, factory as receipt_factory

factory = receipt_factory


def eventually(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        threading.Event().wait(0.005)
    assert predicate(), "projection worker did not reach the expected state"


class ControlledBackend:
    def __init__(self, target):
        self.path = target.path
        self.target = target
        self.available = threading.Event()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.available.set()
        self.release.set()

    def read(self):
        self.entered.set()
        if not self.release.wait(10):
            raise TimeoutError("test backend was not released")
        if not self.available.is_set():
            raise OSError("private backend credential")
        return self.target.read()

    def apply_delta(self, delta):
        if not self.available.is_set():
            raise OSError("private backend credential")
        return self.target.apply_delta(delta)


@pytest.fixture
def backend(tmp_path):
    target = SQLiteProjection(tmp_path / "projection" / "index.sqlite3")
    value = ControlledBackend(target)
    yield value
    value.release.set()
    value.available.set()


def start(adapter, backend):
    return adapter.start_projection_worker(backend, poll_interval=0.02,
                                           retry_initial=0.02, retry_max=0.08)


def test_worker_is_opt_in_and_outage_does_not_block_acknowledged_ingestion(factory, backend):
    adapter = factory()
    first = append(adapter)
    assert adapter._owned_projection_worker is None
    assert backend.target.read()["cursor"] is None
    backend.available.clear()
    worker = start(adapter, backend)
    eventually(lambda: worker.status()["consecutive_failures"] > 0)
    later = append(adapter, text="Durable during projection outage", key="later")
    assert later["revision"] > first["revision"]
    assert append(adapter) == first
    assert backend.target.read()["cursor"] is None
    assert "private backend credential" not in repr(worker.status())
    backend.available.set()
    worker.request_sync()
    eventually(lambda: (worker.status()["last_result"] or {}).get("record_count") == 2)
    assert adapter.projection_status(backend)["matches_pinned_source"] is True
    assert {item["id"] for item in backend.target.read()["items"]} == {
        first["result"]["memory"]["id"], later["result"]["memory"]["id"]}
    adapter.close(persist=False)
    assert worker.status()["state"] == "closed"


def test_blocked_backend_drain_preserves_dependencies_and_retry_close_works(factory, backend, monkeypatch):
    adapter = factory()
    append(adapter)
    backend.release.clear()
    worker = start(adapter, backend)
    assert backend.entered.wait(5)
    closed = []
    real_close = adapter.store.close
    monkeypatch.setattr(adapter.store, "close", lambda: (closed.append("store"), real_close()))
    real_checkpoint_close = adapter.checkpoint_manager.close
    monkeypatch.setattr(adapter.checkpoint_manager, "close",
                        lambda: (closed.append("checkpoint"), real_checkpoint_close())[1])
    try:
        before = time.monotonic()
        with pytest.raises(TimeoutError, match="Projection shutdown"):
            adapter.close(persist=False, projection_timeout=0.02)
        assert time.monotonic() - before < 1.0
        assert closed == []
        assert worker.status()["state"] == "closing"
        assert worker.status()["in_flight"] is True
        with pytest.raises(RuntimeError, match="shutdown"):
            start(adapter, backend)
        assert worker.request_sync() is False
    finally:
        backend.release.set()
    adapter.close(persist=False, projection_timeout=5)
    assert closed == ["checkpoint", "store"]
    assert worker.status()["state"] == "closed"
    attempts = worker.status()["attempts"]
    assert worker.request_sync() is False
    assert worker.status()["attempts"] == attempts


def test_start_is_idempotent_for_same_backend_and_rejects_switching(factory, backend):
    adapter = factory()
    worker = start(adapter, backend)
    with ThreadPoolExecutor(max_workers=16) as executor:
        workers = list(executor.map(lambda _: start(adapter, backend), range(64)))
    assert all(value is worker for value in workers)
    # First settings win, so retrying an active start cannot reschedule it.
    assert adapter.start_projection_worker(backend, poll_interval=20) is worker
    with pytest.raises(ValueError, match="one projection backend"):
        start(adapter, ControlledBackend(backend.target))
    adapter.close(persist=False)
    with pytest.raises(RuntimeError, match="shutdown"):
        start(adapter, backend)


def test_close_without_a_worker_terminally_rejects_later_start(factory, backend):
    adapter = factory()
    adapter.close(persist=False)
    with pytest.raises(RuntimeError, match="shutdown"):
        start(adapter, backend)
    assert not backend.entered.is_set()


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf"), True, "1", None,
                                   threading.TIMEOUT_MAX * 2, 10 ** 1000])
def test_invalid_close_timeout_does_not_begin_shutdown(factory, backend, timeout):
    adapter = factory()
    with pytest.raises(ValueError, match="timeout"):
        adapter.close(persist=False, projection_timeout=timeout)
    assert start(adapter, backend) is not None


def test_legacy_adapter_and_nested_source_ownership_refuse_workers(factory, backend):
    legacy = factory(persistence={"journal": True, "receipts": False, "enable": False})
    with pytest.raises(ValueError, match="opt-in receipt journal"):
        start(legacy, backend)
    adapter = factory(directory=backend.path.parent.parent / "receipt-authority")
    with adapter.mutation_transaction("no-worker-inside-ownership"):
        with pytest.raises(ValueError, match="source ownership"):
            start(adapter, backend)
    assert not backend.entered.is_set()


def test_concurrent_start_and_close_cannot_leave_an_owned_worker_running(factory, tmp_path):
    for index in range(8):
        adapter = factory(directory=tmp_path / f"authority-{index}")
        backend = ControlledBackend(SQLiteProjection(tmp_path / f"projection-{index}" / "index.sqlite3"))
        gate = threading.Barrier(2)

        def race_start():
            gate.wait(timeout=5)
            try:
                return start(adapter, backend)
            except RuntimeError:
                return None

        def race_close():
            gate.wait(timeout=5)
            adapter.close(persist=False)

        with ThreadPoolExecutor(max_workers=2) as executor:
            pending_start = executor.submit(race_start)
            pending_close = executor.submit(race_close)
            worker = pending_start.result(timeout=10)
            pending_close.result(timeout=10)
        if worker is not None:
            assert worker.status()["state"] == "closed"
            assert worker.status()["in_flight"] is False
        with pytest.raises(RuntimeError, match="shutdown"):
            start(adapter, backend)


def test_restart_reconciles_unprojected_receipts_from_authority(factory, backend):
    first = factory()
    receipt = append(first)
    backend.available.clear()
    original = start(first, backend)
    eventually(lambda: original.status()["consecutive_failures"] > 0)
    first.close(persist=False)
    assert backend.target.read()["cursor"] is None
    backend.available.set()
    restarted = factory()
    assert append(restarted) == receipt
    replacement = start(restarted, backend)
    eventually(lambda: (replacement.status()["last_result"] or {}).get("matches_pinned_source") is True)
    assert replacement is not original
    assert [item["id"] for item in backend.target.read()["items"]] == [receipt["result"]["memory"]["id"]]
    assert restarted.projection_status(backend)["matches_pinned_source"] is True
