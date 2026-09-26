"""Bounded scheduling and lifecycle regressions with observable callbacks."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
import threading
import time

import pytest

from daystrom_dml.journal import JournalIntegrityError, JournalSchemaError, RevisionConflict
from daystrom_dml.services.projection import ProjectionError, ProjectionStale, projection_status
from daystrom_dml.services.projection_worker import ProjectionWorker
import daystrom_dml.services.projection_worker as module
from test_projection import pair as projection_pair, write

pair = projection_pair


def until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        threading.Event().wait(0.005)
    raise AssertionError("bounded wait expired")


@pytest.mark.parametrize("field", ["poll_interval", "retry_initial", "retry_max"])
@pytest.mark.parametrize("value", [True, False, None, "1", 0, 1e-300, -1, float("inf"), float("nan"), threading.TIMEOUT_MAX * 2, 10**1000])
def test_invalid_scheduling_limits(pair, field, value):
    with pytest.raises(ValueError):
        ProjectionWorker(*pair, **{field: value})


def test_retry_cap_must_not_precede_first_delay(pair):
    with pytest.raises(ValueError):
        ProjectionWorker(*pair, retry_initial=2, retry_max=1)


@pytest.mark.parametrize("value", [True, None, "1", -1, float("inf"), float("nan"), threading.TIMEOUT_MAX * 2, 10**1000])
def test_invalid_close_timeout_does_not_close_worker(pair, value):
    worker = ProjectionWorker(*pair)
    with pytest.raises(ValueError):
        worker.close(value)
    assert worker.status()["state"] == "new"
    assert worker.close(0)


def test_close_before_start_is_terminal(pair):
    worker = ProjectionWorker(*pair)
    assert not worker.request_sync()
    assert worker.close(0)
    assert worker.close(0)
    assert not worker.request_sync()
    with pytest.raises(RuntimeError):
        worker.start()


def test_start_thread_failure_can_be_retried(pair, monkeypatch):
    worker = ProjectionWorker(*pair, poll_interval=60)
    original = threading.Thread.start
    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", lambda _: (_ for _ in ()).throw(RuntimeError("no thread")))
        with pytest.raises(RuntimeError):
            worker.start()
    assert worker.status()["state"] == "new"
    assert threading.Thread.start is original
    assert worker.start()
    assert worker.close()


def test_single_flight_coalesces_256_wakes_and_starts(pair, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    actual = module.reconcile_incremental

    def blocked(source, backend):
        calls.append(threading.get_ident())
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
        return actual(source, backend)

    monkeypatch.setattr(module, "reconcile_incremental", blocked)
    worker = ProjectionWorker(*pair, poll_interval=60)
    try:
        assert worker.start()
        assert entered.wait(3)
        with ThreadPoolExecutor(max_workers=32) as pool:
            assert all(pool.map(lambda _: worker.start() and worker.request_sync(), range(256)))
        assert len(calls) == 1
        assert worker.status()["pending"]
        release.set()
        until(lambda: worker.status()["successes"] == 2)
        assert len(calls) == 2
        assert not worker.status()["pending"]
    finally:
        release.set()
        assert worker.close()


@pytest.mark.parametrize("error_type", [OSError, TimeoutError, sqlite3.OperationalError, RevisionConflict, ProjectionStale])
def test_retry_backoff_caps_and_wakes_do_not_bypass_delay(pair, monkeypatch, error_type):
    calls = []
    recovered = threading.Event()
    actual = module.reconcile_incremental

    def outage(source, backend):
        calls.append(time.monotonic())
        if len(calls) <= 3:
            raise error_type("database is locked" if error_type is sqlite3.OperationalError else "private credential in backend failure")
        result = actual(source, backend)
        recovered.set()
        return result

    monkeypatch.setattr(module, "reconcile_incremental", outage)
    worker = ProjectionWorker(*pair, poll_interval=60, retry_initial=0.05, retry_max=0.1)
    try:
        worker.start()
        until(lambda: worker.status()["state"] == "retry_wait")
        assert "private credential" not in json.dumps(worker.status())
        # Repeated wakes throughout the outage must preserve .05/.10/.10 waits.
        deadline = time.monotonic() + 2
        while not recovered.is_set() and time.monotonic() < deadline:
            assert worker.request_sync()
            threading.Event().wait(0.001)
        assert recovered.wait(1)
        until(lambda: worker.status()["successes"] >= 1)
        assert all(after - before >= minimum for before, after, minimum in zip(calls, calls[1:], [0.045, 0.095, 0.095]))
        assert worker.status()["consecutive_failures"] == 0
        assert worker.status()["error_type"] is None
    finally:
        assert worker.close()


@pytest.mark.parametrize("error_type", [JournalIntegrityError, JournalSchemaError, ProjectionError, sqlite3.OperationalError, ValueError, RuntimeError, SystemExit, KeyboardInterrupt])
def test_permanent_errors_fail_closed_without_retry_or_secret(pair, monkeypatch, error_type):
    def fail(*_):
        raise error_type("secret memory body")

    monkeypatch.setattr(module, "reconcile_incremental", fail)
    worker = ProjectionWorker(*pair, poll_interval=0.01)
    worker.start()
    until(lambda: worker.status()["state"] == "faulted")
    status = worker.status()
    assert status["attempts"] == 1
    assert status["error_type"] == error_type.__name__
    assert not status["in_flight"]
    assert not worker.request_sync()
    assert "secret memory body" not in json.dumps(status)
    with pytest.raises(RuntimeError):
        worker.start()
    assert worker.close()


def test_close_is_bounded_and_concurrent_close_drains_admitted_callback(pair, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    actual = module.reconcile_incremental

    def blocked(source, backend):
        calls.append(True)
        entered.set()
        assert release.wait(3)
        return actual(source, backend)

    monkeypatch.setattr(module, "reconcile_incremental", blocked)
    worker = ProjectionWorker(*pair, poll_interval=0.01)
    worker.start()
    try:
        assert entered.wait(3)
        worker.request_sync()
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=8) as pool:
            assert not any(pool.map(lambda _: worker.close(0.01), range(8)))
        assert time.monotonic() - started < 1
        assert worker.status()["state"] == "closing"
        assert worker.status()["in_flight"]
        assert not worker.status()["pending"]
        assert not worker.request_sync()
        with pytest.raises(RuntimeError):
            worker.start()
        release.set()
        assert worker.close()
        assert calls == [True]
        assert worker.status()["state"] == "closed"
        assert worker.status()["successes"] == 1
    finally:
        release.set()
        worker.close()


def test_callback_can_close_own_worker_without_deadlock(pair, monkeypatch):
    outcomes = []
    actual = module.reconcile_incremental
    worker = ProjectionWorker(*pair)

    def self_close(source, backend):
        outcomes.append(worker.close(0))
        return actual(source, backend)

    monkeypatch.setattr(module, "reconcile_incremental", self_close)
    worker.start()
    until(lambda: worker.status()["state"] == "closed")
    assert outcomes == [False]
    assert worker.close()


def test_stop_interrupts_long_backoff_and_poll_wait(pair, monkeypatch):
    def fail(*_):
        raise OSError("outage")

    monkeypatch.setattr(module, "reconcile_incremental", fail)
    worker = ProjectionWorker(*pair, retry_initial=60, retry_max=60)
    worker.start()
    until(lambda: worker.status()["state"] == "retry_wait")
    assert worker.close(1)
    assert worker.status()["attempts"] == 1


def test_periodic_poll_catches_source_changes_and_status_is_detached(pair):
    source, target = pair
    worker = ProjectionWorker(source, target, poll_interval=0.02)
    try:
        worker.start()
        until(lambda: worker.status()["successes"] >= 1)
        observed = worker.status()
        assert observed["observation_semantics"] == "historical_pinned_source"
        observed["last_result"]["source"]["source_revision"] = -999
        assert worker.status()["last_result"]["source"]["source_revision"] >= 0
        write(source, "two")
        until(lambda: projection_status(source, target)["matches_pinned_source"])
        assert len(target.read()["items"]) == 2
    finally:
        assert worker.close()


def test_source_drift_report_does_not_busy_loop(pair, monkeypatch):
    calls = []

    def drift(*_):
        calls.append(time.monotonic())
        return {"matches_pinned_source": False}

    monkeypatch.setattr(module, "reconcile_incremental", drift)
    worker = ProjectionWorker(*pair, poll_interval=0.05)
    try:
        worker.start()
        until(lambda: worker.status()["successes"] >= 3)
        assert all(b - a >= 0.045 for a, b in zip(calls, calls[1:]))
        assert not worker.status()["last_result"]["matches_pinned_source"]
    finally:
        assert worker.close()


@pytest.mark.parametrize("code, message, transient", [
    (5 | (1 << 8), "redacted resource", True),
    (6, "redacted resource", True),
    (13, "redacted resource", True),
    (10, "redacted resource", True),
    (14, "redacted resource", True),
    (7, "redacted resource", True),
    (1, "database is locked", False),
    (None, "no such table: state", False),
    (None, "database disk image is malformed", False),
    (None, "unknown operational failure", False),
    (None, "database is locked", True),
])
def test_sqlite_error_classification_fails_closed_for_schema_and_unknown_errors(pair, monkeypatch, code, message, transient):
    error = sqlite3.OperationalError(message)
    if code is not None:
        error.sqlite_errorcode = code

    def fail(*_):
        raise error

    monkeypatch.setattr(module, "reconcile_incremental", fail)
    worker = ProjectionWorker(*pair, retry_initial=60, retry_max=60)
    try:
        worker.start()
        expected = "retry_wait" if transient else "faulted"
        until(lambda: worker.status()["state"] == expected)
        assert worker.status()["attempts"] == 1
        assert worker.status()["error_type"] == "OperationalError"
    finally:
        assert worker.close()


def test_last_success_is_explicitly_historical_during_outage(pair, monkeypatch):
    actual = module.reconcile_incremental
    worker = ProjectionWorker(*pair, poll_interval=60, retry_initial=60, retry_max=60)
    try:
        worker.start()
        until(lambda: worker.status()["successes"] == 1)
        observed = worker.status()["last_result"]

        def fail(*_):
            raise OSError("private server token")

        monkeypatch.setattr(module, "reconcile_incremental", fail)
        assert worker.request_sync()
        until(lambda: worker.status()["state"] == "retry_wait")
        status = worker.status()
        assert status["last_result"] == observed
        assert status["observation_semantics"] == "historical_pinned_source"
        assert status["error_type"] == "OSError"
        assert status["consecutive_failures"] == 1
        assert status["successes"] == 1
    finally:
        assert worker.close()
        monkeypatch.setattr(module, "reconcile_incremental", actual)


def test_raising_exception_code_property_faults_honestly(pair, monkeypatch):
    def broken(_):
        raise RuntimeError("private formatting failure")

    attributes = {"sqlite_errorcode": property(broken)}
    error_type = type("BrokenOperationalError", (sqlite3.OperationalError,), attributes)

    def fail(*_):
        raise error_type("private memory contents")

    monkeypatch.setattr(module, "reconcile_incremental", fail)
    worker = ProjectionWorker(*pair, poll_interval=0.01)
    try:
        worker.start()
        until(lambda: worker.status()["state"] == "faulted")
        status = worker.status()
        assert status["attempts"] == 1
        assert not status["in_flight"]
        assert status["error_type"] == "BrokenOperationalError"
        assert "private" not in json.dumps(status)
        assert not worker.request_sync()
        with pytest.raises(RuntimeError):
            worker.start()
    finally:
        assert worker.close()


def test_blocked_exception_code_property_preserves_bounded_close(pair, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def blocked(_):
        entered.set()
        assert release.wait(3)
        return 5

    attributes = {"sqlite_errorcode": property(blocked)}
    error_type = type("BlockedOperationalError", (sqlite3.OperationalError,), attributes)

    def fail(*_):
        raise error_type("private memory contents")

    monkeypatch.setattr(module, "reconcile_incremental", fail)
    worker = ProjectionWorker(*pair, poll_interval=0.01)
    try:
        worker.start()
        assert entered.wait(3)
        started = time.monotonic()
        assert not worker.close(0.01)
        assert time.monotonic() - started < 0.5
        status = worker.status()
        assert status["state"] == "closing"
        assert status["in_flight"]
        assert status["attempts"] == 1
        release.set()
        assert worker.close()
        assert worker.status()["attempts"] == 1
    finally:
        release.set()
        assert worker.close()


def test_raising_exception_metaclass_uses_fixed_error_type(pair, monkeypatch):
    class BrokenType(type):
        def __getattribute__(cls, key):
            if key == "__name__":
                raise RuntimeError("private type metadata")
            return super().__getattribute__(key)

    class BrokenError(sqlite3.OperationalError, metaclass=BrokenType):
        pass

    def fail(*_):
        raise BrokenError()

    monkeypatch.setattr(module, "reconcile_incremental", fail)
    worker = ProjectionWorker(*pair)
    try:
        worker.start()
        until(lambda: worker.status()["state"] == "faulted")
        assert worker.status()["error_type"] == "BackendError"
        assert not worker.status()["in_flight"]
    finally:
        assert worker.close()


def test_blocking_exception_metaclass_does_not_lock_out_close(pair, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    class BlockingType(type):
        def __getattribute__(cls, key):
            if key == "__name__":
                entered.set()
                assert release.wait(3)
            return super().__getattribute__(key)

    class BlockingError(sqlite3.OperationalError, metaclass=BlockingType):
        pass

    def fail(*_):
        raise BlockingError("database is locked")

    monkeypatch.setattr(module, "reconcile_incremental", fail)
    worker = ProjectionWorker(*pair)
    try:
        worker.start()
        assert entered.wait(3)
        started = time.monotonic()
        assert not worker.close(0.01)
        assert time.monotonic() - started < 0.5
        assert worker.status()["in_flight"]
        release.set()
        assert worker.close()
        assert worker.status()["error_type"] == "BlockingError"
    finally:
        release.set()
        assert worker.close()
