"""Independent scheduling-pressure and damaged-authority regression oracles."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
import sqlite3
import threading
import time

import daystrom_dml.services.projection_worker as module
from daystrom_dml.services.projection_worker import ProjectionWorker
from test_projection import pair as projection_pair
from test_projection_worker import until

pair = projection_pair


def test_concurrent_sustained_wake_storm_preserves_retry_delays(pair, monkeypatch):
    calls = []
    enough_attempts = threading.Event()
    stop = threading.Event()

    def outage(*_):
        calls.append(time.monotonic())
        if len(calls) >= 4:
            enough_attempts.set()
        raise TimeoutError("private backend credential")

    def storm(_):
        while not stop.wait(0.001):
            worker.request_sync()

    monkeypatch.setattr(module, "reconcile_incremental", outage)
    worker = ProjectionWorker(*pair, poll_interval=60, retry_initial=0.05, retry_max=0.1)
    worker.start()
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(storm, index) for index in range(8)]
            try:
                assert enough_attempts.wait(3)
            finally:
                stop.set()
            for future in futures:
                future.result(timeout=3)
        assert calls[1] - calls[0] >= 0.045
        assert all(after - before >= 0.095 for before, after in zip(calls[1:], calls[2:]))
        assert "private backend credential" not in json.dumps(worker.status())
    finally:
        stop.set()
        assert worker.close()


def test_real_authority_corruption_after_success_faults_without_erasing_history(pair):
    source, target = pair
    worker = ProjectionWorker(source, target, poll_interval=60)
    worker.start()
    try:
        until(lambda: worker.status()["successes"] == 1)
        previous = worker.status()["last_result"]
        published = target.read()
        with closing(sqlite3.connect(source.path)) as connection:
            with connection:
                connection.execute("UPDATE state SET checksum=?", ("0" * 64,))
        assert worker.request_sync()
        until(lambda: worker.status()["state"] == "faulted")
        status = worker.status()
        assert status["error_type"] == "JournalIntegrityError"
        assert status["attempts"] == 2
        assert status["successes"] == 1
        assert status["consecutive_failures"] == 1
        assert status["last_result"] == previous
        assert status["observation_semantics"] == "historical_pinned_source"
        assert status["retry_in_seconds"] is None
        assert not worker.request_sync()
        assert target.read() == published
        with closing(sqlite3.connect(source.path)) as connection:
            assert connection.execute("SELECT checksum FROM state").fetchone()[0] == "0" * 64
        status["last_result"]["source"]["source_revision"] = -999
        assert worker.status()["last_result"] == previous
    finally:
        assert worker.close()


def test_unformattable_operational_error_faults_with_honest_terminal_status(pair, monkeypatch):
    class Unformattable:
        def __str__(self):
            raise RuntimeError("exception text cannot be formatted")

    def fail(*_):
        raise sqlite3.OperationalError(Unformattable())

    monkeypatch.setattr(module, "reconcile_incremental", fail)
    worker = ProjectionWorker(*pair)
    worker.start()
    try:
        until(lambda: worker.status()["state"] == "faulted")
        status = worker.status()
        assert status["error_type"] == "OperationalError"
        assert not status["in_flight"]
        assert status["attempts"] == 1
        assert not worker.request_sync()
    finally:
        assert worker.close()


def test_blocked_error_classification_does_not_hold_lifecycle_lock(pair, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    class BlockedMessage:
        def __str__(self):
            entered.set()
            assert release.wait(5)
            return "database is locked"

    def fail(*_):
        raise sqlite3.OperationalError(BlockedMessage())

    monkeypatch.setattr(module, "reconcile_incremental", fail)
    worker = ProjectionWorker(*pair)
    worker.start()
    try:
        assert entered.wait(3)
        with ThreadPoolExecutor(max_workers=1) as executor:
            closing_worker = executor.submit(worker.close, 0.01)
            try:
                # A blocked formatter is still part of the admitted attempt.
                # A timeout here also releases the callback before executor exit.
                assert closing_worker.result(timeout=1) is False
                assert worker.status()["in_flight"]
                assert worker.status()["state"] == "closing"
            finally:
                release.set()
        assert worker.close()
        assert worker.status()["attempts"] == 1
    finally:
        release.set()
        assert worker.close()
