"""Bounded, coalescing reconciliation of a disposable projection.

The journal remains the authority and the restart cursor: there is no volatile
queue of memory mutations. Backend calls happen without source ownership locks.
A blocked backend cannot be cancelled; close returns False until its one admitted
reconciliation (possibly several backend calls) drains. Use a backend with its
own I/O deadlines when bounded drain is needed. Scheduling intervals are at least
one millisecond to avoid deadlines rounding to now and causing busy loops.
"""
from __future__ import annotations

import copy
import math
import sqlite3
import threading
import time

from ..journal import JournalStateStore, RevisionConflict
from .projection import ProjectionStale
from .projection_delta import ProjectionBackend, reconcile_incremental


def _seconds(value: float, name: str, *, allow_zero: bool = False) -> float:
    if (type(value) not in (int, float) or value > threading.TIMEOUT_MAX
            or value < 0 or not math.isfinite(value)
            or (value < 0.001 and not allow_zero)):
        raise ValueError(f"{name} must be finite and within the supported timeout range")
    return float(value)


def _transient(error: BaseException) -> bool:
    if isinstance(error, sqlite3.OperationalError):
        # Syntax/schema problems also use OperationalError. Retry only known
        # resource/contention errors; extended SQLite codes share a low byte.
        code = getattr(error, "sqlite_errorcode", None)
        if type(code) is int:
            # Stable SQLite primary codes: BUSY, LOCKED, FULL, IOERR,
            # CANTOPEN, NOMEM; names are not exported by every Python 3.10.
            return code & 0xFF in (5, 6, 13, 10, 14, 7)
        # Python 3.10 has no sqlite_errorcode. Exact messages are deliberately
        # conservative; unknown errors fault instead of retrying indefinitely.
        return str(error) in {"database is locked", "database table is locked",
                              "database schema is locked", "database or disk is full",
                              "disk I/O error", "unable to open database file", "out of memory"}
    return isinstance(error, (OSError, RevisionConflict, ProjectionStale))


class ProjectionWorker:
    """Explicit, single-flight worker with capped exponential retry delays.

    start is idempotent while active. Closing or a permanent error is terminal;
    recovery from a permanent error requires a new worker after repairing the
    cause. Wake requests coalesce into one bit and never defeat retry backoff.
    Successful reports describe pinned observations, never ongoing freshness.
    """

    def __init__(self, source: JournalStateStore, backend: ProjectionBackend, *,
                 poll_interval: float = 1.0, retry_initial: float = 0.1,
                 retry_max: float = 30.0) -> None:
        self._poll_interval = _seconds(poll_interval, "poll_interval")
        self._retry_initial = _seconds(retry_initial, "retry_initial")
        self._retry_max = _seconds(retry_max, "retry_max")
        if self._retry_max < self._retry_initial:
            raise ValueError("retry_max must be at least retry_initial")
        self._source = source
        self._backend = backend
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._state = "new"
        self._closing = False
        self._pending = False
        self._in_flight = False
        self._next_attempt = 0.0
        self._retry_delay = 0.0
        self._attempts = 0
        self._successes = 0
        self._consecutive_failures = 0
        self._error_type: str | None = None
        self._last_result: dict | None = None
        self._last_completed_at: float | None = None

    def start(self) -> bool:
        with self._condition:
            if self._state in ("faulted", "closing", "closed"):
                raise RuntimeError("Projection worker is terminal")
            if self._thread is not None:
                return True
            thread = threading.Thread(target=self._run, name="dml-projection", daemon=True)
            self._state = "running"
            self._pending = True
            self._thread = thread
            try:
                thread.start()
            except BaseException:
                self._state = "new"
                self._pending = False
                self._thread = None
                raise
            return True

    def request_sync(self) -> bool:
        with self._condition:
            if self._state not in ("running", "retry_wait"):
                return False
            self._pending = True
            self._condition.notify_all()
            return True

    def close(self, timeout: float = 5.0) -> bool:
        """Close admission permanently, waiting at most timeout for admitted I/O.

        A callback may close its own worker: admission closes immediately and
        False reports that the calling callback itself has not yet drained.
        """
        timeout = _seconds(timeout, "timeout", allow_zero=True)
        deadline = time.monotonic() + timeout
        with self._condition:
            self._closing = True
            self._pending = False
            thread = self._thread
            if thread is None or not thread.is_alive():
                self._state = "closed"
                return True
            self._state = "closing"
            self._condition.notify_all()
            if thread is threading.current_thread():
                return False
        # Never join while holding the condition required by callback completion.
        thread.join(max(0.0, deadline - time.monotonic()))
        with self._condition:
            drained = not thread.is_alive()
            if drained:
                self._state = "closed"
            return drained

    def status(self) -> dict:
        """Return detached runtime telemetry; last_result is historical/pinned."""
        with self._condition:
            return {
                "state": self._state, "in_flight": self._in_flight,
                "pending": self._pending, "attempts": self._attempts,
                "successes": self._successes,
                "consecutive_failures": self._consecutive_failures,
                "error_type": self._error_type,
                "retry_in_seconds": (max(0.0, self._next_attempt - time.monotonic())
                                     if self._state == "retry_wait" else None),
                "last_result": copy.deepcopy(self._last_result),
                "last_completed_at_monotonic": self._last_completed_at,
                "observation_semantics": "historical_pinned_source",
            }

    def _run(self) -> None:
        while True:
            with self._condition:
                while True:
                    if self._closing:
                        self._state = "closed"
                        self._condition.notify_all()
                        return
                    remaining = self._next_attempt - time.monotonic()
                    if remaining <= 0 or (self._pending and self._state != "retry_wait"):
                        break
                    self._condition.wait(min(remaining, threading.TIMEOUT_MAX))
                # This lock defines admission; close cannot admit another call.
                self._pending = False
                self._in_flight = True
                self._state = "running"
                self._attempts += 1
            result = None
            error = None
            try:
                result = reconcile_incremental(self._source, self._backend)
            except BaseException as exc:
                # Even SystemExit from an injected backend must leave honest
                # terminal telemetry, not an apparently running dead thread.
                error = exc
            transient = False
            error_type = "BackendError"
            if error is not None:
                try:
                    # Exception attributes/formatters may themselves execute
                    # backend code. Keep admission in flight and the condition
                    # unlocked so close/status remain bounded if they block.
                    candidate_type = type(error).__name__
                    if type(candidate_type) is not str:
                        raise TypeError("Backend exception type name must be a string")
                    error_type = candidate_type
                    transient = _transient(error)
                except BaseException:
                    # Classification failure is permanent. Retain the original
                    # type when available, otherwise use the fixed fallback.
                    transient = False
            with self._condition:
                self._in_flight = False
                self._last_completed_at = time.monotonic()
                if error is None:
                    self._successes += 1
                    self._consecutive_failures = 0
                    self._error_type = None
                    self._retry_delay = 0.0
                    self._last_result = result
                    self._next_attempt = time.monotonic() + self._poll_interval
                else:
                    self._consecutive_failures += 1
                    self._error_type = error_type
                    self._retry_delay = (self._retry_initial if self._retry_delay == 0
                                         else min(self._retry_max, self._retry_delay * 2))
                    self._next_attempt = time.monotonic() + self._retry_delay
                    self._state = "retry_wait" if transient else "faulted"
                if self._closing:
                    self._state = "closed"
                    self._pending = False
                    self._condition.notify_all()
                    return
                if self._state == "faulted":
                    self._pending = False
                    self._condition.notify_all()
                    return
                self._condition.notify_all()
