"""Runtime admission and public-operation guards for the candidate profile."""
from __future__ import annotations

from contextlib import closing, contextmanager
from functools import wraps
import math
import os
from pathlib import Path
import sqlite3
import threading
import time

from ..contracts.profile import ProductionProfileError, validate_profile_authority
from ..journal import JournalIntegrityError, require_patched_sqlite


class ProfileOperationLifetime:
    """Fence and drain one selected-profile adapter without serializing its work.

    Admission covers preparation, authority access and response construction.
    Shutdown does not cancel admitted work: a timed-out caller must retry close,
    while a lost mutation acknowledgement must be resolved using its same key.
    """

    def __init__(self) -> None:
        self._owner_pid = os.getpid()
        self._condition = threading.Condition(threading.Lock())
        self._active: dict[int, int] = {}
        self._closing = False
        self._closed = False
        self._cleanup_owner: int | None = None

    def _require_owner(self) -> None:
        # Never touch an inherited mutex: its owning thread may not exist here.
        if os.getpid() != self._owner_pid:
            raise RuntimeError("Inherited profile adapter cannot be used after fork; create a fresh adapter")

    @contextmanager
    def operation(self):
        self._require_owner()
        ident = threading.get_ident()
        with self._condition:
            if self._closing:
                raise RuntimeError("Selected profile is closing or closed; create a fresh adapter")
            self._active[ident] = self._active.get(ident, 0) + 1
        try:
            yield
        finally:
            self._require_owner()
            with self._condition:
                remaining = self._active[ident] - 1
                if remaining:
                    self._active[ident] = remaining
                else:
                    del self._active[ident]
                self._condition.notify_all()

    @contextmanager
    def shutdown(self, *, timeout: float):
        self._require_owner()
        deadline = time.monotonic() + timeout
        with self._condition:
            if threading.get_ident() in self._active:
                raise RuntimeError("Cannot close a profile adapter from an admitted operation")
            if threading.get_ident() == self._cleanup_owner:
                raise RuntimeError("Cannot close a profile adapter from its dependency cleanup")
            self._closing = True
            while self._active or self._cleanup_owner is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Profile shutdown is incomplete; retry close after admitted operations drain")
                self._condition.wait(remaining)
            cleanup = not self._closed
            if cleanup:
                self._cleanup_owner = threading.get_ident()
        if not cleanup:
            yield False
            return
        succeeded = False
        try:
            yield True
            succeeded = True
        finally:
            with self._condition:
                self._closed = succeeded
                self._cleanup_owner = None
                self._condition.notify_all()


def admitted_profile_operation(method):
    """Keep legacy lifetimes unchanged; selected profiles reject use after close."""
    @wraps(method)
    def guarded(self, *args, **kwargs):
        lifetime = getattr(self, "_profile_operation_lifetime", None)
        if lifetime is None:
            return method(self, *args, **kwargs)
        with lifetime.operation():
            return method(self, *args, **kwargs)
    return guarded


def outside_production_profile(method):
    """Keep a legacy capability unavailable on a selected profile instance."""
    @wraps(method)
    def guarded(self, *args, **kwargs):
        if getattr(self, "_production_profile", None) is not None:
            raise ProductionProfileError("Operation is outside the selected production profile")
        return method(self, *args, **kwargs)
    return guarded


def preflight_profile_authority(profile_id: str | None, storage_dir: Path,
                                *, outbox_enabled: bool) -> None:
    """Check an existing SQLite view before provider initialization.

    Read-only SQLite access must honor committed WAL content. SQLite may use
    transient shared-memory/locking sidecars; this is not an immutable-file read.
    The journal repeats schema admission under its initialization lock.
    """
    if profile_id is None:
        return
    try:
        require_patched_sqlite()
    except JournalIntegrityError as exc:
        raise ProductionProfileError(str(exc)) from exc
    path = storage_dir.expanduser().resolve() / "dml_state.sqlite3"
    if path.with_name(path.name + ".migration.json").exists():
        raise ProductionProfileError("Incomplete journal migration requires recovery")
    if not path.exists():
        if path.with_name(path.name + ".identity.json").exists():
            raise ProductionProfileError("Previously initialized journal is missing")
        return
    if not path.is_file():
        raise ProductionProfileError("Profile authority must be a SQLite file")
    try:
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30)) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    except sqlite3.Error as exc:
        raise ProductionProfileError("Profile authority cannot be inspected") from exc
    validate_profile_authority(profile_id, version, outbox_enabled)
    if journal_mode != "wal":
        raise ProductionProfileError("Selected profile requires SQLite WAL journal mode")


def validate_profile_retrieval(prompt, *, scope, top_k, kinds, phase,
                               include_quarantined, as_of, dpm_identifiers) -> None:
    """Reject malformed scoped read inputs before embedding or ownership."""
    if type(prompt) is not str or not prompt.strip() or len(prompt.encode("utf-8")) > 1024 * 1024:
        raise ProductionProfileError("Profile query must be nonempty text of at most 1 MiB")
    for index, value in enumerate(scope):
        if value is None and index:
            continue
        if type(value) is not str or not value.strip() or len(value.encode("utf-8")) > 256:
            raise ProductionProfileError("Profile scope requires exact nonempty string identifiers")
    if top_k is not None and (type(top_k) is not int or not 1 <= top_k <= 10):
        raise ProductionProfileError("Profile top_k must be an integer from 1 through 10")
    if kinds is not None and (type(kinds) is not list or len(kinds) > 256 or any(
            type(value) is not str or not value.strip() or len(value.encode("utf-8")) > 256
            for value in kinds)):
        raise ProductionProfileError("Profile kinds must be a bounded list of string identifiers")
    if phase is not None and (type(phase) is not str or phase not in {"plan", "build", "execute", "debug", "reflect"}):
        raise ProductionProfileError("Profile phase is invalid")
    if type(include_quarantined) is not bool:
        raise ProductionProfileError("Profile quarantine selection must be boolean")
    if as_of is not None:
        try:
            finite_time = type(as_of) in (int, float) and math.isfinite(as_of)
        except OverflowError:
            finite_time = False
        if not finite_time:
            raise ProductionProfileError("Profile as_of must be a finite number")
    if any(value is not None for value in dpm_identifiers):
        raise ProductionProfileError("DPM is outside the selected production profile")
