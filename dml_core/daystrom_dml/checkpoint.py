"""Crash-conscious semantic checkpoint publication and bounded worker lifecycle."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, Optional

from .atomic_io import _sync_directory, atomic_write_text
from .store_lock import store_write_lock

LOGGER = logging.getLogger(__name__)
StateProvider = Callable[[], Dict[str, object]]
_CHECKPOINT_NAME = re.compile(r"checkpoint-([0-9]{20,})-[0-9a-f]{32}-([0-9a-f]{64})\.json\Z")
_ORDER_NAME = re.compile(r"checkpoint-([0-9]{20,})-[0-9a-f]{32}(?:-[0-9a-f]{64})?\.json\Z")


class CheckpointClosedError(RuntimeError):
    """The manager has permanently stopped accepting publications."""


class CheckpointIntegrityError(ValueError):
    """A checkpoint failed integrity validation; retention was not attempted."""


def _validate_json(value: object) -> None:
    """Reject coercions and non-finite values that silently change saved state."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Checkpoint object keys must be strings")
        for child in value.values():
            _validate_json(child)
        return
    if isinstance(value, list):
        for child in value:
            _validate_json(child)
        return
    raise ValueError("Checkpoint state must contain only finite JSON values")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate checkpoint object key")
        result[key] = value
    return result


class CheckpointManager:
    """Publish frozen JSON snapshots; close is terminal and reports actual drain.

    Providers must return an owned snapshot. Capture and serialization happen
    before the directory lock, which is never held while calling the provider.
    Retention applies only to this format's integrity-checked files. Legacy or
    unrecognized files are preserved for explicit operator migration/removal.
    """

    FAULT_POINTS = (
        "before_publish", "after_publish", "before_prune",
        "before_delete", "after_delete", "after_prune",
    )

    def __init__(
        self,
        directory: Path,
        provider: StateProvider,
        *,
        interval_seconds: int = 0,
        retention: int = 3,
        start: bool = True,
        fault_hook: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.directory = Path(directory).resolve()
        self.provider = provider
        self.interval_seconds = max(0, int(interval_seconds))
        self.retention = max(0, int(retention))
        self._fault_hook = fault_hook
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._status_lock = threading.RLock()
        self._condition = threading.Condition(self._status_lock)
        self._closed = False
        self._in_flight = 0
        self._operation_threads: dict[int, int] = {}
        self._last_error: Optional[str] = None
        self._last_checkpoint: Optional[str] = None
        self._last_publication: Optional[str] = None
        self._publication_outcome = "none"
        self._retention_error: Optional[str] = None
        self._directory_epoch = 0
        self._publication_epoch = 0
        self._retention_epoch = 0
        self.directory.mkdir(parents=True, exist_ok=True)
        if start:
            self.start()

    def start(self) -> None:
        """Start at most one worker; a closed manager cannot be restarted."""
        with self._condition:
            if self._closed:
                raise CheckpointClosedError("Checkpoint manager is closed")
            if self.interval_seconds > 0 and not (self._thread and self._thread.is_alive()):
                thread = threading.Thread(target=self._loop, name="dml-checkpoint", daemon=True)
                self._thread = thread
                try:
                    thread.start()
                except BaseException:
                    self._thread = None
                    raise

    def close(self, timeout: float = 1.0) -> bool:
        """Stop admissions and wait up to timeout seconds; False means draining.

        Python cannot cancel a blocked provider. Once it returns, it cannot
        begin publication. A publication already started may finish while this
        method reports False. True guarantees no active operation or worker.
        """
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("Checkpoint close timeout must be finite and nonnegative")
        deadline = time.monotonic() + timeout
        with self._condition:
            self._closed = True
            self._stop_event.set()
            if self._operation_threads.get(threading.get_ident(), 0):
                return False
            while self._in_flight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            thread = self._thread
        if thread is threading.current_thread():
            return False
        if thread is not None:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
            return not thread.is_alive()
        return True

    def checkpoint(self) -> Path:
        """Return a committed snapshot even if subsequent retention fails.

        Publication failures raise; if replacement already happened, status is
        uncertain and the path may exist without a durability acknowledgment.
        Previously committed files are never pruned after a publication error.
        """
        ident = threading.get_ident()
        with self._condition:
            if self._closed:
                raise CheckpointClosedError("Checkpoint manager is closed")
            self._in_flight += 1
            self._operation_threads[ident] = self._operation_threads.get(ident, 0) + 1
        path: Optional[Path] = None
        published = False
        directory_epoch: Optional[int] = None
        phase = "capture"
        try:
            payload = self.provider()
            phase = "serialization"
            if not isinstance(payload, dict):
                raise ValueError("Checkpoint provider must return a JSON object")
            # Freeze before waiting for another manager's directory ownership.
            # Keep the established JSON encoding of tuple values and numeric
            # keys, but reject collisions such as {1: ..., "1": ...}.
            serialized = json.dumps(payload, indent=2, allow_nan=False)
            _validate_json(json.loads(serialized, object_pairs_hook=_unique_object))
            digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            phase = "directory_lock"
            with store_write_lock(self.directory, operation="checkpoint", timeout_ms=30000):
                with self._condition:
                    self._directory_epoch += 1
                    directory_epoch = self._directory_epoch
                phase = "publication"
                self._fault("before_publish")
                with self._condition:
                    if self._closed:
                        raise CheckpointClosedError("Checkpoint manager closed before publication")
                    # This is the publication admission point. close() observes
                    # this operation as in flight until its outcome is recorded.
                order = max(time.time_ns(), self._latest_order() + 1)
                path = self.directory / f"checkpoint-{order:020d}-{uuid.uuid4().hex}-{digest}.json"
                atomic_write_text(path, serialized)
                published = True
                with self._condition:
                    self._publication_epoch = directory_epoch
                    self._last_error = None
                    self._last_checkpoint = path.name
                    self._last_publication = path.name
                    self._publication_outcome = "published"
                phase = "retention"
                self._fault("after_publish")
                self._fault("before_prune")
                self._prune_history()
                self._fault("after_prune")
                with self._condition:
                    self._retention_epoch = directory_epoch
                    self._retention_error = None
            return path
        except BaseException as exc:
            uncertain = False
            if not published and path is not None:
                try:
                    uncertain = path.exists()
                except OSError:
                    uncertain = True
            with self._condition:
                # Lock exit can let a newer operation progress before this
                # handler runs. Only a newer *recorded outcome* supersedes this
                # failure. Publication alone does not repair failed retention;
                # keep their completed-outcome epochs separate.
                if published:
                    assert directory_epoch is not None
                    if directory_epoch >= self._retention_epoch:
                        self._retention_epoch = directory_epoch
                        self._retention_error = type(exc).__name__
                elif directory_epoch is None or directory_epoch >= self._publication_epoch:
                    if directory_epoch is not None:
                        self._publication_epoch = directory_epoch
                    self._last_error = type(exc).__name__
                    self._last_publication = path.name if path is not None else None
                    self._publication_outcome = "uncertain" if uncertain else "not_published"
            # Never log provider text, serialized memory, or arbitrary exception
            # messages/tracebacks in operator logs.
            LOGGER.error("DML checkpoint phase=%s error_type=%s", phase, type(exc).__name__)
            if published and path is not None and isinstance(exc, Exception):
                return path
            raise
        finally:
            with self._condition:
                self._in_flight -= 1
                count = self._operation_threads[ident] - 1
                if count:
                    self._operation_threads[ident] = count
                else:
                    del self._operation_threads[ident]
                self._condition.notify_all()

    def status(self) -> dict:
        """Payload-free publication, maintenance, and lifecycle evidence."""
        with self._condition:
            worker_alive = bool(self._thread and self._thread.is_alive())
            draining = bool(self._in_flight or worker_alive)
            lifecycle = ("closing" if draining else "closed") if self._closed else ("running" if worker_alive else "idle")
            return {
                "status": "degraded" if self._last_error or self._retention_error else "ok",
                "last_error": self._last_error,
                "last_checkpoint": self._last_checkpoint,
                "last_publication": self._last_publication,
                "publication_outcome": self._publication_outcome,
                "retention_error": self._retention_error,
                "lifecycle": lifecycle,
                "in_flight": self._in_flight,
                "worker_alive": worker_alive,
            }

    def _loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                self.checkpoint()
            except Exception:
                continue

    def _fault(self, point: str) -> None:
        if self._fault_hook is not None:
            self._fault_hook(point)

    def _latest_order(self) -> int:
        orders = [int(match.group(1)) for path in self.directory.glob("checkpoint-*.json")
                  if (match := _ORDER_NAME.fullmatch(path.name))]
        return max(orders, default=0)

    def _prune_history(self) -> None:
        if self.retention <= 0:
            return
        checkpoints: list[tuple[int, str, Path]] = []
        for path in self.directory.glob("checkpoint-*.json"):
            match = _CHECKPOINT_NAME.fullmatch(path.name)
            if match is None:
                continue  # Legacy and unrecognized artifacts require explicit cleanup.
            if path.is_symlink() or not path.is_file():
                raise CheckpointIntegrityError("Checkpoint candidate is not a regular file")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != match.group(2):
                raise CheckpointIntegrityError("Checkpoint content digest mismatch")
            try:
                payload = json.loads(raw, object_pairs_hook=_unique_object)
                if not isinstance(payload, dict):
                    raise ValueError("Expected an object")
                _validate_json(payload)
            except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
                raise CheckpointIntegrityError("Checkpoint JSON is invalid") from exc
            checkpoints.append((int(match.group(1)), path.name, path))
        checkpoints.sort()
        # Complete all validation before deleting anything. Invalid snapshots
        # cannot count toward retention and evict the only usable recovery copy.
        for _, _, stale in checkpoints[:-self.retention]:
            self._fault("before_delete")
            stale.unlink()
            self._fault("after_delete")
        if len(checkpoints) > self.retention:
            _sync_directory(self.directory)
