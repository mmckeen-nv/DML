"""Checkpointing utilities for the Daystrom Memory Lattice."""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Callable, Dict, Optional

from .atomic_io import atomic_write_text
from .store_lock import store_write_lock

LOGGER = logging.getLogger(__name__)

StateProvider = Callable[[], Dict[str, object]]


class CheckpointManager:
    """Persist lattice state to disk on demand and at fixed intervals."""

    def __init__(
        self,
        directory: Path,
        provider: StateProvider,
        *,
        interval_seconds: int = 0,
        retention: int = 3,
        start: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.provider = provider
        self.interval_seconds = max(0, int(interval_seconds))
        self.retention = max(0, int(retention))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._status_lock = threading.RLock()
        self._last_error: Optional[str] = None
        self._last_checkpoint: Optional[str] = None
        self.directory.mkdir(parents=True, exist_ok=True)
        if start:
            self.start()

    def start(self) -> None:
        if self.interval_seconds > 0 and not (self._thread and self._thread.is_alive()):
            self._thread = threading.Thread(
                target=self._loop,
                name="dml-checkpoint",
                daemon=True,
            )
            self._thread.start()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Stop the background loop if it is running."""

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    # operations
    # ------------------------------------------------------------------
    def checkpoint(self) -> Path:
        """Create a checkpoint immediately and return its path."""
        try:
            # Obtain an owned provider snapshot before taking the checkpoint
            # directory lock; no code takes these locks in reverse order.
            payload = self.provider()
            with store_write_lock(self.directory, operation="checkpoint", timeout_ms=30000):
                path = self.directory / f"checkpoint-{time.time_ns():020d}-{uuid.uuid4().hex}.json"
                atomic_write_text(path, json.dumps(payload, indent=2, allow_nan=False))
                self._prune_history()
        except Exception as exc:
            with self._status_lock:
                self._last_error = type(exc).__name__
            LOGGER.exception("DML checkpoint failed")
            raise
        with self._status_lock:
            self._last_error = None
            self._last_checkpoint = path.name
        return path

    def status(self) -> dict:
        """Payload-free operational evidence, also for background failures."""
        with self._status_lock:
            return {"status": "degraded" if self._last_error else "ok",
                    "last_error": self._last_error, "last_checkpoint": self._last_checkpoint}

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _loop(self) -> None:  # pragma: no cover - exercised indirectly via tests
        while not self._stop_event.wait(self.interval_seconds):
            try:
                self.checkpoint()
            except Exception:
                continue

    def _prune_history(self) -> None:
        if self.retention <= 0:
            return
        checkpoints = sorted(self.directory.glob("checkpoint-*.json"), key=lambda path: (path.stat().st_mtime_ns, path.name))
        if len(checkpoints) <= self.retention:
            return
        for stale in checkpoints[:-self.retention]:
            with suppress(FileNotFoundError):
                stale.unlink()
