"""Cross-platform advisory locking for shared DML stores."""
from __future__ import annotations

import errno
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO

from .atomic_io import atomic_write_text

if os.name == "nt":  # pragma: no cover - exercised on Windows CI
    import msvcrt
else:  # pragma: no cover - platform-specific import
    import fcntl

_CONTENTION_ERRNOS = {errno.EACCES, errno.EAGAIN}
if hasattr(errno, "EDEADLK"):
    _CONTENTION_ERRNOS.add(errno.EDEADLK)


def _ensure_lock_byte(handle: TextIO) -> None:
    """Ensure Windows has a stable byte range to lock."""

    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write("\0")
        handle.flush()
    handle.seek(0)


def acquire_file_lock(handle: TextIO) -> None:
    """Acquire a non-blocking exclusive lock or raise ``BlockingIOError``."""

    try:
        if os.name == "nt":  # pragma: no cover - exercised on Windows CI
            _ensure_lock_byte(handle)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in _CONTENTION_ERRNOS or getattr(exc, "winerror", None) in {33, 36, 158}:
            raise BlockingIOError(exc.errno or errno.EAGAIN, str(exc)) from exc
        raise


def release_file_lock(handle: TextIO) -> None:
    """Release a lock previously acquired with :func:`acquire_file_lock`."""

    if os.name == "nt":  # pragma: no cover - exercised on Windows CI
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def store_write_lock(
    storage_dir: str | os.PathLike[str],
    *,
    operation: str,
    timeout_ms: int = 30000,
) -> Iterator[dict]:
    """Serialize mutations performed by independent DML processes."""

    root = Path(storage_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".dml_store.lock"
    metadata_path = root / ".dml_store.lock.json"
    started = time.perf_counter()
    handle = lock_path.open("a+", encoding="utf-8")
    acquired = False
    try:
        while True:
            try:
                acquire_file_lock(handle)
                acquired = True
                break
            except BlockingIOError:
                waited_ms = (time.perf_counter() - started) * 1000.0
                if timeout_ms <= 0 or waited_ms >= timeout_ms:
                    raise TimeoutError(
                        f"Timed out waiting for DML store lock {lock_path} during {operation}"
                    )
                time.sleep(min(0.05, max(0.005, (timeout_ms - waited_ms) / 1000.0)))
        metadata = {
            "operation": operation,
            "pid": os.getpid(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "lock_path": str(lock_path),
        }
        atomic_write_text(metadata_path, json.dumps(metadata, indent=2, sort_keys=True))
        yield metadata
    finally:
        if acquired:
            try:
                metadata_path.unlink(missing_ok=True)
            finally:
                release_file_lock(handle)
        handle.close()
