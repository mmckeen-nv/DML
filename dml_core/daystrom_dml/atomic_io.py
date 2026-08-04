"""Portable crash-conscious file replacement helpers."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Callable

_WINDOWS_RETRY_ERRORS = {5, 32, 33}  # access denied / sharing / lock violation


def replace_with_retry(source: Path, target: Path, *, timeout: float = 2.0) -> None:
    """Replace ``target`` with bounded retries for transient Windows sharing errors."""

    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        try:
            os.replace(source, target)
            return
        except OSError as exc:
            if os.name != "nt" or getattr(exc, "winerror", None) not in _WINDOWS_RETRY_ERRORS:
                raise
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.025)


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    """Durably write bytes using a unique sibling and atomic name replacement."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(tmp, target)
        _sync_directory(target.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Durably write text using a unique sibling and atomic name replacement."""

    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_via(path: str | Path, writer: Callable[[Path], None]) -> Path:
    """Atomically install output produced by a path-based binary writer (for FAISS)."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    tmp = Path(raw_tmp)
    try:
        writer(tmp)
        # Windows' CRT rejects ``fsync`` on a read-only descriptor even though
        # POSIX accepts it. Open the completed artifact read/write so the same
        # durability barrier works on every supported platform.
        with tmp.open("rb+") as handle:
            os.fsync(handle.fileno())
        replace_with_retry(tmp, target)
        _sync_directory(target.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def _sync_directory(path: Path) -> None:
    """Sync directory entries where the platform exposes directory descriptors."""

    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
