"""Opt-in, hardened storage for runtime credentials."""
from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path


def persist_secret(secret: str, path: Path, *, repository_root: Path) -> None:
    """Atomically persist ``secret`` to an explicitly external file.

    Credential files are deliberately forbidden inside the source checkout.  The
    caller must opt in and provide an absolute path (normally a mounted secret
    volume) outside ``repository_root``.
    """

    if not path.is_absolute():
        raise ValueError("credential persistence path must be absolute")

    root = repository_root.resolve()
    resolved = path.resolve(strict=False)
    if resolved == root or root in resolved.parents:
        raise ValueError("credential persistence path must be outside the repository")

    if path.is_symlink():
        raise ValueError("credential persistence path must not be a symlink")
    if path.exists() and not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
        raise ValueError("credential persistence target must be a regular file")

    parent = path.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("credential persistence parent must be a real directory")

    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(secret.strip() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        path.chmod(0o600)
    finally:
        if fd >= 0:
            os.close(fd)
        temporary_path.unlink(missing_ok=True)
