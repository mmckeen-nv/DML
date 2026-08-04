"""Security tests for opt-in credential persistence."""
from __future__ import annotations

import stat
from pathlib import Path

import pytest

from daystrom_dml.secret_storage import persist_secret


def test_persist_secret_is_atomic_and_owner_only(tmp_path: Path) -> None:
    repository = tmp_path / "checkout"
    repository.mkdir()
    target = tmp_path / "secrets" / "ngc-key"

    persist_secret("super-secret", target, repository_root=repository)

    assert target.read_text(encoding="utf-8") == "super-secret\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not list(target.parent.glob(f".{target.name}.*"))


def test_persist_secret_rejects_repository_path(tmp_path: Path) -> None:
    repository = tmp_path / "checkout"
    repository.mkdir()

    with pytest.raises(ValueError, match="outside the repository"):
        persist_secret("secret", repository / "key", repository_root=repository)


def test_persist_secret_rejects_relative_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        persist_secret("secret", Path("key"), repository_root=tmp_path)


def test_persist_secret_rejects_symlink(tmp_path: Path) -> None:
    repository = tmp_path / "checkout"
    repository.mkdir()
    real_target = tmp_path / "real-key"
    real_target.write_text("original", encoding="utf-8")
    link = tmp_path / "linked-key"
    link.symlink_to(real_target)

    with pytest.raises(ValueError, match="symlink"):
        persist_secret("replacement", link, repository_root=repository)

    assert real_target.read_text(encoding="utf-8") == "original"


def test_persist_secret_rejects_non_regular_target(tmp_path: Path) -> None:
    repository = tmp_path / "checkout"
    repository.mkdir()
    target = tmp_path / "key-directory"
    target.mkdir()

    with pytest.raises(ValueError, match="regular file"):
        persist_secret("secret", target, repository_root=repository)
