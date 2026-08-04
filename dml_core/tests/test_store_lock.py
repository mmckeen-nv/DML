"""Cross-platform store-lock regression tests."""
from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys
import time

import pytest

from daystrom_dml.store_lock import store_write_lock


def test_two_writers_cannot_hold_same_store_lock(tmp_path: Path):
    with store_write_lock(tmp_path, operation="writer-one", timeout_ms=0):
        with pytest.raises(TimeoutError):
            with store_write_lock(tmp_path, operation="writer-two", timeout_ms=0):
                pass


def test_store_lock_releases_after_exception(tmp_path: Path):
    with pytest.raises(RuntimeError):
        with store_write_lock(tmp_path, operation="crashing-writer", timeout_ms=0):
            raise RuntimeError("simulated failure")

    with store_write_lock(tmp_path, operation="recovered-writer", timeout_ms=0):
        assert (tmp_path / ".dml_store.lock").exists()


def _locking_child(tmp_path: Path, ready_path: Path) -> subprocess.Popen[str]:
    code = """
import os, pathlib, time
from daystrom_dml.store_lock import store_write_lock
root = pathlib.Path(os.environ['DML_LOCK_ROOT'])
ready = pathlib.Path(os.environ['DML_LOCK_READY'])
with store_write_lock(root, operation='spawned-child', timeout_ms=2000):
    ready.write_text('ready', encoding='utf-8')
    time.sleep(30)
"""
    env = os.environ.copy()
    env["DML_LOCK_ROOT"] = str(tmp_path)
    env["DML_LOCK_READY"] = str(ready_path)
    core_path = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (core_path, env.get("PYTHONPATH", "")) if part
    )
    return subprocess.Popen([sys.executable, "-c", code], env=env, text=True)


def test_lock_contends_across_spawned_processes_and_recovers_after_termination(tmp_path: Path):
    ready = tmp_path / "child.ready"
    child = _locking_child(tmp_path, ready)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "spawned lock holder did not become ready"
        with pytest.raises(TimeoutError):
            with store_write_lock(tmp_path, operation="parent-contender", timeout_ms=100):
                pass
    finally:
        child.terminate()
        child.wait(timeout=5)

    with store_write_lock(tmp_path, operation="parent-after-termination", timeout_ms=1000):
        assert (tmp_path / ".dml_store.lock").exists()
