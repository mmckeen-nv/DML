"""Independent checkpoint crash, shutdown and maintenance-outcome harness."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

from daystrom_dml import checkpoint as module
from daystrom_dml.checkpoint import CheckpointManager
from daystrom_dml.dml_adapter import DMLAdapter


def snapshots(directory):
    return sorted(directory.glob("checkpoint-*.json"))


@pytest.mark.parametrize("point, generations", [
    ("before_publish", {0}), ("after_publish", {0, 1}), ("before_prune", {0, 1}),
    ("before_delete", {0, 1}), ("after_delete", {1}), ("after_prune", {1}),
])
def test_process_death_at_publication_and_retention_boundaries(tmp_path, point, generations):
    CheckpointManager(tmp_path, lambda: {"generation": 0}, retention=1).checkpoint()
    code = '''
import os, signal, sys
from pathlib import Path
from daystrom_dml.checkpoint import CheckpointManager
point, root = sys.argv[1:]
def crash(here):
    if here == point:
        if os.name == 'posix':
            os.kill(os.getpid(), signal.SIGKILL)
        os._exit(79)
CheckpointManager(Path(root), lambda: {'generation':1}, retention=1, fault_hook=crash).checkpoint()
raise AssertionError('crash hook not reached')
'''
    child = subprocess.run([sys.executable, "-c", code, point, str(tmp_path)],
                           capture_output=True, text=True, timeout=30)
    assert child.returncode == (-9 if os.name == "posix" else 79), child.stderr
    assert {json.loads(path.read_text())["generation"] for path in snapshots(tmp_path)} == generations
    latest = CheckpointManager(tmp_path, lambda: {"generation": 2}, retention=1).checkpoint()
    assert snapshots(tmp_path) == [latest]
    assert json.loads(latest.read_text()) == {"generation": 2}


def test_close_waits_for_manual_provider_and_prevents_later_publication(tmp_path):
    entered, release = threading.Event(), threading.Event()
    def provider():
        entered.set()
        assert release.wait(5)
        return {"generation": 1}
    manager = CheckpointManager(tmp_path, provider)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(manager.checkpoint)
        assert entered.wait(2)
        try:
            assert manager.close(timeout=.01) is False
            status = manager.status()
            assert status["lifecycle"] == "closing" and status["in_flight"] == 1
        finally:
            release.set()
        with pytest.raises(RuntimeError):
            future.result(timeout=3)
    assert manager.close(timeout=1) is True
    assert manager.status()["lifecycle"] == "closed"
    assert snapshots(tmp_path) == []
    for action in (manager.start, manager.checkpoint):
        with pytest.raises(RuntimeError):
            action()


def test_close_during_started_atomic_publication_reports_undrained_then_completes(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    real = module.atomic_write_text
    def blocked_write(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return real(*args, **kwargs)
    monkeypatch.setattr(module, "atomic_write_text", blocked_write)
    manager = CheckpointManager(tmp_path, lambda: {"generation": 1})
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(manager.checkpoint)
        assert entered.wait(2)
        try:
            assert manager.close(timeout=.01) is False
        finally:
            release.set()
        path = future.result(timeout=3)
    assert path.exists()
    assert manager.close(timeout=1) is True
    assert manager.status()["in_flight"] == 0


def test_concurrent_start_creates_one_worker(tmp_path, monkeypatch):
    real_thread = threading.Thread
    constructed = []
    guard = threading.Lock()
    def tracked_thread(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        if kwargs.get("name") == "dml-checkpoint":
            with guard:
                constructed.append(thread)
        return thread
    monkeypatch.setattr(module.threading, "Thread", tracked_thread)
    manager = CheckpointManager(tmp_path, lambda: {}, interval_seconds=100, start=False)
    barrier = threading.Barrier(16)
    def start(_):
        barrier.wait(timeout=3)
        manager.start()
    try:
        with ThreadPoolExecutor(max_workers=16) as executor:
            list(executor.map(start, range(16)))
        assert len(constructed) == 1
        assert constructed[0].is_alive()
    finally:
        assert manager.close(timeout=2) is True
    assert not constructed[0].is_alive()


def test_published_checkpoint_survives_retention_failure_and_reports_both_outcomes(tmp_path, monkeypatch):
    manager = CheckpointManager(tmp_path, lambda: {"generation": 1}, retention=1)
    old = manager.checkpoint()
    original = manager._prune_history
    def fail(*args, **kwargs):
        raise OSError("SENSITIVE_RETENTION_DETAIL")
    monkeypatch.setattr(manager, "_prune_history", fail)
    new = manager.checkpoint()
    assert old.exists() and new.exists() and new != old
    status = manager.status()
    assert status["status"] == "degraded" and status["publication_outcome"] == "published"
    assert status["last_checkpoint"] == new.name and status["retention_error"] == "OSError"
    monkeypatch.setattr(manager, "_prune_history", original)
    recovered = manager.checkpoint()
    assert snapshots(tmp_path) == [recovered]
    assert manager.status()["status"] == "ok"


def test_clock_rollback_does_not_prune_just_published_file(tmp_path, monkeypatch):
    now = iter((9000, 1000, 1))
    monkeypatch.setattr(module.time, "time_ns", lambda: next(now))
    manager = CheckpointManager(tmp_path, lambda: {}, retention=1)
    paths = [manager.checkpoint() for _ in range(3)]
    assert snapshots(tmp_path) == [paths[-1]]
    assert paths == sorted(paths)


def test_corrupt_recognized_checkpoint_never_causes_good_snapshots_to_be_pruned(tmp_path):
    manager = CheckpointManager(tmp_path, lambda: {"generation": 1}, retention=2)
    good, corrupt = manager.checkpoint(), manager.checkpoint()
    corrupt.write_text('{"generation":999}')  # valid JSON, invalid published digest
    new = manager.checkpoint()
    assert good.exists() and corrupt.exists() and new.exists()
    assert manager.status()["status"] == "degraded"
    assert manager.status()["retention_error"]


def test_unrecognized_and_legacy_checkpoints_are_preserved_outside_new_retention_quota(tmp_path):
    legacy = tmp_path / "checkpoint-1600000000.json"
    unknown = tmp_path / "checkpoint-unrecognized.json"
    legacy.write_text('{"old":"evidence"}')
    unknown.write_text('broken imported bytes')
    manager = CheckpointManager(tmp_path, lambda: {"generation": 1}, retention=1)
    old, new = manager.checkpoint(), manager.checkpoint()
    assert legacy.read_text() == '{"old":"evidence"}'
    assert unknown.read_text() == 'broken imported bytes'
    assert not old.exists() and new.exists()


def test_payload_is_frozen_before_waiting_for_directory_lock(tmp_path, monkeypatch):
    payload = {"generation": 1, "memories": ["original"]}
    real = module.store_write_lock
    @contextmanager
    def interleaved(*args, **kwargs):
        payload["memories"].append("changed while waiting")
        with real(*args, **kwargs) as owned:
            yield owned
    monkeypatch.setattr(module, "store_write_lock", interleaved)
    path = CheckpointManager(tmp_path, lambda: payload).checkpoint()
    assert json.loads(path.read_text()) == {"generation": 1, "memories": ["original"]}


def test_atomic_publication_error_is_uncertain_and_preserves_prior_snapshots(tmp_path, monkeypatch):
    manager = CheckpointManager(tmp_path, lambda: {"generation": 1}, retention=1)
    old = manager.checkpoint()
    real = module.atomic_write_text
    def committed_then_error(*args, **kwargs):
        real(*args, **kwargs)
        raise OSError("SENSITIVE_FSYNC_DETAIL")
    monkeypatch.setattr(module, "atomic_write_text", committed_then_error)
    with pytest.raises(OSError):
        manager.checkpoint()
    assert old.exists() and len(snapshots(tmp_path)) == 2
    assert manager.status()["publication_outcome"] == "uncertain"


@pytest.mark.parametrize("value", [["unsupported root"], {"value": float("nan")}])
def test_invalid_provider_payload_does_not_publish(tmp_path, value):
    manager = CheckpointManager(tmp_path, lambda: value)
    with pytest.raises((ValueError, TypeError)):
        manager.checkpoint()
    assert snapshots(tmp_path) == []


def test_checkpoint_failure_logs_exclude_provider_exception_text(tmp_path, caplog):
    def provider():
        raise RuntimeError("SECRET_MEMORY_CANARY")
    manager = CheckpointManager(tmp_path, provider)
    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError):
        manager.checkpoint()
    assert "SECRET_MEMORY_CANARY" not in caplog.text
    assert "SECRET_MEMORY_CANARY" not in json.dumps(manager.status())
    assert manager.status()["last_error"] == "RuntimeError"


def test_adapter_drains_checkpoints_before_persisting_or_closing_dependencies():
    calls = []
    class Manager:
        drained = False
        def close(self):
            calls.append("checkpoint-close")
            return self.drained
    class Store:
        def close(self):
            calls.append("store-close")
    adapter = DMLAdapter.__new__(DMLAdapter)
    adapter._projection_lifecycle_lock = threading.RLock()
    adapter._projection_workers_closing = False
    adapter._owned_projection_worker = None
    adapter.checkpoint_manager = Manager()
    adapter.store = Store()
    adapter.metrics_enabled = False
    adapter._stop_persistence_loop = lambda: calls.append("persistence-loop-stop")
    adapter._persist_all = lambda: calls.append("persist")
    @contextmanager
    def transaction(_):
        yield
    adapter._mutation_transaction = transaction
    with pytest.raises(TimeoutError):
        adapter.close()
    assert calls == ["persistence-loop-stop", "checkpoint-close"]
    calls.clear()
    adapter.checkpoint_manager.drained = True
    adapter.close()
    assert calls == ["persistence-loop-stop", "checkpoint-close", "persist", "store-close"]


def test_processes_publish_unique_order_and_retain_latest_publications(tmp_path):
    barrier = tmp_path / "start"
    code = '''
import json, sys, time
from pathlib import Path
from daystrom_dml import checkpoint as module
from daystrom_dml.checkpoint import CheckpointManager
root, gate, ident = sys.argv[1:]
module.time.time_ns = lambda: 1000
deadline = time.monotonic() + 20
while not Path(gate).exists():
    if time.monotonic() > deadline:
        raise TimeoutError('barrier')
    time.sleep(.005)
payload = {'client':int(ident),'step':0}
manager = CheckpointManager(Path(root), lambda: dict(payload), retention=3)
receipts = []
for step in range(6):
    payload['step'] = step
    path = manager.checkpoint()
    receipts.append({'name':path.name,'payload':dict(payload)})
print(json.dumps(receipts))
'''
    children = [subprocess.Popen([sys.executable, "-c", code, str(tmp_path), str(barrier), str(i)],
                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for i in range(6)]
    receipts = []
    try:
        barrier.touch()
        for child in children:
            stdout, stderr = child.communicate(timeout=30)
            assert child.returncode == 0, stderr
            receipts.extend(json.loads(stdout))
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)
    names = sorted(receipt["name"] for receipt in receipts)
    assert len(names) == len(set(names)) == 36
    assert [int(name.split("-")[1]) for name in names] == list(range(1000, 1036))
    assert [path.name for path in snapshots(tmp_path)] == names[-3:]
    expected = {receipt["name"]: receipt["payload"] for receipt in receipts}
    for path in snapshots(tmp_path):
        assert json.loads(path.read_text()) == expected[path.name]


def test_json_compatible_numeric_keys_are_preserved_but_collisions_are_rejected(tmp_path):
    good = CheckpointManager(tmp_path, lambda: {"levels": {0: 5, 1: 2}}).checkpoint()
    assert json.loads(good.read_text()) == {"levels": {"0": 5, "1": 2}}
    manager = CheckpointManager(tmp_path, lambda: {"levels": {0: 5, "0": 2}})
    with pytest.raises((ValueError, TypeError)):
        manager.checkpoint()
    assert snapshots(tmp_path) == [good]


def test_manual_adapter_checkpoints_share_close_boundary_when_periodic_worker_disabled(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    gather = DMLAdapter._gather_checkpoint_state
    def blocked_provider(instance):
        entered.set()
        assert release.wait(5)
        return gather(instance)
    class Embedder:
        def embed(self, text):
            return np.ones(4, dtype=np.float32)
    monkeypatch.setattr(DMLAdapter, "_gather_checkpoint_state", blocked_provider)
    adapter = DMLAdapter(config_overrides={"storage_dir": str(tmp_path), "model_name": "dummy",
        "embedding_model": None, "checkpoint_interval_seconds": 0,
        "dpm": {"enable": False}, "rag_store": {"enable": False}},
        embedder=Embedder(), start_aging_loop=False)
    assert adapter.checkpoint_manager is not None
    assert adapter.checkpoint_manager.interval_seconds == 0
    store_close, closed = adapter.store.close, []
    def track_store_close():
        closed.append(True)
        return store_close()
    monkeypatch.setattr(adapter.store, "close", track_store_close)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(adapter.create_checkpoint)
        assert entered.wait(2)
        try:
            with pytest.raises(TimeoutError):
                adapter.close(persist=False)
            assert closed == [], "a still-running provider owns its store dependencies"
        finally:
            release.set()
        with pytest.raises(RuntimeError):
            future.result(timeout=3)
    adapter.close(persist=False)
    assert closed == [True]
    assert snapshots(adapter.checkpoint_dir) == []
    with pytest.raises(RuntimeError):
        adapter.create_checkpoint()


def test_directory_is_anchored_before_working_directory_changes(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    manager = CheckpointManager(Path("checkpoints"), lambda: {})
    monkeypatch.chdir(second)
    result = manager.checkpoint()
    assert result.parent == first / "checkpoints"
    assert not (second / "checkpoints").exists()


def test_recognized_symlink_is_not_a_trusted_retention_candidate(tmp_path):
    directory = tmp_path / "checkpoints"
    manager = CheckpointManager(directory, lambda: {"generation": 1}, retention=1)
    original = manager.checkpoint()
    external = tmp_path / "external.json"
    data = original.read_bytes()
    external.write_bytes(data)
    original.unlink()
    try:
        original.symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this runner")
    new = manager.checkpoint()
    assert original.is_symlink() and new.exists()
    assert external.read_bytes() == data
    assert manager.status()["status"] == "degraded"


def test_old_retention_failure_cannot_overwrite_newer_success_status(tmp_path, monkeypatch):
    role = threading.local()
    old_released, new_completed = threading.Event(), threading.Event()
    manager = CheckpointManager(tmp_path, lambda: {"generation": 1}, retention=1)
    real_lock, real_prune = module.store_write_lock, manager._prune_history
    @contextmanager
    def release_interleaving(*args, **kwargs):
        try:
            with real_lock(*args, **kwargs) as owned:
                yield owned
        finally:
            if role.name == "old":
                old_released.set()
                assert new_completed.wait(5)
    def prune():
        if role.name == "old":
            raise OSError("older retention failure")
        return real_prune()
    monkeypatch.setattr(module, "store_write_lock", release_interleaving)
    monkeypatch.setattr(manager, "_prune_history", prune)
    def attempt(name):
        role.name = name
        path = manager.checkpoint()
        if name == "new":
            new_completed.set()
        return path
    with ThreadPoolExecutor(max_workers=2) as executor:
        older = executor.submit(attempt, "old")
        assert old_released.wait(2)
        newer = executor.submit(attempt, "new")
        try:
            latest = newer.result(timeout=3)
        finally:
            new_completed.set()
        older.result(timeout=3)
    status = manager.status()
    assert status["last_checkpoint"] == latest.name
    assert status["publication_outcome"] == "published"
    assert status["retention_error"] is None
    assert status["status"] == "ok"


def test_new_publication_cannot_hide_retention_failure_until_maintenance_completes(tmp_path, monkeypatch):
    role = threading.local()
    old_released, new_waiting, release_new = threading.Event(), threading.Event(), threading.Event()
    def pause_new_retention(point):
        if role.name == "new" and point == "before_prune":
            new_waiting.set()
            assert release_new.wait(5)
    manager = CheckpointManager(tmp_path, lambda: {"generation": 1}, retention=1, fault_hook=pause_new_retention)
    real_lock, real_prune = module.store_write_lock, manager._prune_history
    @contextmanager
    def interleaved_release(*args, **kwargs):
        try:
            with real_lock(*args, **kwargs) as owned:
                yield owned
        finally:
            if role.name == "old":
                old_released.set()
                assert new_waiting.wait(5)
    def prune():
        if role.name == "old":
            raise OSError("retention still needs repair")
        return real_prune()
    monkeypatch.setattr(module, "store_write_lock", interleaved_release)
    monkeypatch.setattr(manager, "_prune_history", prune)
    def attempt(name):
        role.name = name
        return manager.checkpoint()
    with ThreadPoolExecutor(max_workers=2) as executor:
        older = executor.submit(attempt, "old")
        assert old_released.wait(2)
        newer = executor.submit(attempt, "new")
        try:
            assert new_waiting.wait(2)
            older.result(timeout=3)
            assert manager.status()["retention_error"] == "OSError"
            assert manager.status()["status"] == "degraded"
        finally:
            release_new.set()
        newer.result(timeout=3)
    assert manager.status()["retention_error"] is None
    assert manager.status()["status"] == "ok"
