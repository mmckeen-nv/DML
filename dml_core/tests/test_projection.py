"""Independent authority oracle for disposable transactional vector projections."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading

import numpy as np
import pytest

from daystrom_dml.journal import FAULT_POINTS, INIT_FAULT_POINTS, JournalIntegrityError, JournalStateStore
from daystrom_dml.services.projection import (
    ProjectionError, ProjectionStale, SQLiteProjection, prepare_source, projection_status, query_projection, reconcile,
)
from daystrom_dml.services.receipt_ingestion import ReceiptEmbeddingCompatibilityError, append_receipted, canonical_request

IDENTITY = {"backend": "test.reference", "revision": "v1", "model": None, "mode": "native"}
SCOPE = {"tenant_id": "tenant", "client_id": None, "session_id": None, "instance_id": None}


def write(source, key="one", *, text="acknowledged memory", meta=None, scope=None):
    request, digest = canonical_request(text, **(scope or SCOPE), meta=meta)
    return append_receipted(source, request=request, request_digest=digest, key=key,
        embed=lambda _: np.array([1., 0.]), embedding_space=lambda: IDENTITY,
        capacity=1000, hydrate=lambda *_: None, degraded=lambda _: None)


@pytest.fixture
def pair(tmp_path):
    source = JournalStateStore(tmp_path / "authority" / "source.sqlite3", receipt_mode=True)
    write(source)
    return source, SQLiteProjection(tmp_path / "projection" / "index.sqlite3")


def query(source, target, **overrides):
    return query_projection(source, target, **{"vector": [1., 0.], "embedding_identity": IDENTITY,
        "scope": SCOPE, "as_of": 100., "top_k": 10, **overrides})


def run_child(code, *args):
    return subprocess.run([sys.executable, "-c", code, *map(str, args)],
                          capture_output=True, text=True, timeout=45)


@pytest.mark.parametrize("point", FAULT_POINTS)
def test_killed_publication_never_marks_partial_projection_current(pair, point):
    source, target = pair
    acknowledged = prepare_source(source)
    reconcile(source, target)
    prior = target.read()
    write(source, "two", text="second acknowledged memory")
    current = prepare_source(source)
    result = run_child('''
import os, signal, sys
from pathlib import Path
from daystrom_dml.journal import JournalStateStore
from daystrom_dml.services.projection import SQLiteProjection, reconcile
source_path, target_path, point = sys.argv[1:]
def crash(here):
    if here == point:
        if os.name == 'posix': os.kill(os.getpid(), signal.SIGKILL)
        os._exit(79)
source = JournalStateStore(Path(source_path))
reconcile(source, SQLiteProjection(Path(target_path), fault_hook=crash))
raise AssertionError('fault hook not reached')
''', source.path, target.path, point)
    assert result.returncode == (-9 if os.name == "posix" else 79), result.stderr
    assert prepare_source(source) == current, "projection failure must never modify authority"
    reopened = SQLiteProjection(target.path)
    if point != "after_commit":
        assert reopened.read() == prior
        with pytest.raises(ProjectionStale):
            query(source, reopened)
    else:
        assert reopened.read() != prior
        query(source, reopened)
    reconcile(source, reopened)
    query(source, reopened)
    assert prepare_source(source) == current
    assert acknowledged != current


def test_lost_publication_acknowledgement_reconciles_without_duplicating(pair):
    source, target = pair
    once = []
    def fail(point):
        if point == "after_commit" and not once:
            once.append(True)
            raise OSError("private disk details")
    failing = SQLiteProjection(target.path, fault_hook=fail)
    reconcile(source, failing)
    before = failing.read()
    reconcile(source, failing)
    assert failing.read() == before
    assert JournalStateStore(target.path).read_snapshot()[0] == 2
    query(source, failing)


def test_delayed_old_publisher_cannot_replace_new_snapshot_or_resurrect_delete(pair):
    source, target = pair
    old = prepare_source(source)
    reconcile(source, target)
    revision, payload = source.read_snapshot()
    payload["items"] = []
    source.save(payload, expected_revision=revision, operation="trusted-low-level-delete")
    reconcile(source, target)
    after_delete = target.read()
    with pytest.raises(ProjectionStale):
        target.publish(old)
    assert target.read() == after_delete
    query(source, target)


def test_source_writes_continue_during_stalled_projection_io(pair):
    source, target = pair
    entered, release = threading.Event(), threading.Event()
    def delay(point):
        if point == "before_publish":
            entered.set()
            assert release.wait(10)
    delayed = SQLiteProjection(target.path, fault_hook=delay)
    with ThreadPoolExecutor(max_workers=2) as pool:
        sync = pool.submit(reconcile, source, delayed)
        assert entered.wait(5)
        try:
            receipt = pool.submit(write, source, "two").result(timeout=5)
            assert receipt["revision"] == 2
        finally:
            release.set()
        result = sync.result(timeout=10)
    # A stale snapshot may finish atomically, but it cannot serve newer authority.
    assert result["matches_pinned_source"] is False
    assert result["published"]["source_revision"] == 1
    assert result["source"]["source_revision"] == 2
    with pytest.raises((ProjectionError, JournalIntegrityError)):
        query(source, target)
    reconcile(source, target)
    query(source, target)


def test_concurrent_duplicate_publishers_publish_one_target_revision(pair):
    source, target = pair
    snapshot = prepare_source(source)
    clients = int(os.environ.get("DML_STRESS_CLIENTS", "32"))
    barrier = threading.Barrier(clients)
    def publish(_):
        barrier.wait(timeout=30)
        return target.publish(snapshot)
    with ThreadPoolExecutor(max_workers=clients) as pool:
        cursors = list(pool.map(publish, range(clients)))
    assert cursors == [cursors[0]] * clients
    assert JournalStateStore(target.path).read_snapshot()[0] == 2
    query(source, target)


def test_foreign_target_and_source_binding_are_preserved(pair, tmp_path):
    source, target = pair
    foreign_path = tmp_path / "foreign" / "index.sqlite3"
    foreign = JournalStateStore(foreign_path)
    foreign.save({"items": [], "lineage": [], "foreign": True}, expected_revision=0)
    before = foreign.read_snapshot()
    with pytest.raises(ProjectionError):
        reconcile(source, SQLiteProjection(foreign_path))
    assert foreign.read_snapshot() == before
    reconcile(source, target)
    before = target.read()
    other = JournalStateStore(tmp_path / "other" / "source.sqlite3", receipt_mode=True)
    write(other)
    with pytest.raises(ProjectionError):
        reconcile(other, target)
    assert target.read() == before


@pytest.mark.parametrize("damage", ["missing", "record", "cursor"])
def test_damaged_projection_fails_closed_and_authority_receipt_survives(pair, damage):
    source, target = pair
    reconcile(source, target)
    authority = source.read_snapshot()
    if damage == "missing":
        target.path.unlink()
    else:
        with closing(sqlite3.connect(target.path)) as connection:
            if damage == "record":
                connection.execute("UPDATE records SET checksum=?", ("0" * 64,))
            else:
                connection.execute("UPDATE state SET checksum=?", ("0" * 64,))
            connection.commit()
    with pytest.raises((ProjectionError, JournalIntegrityError)):
        query(source, target)
    with pytest.raises(JournalIntegrityError):
        reconcile(source, target)
    assert source.read_snapshot() == authority
    assert write(source)["revision"] == 1


@pytest.mark.parametrize("overrides", [
    {"vector": [float("nan"), 0.]}, {"vector": [0., 0.]}, {"vector": [1.]},
    {"embedding_identity": {**IDENTITY, "revision": "other"}},
    {"scope": {"tenant_id": "tenant"}}, {"scope": {**SCOPE, "unknown": None}},
    {"top_k": True}, {"top_k": -1}, {"as_of": float("nan")},
])
def test_invalid_projection_query_cannot_serve_results(pair, overrides):
    source, target = pair
    reconcile(source, target)
    with pytest.raises((ProjectionError, ReceiptEmbeddingCompatibilityError)):
        query(source, target, **overrides)


def test_authority_and_projection_cannot_share_directory(pair):
    source, _ = pair
    same_directory = source.path.parent / "projection.sqlite3"
    with pytest.raises(ProjectionError):
        reconcile(source, SQLiteProjection(same_directory))
    assert source.read_snapshot()[0] == 1


def test_cli_does_not_create_missing_authority_or_target_on_read(pair, tmp_path, capsys):
    from scripts.dml_projection import main
    source, target = pair
    missing = tmp_path / "missing.sqlite3"
    assert main([str(missing), str(target.path), "sync"]) == 2
    assert not missing.exists()
    absent_target = tmp_path / "absent" / "projection.sqlite3"
    assert main([str(source.path), str(absent_target), "status"]) == 2
    assert not absent_target.exists()
    out = capsys.readouterr().out
    assert "acknowledged memory" not in out
    assert str(tmp_path) not in out


def test_process_publishers_share_one_idempotent_snapshot(pair, tmp_path):
    source, target = pair
    gate = tmp_path / "start"
    code = '''
import json,sys,time
from pathlib import Path
from daystrom_dml.journal import JournalStateStore
from daystrom_dml.services.projection import SQLiteProjection, prepare_source
source_path,target_path,gate=sys.argv[1:]
snapshot=prepare_source(JournalStateStore(Path(source_path)))
projection=SQLiteProjection(Path(target_path))
deadline=time.monotonic()+30
while not Path(gate).exists():
    if time.monotonic()>deadline: raise TimeoutError('parent barrier')
    time.sleep(.005)
print(json.dumps(projection.publish(snapshot),sort_keys=True))
'''
    children = [subprocess.Popen([sys.executable, "-c", code, str(source.path), str(target.path), str(gate)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(8)]
    cursors = []
    try:
        gate.touch()
        for process in children:
            out, err = process.communicate(timeout=45)
            assert process.returncode == 0, err
            cursors.append(json.loads(out))
    finally:
        for process in children:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)
    assert cursors == [cursors[0]] * 8
    assert JournalStateStore(target.path).read_snapshot()[0] == 2
    query(source, target)


def test_target_disk_quota_rolls_back_cursor_and_records(pair, monkeypatch):
    from contextlib import contextmanager
    source, target = pair
    reconcile(source, target)
    before = target.read()
    write(source, "large", text="x" * 500_000)
    connect = target.journal._connect
    @contextmanager
    def limited():
        with connect() as connection:
            pages = connection.execute("PRAGMA page_count").fetchone()[0]
            connection.execute(f"PRAGMA max_page_count={pages}")
            yield connection
    monkeypatch.setattr(target.journal, "_connect", limited)
    with pytest.raises(sqlite3.OperationalError, match="full"):
        reconcile(source, target)
    assert target.read() == before
    assert source.read_snapshot()[0] == 2
    with pytest.raises(ProjectionStale):
        query(source, target)


def test_publication_freezes_input_before_waiting_for_backend(pair):
    source, target = pair
    snapshot = prepare_source(source)
    expected = json.loads(json.dumps(snapshot))
    def mutate(point):
        if point == "before_publish":
            snapshot["source_state"]["items"][0]["text"] = "caller mutation"
    projection = SQLiteProjection(target.path, fault_hook=mutate)
    projection.publish(snapshot)
    assert projection.read()["items"] == expected["source_state"]["items"]
    assert query(source, projection)["results"][0]["memory"]["text"] == "acknowledged memory"


@pytest.mark.parametrize("point", (*INIT_FAULT_POINTS, *FAULT_POINTS))
def test_killed_projection_creation_is_complete_or_explicitly_unusable(tmp_path, point):
    source = JournalStateStore(tmp_path / "authority" / "source.sqlite3", receipt_mode=True)
    receipt = write(source)
    before = source.read_snapshot()
    target = tmp_path / "projection" / "index.sqlite3"
    result = run_child('''
import os,signal,sys
from pathlib import Path
from daystrom_dml.services.projection import SQLiteProjection
filename,point=sys.argv[1:]
def crash(here):
    if here==point:
        if os.name=='posix': os.kill(os.getpid(),signal.SIGKILL)
        os._exit(79)
SQLiteProjection(Path(filename),fault_hook=crash)
raise AssertionError('initialization hook not reached')
''', target, point)
    assert result.returncode == (-9 if os.name == "posix" else 79), result.stderr
    if point in {"init_before_connect", "after_commit"}:
        projection = SQLiteProjection(target)
        assert projection.read()["cursor"] is None
        reconcile(source, projection)
        query(source, projection)
    else:
        with pytest.raises((ProjectionError, JournalIntegrityError)):
            SQLiteProjection(target)
        # Preserve the incomplete target; reconstruct into a new directory.
        replacement = SQLiteProjection(tmp_path / "replacement" / "index.sqlite3")
        reconcile(source, replacement)
        query(source, replacement)
    assert source.read_snapshot() == before
    assert write(source) == receipt


def test_schema_two_target_is_rejected_without_mutation(pair, tmp_path):
    source, target = pair
    envelope = target.read()
    wrong = JournalStateStore(tmp_path / "wrong" / "projection.sqlite3", receipt_mode=True)
    wrong.save(envelope, expected_revision=0)
    before = wrong.read_snapshot()
    with pytest.raises(ProjectionError):
        SQLiteProjection(wrong.path)
    assert wrong.read_snapshot() == before
    assert source.read_snapshot()[0] == 1


def test_verified_snapshot_rechecks_identity_inside_its_read_transaction(pair, monkeypatch):
    from contextlib import contextmanager
    source, _ = pair
    connect = source._connect
    @contextmanager
    def identity_changes_after_connect_verification():
        with connect() as connection:
            with closing(sqlite3.connect(source.path)) as peer:
                peer.execute("UPDATE identity SET store_id=?", ("f" * 32,))
                peer.commit()
            yield connection
    monkeypatch.setattr(source, "_connect", identity_changes_after_connect_verification)
    with pytest.raises(JournalIntegrityError):
        source.verified_snapshot()
