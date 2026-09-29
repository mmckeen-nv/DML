"""Independent packet, replay, crash and concurrency oracles for delta delivery."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from copy import deepcopy
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time

import pytest

from daystrom_dml.journal import FAULT_POINTS, JournalStateStore
from daystrom_dml.services.projection import ProjectionError, ProjectionStale, SQLiteProjection, reconcile
from daystrom_dml.services.projection_delta import prepare_delta, reconcile_incremental
from test_projection import query, run_child, write


@pytest.fixture
def pair(tmp_path):
    source = JournalStateStore(tmp_path / "authority" / "source.sqlite3", receipt_mode=True)
    write(source)
    return source, SQLiteProjection(tmp_path / "projection" / "index.sqlite3")


def mutate(source, fn):
    revision, state = source.read_snapshot()
    fn(state)
    source.save(state, expected_revision=revision, operation="trusted-test-lifecycle")


def resign(packet):
    body = {key: value for key, value in packet.items() if key != "checksum"}
    packet["checksum"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return packet


def test_bootstrap_then_noop_has_no_additional_target_commit(pair):
    source, target = pair
    packet = prepare_delta(source, target)
    assert len(packet["upserts"]) == 1 and packet["deletes"] == []
    result = reconcile_incremental(source, target)
    assert result["matches_pinned_source"] is True
    before = target.journal.read_snapshot()
    packet = prepare_delta(source, target)
    assert packet["upserts"] == [] and packet["deletes"] == []
    assert target.apply_delta(packet) == packet["next_cursor"]
    assert target.journal.read_snapshot() == before
    query(source, target)


def test_delta_carries_changed_payloads_and_preserves_full_snapshot_equivalence(pair, tmp_path):
    source, target = pair
    for ident in range(1, 5):
        write(source, str(ident), text=f"large unchanged memory {ident} " + "x" * 1000)
    reconcile_incremental(source, target)
    def change(state):
        state["items"] = [state["items"][4], state["items"][0], state["items"][2], state["items"][3]]
        state["items"][1]["text"] = "updated memory"
    mutate(source, change)
    packet = prepare_delta(source, target)
    assert [record["id"] for record in packet["upserts"]] == [0]
    assert packet["deletes"] == [1]
    assert packet["ordered_ids"] == [4, 0, 2, 3]
    assert "large unchanged memory" not in json.dumps(packet)
    target.apply_delta(packet)
    reference = SQLiteProjection(tmp_path / "reference" / "index.sqlite3")
    reconcile(source, reference)
    assert target.read() == reference.read()
    assert query(source, target) == query(source, reference)


def test_lineage_only_revision_advances_cursor_without_resending_live_records(pair):
    source, target = pair
    reconcile_incremental(source, target)
    mutate(source, lambda state: state.update(lineage=deepcopy(state["items"])))
    packet = prepare_delta(source, target)
    assert packet["upserts"] == [] and packet["deletes"] == []
    assert packet["next_cursor"]["source_revision"] > packet["base_cursor"]["source_revision"]
    target.apply_delta(packet)
    assert query(source, target)["source"] == packet["next_cursor"]


def test_delayed_packet_cannot_restore_deleted_records(pair):
    source, target = pair
    reconcile_incremental(source, target)
    write(source, "two")
    delayed = prepare_delta(source, target)
    mutate(source, lambda state: state.update(items=[]))
    reconcile_incremental(source, target)
    before = target.read()
    with pytest.raises(ProjectionError):
        target.apply_delta(delayed)
    assert target.read() == before
    assert query(source, target)["results"] == []


@pytest.mark.parametrize("damage", ["checksum", "base", "target", "duplicate", "unknown-delete", "overlap", "missing-id", "unknown-id", "identity", "rollback", "unknown-format"])
def test_malformed_or_wrong_base_packet_never_mutates_target(pair, damage):
    source, target = pair
    reconcile_incremental(source, target)
    write(source, "two")
    packet = prepare_delta(source, target)
    if damage == "checksum":
        packet["upserts"][0]["text"] = "wire corruption"
    elif damage == "base":
        packet["base_envelope_digest"] = "0" * 64
    elif damage == "target":
        packet["target_envelope_digest"] = "0" * 64
    elif damage == "duplicate":
        packet["upserts"].append(deepcopy(packet["upserts"][0]))
    elif damage == "unknown-delete":
        packet["deletes"] = [900]
    elif damage == "overlap":
        packet["deletes"] = [packet["upserts"][0]["id"]]
    elif damage == "missing-id":
        packet["ordered_ids"] = packet["ordered_ids"][:-1]
    elif damage == "unknown-id":
        packet["ordered_ids"].append(900)
    elif damage == "identity":
        packet["next_cursor"]["source_store_id"] = "f" * 32
    elif damage == "rollback":
        packet["next_cursor"]["source_revision"] = 0
    elif damage == "unknown-format":
        packet["delta_format"] = "dml-projection-delta-v2"
    if damage != "checksum":
        resign(packet)
    before = target.journal.read_snapshot()
    with pytest.raises(ProjectionError):
        target.apply_delta(packet)
    assert target.journal.read_snapshot() == before


def test_packet_is_frozen_before_backend_hook(pair):
    source, target = pair
    packet = prepare_delta(source, target)
    desired = deepcopy(packet["next_cursor"])
    def mutate_packet(point):
        if point == "delta_before_apply":
            packet["upserts"][0]["text"] = "caller mutation"
            packet["ordered_ids"].clear()
    target.fault_hook = mutate_packet
    target.journal._fault_hook = mutate_packet
    assert target.apply_delta(packet) == desired
    assert query(source, target)["results"][0]["memory"]["text"] == "acknowledged memory"


@pytest.mark.parametrize("point", FAULT_POINTS)
def test_process_death_during_delta_keeps_cursor_and_records_atomic(pair, point):
    source, target = pair
    reconcile_incremental(source, target)
    prior = target.read()
    write(source, "two")
    before_source = source.read_snapshot()
    result = run_child('''
import os,signal,sys
from pathlib import Path
from daystrom_dml.journal import JournalStateStore
from daystrom_dml.services.projection import SQLiteProjection
from daystrom_dml.services.projection_delta import reconcile_incremental
source_path,target_path,point=sys.argv[1:]
def crash(here):
    if here==point:
        if os.name=='posix': os.kill(os.getpid(),signal.SIGKILL)
        os._exit(79)
source=JournalStateStore(Path(source_path))
reconcile_incremental(source,SQLiteProjection(Path(target_path),fault_hook=crash))
raise AssertionError('delta hook not reached')
''', source.path, target.path, point)
    assert result.returncode == (-9 if os.name == "posix" else 79), result.stderr
    reopened = SQLiteProjection(target.path)
    if point != "after_commit":
        assert reopened.read() == prior
        with pytest.raises(ProjectionStale):
            query(source, reopened)
    else:
        query(source, reopened)
    reconcile_incremental(source, reopened)
    assert len(query(source, reopened)["results"]) == 2
    assert source.read_snapshot() == before_source


def test_after_commit_exception_reconciles_and_replay_does_not_duplicate(pair):
    source, target = pair
    packet = prepare_delta(source, target)
    def fail(point):
        if point == "after_commit":
            raise OSError("private backend detail")
    target.journal._fault_hook = fail
    assert target.apply_delta(packet) == packet["next_cursor"]
    before = target.journal.read_snapshot()
    assert target.apply_delta(packet) == packet["next_cursor"]
    assert target.journal.read_snapshot() == before


def test_delta_disk_full_preserves_prior_projection_and_authority_receipts(pair, monkeypatch):
    source, target = pair
    reconcile_incremental(source, target)
    before = target.read()
    receipt = write(source, "large", text="x" * 500_000)
    packet = prepare_delta(source, target)
    connect = target.journal._connect
    @contextmanager
    def limited():
        with connect() as connection:
            pages = connection.execute("PRAGMA page_count").fetchone()[0]
            connection.execute(f"PRAGMA max_page_count={pages}")
            yield connection
    monkeypatch.setattr(target.journal, "_connect", limited)
    with pytest.raises(sqlite3.OperationalError, match="full"):
        target.apply_delta(packet)
    assert target.read() == before
    assert write(source, "large", text="x" * 500_000) == receipt


def test_duplicate_concurrent_deliveries_commit_once(pair):
    source, target = pair
    packet = prepare_delta(source, target)
    clients = int(os.environ.get("DML_STRESS_CLIENTS", "32"))
    barrier = threading.Barrier(clients)
    before = target.journal.read_snapshot()[0]
    def apply(_):
        barrier.wait(timeout=30)
        return target.apply_delta(packet)
    with ThreadPoolExecutor(max_workers=clients) as pool:
        cursors = list(pool.map(apply, range(clients)))
    assert cursors == [packet["next_cursor"]] * clients
    assert target.journal.read_snapshot()[0] == before + 1


def test_source_commit_proceeds_while_delta_backend_stalls(pair):
    source, target = pair
    entered, release = threading.Event(), threading.Event()
    def delay(point):
        if point == "delta_before_apply":
            entered.set()
            assert release.wait(10)
    target.fault_hook = delay
    with ThreadPoolExecutor(max_workers=2) as pool:
        sync = pool.submit(reconcile_incremental, source, target)
        assert entered.wait(5)
        try:
            receipt = pool.submit(write, source, "two").result(timeout=5)
            assert receipt["revision"] == 2
        finally:
            release.set()
        result = sync.result(timeout=10)
    assert result["published"]["source_revision"] == 1
    assert result["source"]["source_revision"] == 2
    assert result["matches_pinned_source"] is False


@pytest.mark.parametrize("before,after", [(True, 1), (1, 1.0)])
def test_json_metadata_type_changes_are_transported(pair, before, after):
    source, target = pair
    mutate(source, lambda state: state["items"][0]["meta"].update(value=before))
    reconcile_incremental(source, target)
    mutate(source, lambda state: state["items"][0]["meta"].update(value=after))
    packet = prepare_delta(source, target)
    assert len(packet["upserts"]) == 1
    assert type(packet["upserts"][0]["meta"]["value"]) is type(after)
    target.apply_delta(packet)
    assert type(target.read()["items"][0]["meta"]["value"]) is type(after)
    query(source, target)


@pytest.mark.parametrize("broken", ["no-publication", "wrong-content"])
def test_correct_looking_acknowledgement_requires_verified_publication(pair, broken):
    source, target = pair
    before = source.read_snapshot()
    class BrokenBackend:
        path = target.path
        def read(self):
            return target.read()
        def apply_delta(self, packet):
            if broken == "wrong-content":
                target.apply_delta(packet)
                revision, payload = target.journal.read_snapshot()
                payload["items"][0]["text"] = "backend stored incorrect contents"
                target.journal.save(payload, expected_revision=revision)
            return deepcopy(packet["next_cursor"])
    with pytest.raises(ProjectionError):
        reconcile_incremental(source, BrokenBackend())
    assert source.read_snapshot() == before
    assert write(source)["revision"] == 1


def test_later_verified_publication_can_supersede_the_acknowledged_revision(pair):
    source, target = pair
    class AdvancingBackend:
        path = target.path
        def read(self):
            return target.read()
        def apply_delta(self, packet):
            acknowledged = target.apply_delta(packet)
            write(source, "two")
            target.apply_delta(prepare_delta(source, target))
            return acknowledged
    result = reconcile_incremental(source, AdvancingBackend())
    assert result["published"]["source_revision"] == 1
    assert result["projection"]["source_revision"] == 2
    assert result["matches_pinned_source"] is True


def test_process_duplicate_deliveries_commit_one_delta(pair, tmp_path):
    source, target = pair
    gate = tmp_path / "start"
    code = '''
import json,sys,time
from pathlib import Path
from daystrom_dml.journal import JournalStateStore
from daystrom_dml.services.projection import SQLiteProjection
from daystrom_dml.services.projection_delta import prepare_delta
source_path,target_path,gate,ident=sys.argv[1:]
backend=SQLiteProjection(Path(target_path))
packet=prepare_delta(JournalStateStore(Path(source_path)),backend)
Path(gate+'.ready-'+ident).touch()
deadline=time.monotonic()+30
while not Path(gate).exists():
    if time.monotonic()>deadline: raise TimeoutError('parent barrier')
    time.sleep(.005)
print(json.dumps(backend.apply_delta(packet),sort_keys=True))
'''
    children = [subprocess.Popen([sys.executable, "-c", code, str(source.path), str(target.path), str(gate), str(i)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for i in range(8)]
    cursors = []
    try:
        # Ensure all packets share the same unbound base, before any delivery.
        deadline = time.monotonic() + 30
        while not all((tmp_path / f"start.ready-{i}").exists() for i in range(8)):
            if time.monotonic() > deadline:
                raise TimeoutError("publishers did not prepare packets")
            time.sleep(.005)
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
    assert target.journal.read_snapshot()[0] == 2
    query(source, target)


def test_competing_packets_from_same_base_reject_incomplete_update(pair):
    source, target = pair
    reconcile_incremental(source, target)
    write(source, "two")
    earlier = prepare_delta(source, target)
    write(source, "three")
    later = prepare_delta(source, target)
    assert earlier["base_cursor"] == later["base_cursor"]
    target.apply_delta(earlier)
    before = target.read()
    with pytest.raises(ProjectionStale):
        target.apply_delta(later)
    assert target.read() == before
    # Regenerate against the new base rather than applying a partial old packet.
    fresh = prepare_delta(source, target)
    assert len(fresh["upserts"]) == 1
    target.apply_delta(fresh)
    assert len(query(source, target)["results"]) == 3


def test_empty_authority_bootstrap_is_idempotent(tmp_path):
    source = JournalStateStore(tmp_path / "authority" / "source.sqlite3", receipt_mode=True)
    target = SQLiteProjection(tmp_path / "target" / "index.sqlite3")
    first = reconcile_incremental(source, target)
    before = target.journal.read_snapshot()
    again = reconcile_incremental(source, target)
    assert first["matches_pinned_source"] is again["matches_pinned_source"] is True
    assert first["published"]["source_revision"] == 0
    assert target.journal.read_snapshot() == before


def test_corrupt_authority_prevents_backend_delivery(pair):
    from daystrom_dml.journal import JournalIntegrityError
    source, target = pair
    before = target.read()
    with closing(sqlite3.connect(source.path)) as connection:
        connection.execute("UPDATE state SET checksum=?", ("0" * 64,))
        connection.commit()
    with pytest.raises(JournalIntegrityError):
        reconcile_incremental(source, target)
    assert target.read() == before
