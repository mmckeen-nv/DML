"""Component durability behavior with public runtime capabilities only."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from daystrom_dml.atomic_io import atomic_write_text
from daystrom_dml.journal import JournalStateStore, RevisionConflict
from daystrom_dml.persistence import load_state, save_state
from daystrom_dml.services.persistence import (
    LatticePersistence,
    PersistenceCoordinator,
    PersistenceState,
)
from daystrom_dml.services.transactions import PersistenceCommitError


class LatticeRuntime:
    def __init__(self, events):
        self.events = events
        self.data = {"schema_version": 1, "items": [], "lineage": [], "next_id": 0,
                     "repair_queue": []}

    def snapshot_state(self):
        return copy.deepcopy(self.data)

    def import_state(self, payload):
        self.events.append("restore_lattice")
        self.data = copy.deepcopy(payload)

    def export_state(self):
        self.events.append("export_lattice")
        return copy.deepcopy(self.data)

    def items(self):
        self.events.append("lattice_items")
        return []


class RAGRuntime:
    def __init__(self, events):
        self.events = events
        self.data = {"documents": [{"text": "original", "meta": {"source": "test"}}],
                     "backends": [{"id": "test", "available": False, "error": "offline"}]}

    def snapshot_state(self):
        return copy.deepcopy(self.data)

    def restore_state(self, payload):
        self.events.append("restore_rag")
        self.data = copy.deepcopy(payload)

    def import_state(self, payload):
        self.events.append("import_rag")
        self.data = copy.deepcopy(payload)

    def export_state(self):
        return {"documents": copy.deepcopy(self.data["documents"])}


class PersistentRAG:
    def __init__(self, path, events):
        self.manifest_path = path
        self.events = events
        self.data = {"records": [{"text": "original"}]}
        self.load_result = True

    def snapshot_state(self):
        return copy.deepcopy(self.data)

    def restore_state(self, payload):
        self.events.append("restore_persistent")
        self.data = copy.deepcopy(payload)

    def persist(self):
        self.events.append("write_persistent")
        atomic_write_text(self.manifest_path, json.dumps(self.data))

    def load(self):
        self.events.append("load_persistent")
        if self.load_result:
            self.data = json.loads(self.manifest_path.read_text())
        return self.load_result


@pytest.fixture
def make_coordinator(tmp_path):
    def make(*, persistent=True, jsonl=False, journal=False):
        events, commits = [], set()
        runtime, rag = LatticeRuntime(events), RAGRuntime(events)
        persistent_store = PersistentRAG(tmp_path / "manifest.json", events) if persistent else None
        provider = [persistent_store]
        authority = JournalStateStore(tmp_path / "journal.sqlite3") if journal else None
        lattice = LatticePersistence(
            json_path=tmp_path / "lattice.json", jsonl_path=tmp_path / "lattice.jsonl",
            use_jsonl=jsonl, journal=authority, read_jsonl=load_state,
            write_jsonl=save_state, write_text=atomic_write_text,
        )
        state = PersistenceState()

        def write_text(path, text):
            events.append("write_legacy")
            atomic_write_text(path, text)

        coordinator = PersistenceCoordinator(
            state=state, lattice=lattice, runtime=runtime, rag=rag,
            rag_path=tmp_path / "rag.json", persistent_rag=lambda: provider[0],
            invalidate_cache=lambda: events.append("invalidate"), write_text=write_text,
            mark_committed=commits.add, require_legacy=lambda: None,
            operation=lambda: "component-test",
        )
        return SimpleNamespace(service=coordinator, state=state, lattice=lattice,
                               runtime=runtime, rag=rag, persistent=persistent_store,
                               provider=provider, events=events, commits=commits)
    return make


def test_restores_all_runtime_before_ordered_compensation(make_coordinator):
    rig = make_coordinator()
    snapshot = rig.service.capture_snapshot()
    rig.runtime.data["next_id"] = 10
    rig.rag.data["documents"][0]["text"] = "new"
    rig.persistent.data["records"][0]["text"] = "new"

    def persist_lattice():
        assert rig.runtime.data == snapshot["dml"]
        assert rig.rag.data == snapshot["rag"]
        assert rig.persistent.data == snapshot["persistent_rag"]
        rig.events.append("write_lattice")

    rig.service.rollback(snapshot, {"dml", "persistent_rag", "rag"},
                         persist_lattice=persist_lattice)

    assert rig.events == ["restore_lattice", "restore_rag", "restore_persistent", "invalidate",
                          "write_lattice", "write_persistent", "write_legacy"]
    assert json.loads(rig.service.rag_path.read_text()) == {"documents": snapshot["rag"]["documents"]}
    assert rig.state.observed_rag == rig.service.path_stamp(rig.service.rag_path)
    assert rig.state.observed_persistent_rag == rig.service.path_stamp(rig.persistent.manifest_path)


@pytest.mark.parametrize("committed", [set(), {"dml"}, {"rag"}, {"persistent_rag"}])
def test_compensates_only_recorded_components(make_coordinator, committed):
    rig = make_coordinator()
    rig.service.rollback(rig.service.capture_snapshot(), committed,
                         persist_lattice=lambda: rig.events.append("write_lattice"))
    assert ("write_lattice" in rig.events) == ("dml" in committed)
    assert ("write_legacy" in rig.events) == ("rag" in committed)
    assert ("write_persistent" in rig.events) == ("persistent_rag" in committed)


def test_compensation_failure_stops_later_writes_after_runtime_restore(make_coordinator):
    rig = make_coordinator()
    failure = OSError("lattice compensation unavailable")

    def fail():
        raise failure

    with pytest.raises(OSError) as caught:
        rig.service.rollback(rig.service.capture_snapshot(), {"dml", "persistent_rag", "rag"},
                             persist_lattice=fail)
    assert caught.value is failure
    assert rig.events == ["restore_lattice", "restore_rag", "restore_persistent", "invalidate"]


def test_rag_publish_before_legacy_failure_remains_tracked(make_coordinator, monkeypatch):
    rig = make_coordinator()

    def fail(*args):
        raise PermissionError("legacy write blocked")

    monkeypatch.setattr(rig.service, "_write_text", fail)
    with pytest.raises(PersistenceCommitError) as caught:
        rig.service.persist_rag()
    assert isinstance(caught.value.__cause__, PermissionError)
    assert rig.commits == {"persistent_rag"}
    assert rig.state.observed_persistent_rag is not None
    assert rig.state.observed_rag is None
    assert rig.service.durability_status()["failures"] == {"rag": "PermissionError: legacy write blocked"}


def test_health_is_detached_and_success_clears_only_its_component(make_coordinator, monkeypatch):
    rig = make_coordinator(persistent=False)
    real_commit = rig.lattice.commit

    def fail(**kwargs):
        raise OSError("full")

    rig.service.record_failure("rollback", RuntimeError("prior failure"))
    monkeypatch.setattr(rig.lattice, "commit", fail)
    with pytest.raises(PersistenceCommitError):
        rig.service.persist_lattice()
    assert not rig.commits
    detached = rig.service.durability_status()
    detached["failures"].clear()
    assert "dml" in rig.state.failures

    monkeypatch.setattr(rig.lattice, "commit", real_commit)
    rig.service.persist_lattice()
    assert rig.service.durability_status() == {
        "status": "degraded", "failures": {"rollback": "RuntimeError: prior failure"},
    }


def test_jsonl_path_uses_items_without_lattice_export(make_coordinator):
    rig = make_coordinator(persistent=False, jsonl=True)
    rig.service.persist_all()
    assert rig.events == ["lattice_items", "write_legacy"]
    assert rig.commits == {"dml", "rag"}
    assert load_state(rig.lattice.path) == []


def test_legacy_guard_failure_is_not_reported_as_a_storage_failure(make_coordinator, monkeypatch):
    rig = make_coordinator()

    def forbid():
        raise ValueError("receipt-only authority")

    monkeypatch.setattr(rig.service, "_require_legacy", forbid)
    with pytest.raises(ValueError, match="receipt-only"):
        rig.service.persist_lattice()
    assert not rig.events and not rig.commits
    assert rig.service.durability_status() == {"status": "ok", "failures": {}}


def test_refresh_imports_components_and_preserves_unrelated_health(make_coordinator):
    rig = make_coordinator()
    payload = rig.runtime.snapshot_state()
    payload["next_id"] = 7
    rig.lattice.commit(payload=payload, items=None, expected_revision=0, operation="external")
    atomic_write_text(rig.service.rag_path, json.dumps({"documents": []}))
    rig.persistent.persist()
    rig.events.clear()
    rig.service.record_failure("receipt_runtime", ValueError("stale"))
    rig.service.record_failure("rag", OSError("keep"))

    assert rig.service.refresh()
    assert rig.runtime.data == payload
    assert rig.rag.data == {"documents": []}
    assert rig.events == ["restore_lattice", "invalidate", "import_rag", "load_persistent"]
    assert rig.state.failures == {"rag": "OSError: keep"}
    rig.events.clear()
    assert not rig.service.refresh()
    assert not rig.events


def test_refresh_does_not_acknowledge_unsuccessful_persistent_load(make_coordinator):
    rig = make_coordinator()
    rig.persistent.persist()
    rig.persistent.load_result = False
    assert not rig.service.refresh()
    assert rig.state.observed_persistent_rag is None
    rig.persistent.load_result = True
    assert rig.service.refresh()
    assert rig.state.observed_persistent_rag is not None


def test_refresh_fails_closed_when_initialized_lattice_disappears(make_coordinator):
    rig = make_coordinator(persistent=False)
    rig.service.persist_lattice()
    rig.lattice.path.unlink()
    with pytest.raises(FileNotFoundError, match="Previously initialized"):
        rig.service.refresh()


def test_refresh_uses_revision_of_loaded_snapshot(make_coordinator, monkeypatch):
    rig = make_coordinator(persistent=False, journal=True)
    rig.service.persist_lattice()
    first = copy.deepcopy(rig.runtime.data)
    rig.state.observed_lattice = None
    original_load = rig.lattice.load_with_revision

    def load_then_peer_write(**kwargs):
        loaded, loaded_revision = original_load(**kwargs)
        peer = JournalStateStore(rig.lattice.path)
        newer = copy.deepcopy(loaded)
        newer["next_id"] = 1
        peer.save(newer, expected_revision=1, operation="peer")
        return loaded, loaded_revision

    monkeypatch.setattr(rig.lattice, "load_with_revision", load_then_peer_write)
    assert rig.service.refresh()
    assert rig.runtime.data == first
    assert rig.state.observed_lattice[0] == 1
    assert rig.lattice.journal.stamp()[0] == 2


def test_current_optional_persistent_store_is_resolved_per_operation(make_coordinator):
    rig = make_coordinator(persistent=False)
    assert rig.service.capture_snapshot()["persistent_rag"] is None
    replacement = PersistentRAG(rig.service.rag_path.with_name("replacement.json"), rig.events)
    rig.provider[0] = replacement
    assert rig.service.capture_snapshot()["persistent_rag"] == replacement.data
    rig.service.persist_rag()
    assert replacement.manifest_path.exists()


def test_journal_late_acknowledgement_failure_uses_verified_revision_for_compensation(
    make_coordinator, monkeypatch,
):
    rig = make_coordinator(persistent=False, journal=True)
    rig.service.persist_lattice()
    snapshot = rig.service.capture_snapshot()
    rig.commits.clear()
    rig.runtime.data["next_id"] = 3
    commit = rig.lattice.commit

    def publish_then_raise(**kwargs):
        commit(**kwargs)
        raise OSError("lost acknowledgement")

    monkeypatch.setattr(rig.lattice, "commit", publish_then_raise)
    with pytest.raises(PersistenceCommitError):
        rig.service.persist_lattice()
    assert rig.commits == {"dml"}
    assert rig.state.observed_lattice[0] == 2
    monkeypatch.setattr(rig.lattice, "commit", commit)
    rig.service.rollback(snapshot, rig.commits)
    assert rig.lattice.journal.read_snapshot() == (3, snapshot["dml"])


def test_explicit_cas_conflict_never_claims_matching_peer_state(make_coordinator):
    rig = make_coordinator(persistent=False, journal=True)
    rig.service.persist_lattice()
    rig.commits.clear()
    rig.runtime.data["next_id"] = 4
    peer = JournalStateStore(rig.lattice.path)
    peer.save(copy.deepcopy(rig.runtime.data), expected_revision=1, operation="peer")

    with pytest.raises(PersistenceCommitError) as caught:
        rig.service.persist_lattice()
    assert isinstance(caught.value.__cause__, RevisionConflict)
    assert not rig.commits
    assert rig.state.observed_lattice[0] == 1
    assert peer.read_snapshot()[1] == rig.runtime.data


def test_unknown_journal_outcome_fails_compensation_without_replacing_authority(
    make_coordinator, monkeypatch,
):
    rig = make_coordinator(persistent=False, journal=True)
    rig.service.persist_lattice()
    snapshot = rig.service.capture_snapshot()
    rig.commits.clear()
    rig.runtime.data["next_id"] = 9
    commit = rig.lattice.commit

    def publish_then_raise(**kwargs):
        commit(**kwargs)
        raise OSError("acknowledgement failed")

    def unavailable():
        raise OSError("authority unreadable")

    monkeypatch.setattr(rig.lattice, "commit", publish_then_raise)
    monkeypatch.setattr(rig.lattice.journal, "read_snapshot", unavailable)
    with pytest.raises(PersistenceCommitError):
        rig.service.persist_lattice()
    assert rig.commits == {"dml"}
    with pytest.raises(OSError, match="authority unreadable"):
        rig.service.rollback(snapshot, rig.commits)
    assert rig.runtime.data == snapshot["dml"]
    assert JournalStateStore(rig.lattice.path).read_snapshot()[1]["next_id"] == 9
