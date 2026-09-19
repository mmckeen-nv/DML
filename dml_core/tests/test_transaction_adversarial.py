"""Independent oracles for legacy compensation and transaction extraction.

These tests cover caught exceptions, not atomic commits across multiple files
or recovery after process death. Receipt transactions retain separate semantics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import daystrom_dml.dml_adapter as adapter_module
from daystrom_dml.dml_adapter import (
    DMLAdapter, PersistenceCommitError, PersistenceRollbackError,
)
from daystrom_dml.journal import JournalStateStore


class FixedEmbedder:
    receipt_embedding_identity = "transaction-adversarial-v1"

    def embed(self, _text):
        return np.ones(4, dtype=np.float32)


@pytest.fixture
def factory(tmp_path):
    instances = []

    def build(mode="jsonl", **overrides):
        instance = DMLAdapter(
            config_overrides={
                "storage_dir": str(tmp_path / mode),
                "model_name": "dummy", "embedding_model": None,
                "checkpoint_interval_seconds": 0,
                "persistence": {
                    "enable": mode == "jsonl", "path": "state.jsonl", "interval_sec": 0,
                    "journal": mode in {"journal", "receipt"}, "receipts": mode == "receipt",
                },
                "rag_store": {"enable": False},
                "dpm": {"enable": False, "include_in_context": False},
                **overrides,
            },
            embedder=FixedEmbedder(), start_aging_loop=False,
        )
        instances.append(instance)
        return instance

    yield build
    for instance in reversed(instances):
        instance.close(persist=False)


@pytest.mark.parametrize("mode", ["json", "jsonl", "journal"])
def test_published_lattice_then_error_is_compensated(factory, monkeypatch, mode):
    adapter = factory(mode)
    original = adapter.lattice_persistence.commit
    attempted = 0

    def publish_then_error(**kwargs):
        nonlocal attempted
        result = original(**kwargs)
        attempted += 1
        if attempted == 1:
            raise OSError("acknowledgement failed after lattice publication")
        return result

    monkeypatch.setattr(adapter.lattice_persistence, "commit", publish_then_error)
    with pytest.raises(PersistenceCommitError):
        adapter.ingest("failed mutation must not remain authoritative")
    assert adapter.memory_count() == 0
    reloaded = factory(mode)
    assert reloaded.memory_count() == 0
    assert reloaded.rag_store.catalog_summary()["count"] == 0


def test_published_legacy_rag_then_error_is_compensated(factory, monkeypatch):
    adapter = factory()
    original = adapter_module.atomic_write_text
    attempted = 0

    def publish_then_error(path, content, *args, **kwargs):
        nonlocal attempted
        result = original(path, content, *args, **kwargs)
        if Path(path) == adapter.rag_state_path:
            attempted += 1
            if attempted == 1:
                raise OSError("acknowledgement failed after RAG publication")
        return result

    monkeypatch.setattr(adapter_module, "atomic_write_text", publish_then_error)
    with pytest.raises(PersistenceCommitError):
        adapter.ingest("RAG must not survive failed mutation")
    assert adapter.memory_count() == 0
    assert adapter.rag_store.catalog_summary()["count"] == 0
    reloaded = factory()
    assert reloaded.memory_count() == 0
    assert reloaded.rag_store.catalog_summary()["count"] == 0


def test_nested_batch_success_is_compensated_when_outer_body_fails(factory):
    adapter = factory()
    with pytest.raises(RuntimeError, match="outer failure"):
        with adapter.atomic_batch("outer"):
            with adapter.atomic_batch("inner"):
                adapter.ingest("inner durable success", persist=False)
            raise RuntimeError("outer failure")
    assert adapter.memory_count() == 0
    assert factory().memory_count() == 0
    assert factory().rag_store.catalog_summary()["count"] == 0


def test_refresh_failure_unwinds_ownership_before_next_transaction(factory, monkeypatch):
    adapter = factory()
    original = adapter.refresh_if_changed

    def fail():
        raise RuntimeError("refresh unavailable")

    monkeypatch.setattr(adapter, "refresh_if_changed", fail)
    with pytest.raises(RuntimeError, match="refresh unavailable"):
        with adapter.mutation_transaction("failed-refresh"):
            pytest.fail("body must not run after failed refresh")
    assert getattr(adapter._mutation_local, "depth", 0) == 0
    monkeypatch.setattr(adapter, "refresh_if_changed", original)
    with adapter.mutation_transaction("next"):
        assert getattr(adapter._mutation_local, "depth", 0) == 1


def test_nested_ownership_refreshes_once_and_preserves_outer_operation(factory, monkeypatch):
    adapter = factory()
    refreshes = []
    original = adapter.refresh_if_changed

    def refresh():
        refreshes.append(True)
        return original()

    monkeypatch.setattr(adapter, "refresh_if_changed", refresh)
    with adapter.mutation_transaction("outer"):
        with adapter.mutation_transaction("inner"):
            assert adapter._mutation_local.operation == "outer"
            assert adapter._mutation_local.depth == 2
        assert adapter._mutation_local.depth == 1
    assert adapter._mutation_local.depth == 0
    assert refreshes == [True]


def test_receipt_and_lattice_only_guards_precede_any_writer(factory, monkeypatch):
    adapter = factory("receipt")
    writes = []
    monkeypatch.setattr(adapter.lattice_persistence, "commit", lambda **kwargs: writes.append(kwargs))
    for operation in (adapter._persist_dml_state, adapter._persist_all,
                      adapter._gather_checkpoint_state):
        with pytest.raises(ValueError):
            operation()
    with pytest.raises(ValueError):
        with adapter.atomic_batch("forbidden"):
            pytest.fail("receipt batch body must not run")
    assert writes == []
    assert adapter._journal.revision == 0


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_baseexception_rolls_back_and_releases_ownership(factory, interrupt):
    adapter = factory()
    error = interrupt("cancel mutation")
    with pytest.raises(interrupt) as caught:
        with adapter.atomic_batch("interrupted"):
            adapter.ingest("uncommitted cancelled body", persist=False)
            raise error
    assert caught.value is error
    assert adapter.memory_count() == 0
    assert adapter.rag_store.catalog_summary()["count"] == 0
    assert getattr(adapter._mutation_local, "depth", 0) == 0
    assert getattr(adapter._mutation_local, "committed_components", None) is None
    adapter.ingest("next mutation succeeds")
    assert [item.text for item in adapter.store.items()] == ["next mutation succeeds"]


def test_outer_rollback_retries_components_after_inner_compensation_fails(factory, monkeypatch):
    adapter = factory()
    original_commit = adapter.lattice_persistence.commit
    original_write = adapter_module.atomic_write_text
    commits = 0
    rag_writes = 0

    def fail_first_compensation(**kwargs):
        nonlocal commits
        commits += 1
        if commits == 2:
            raise OSError("inner rollback temporarily unavailable")
        return original_commit(**kwargs)

    def fail_first_rag(path, content, *args, **kwargs):
        nonlocal rag_writes
        if Path(path) == adapter.rag_state_path:
            rag_writes += 1
            if rag_writes == 1:
                raise OSError("RAG failed before publication")
        return original_write(path, content, *args, **kwargs)

    monkeypatch.setattr(adapter.lattice_persistence, "commit", fail_first_compensation)
    monkeypatch.setattr(adapter_module, "atomic_write_text", fail_first_rag)
    with pytest.raises(PersistenceRollbackError):
        with adapter.atomic_batch("outer"):
            with adapter.atomic_batch("inner"):
                adapter.ingest("failed inner mutation", persist=False)
    assert commits >= 3, "outer rollback must retry the inner published component"
    assert adapter.memory_count() == 0
    assert factory().memory_count() == 0


def test_inner_successful_compensation_is_still_tracked_by_outer_frame(factory):
    adapter = factory()
    with pytest.raises(RuntimeError, match="inner body failed"):
        with adapter.atomic_batch("outer"):
            adapter.ingest("outer runtime changes", persist=False)
            with adapter.atomic_batch("inner"):
                adapter.ingest("inner durably published changes")
                raise RuntimeError("inner body failed")
    assert adapter.memory_count() == 0
    reloaded = factory()
    assert reloaded.memory_count() == 0
    assert reloaded.rag_store.catalog_summary()["count"] == 0


def test_journal_compensation_does_not_erase_intervening_authority(factory, monkeypatch):
    adapter = factory("journal")
    original = adapter.lattice_persistence.commit
    attempts = 0

    def publish_then_peer_then_error(**kwargs):
        nonlocal attempts
        result = original(**kwargs)
        attempts += 1
        if attempts == 1:
            revision, payload = adapter._journal.read_snapshot()
            payload["items"][0]["text"] = "independently committed authority"
            adapter._journal.save(payload, expected_revision=revision, operation="external-direct-writer")
            raise OSError("acknowledgement lost before observing peer commit")
        return result

    monkeypatch.setattr(adapter.lattice_persistence, "commit", publish_then_peer_then_error)
    with pytest.raises(PersistenceRollbackError):
        adapter.ingest("original failed mutation")
    assert adapter._journal.load()["items"][0]["text"] == "independently committed authority"
    assert adapter.durability_status()["status"] == "degraded"


def test_journal_cas_rejection_does_not_erase_equal_peer_payload(factory, monkeypatch):
    adapter = factory("journal")
    peer = JournalStateStore(adapter._journal.path)
    original = adapter.lattice_persistence.commit

    def peer_commits_identical_payload_first(**kwargs):
        peer.save(kwargs["payload"], expected_revision=kwargs["expected_revision"],
                  operation="independent-identical-peer")
        return original(**kwargs)

    monkeypatch.setattr(adapter.lattice_persistence, "commit", peer_commits_identical_payload_first)
    with pytest.raises(PersistenceCommitError):
        adapter.ingest("identical independent memory")
    revision, payload = peer.read_snapshot()
    assert revision == 1
    assert payload["items"][0]["text"] == "identical independent memory"
    assert adapter.memory_count() == 0
    assert "rollback" not in adapter.durability_status()["failures"]


@pytest.mark.parametrize("mode", ["json", "jsonl"])
def test_equal_stat_stamp_cannot_prove_failed_writer_did_not_publish(factory, monkeypatch, mode):
    adapter = factory(mode)
    adapter._persist_all()
    unchanged_stamp = adapter._state_stamp()
    original = adapter.lattice_persistence.commit
    attempts = 0

    def publish_then_error(**kwargs):
        nonlocal attempts
        result = original(**kwargs)
        attempts += 1
        if attempts == 1:
            raise OSError("published despite indistinguishable stat stamp")
        return result

    monkeypatch.setattr(adapter.lattice_persistence, "stamp", lambda: unchanged_stamp)
    monkeypatch.setattr(adapter.lattice_persistence, "commit", publish_then_error)
    with pytest.raises(PersistenceCommitError):
        adapter.ingest("publication identity must be stronger than stat")
    assert adapter.memory_count() == 0
    assert factory(mode).memory_count() == 0


def test_postpublication_stamp_error_still_compensates(factory, monkeypatch):
    adapter = factory()
    original_write = adapter.lattice_persistence._write_jsonl
    original_stamp = adapter.lattice_persistence.stamp
    written = 0
    fail_stamp = False

    def write(*args, **kwargs):
        nonlocal written, fail_stamp
        result = original_write(*args, **kwargs)
        written += 1
        fail_stamp = written == 1
        return result

    def stamp():
        nonlocal fail_stamp
        if fail_stamp:
            fail_stamp = False
            raise OSError("stat unavailable immediately after publication")
        return original_stamp()

    monkeypatch.setattr(adapter.lattice_persistence, "_write_jsonl", write)
    monkeypatch.setattr(adapter.lattice_persistence, "stamp", stamp)
    with pytest.raises(PersistenceCommitError):
        adapter.ingest("published but stat failed")
    assert factory().memory_count() == 0


@pytest.mark.parametrize("mode", ["jsonl", "journal"])
def test_interrupt_after_lattice_publication_is_compensated_without_rewrapping(factory, monkeypatch, mode):
    adapter = factory(mode)
    original = adapter.lattice_persistence.commit
    interruption = KeyboardInterrupt("interrupted publication acknowledgement")
    attempts = 0

    def publish_then_interrupt(**kwargs):
        nonlocal attempts
        result = original(**kwargs)
        attempts += 1
        if attempts == 1:
            raise interruption
        return result

    monkeypatch.setattr(adapter.lattice_persistence, "commit", publish_then_interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        adapter.ingest("interrupted after publication")
    assert caught.value is interruption
    assert adapter.memory_count() == 0
    assert factory(mode).memory_count() == 0


def test_persistent_rag_publication_then_error_is_compensated(factory, monkeypatch):
    pytest.importorskip("faiss")
    config = {"enable": True, "path": "persistent.faiss", "meta_path": "persistent.json",
              "backend": "faiss", "dim": 4}
    adapter = factory(rag_store=config)
    original = adapter.persistent_rag_store.persist
    attempts = 0

    def persist():
        nonlocal attempts
        result = original()
        attempts += 1
        if attempts == 1:
            raise OSError("manifest publication acknowledgement failed")
        return result

    monkeypatch.setattr(adapter.persistent_rag_store, "persist", persist)
    with pytest.raises(PersistenceCommitError):
        adapter.ingest("persistent RAG failed publication")
    reloaded = factory(rag_store=config)
    assert reloaded.memory_count() == 0
    assert reloaded.persistent_rag_store.search(np.ones(4), top_k=2) == []


def test_refresh_pins_the_journal_revision_actually_imported(factory, monkeypatch):
    adapter = factory("journal")
    adapter.ingest("initial authority")
    peer = JournalStateStore(adapter._journal.path)
    revision, payload = peer.read_snapshot()
    payload["items"][0]["text"] = "first peer generation"
    peer.save(payload, expected_revision=revision, operation="peer-one")
    expected_imported_revision = peer.revision
    original_load = adapter.lattice_persistence.load
    interleaved = False

    def load(**kwargs):
        nonlocal interleaved
        imported = original_load(**kwargs)
        if not interleaved:
            interleaved = True
            revision, newer = peer.read_snapshot()
            newer["items"][0]["text"] = "second peer generation"
            peer.save(newer, expected_revision=revision, operation="peer-two")
        return imported

    monkeypatch.setattr(adapter.lattice_persistence, "load", load)
    adapter.query_cache.get("must-invalidate", lambda _text: np.ones(4))
    assert adapter.refresh_if_changed()
    assert adapter._last_observed_state[0] == expected_imported_revision
    assert adapter.store.items()[0].text == "first peer generation"
    assert not adapter.query_cache.values
    assert adapter.refresh_if_changed()
    assert adapter.store.items()[0].text == "second peer generation"


def test_failed_import_does_not_advance_stamp_or_clear_runtime_degradation(factory, monkeypatch):
    adapter = factory()
    peer = factory()
    peer.ingest("externally persisted authority")
    before = adapter._last_observed_state
    adapter._durability_failures["receipt_runtime"] = "RuntimeError"
    original = adapter.store.import_state

    def fail(_payload):
        raise RuntimeError("hydration unavailable")

    monkeypatch.setattr(adapter.store, "import_state", fail)
    with pytest.raises(RuntimeError, match="hydration unavailable"):
        adapter.refresh_if_changed()
    assert adapter._last_observed_state == before
    assert adapter.durability_status()["failures"]["receipt_runtime"] == "RuntimeError"
    monkeypatch.setattr(adapter.store, "import_state", original)
    assert adapter.refresh_if_changed()
    assert "receipt_runtime" not in adapter.durability_status()["failures"]
    assert adapter.store.items()[0].text == "externally persisted authority"
