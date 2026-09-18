"""Public transaction histories captured before coordinator extraction.

These oracles use real stores and acknowledgements, and check both runtime and
reopened state. Faults are injected at provider or storage boundaries rather
than replacing the transaction implementation under test.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import daystrom_dml.dml_adapter as adapter_module
from daystrom_dml.dml_adapter import (
    DMLAdapter,
    PersistenceCommitError,
    PersistenceRollbackError,
)
from daystrom_dml.services.outbox_migration import upgrade_outbox_journal
from daystrom_dml.store_lock import store_write_lock


LEGACY = ["json", "jsonl", "j1"]
RECEIPTS = ["r2", "r3", "r4"]


class FixedEmbedder:
    receipt_embedding_identity = "transaction-fixture-v1"

    def __init__(self):
        self.calls = []
        self.callback = None

    def embed(self, text):
        self.calls.append(text)
        if self.callback is not None:
            self.callback(text)
        return np.array([1, 0, 0, 0], dtype=np.float32)


class LiteralSummarizer:
    def summarize(self, text, max_len=256):
        return text[:max_len]


@pytest.fixture
def factory(tmp_path):
    instances = []

    def build(profile="json", directory=None):
        directory = Path(directory or tmp_path / f"s{len(instances)}")
        if profile == "r4" and not (directory / "dml_state.sqlite3").exists():
            source = build("r2", directory.with_name(directory.name + "s"))
            source.close(persist=False)
            upgrade_outbox_journal(source._journal.path, directory / "dml_state.sqlite3")
        adapter = DMLAdapter(
            config_overrides={
                "storage_dir": str(directory), "model_name": "dummy", "embedding_model": None,
                "checkpoint_interval_seconds": 0, "metrics_enabled": False,
                "persistence": {
                    "enable": profile == "jsonl", "interval_sec": 0,
                    "journal": profile in ["j1", *RECEIPTS],
                    "receipts": profile in RECEIPTS, "outbox": profile in ("r3", "r4"),
                },
                "rag_store": {"enable": False},
                "dpm": {"enable": False, "include_in_context": False},
                "theta_merge": 2.0, "eta": 0.0, "gamma": 0.0, "kappa": 0.0,
                "similarity_threshold": 0.0, "enable_quality_on_retrieval": False,
                "ann_min_items": 0, "token_budget": 600, "dml_context_max_items": 10,
            },
            embedder=FixedEmbedder(), summarizer=LiteralSummarizer(), start_aging_loop=False,
        )
        instances.append(adapter)
        if profile in RECEIPTS:
            assert adapter._journal.schema_version == int(profile[1:])
        return adapter

    yield build
    for adapter in reversed(instances):
        adapter.close(persist=False)


def texts(adapter):
    return [item.text for item in adapter.store.items()]


def acknowledged(adapter):
    """Snapshot the selected authority without relying on runtime counters."""
    if adapter._journal is not None:
        return adapter._journal.read_snapshot()
    return adapter._active_state_path().read_bytes()


def assert_owned(adapter):
    with pytest.raises(TimeoutError):
        with store_write_lock(adapter._active_state_path().parent, operation="probe", timeout_ms=0):
            pytest.fail("operation ran outside store ownership")


@pytest.mark.parametrize("profile", LEGACY)
def test_refresh_precedes_rollback_snapshot_and_provider(factory, profile):
    writer = factory(profile)
    writer.ingest("base")
    reader = factory(profile, writer.storage_dir)
    writer.ingest("peer")
    authority = acknowledged(writer)
    rag_bytes = writer.rag_state_path.read_bytes()
    provider_error = RuntimeError("provider rejected")

    def fail_provider(text):
        assert_owned(reader)
        assert texts(reader) == ["base", "peer"]
        if text == "rejected":
            raise provider_error

    reader.embedder.callback = fail_provider
    with pytest.raises(RuntimeError) as raised:
        reader.ingest("rejected")

    assert raised.value is provider_error
    assert raised.value.__cause__ is None
    assert texts(reader) == ["base", "peer"]
    assert acknowledged(reader) == authority
    assert reader.rag_state_path.read_bytes() == rag_bytes
    assert reader.refresh_if_changed() is False
    assert reader.durability_status() == {"status": "ok", "failures": {}}
    assert texts(factory(profile, writer.storage_dir)) == ["base", "peer"]


def test_ownership_refresh_is_reentrant_and_outer_operation_survives(factory, monkeypatch):
    adapter = factory("j1")
    real_lock = adapter_module.store_write_lock
    real_refresh = adapter.refresh_if_changed
    events = []

    @contextmanager
    def observed_lock(root, **kwargs):
        events.append(("acquire", kwargs["operation"]))
        with real_lock(root, **kwargs):
            yield
        events.append(("release", kwargs["operation"]))

    def refresh():
        assert_owned(adapter)
        events.append(("refresh", None))
        with adapter.mutation_transaction("refresh-child"):
            events.append(("refresh-child", None))
        return real_refresh()

    monkeypatch.setattr(adapter_module, "store_write_lock", observed_lock)
    monkeypatch.setattr(adapter, "refresh_if_changed", refresh)
    inner_error = RuntimeError("inner")
    with adapter.mutation_transaction("outer"):
        with pytest.raises(RuntimeError) as raised:
            with adapter.mutation_transaction("inner"):
                raise inner_error
        assert raised.value is inner_error
        adapter.ingest("survives nested ownership")

    assert events == [("acquire", "outer"), ("refresh", None),
                      ("refresh-child", None), ("release", "outer")]
    assert adapter._journal.decisions()[-1]["operation"] == "outer"
    adapter.ingest("new operation")
    assert adapter._journal.decisions()[-1]["operation"] == "ingest"
    assert events[-4:] == [("acquire", "ingest"), ("refresh", None),
                           ("refresh-child", None), ("release", "ingest")]


@pytest.mark.parametrize("profile", LEGACY)
@pytest.mark.parametrize("inner", ["batch", "decorated"], ids=["b", "d"])
def test_outer_abort_compensates_successful_inner_commit(factory, profile, inner):
    adapter = factory(profile)
    adapter.ingest("base", meta={"source": {"tags": ["base"]}})
    original_items = deepcopy([item.to_dict() for item in adapter.store.items()])
    original_rag = deepcopy(adapter.rag_store.export_state())
    revision = adapter._journal.revision if adapter._journal else None
    failure = RuntimeError("outer rejected")

    with pytest.raises(RuntimeError) as raised:
        with adapter.atomic_batch("outer"):
            adapter.store.items()[0].meta["source"]["tags"].append("temporary")
            adapter.ingest("outer", persist=False)
            if inner == "batch":
                with adapter.atomic_batch("inner"):
                    adapter.ingest("inner", persist=False)
            else:
                adapter.ingest("inner")
            raise failure

    assert raised.value is failure
    assert [item.to_dict() for item in adapter.store.items()] == original_items
    assert adapter.rag_store.export_state() == original_rag
    assert adapter.refresh_if_changed() is False
    assert adapter.durability_status() == {"status": "ok", "failures": {}}
    reopened = factory(profile, adapter.storage_dir)
    assert [item.to_dict() for item in reopened.store.items()] == original_items
    assert reopened.rag_store.export_state() == original_rag
    if revision is not None:
        assert adapter._journal.revision == revision + 2
        assert [entry["operation"] for entry in adapter._journal.decisions()][-2:] == ["outer", "outer"]
    adapter.ingest("later")
    assert texts(adapter) == ["base", "later"]
    assert adapter.store.items()[-1].id == original_items[0]["id"] + 1


@pytest.mark.parametrize("profile", LEGACY)
def test_caught_inner_abort_restores_savepoint_and_outer_can_commit(factory, profile):
    adapter = factory(profile)
    adapter.ingest("base")
    before = acknowledged(adapter)
    error = ValueError("discard inner")

    with adapter.atomic_batch("outer"):
        adapter.ingest("outer", persist=False)
        with pytest.raises(ValueError) as raised:
            with adapter.atomic_batch("inner"):
                adapter.ingest("discarded", persist=False)
                raise error
        assert raised.value is error
        assert acknowledged(adapter) == before
        assert texts(adapter) == ["base", "outer"]
        adapter.ingest("kept", persist=False)
        assert acknowledged(adapter) == before

    reopened = factory(profile, adapter.storage_dir)
    assert texts(reopened) == ["base", "outer", "kept"]
    assert [item.id for item in reopened.store.items()] == [0, 1, 2]
    assert reopened.rag_store.catalog_summary()["count"] == 3
    assert adapter.refresh_if_changed() is False


@pytest.mark.parametrize("profile", LEGACY)
def test_uncommitted_abort_restores_runtime_without_durable_replacement(factory, profile):
    adapter = factory(profile)
    adapter.ingest("base")
    before = acknowledged(adapter)
    rag_bytes = adapter.rag_state_path.read_bytes()
    failure = RuntimeError("body failure")
    with pytest.raises(RuntimeError) as raised:
        with adapter.atomic_batch("abort"):
            adapter.ingest("uncommitted", persist=False)
            raise failure

    assert raised.value is failure
    assert texts(adapter) == ["base"]
    assert acknowledged(adapter) == before
    assert adapter.rag_state_path.read_bytes() == rag_bytes
    assert adapter.refresh_if_changed() is False
    assert texts(factory(profile, adapter.storage_dir)) == ["base"]


@pytest.mark.parametrize("profile", LEGACY)
def test_rollback_error_preserves_original_and_compensation_cause(factory, monkeypatch, profile):
    adapter = factory(profile)
    adapter.ingest("base")
    real_commit = adapter.lattice_persistence.commit
    real_write = adapter_module.atomic_write_text
    original = PermissionError("RAG denied")
    compensation = OSError("compensation denied")
    attempts = []

    def commit(**kwargs):
        attempts.append(kwargs["operation"])
        if len(attempts) == 2:
            raise compensation
        return real_commit(**kwargs)

    def write(path, *args, **kwargs):
        if Path(path) == adapter.rag_state_path:
            raise original
        return real_write(path, *args, **kwargs)

    monkeypatch.setattr(adapter.lattice_persistence, "commit", commit)
    monkeypatch.setattr(adapter_module, "atomic_write_text", write)
    with pytest.raises(PersistenceRollbackError) as raised:
        with adapter.atomic_batch("failed"):
            adapter.ingest("unacknowledged", persist=False)

    failure = raised.value
    assert isinstance(failure.original_error, PersistenceCommitError)
    assert failure.original_error.__cause__ is original
    assert failure.rollback_error is compensation
    assert failure.__cause__ is failure.original_error
    assert attempts == ["failed", "failed"]
    assert texts(adapter) == ["base"]
    status = adapter.durability_status()
    assert status == {"status": "degraded", "failures": {
        "rag": "PermissionError: RAG denied", "dml": "OSError: compensation denied",
        "rollback": "OSError: compensation denied",
    }}
    status["failures"].clear()
    assert adapter.durability_status()["status"] == "degraded"


@pytest.mark.parametrize("profile", RECEIPTS)
def test_receipt_legacy_refusal_precedes_ownership_and_provider(factory, monkeypatch, profile):
    adapter = factory(profile)
    before = acknowledged(adapter)

    def forbidden(*_args, **_kwargs):
        pytest.fail("legacy refusal must happen before ownership or embedding")

    monkeypatch.setattr(adapter, "refresh_if_changed", forbidden)
    adapter.embedder.callback = forbidden
    with pytest.raises(ValueError, match="legacy writes"):
        adapter.ingest("forbidden")
    with pytest.raises(ValueError, match="legacy writes"):
        adapter.ingest_memory_batch([{"text": "forbidden", "tenant_id": "owner"}])
    with pytest.raises(ValueError, match="legacy writes"):
        with adapter.atomic_batch("forbidden"):
            pytest.fail("receipt store admitted a legacy batch")
    assert acknowledged(adapter) == before
    assert adapter.embedder.calls == []
    assert adapter.durability_status() == {"status": "ok", "failures": {}}


@pytest.mark.parametrize("profile", RECEIPTS)
def test_owned_retrieval_repairs_hydration_and_sees_provider_time_commit(factory, monkeypatch, profile):
    adapter = factory(profile)
    real_import = adapter.store.import_state

    def fail_import(_payload):
        raise RuntimeError("private hydration details")

    monkeypatch.setattr(adapter.store, "import_state", fail_import)
    receipt = adapter.ingest_memory_receipted("first", idempotency_key="a", tenant_id="owner")
    assert adapter.durability_status() == {
        "status": "degraded", "failures": {"receipt_runtime": "RuntimeError"}}
    assert texts(adapter) == []
    peer = factory(profile, adapter.storage_dir)
    events = []
    peer_receipts = []

    def provider(text):
        assert text == "query"
        with store_write_lock(adapter._active_state_path().parent, operation="provider-probe", timeout_ms=0):
            events.append("provider")
        peer_receipts.append(peer.ingest_memory_receipted(
            "second", idempotency_key="b", tenant_id="owner"))

    def import_owned(payload):
        assert_owned(adapter)
        events.append("hydrate")
        return real_import(payload)

    monkeypatch.setattr(adapter.store, "import_state", import_owned)
    adapter.embedder.callback = provider
    report = adapter.retrieve_context("query", tenant_id="owner", top_k=2)

    assert events == ["provider", "hydrate"]
    assert [entry["text"] for entry in report["items"]] == ["first", "second"]
    assert report["decision"]["store_revision"] == peer_receipts[0]["revision"]
    assert adapter._journal.revision == peer_receipts[0]["revision"]
    assert adapter.refresh_if_changed() is False
    assert adapter.durability_status() == {"status": "ok", "failures": {}}
    assert adapter.ingest_memory_receipted("first", idempotency_key="a", tenant_id="owner") == receipt
    assert adapter.embedder.calls == ["first", "query"]
