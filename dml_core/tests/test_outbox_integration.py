"""Independent API regressions for explicit transactional outbox adoption."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from daystrom_dml.journal import JournalSchemaError, JournalStateStore
from daystrom_dml.services.projection import SQLiteProjection
from daystrom_dml.settings import PersistenceSettings
from test_receipt_adapter import CountingEmbedder, append, factory as receipt_factory

factory = receipt_factory


def build(factory, **kwargs):
    return factory(persistence={"journal": True, "receipts": True, "outbox": True}, **kwargs)


def test_outbox_setting_is_disabled_by_default():
    assert PersistenceSettings().outbox is False


@pytest.mark.parametrize("journal,receipts", [(False, False), (False, True), (True, False)])
def test_outbox_requires_explicit_journal_and_receipt_configuration(factory, tmp_path, journal, receipts):
    with pytest.raises(ValueError, match="outbox"):
        factory(persistence={"journal": journal, "receipts": receipts, "outbox": True})
    assert not (tmp_path / "dml_state.sqlite3").exists()


@pytest.mark.parametrize("receipts,schema", [(False, 1), (True, 2)])
def test_existing_configuration_does_not_silently_enable_outbox(factory, receipts, schema):
    adapter = factory(persistence={"journal": True, "receipts": receipts})
    assert adapter._journal.schema_version == schema
    with pytest.raises(ValueError, match="outbox"):
        adapter.deliver_outbox(object())
    with pytest.raises(JournalSchemaError):
        adapter._journal.outbox_events()


def test_schema_two_requires_explicit_migration_before_enabling_outbox(factory):
    old = factory()
    receipt = append(old)
    before = old._journal.read_snapshot()
    old.close(persist=False)
    with pytest.raises(JournalSchemaError):
        build(factory)
    reopened = factory()
    assert reopened._journal.schema_version == 2
    assert reopened._journal.read_snapshot() == before
    assert append(reopened) == receipt


def test_append_receipt_and_event_survive_retry_and_adapter_restart(factory):
    adapter = build(factory)
    assert adapter._journal.schema_version == 3
    receipt = append(adapter)
    page = adapter._journal.outbox_events()
    assert page["head_revision"] == receipt["revision"] == 1
    assert len(page["events"]) == 1
    event = page["events"][0]
    assert event["source_revision"] == receipt["revision"]
    assert event["state"]["items"] == adapter._journal.load()["items"]
    assert event["state"]["items"][0]["id"] == receipt["result"]["memory"]["id"]
    assert event["receipt"]["key"] == "request-1"
    assert append(adapter) == receipt
    assert adapter._journal.outbox_events() == page
    adapter.close()  # Default persistence must not introduce a legacy mutation.
    embedder = CountingEmbedder(fail=True)
    reopened = build(factory, embedder=embedder)
    assert append(reopened) == receipt
    assert embedder.calls == 0
    assert reopened._journal.outbox_events() == page


@pytest.mark.parametrize("receipts,outbox", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("operation", ["ingest", "persist", "maintenance", "checkpoint", "unscoped_query"])
def test_schema_three_preserves_receipt_protections_even_when_flags_disabled(factory, receipts, outbox, operation):
    initial = build(factory)
    append(initial)
    initial.close(persist=False)
    adapter = factory(persistence={"journal": True, "receipts": receipts, "outbox": outbox})
    before = adapter._journal.outbox_events()
    calls = adapter.embedder.calls
    action = {
        "ingest": lambda: adapter.ingest_memory("legacy", tenant_id="owner"),
        "persist": adapter._persist_dml_state,
        "maintenance": adapter.run_maintenance,
        "checkpoint": adapter.create_checkpoint,
        "unscoped_query": lambda: adapter.retrieve_context("preference"),
    }[operation]
    with pytest.raises(ValueError):
        action()
    assert adapter.embedder.calls == calls
    assert adapter._journal.outbox_events() == before
    assert adapter.enable_quality_on_retrieval is False
    assert adapter._persistence_thread is None


def test_schema_three_coalesced_projection_and_scoped_query(factory, tmp_path):
    adapter = build(factory)
    receipt = append(adapter)
    target = SQLiteProjection(tmp_path / "projection" / "index.sqlite")
    assert adapter.sync_projection(target)["matches_pinned_source"] is True
    result = adapter.query_projection("preference", backend=target, tenant_id="owner", as_of=100)
    assert [hit["memory"]["id"] for hit in result["results"]] == [receipt["result"]["memory"]["id"]]
    assert adapter.projection_status(target)["matches_pinned_source"] is True
    assert len(adapter._journal.outbox_events()["events"]) == 1


def test_schema_three_projection_cli_keeps_existing_sync_modes(factory, tmp_path):
    adapter = build(factory)
    append(adapter)
    script = Path(__file__).resolve().parents[1] / "scripts" / "dml_projection.py"
    target = tmp_path / "projection" / "index.sqlite"
    for flags in [("sync",), ("sync", "--incremental"), ("status",)]:
        response = subprocess.run([sys.executable, str(script), str(adapter._journal.path), str(target), *flags],
                                  capture_output=True, text=True, timeout=30)
        assert response.returncode == 0, response.stdout + response.stderr
        assert json.loads(response.stdout)["matches_pinned_source"] is True
    assert JournalStateStore(adapter._journal.path).schema_version == 3


def test_outbox_delivery_refuses_nested_source_ownership_before_consumer_io(factory):
    adapter = build(factory)
    append(adapter)
    with adapter.mutation_transaction("nested-consumer"):
        with pytest.raises(ValueError, match="ownership"):
            adapter.deliver_outbox(object())


def test_outbox_adapter_explicit_bounded_delivery_and_consumer_restart(factory, tmp_path):
    from daystrom_dml.services.outbox_delivery import SQLiteOutboxConsumer

    adapter = build(factory)
    receipts = [append(adapter, key=str(index), text=f"Memory {index}") for index in range(3)]
    consumer = SQLiteOutboxConsumer(tmp_path / "consumer" / "state.sqlite")
    assert consumer.read()["cursor"] is None
    first = adapter.deliver_outbox(consumer, limit=2)
    assert first["delivered_count"] == 2
    assert first["backlog"] == 1
    assert first["consumer_cursor"]["source_revision"] == receipts[1]["revision"]
    reopened = SQLiteOutboxConsumer(consumer.path)
    final = adapter.deliver_outbox(reopened, limit=2)
    assert final["delivered_count"] == 1
    assert final["matches_observed_source"] is True
    assert reopened.read()["last_event"]["state"] == adapter._journal.load()
    assert adapter.deliver_outbox(reopened)["delivered_count"] == 0


def test_consumer_outage_preserves_receipts_and_delivers_after_recovery(factory, tmp_path, monkeypatch):
    from daystrom_dml.services.outbox_delivery import SQLiteOutboxConsumer

    adapter = build(factory)
    receipt = append(adapter)
    consumer = SQLiteOutboxConsumer(tmp_path / "consumer" / "state.sqlite")
    apply = consumer.apply_event

    def fail(_event):
        raise OSError("private consumer credential")

    monkeypatch.setattr(consumer, "apply_event", fail)
    before = adapter._journal.outbox_events()
    with pytest.raises(OSError):
        adapter.deliver_outbox(consumer)
    assert adapter._journal.outbox_events() == before
    assert append(adapter) == receipt
    later = append(adapter, key="later", text="Accepted during outage")
    monkeypatch.setattr(consumer, "apply_event", apply)
    assert adapter.deliver_outbox(consumer)["consumer_cursor"]["source_revision"] == later["revision"]


def test_consumer_io_does_not_hold_source_ownership(factory, tmp_path):
    from daystrom_dml.services.outbox_delivery import SQLiteOutboxConsumer
    from daystrom_dml.store_lock import store_write_lock

    adapter = build(factory)
    append(adapter)
    target = SQLiteOutboxConsumer(tmp_path / "consumer" / "state.sqlite")
    probes = []

    class ProbeConsumer:
        path = target.path

        def read(self):
            with store_write_lock(adapter._journal.path.parent, operation="consumer-read-probe", timeout_ms=100):
                probes.append("read")
            return target.read()

        def apply_event(self, event):
            with store_write_lock(adapter._journal.path.parent, operation="consumer-write-probe", timeout_ms=100):
                probes.append("apply")
            return target.apply_event(event)

    assert adapter.deliver_outbox(ProbeConsumer())["matches_observed_source"] is True
    assert "read" in probes and "apply" in probes


def test_existing_schema_three_requires_explicit_outbox_flag_for_delivery(factory):
    initial = build(factory)
    receipt = append(initial)
    initial.close(persist=False)
    adapter = factory()
    assert append(adapter) == receipt
    with pytest.raises(ValueError, match="opt-in"):
        adapter.deliver_outbox(object())


def test_outbox_cli_json_bounds_and_status(factory, tmp_path):
    adapter = build(factory)
    append(adapter)
    append(adapter, key="second", text="A second memory")
    script = Path(__file__).resolve().parents[1] / "scripts" / "dml_outbox.py"
    target = tmp_path / "consumer" / "state.sqlite"

    def cli(*args):
        response = subprocess.run([sys.executable, str(script), str(adapter._journal.path), str(target), *args],
                                  capture_output=True, text=True, timeout=30)
        return response, json.loads(response.stdout)

    response, result = cli("status")
    assert response.returncode == 2
    assert result == {"ok": False, "error": "FileNotFoundError"}
    assert not target.exists()
    response, result = cli("sync", "--limit", "1")
    assert response.returncode == 0, response.stdout + response.stderr
    assert result["delivered_count"] == 1 and result["backlog"] == 1
    response, result = cli("sync")
    assert response.returncode == 0
    assert result["delivered_count"] == 1 and result["backlog"] == 0
    response, result = cli("status")
    assert response.returncode == 0
    assert result["matches_observed_source"] is True


def test_outbox_cli_consumer_error_is_sanitized(factory, tmp_path, monkeypatch, capsys):
    from scripts import dml_outbox

    adapter = build(factory)
    receipt = append(adapter)

    def unavailable(*_args, **_kwargs):
        raise OSError("private tenant contents and credentials")

    monkeypatch.setattr(dml_outbox, "deliver_outbox", unavailable)
    result = dml_outbox.main([str(adapter._journal.path), str(tmp_path / "consumer" / "state.sqlite"), "sync"])
    captured = capsys.readouterr()
    assert result == 2
    assert json.loads(captured.out) == {"ok": False, "error": "OSError"}
    assert "private" not in captured.out + captured.err
    assert append(adapter) == receipt


def test_schema_three_query_bypasses_identity_free_cache(factory):
    from daystrom_dml.services.receipt_ingestion import ReceiptEmbeddingCompatibilityError

    adapter = build(factory)
    append(adapter)
    initial_calls = adapter.embedder.calls
    first = adapter._embed_query("same query")
    second = adapter._embed_query("same query")
    assert adapter.embedder.calls == initial_calls + 2
    assert first.tolist() == second.tolist()
    adapter.embedder.receipt_embedding_identity = "different-model-revision"
    with pytest.raises(ReceiptEmbeddingCompatibilityError):
        adapter._embed_query("same query")
    assert adapter.embedder.calls == initial_calls + 2
