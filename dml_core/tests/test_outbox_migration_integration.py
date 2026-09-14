"""Migration adoption must preserve receipt protections across public boundaries."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time

import pytest

from daystrom_dml.journal import JournalStateStore
from daystrom_dml.services.outbox_migration import upgrade_outbox_journal
from daystrom_dml.services.projection import SQLiteProjection
from test_receipt_adapter import CountingEmbedder, append, factory as receipt_factory

factory = receipt_factory


@pytest.fixture
def migrated(factory, tmp_path):
    source = factory(directory=tmp_path / "old")
    receipts = [append(source), append(source, key="second", text="Another durable memory")]
    before = source._journal.verified_snapshot()
    decisions = source._journal.decisions()
    source.close(persist=False)
    destination = tmp_path / "migrated" / "dml_state.sqlite3"
    report = upgrade_outbox_journal(source._journal.path, destination)
    assert source._journal.verified_snapshot() == before
    assert source._journal.decisions() == decisions
    return source._journal, destination, receipts, report


def adopt(factory, migrated, *, receipts=True, outbox=True, embedder=None):
    return factory(directory=migrated[1].parent, embedder=embedder,
                   persistence={"journal": True, "receipts": receipts, "outbox": outbox})


def script(name, *args):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    response = subprocess.run([sys.executable, str(path), *map(str, args)],
                              capture_output=True, text=True, timeout=30)
    return response, json.loads(response.stdout)


def test_explicit_adoption_preserves_historical_receipt_before_embedding(factory, migrated):
    embedder = CountingEmbedder(fail=True)
    adapter = adopt(factory, migrated, embedder=embedder)
    assert adapter._journal.schema_version == 4
    baseline = adapter._journal.outbox_events()
    assert append(adapter) == migrated[2][0]
    assert append(adapter, key="second", text="Another durable memory") == migrated[2][1]
    assert embedder.calls == 0
    assert adapter._journal.outbox_events() == baseline
    assert len(baseline["events"]) == 1
    assert baseline["events"][0]["source_revision"] == migrated[3]["boundary_revision"]
    adapter.close()  # Must not append an incidental persistence mutation.
    assert JournalStateStore(migrated[1]).outbox_events() == baseline


@pytest.mark.parametrize("receipts,outbox", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("operation", ["ingest", "fast_ingest", "persist", "maintenance", "aging", "checkpoint", "unscoped_query", "workflow"])
def test_migrated_store_keeps_legacy_protections_when_flags_change(factory, migrated, receipts, outbox, operation):
    adapter = adopt(factory, migrated, receipts=receipts, outbox=outbox)
    before = adapter._journal.outbox_events()
    calls = adapter.embedder.calls
    action = {
        "ingest": lambda: adapter.ingest_memory("legacy", tenant_id="owner"),
        "fast_ingest": lambda: adapter.ingest_fast("legacy"),
        "persist": adapter._persist_dml_state,
        "maintenance": adapter.run_maintenance,
        "aging": adapter._run_aging,
        "checkpoint": adapter.create_checkpoint,
        "unscoped_query": lambda: adapter.retrieve_context("preference"),
        "workflow": lambda: adapter.record_agent_workflow("task", ["step"], "success"),
    }[operation]
    with pytest.raises(ValueError):
        action()
    assert adapter.embedder.calls == calls
    assert adapter._journal.outbox_events() == before
    assert adapter.enable_quality_on_retrieval is False
    assert adapter._persistence_thread is None


def test_migrated_store_requires_delivery_opt_in_and_retains_receipt_retry(factory, migrated):
    adapter = adopt(factory, migrated, outbox=False)
    assert append(adapter) == migrated[2][0]
    with pytest.raises(ValueError, match="opt-in"):
        adapter.deliver_outbox(object())


def test_migrated_append_worker_and_verified_scoped_query(factory, migrated, tmp_path):
    adapter = adopt(factory, migrated)
    target = SQLiteProjection(tmp_path / "projection" / "state.sqlite")
    worker = adapter.start_projection_worker(target, poll_interval=0.01, retry_initial=0.01, retry_max=0.04)
    receipt = append(adapter, key="new", text="Accepted after explicit migration")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = worker.status()["last_result"] or {}
        if result.get("matches_pinned_source") is True and result.get("projection", {}).get("source_revision") == receipt["revision"]:
            break
        threading.Event().wait(0.005)
    assert adapter.projection_status(target)["matches_pinned_source"] is True
    result = adapter.query_projection("preference", backend=target, tenant_id="owner", as_of=100)
    assert {hit["memory"]["id"] for hit in result["results"]} == {
        entry["result"]["memory"]["id"] for entry in migrated[2] + [receipt]}
    assert adapter.query_projection("preference", backend=target, tenant_id="other", as_of=100)["results"] == []
    adapter.close(persist=False)
    assert worker.status()["state"] == "closed"
    assert migrated[0].schema_version == 2
    assert migrated[0].revision == migrated[3]["source_revision"]


def test_migrated_query_bypasses_cache_and_rejects_embedding_drift(factory, migrated):
    from daystrom_dml.services.receipt_ingestion import ReceiptEmbeddingCompatibilityError

    adapter = adopt(factory, migrated, receipts=False, outbox=False)
    calls = adapter.embedder.calls
    assert adapter._embed_query("query").tolist() == adapter._embed_query("query").tolist()
    assert adapter.embedder.calls == calls + 2
    adapter.embedder.receipt_embedding_identity = "changed-runtime"
    with pytest.raises(ReceiptEmbeddingCompatibilityError):
        adapter._embed_query("query")
    assert adapter.embedder.calls == calls + 2


def test_migration_cli_reports_verified_boundary_without_cutover(factory, tmp_path):
    old = factory(directory=tmp_path / "source")
    receipt = append(old)
    before = old._journal.verified_snapshot()
    destination = tmp_path / "destination" / "dml_state.sqlite3"
    response, report = script("dml_journal.py", "enable-outbox", old._journal.path, destination)
    assert response.returncode == 0, response.stdout + response.stderr
    assert report["schema_version"] == 4
    assert report["source_store_id"] == before[0]
    assert report["source_revision"] == receipt["revision"]
    assert report["destination_revision"] == report["boundary_revision"] == receipt["revision"] + 1
    assert report["preserved_receipt_count"] == 1
    assert report["preserved_decision_count"] == before[1]
    assert report["origin"]["source_schema_version"] == 2
    assert old._journal.verified_snapshot() == before
    assert old._journal.schema_version == 2
    assert JournalStateStore(destination, receipt_mode=True, outbox_mode=True).schema_version == 4


@pytest.mark.parametrize("schema", [1, 3, 4, 99])
def test_migration_cli_rejects_non_source_versions_without_destination(factory, migrated, tmp_path, schema):
    if schema == 4:
        source = migrated[1]
    else:
        source = tmp_path / f"schema-{schema}" / "state.sqlite"
        JournalStateStore(source, receipt_mode=schema == 3, outbox_mode=schema == 3)
        if schema == 99:
            with sqlite3.connect(source) as connection:
                connection.execute("PRAGMA user_version=99")
    destination = tmp_path / "rejected" / "state.sqlite"
    response, result = script("dml_journal.py", "enable-outbox", source, destination)
    assert response.returncode == 2
    assert result["ok"] is False
    assert set(result) == {"ok", "error"}
    assert str(source) not in response.stdout + response.stderr
    assert not destination.exists()


def test_migration_cli_sanitizes_storage_errors(migrated, tmp_path, monkeypatch, capsys):
    from daystrom_dml.services import outbox_migration
    from scripts import dml_journal

    def fail(*_args, **_kwargs):
        raise OSError("private tenant contents and storage credential")

    monkeypatch.setattr(outbox_migration, "upgrade_outbox_journal", fail)
    code = dml_journal.main(["enable-outbox", str(migrated[0].path), str(tmp_path / "new" / "state.sqlite")])
    captured = capsys.readouterr()
    assert code == 2
    assert json.loads(captured.out) == {"ok": False, "error": "OSError"}
    assert "private" not in captured.out + captured.err


def test_migrated_projection_cli_full_incremental_and_query(factory, migrated, tmp_path):
    adapter = adopt(factory, migrated)
    target = tmp_path / "projection" / "state.sqlite"
    for args in [("sync",), ("sync", "--incremental"), ("status",)]:
        response, result = script("dml_projection.py", migrated[1], target, *args)
        assert response.returncode == 0, response.stdout + response.stderr
        assert result["matches_pinned_source"] is True
    request = tmp_path / "query.json"
    request.write_text(json.dumps({"vector": [1, 1, 1, 1], "embedding_identity": adapter._receipt_embedding_space(),
                                  "scope": {"tenant_id": "owner", "client_id": None, "session_id": None, "instance_id": None},
                                  "as_of": 100}), encoding="utf-8")
    response, result = script("dml_projection.py", migrated[1], target, "query", request)
    assert response.returncode == 0, response.stdout + response.stderr
    assert len(result["results"]) == 2


def test_migrated_outbox_cli_delivers_baseline_and_new_events_in_bounded_pages(factory, migrated, tmp_path):
    adapter = adopt(factory, migrated)
    final = append(adapter, key="new", text="Post migration append")
    target = tmp_path / "consumer" / "state.sqlite"
    response, first = script("dml_outbox.py", migrated[1], target, "sync", "--limit", "1")
    assert response.returncode == 0, response.stdout + response.stderr
    assert first["delivered_count"] == 1 and first["backlog"] == 1
    assert first["consumer_cursor"]["source_revision"] == migrated[3]["boundary_revision"]
    assert first["history_coverage"]["legacy_operations_delivered"] is False
    response, last = script("dml_outbox.py", migrated[1], target, "sync", "--limit", "1")
    assert response.returncode == 0, response.stdout + response.stderr
    assert last["delivered_count"] == 1 and last["backlog"] == 0
    assert last["consumer_cursor"]["source_revision"] == final["revision"]
    response, status = script("dml_outbox.py", migrated[1], target, "status")
    assert response.returncode == 0
    assert status["matches_observed_source"] is True
    from daystrom_dml.services.outbox_delivery import SQLiteOutboxConsumer
    envelope = SQLiteOutboxConsumer(target).read()
    assert envelope["schema_version"] == 1
    assert envelope["consumer_format"] == "dml-sqlite-outbox-consumer-v2"
    assert envelope["last_event"]["state"] == adapter._journal.load()
    assert len(envelope["event_checksums"]) == 2


def test_migrated_delivery_requires_unlocked_source_before_consumer_io(factory, migrated, tmp_path):
    from daystrom_dml.services.outbox_delivery import SQLiteOutboxConsumer
    from daystrom_dml.store_lock import store_write_lock

    adapter = adopt(factory, migrated)
    consumer = SQLiteOutboxConsumer(tmp_path / "consumer" / "state.sqlite")
    probes = []

    class ProbeConsumer:
        path = consumer.path

        def read(self):
            with store_write_lock(adapter._journal.path.parent, operation="migrated-read", timeout_ms=100):
                probes.append("read")
            return consumer.read()

        def apply_event(self, event):
            with store_write_lock(adapter._journal.path.parent, operation="migrated-apply", timeout_ms=100):
                probes.append("apply")
            return consumer.apply_event(event)

    with adapter.mutation_transaction("nested-delivery"):
        with pytest.raises(ValueError, match="ownership"):
            adapter.deliver_outbox(ProbeConsumer())
    assert probes == []
    result = adapter.deliver_outbox(ProbeConsumer(), limit=1)
    assert result["delivered_count"] == 1
    assert result["matches_observed_source"] is True
    assert "read" in probes and "apply" in probes


@pytest.mark.parametrize("name", ["dml_projection.py", "dml_outbox.py"])
def test_delivery_cli_rejects_unknown_future_source_before_target_creation(tmp_path, name):
    source = tmp_path / "future" / "state.sqlite"
    JournalStateStore(source)
    with sqlite3.connect(source) as connection:
        connection.execute("PRAGMA user_version=99")
    target = tmp_path / "never-created" / "state.sqlite"
    response, result = script(name, source, target, "sync")
    assert response.returncode == 2
    assert result == {"ok": False, "error": "JournalSchemaError"}
    assert not target.exists()


def test_migration_cli_refuses_existing_target_without_changing_it(migrated):
    before = JournalStateStore(migrated[1]).verified_snapshot()
    response, result = script("dml_journal.py", "enable-outbox", migrated[0].path, migrated[1])
    assert response.returncode == 2
    assert result["ok"] is False
    assert str(migrated[1]) not in response.stdout + response.stderr
    assert JournalStateStore(migrated[1]).verified_snapshot() == before


def test_migration_cli_missing_source_has_sanitized_error_and_no_new_store(tmp_path):
    source = tmp_path / "secret-source" / "state.sqlite"
    target = tmp_path / "target" / "state.sqlite"
    response, result = script("dml_journal.py", "enable-outbox", source, target)
    assert response.returncode == 2
    assert result == {"ok": False, "error": "FileNotFoundError"}
    assert "secret-source" not in response.stdout + response.stderr
    assert not source.exists() and not target.exists()
