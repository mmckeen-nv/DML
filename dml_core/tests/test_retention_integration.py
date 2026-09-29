"""Public retention inspection: scoped, read-only, detached and payload-free."""
from __future__ import annotations

from contextlib import closing
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from daystrom_dml.contracts.retention import retention_contract
from daystrom_dml.dml_adapter import DMLAdapter
from daystrom_dml.journal import JournalIntegrityError
from daystrom_dml.provider_server import create_app
from daystrom_dml.services.outbox_migration import upgrade_outbox_journal
from daystrom_dml.services.receipt_lifecycle import ReceiptMemoryNotFound, memory_digest
from daystrom_dml.services.retention import RetentionInspectionUnsupported, UNINSPECTED_SURFACES
from scripts import dml_journal
from test_receipt_adapter import append, factory as receipt_factory


factory = receipt_factory
ENDPOINT = "/api/memory/retention/inspect"
SCOPE = {"tenant_id": "owner", "client_id": "client", "session_id": "session", "instance_id": "instance"}
SOURCE = "private-sentinel-original-preference"
CHANGED = "private-sentinel-corrected-preference"
DERIVED = "private-sentinel-approved-derivation"


@pytest.fixture(params=[2, 3, 4], ids=lambda value: f"schema-{value}")
def authority(request, factory, tmp_path):
    adapter = factory(directory=tmp_path / "authority", persistence={
        "journal": True, "receipts": True, "outbox": request.param == 3})
    if request.param == 4:
        adapter.close(persist=False)
        destination = tmp_path / "migrated" / "dml_state.sqlite3"
        upgrade_outbox_journal(adapter._journal.path, destination)
        adapter = factory(directory=destination.parent,
            persistence={"journal": True, "receipts": True, "outbox": True})
    receipt = append(adapter, text=SOURCE, key="private-ingestion-key", **SCOPE,
        meta={"source_trust": "trusted", "provenance": "private-provenance-sentinel"})
    return adapter, receipt


def inspect(adapter, ident):
    return adapter.inspect_memory_retention(ident, **SCOPE)


def unavailable(*_args, **_kwargs):
    raise AssertionError("Retention inspection contacted an unrelated runtime component")


def logical_tables(journal):
    with closing(sqlite3.connect(journal.path)) as connection:
        names = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return {name: connection.execute(f'SELECT * FROM "{name}" ORDER BY rowid').fetchall()
                for name in names}


def test_retired_source_remains_known_in_receipts_snapshot_and_promotion_proofs(authority):
    adapter, receipt = authority
    original = receipt["result"]["memory"]
    updated = adapter.update_memory_receipted(original["id"], text=CHANGED,
        expected_memory_digest=memory_digest(original), reason="private-update-reason",
        idempotency_key="private-update-key", **SCOPE)["result"]["memory"]
    adapter.promote_memories_receipted([
        {"memory_id": updated["id"], "expected_memory_digest": memory_digest(updated)}],
        text=DERIVED, reason="private-promotion-reason", idempotency_key="private-promotion-key", **SCOPE)
    adapter.retire_memory_receipted(updated["id"], expected_memory_digest=memory_digest(updated),
        reason="private-retirement-reason", idempotency_key="private-retirement-key", **SCOPE)
    before = logical_tables(adapter._journal)
    revision = adapter._journal.revision
    capacity = adapter.store.capacity
    calls = adapter.embedder.calls

    report = inspect(adapter, original["id"])

    assert report["schema_version"] == "dml-retention-report-v1"
    assert report["source"]["revision"] == revision
    assert report["source"]["journal_schema_version"] == adapter._journal.schema_version
    assert report["surfaces"]["current_items"] == {"direct_records": 1, "embedded_source_records": 1}
    assert report["surfaces"]["receipts"] == {"direct_records": 3, "embedded_source_records": 1}
    assert report["surfaces"]["current_lineage"] == {"direct_records": 0, "embedded_source_records": 0}
    assert report["surfaces"]["journal_snapshot"] == {
        "direct_records": 0 if adapter._journal.schema_version == 4 else 1,
        "embedded_source_records": 0,
    }
    assert report["surfaces"]["outbox_states"]["direct_records"] == (
        4 if adapter._journal.schema_version in (3, 4) else 0)
    assert report["surfaces"]["outbox_states"]["embedded_source_records"] == (
        2 if adapter._journal.schema_version in (3, 4) else 0)
    assert report["physical_erasure_supported"] is False
    assert report["retirement_is_erasure"] is False
    assert report["erasure_proven"] is False
    assert report["uninspected_surfaces"] == list(UNINSPECTED_SURFACES)
    assert report["known_reference_count"] == sum(
        sum(surface.values()) for surface in report["surfaces"].values())
    assert report["known_reference_count"] == {2: 7, 3: 13, 4: 12}[adapter._journal.schema_version]
    encoded = json.dumps(report)
    for secret in (SOURCE, CHANGED, DERIVED, "private-ingestion-key", "private-update-key",
                   "private-promotion-key", "private-retirement-key", "private-provenance-sentinel",
                   "private-update-reason", "private-promotion-reason", "private-retirement-reason",
                   str(adapter._journal.path)):
        assert secret not in encoded
    assert logical_tables(adapter._journal) == before
    assert adapter.store.capacity == capacity
    assert adapter.embedder.calls == calls


def test_history_only_memory_remains_inspectable_after_live_record_removal(authority):
    adapter, receipt = authority
    original = receipt["result"]["memory"]
    revision, state = adapter._journal.read_snapshot()
    state["items"] = []
    # A trusted low-level import/maintenance operation can remove a live item;
    # this is fixture construction, not a public erase operation.
    adapter._journal.save(state, expected_revision=revision, operation="test-remove-live-record")
    report = inspect(adapter, original["id"])
    assert report["surfaces"]["current_items"]["direct_records"] == 0
    assert report["surfaces"]["receipts"]["direct_records"] == 1
    assert report["known_reference_count"] >= 1
    assert report["erasure_proven"] is False


def test_inspection_never_embeds_hydrates_or_takes_adapter_writer_ownership(authority, monkeypatch):
    adapter, receipt = authority
    ident = receipt["result"]["memory"]["id"]
    expected = inspect(adapter, ident)
    before = logical_tables(adapter._journal)
    read_hooks = []

    def reject_mutation_hook(point):
        assert point in {"retention_after_snapshot", "retention_after_history"}
        read_hooks.append(point)

    with monkeypatch.context() as guarded:
        guarded.setattr(adapter.embedder, "embed", unavailable)
        guarded.setattr(adapter, "_receipt_embedding_space", unavailable)
        guarded.setattr(adapter.store, "import_state", unavailable)
        guarded.setattr(adapter.query_cache, "clear", unavailable)
        guarded.setattr(adapter, "_mutation_transaction", unavailable)
        guarded.setattr(adapter._journal, "save", unavailable)
        guarded.setattr(adapter._journal, "save_with_receipt", unavailable)
        guarded.setattr(adapter._journal, "_fault_hook", reject_mutation_hook)
        assert inspect(adapter, ident) == expected
    assert read_hooks == ["retention_after_snapshot", "retention_after_history"]
    assert logical_tables(adapter._journal) == before


def test_report_is_detached_and_request_preserves_exact_full_scope(authority):
    adapter, receipt = authority
    ident = receipt["result"]["memory"]["id"]
    expected = inspect(adapter, ident)
    changed = inspect(adapter, ident)
    changed["request"]["scope"]["tenant_id"] = "other"
    changed["source"].clear()
    changed["surfaces"]["receipts"]["direct_records"] = -1
    changed["uninspected_surfaces"].clear()
    assert inspect(adapter, ident) == expected
    assert expected["request"]["scope"] == SCOPE


@pytest.mark.parametrize("member", list(SCOPE))
def test_api_wrong_scope_and_missing_id_are_indistinguishable(authority, member):
    adapter, receipt = authority
    client = TestClient(create_app(adapter_factory=lambda: adapter))
    ident = receipt["result"]["memory"]["id"]
    other = client.post(ENDPOINT, json={"memory_id": ident, **SCOPE, member: "different"})
    absent = client.post(ENDPOINT, json={"memory_id": ident + 1000, **SCOPE})
    assert other.status_code == absent.status_code == 404
    assert other.json() == absent.json() == {"detail": {"code": "receipt_memory_not_found"}}


def test_api_result_matches_adapter_without_runtime_mutation(authority):
    adapter, receipt = authority
    ident = receipt["result"]["memory"]["id"]
    expected = inspect(adapter, ident)
    before = logical_tables(adapter._journal)
    response = TestClient(create_app(adapter_factory=lambda: adapter)).post(
        ENDPOINT, json={"memory_id": ident, **SCOPE})
    assert response.status_code == 200
    assert response.json() == expected
    assert logical_tables(adapter._journal) == before


@pytest.mark.parametrize("body", [
    {}, {"tenant_id": "owner"}, {"memory_id": 0},
    {"memory_id": True, "tenant_id": "owner"},
    {"memory_id": "0", "tenant_id": "owner"},
    {"memory_id": 0.0, "tenant_id": "owner"},
    {"memory_id": -1, "tenant_id": "owner"},
    {"memory_id": 0, "tenant_id": 7},
    {"memory_id": 0, "tenant_id": None},
    {"memory_id": 0, "tenant_id": "owner", "client_id": 7},
    {"memory_id": 0, "tenant_id": "owner", "session_id": False},
    {"memory_id": 0, "tenant_id": "owner", "instance_id": []},
    {"memory_id": 0, "tenant_id": "owner", "text": "private-forbidden-payload"},
    {"memory_id": 0, "tenant_id": "owner", "idempotency_key": "private-forbidden-key"},
], ids=["empty", "no-id", "no-scope", "bool-id", "string-id", "float-id", "negative-id",
        "integer-tenant", "null-tenant", "integer-client", "bool-session", "list-instance",
        "extra-text", "extra-key"])
def test_api_strict_validation_never_calls_adapter_or_echoes_input(factory, monkeypatch, body):
    adapter = factory()
    monkeypatch.setattr(adapter, "inspect_memory_retention", unavailable)
    response = TestClient(create_app(adapter_factory=lambda: adapter)).post(ENDPOINT, json=body)
    assert response.status_code == 422
    assert response.json() == {"detail": {"code": "retention_validation_failed"}}


@pytest.mark.parametrize("content", [
    '{"memory_id": 0, "tenant_id": "private-malformed-json"',
    json.dumps({"memory_id": 0, "tenant_id": "private-oversized-scope" * 1000}),
    json.dumps({"memory_id": 0, "tenant_id": "owner", "text": "private-oversized-payload" * 10000}),
], ids=["malformed-json", "oversized-scope", "oversized-extra"])
def test_api_invalid_raw_body_never_echoes_secrets(factory, monkeypatch, content):
    adapter = factory()
    monkeypatch.setattr(adapter, "inspect_memory_retention", unavailable)
    response = TestClient(create_app(adapter_factory=lambda: adapter)).post(
        ENDPOINT, content=content, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json() == {"detail": {"code": "retention_validation_failed"}}


def test_other_endpoints_keep_existing_validation_response(factory):
    response = TestClient(create_app(adapter_factory=factory)).post(
        "/api/memory/retire/receipt", json={"memory_id": "invalid"})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, list)
    assert any(error["loc"] == ["body", "memory_id"] for error in detail)


@pytest.mark.parametrize("value", [" ", "é" * 129], ids=["whitespace", "utf8-byte-limit"])
def test_api_semantically_invalid_scope_has_fixed_400(factory, value):
    response = TestClient(create_app(adapter_factory=factory)).post(
        ENDPOINT, json={"memory_id": 0, "tenant_id": value})
    assert response.status_code == 400
    assert response.json() == {"detail": {"code": "invalid_or_unsupported_retention_request"}}


@pytest.mark.parametrize("error,status,code", [
    (ReceiptMemoryNotFound, 404, "receipt_memory_not_found"),
    (RetentionInspectionUnsupported, 409, "retention_inspection_unsupported"),
    (JournalIntegrityError, 503, "retention_outcome_unavailable"),
    (sqlite3.OperationalError, 503, "retention_outcome_unavailable"),
    (sqlite3.DatabaseError, 503, "retention_outcome_unavailable"),
    (PermissionError, 503, "retention_outcome_unavailable"),
    (TimeoutError, 503, "retention_outcome_unavailable"),
    (ValueError, 400, "invalid_or_unsupported_retention_request"),
], ids=["missing", "unsupported-proof", "integrity", "sqlite-io", "sqlite-corrupt",
        "permission", "timeout", "profile"])
def test_api_storage_failures_have_fixed_codes_without_exception_messages(factory, monkeypatch, error, status, code):
    adapter = factory()

    def fail(*_args, **_kwargs):
        raise error("private-path /credentials/private-memory")

    monkeypatch.setattr(adapter, "inspect_memory_retention", fail)
    response = TestClient(create_app(adapter_factory=lambda: adapter)).post(
        ENDPOINT, json={"memory_id": 0, **SCOPE})
    assert response.status_code == status
    assert response.json() == {"detail": {"code": code}}


def test_api_existing_bearer_auth_is_required(factory, monkeypatch):
    adapter = factory()
    receipt = append(adapter, **SCOPE)
    monkeypatch.setenv("DML_API_TOKEN", "retention-test-token")
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    client = TestClient(create_app(adapter_factory=lambda: adapter))
    body = {"memory_id": receipt["result"]["memory"]["id"], **SCOPE}
    assert client.post(ENDPOINT, json=body).status_code == 401
    assert client.post(ENDPOINT, json=body, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post(ENDPOINT, json=body,
        headers={"Authorization": "Bearer retention-test-token"}).status_code == 200


@pytest.mark.parametrize("config", [
    {"journal": False, "receipts": False},
    {"journal": True, "receipts": False},
], ids=["no-journal", "legacy-journal"])
def test_adapter_requires_configured_receipt_profile(factory, config):
    adapter = factory(persistence=config)
    with pytest.raises(ValueError, match="configured receipt journal"):
        adapter.inspect_memory_retention(0, **SCOPE)
    response = TestClient(create_app(adapter_factory=lambda: adapter)).post(
        ENDPOINT, json={"memory_id": 0, **SCOPE})
    assert response.status_code == 400


def test_missing_initialized_database_returns_unavailable_without_recreation(authority):
    adapter, receipt = authority
    ident = receipt["result"]["memory"]["id"]
    path = adapter._journal.path
    path.unlink()
    response = TestClient(create_app(adapter_factory=lambda: adapter)).post(
        ENDPOINT, json={"memory_id": ident, **SCOPE})
    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "retention_outcome_unavailable"}}
    assert not path.exists()


def test_contract_is_detached_data_independent_and_honest():
    expected = retention_contract()
    changed = retention_contract()
    changed["journal_schema_versions"].clear()
    changed["retirement"].clear()
    changed["inspection"]["surfaces"].append("unverified")
    changed["uninspected_surfaces"].clear()
    assert retention_contract() == expected
    assert expected["physical_erasure_supported"] is expected["erasure_proven"] is False
    assert expected["retirement_is_erasure"] is False
    assert expected["history"]["retention"] == "indefinite_under_supported_operations"
    assert expected["uninspected_surfaces"] == list(UNINSPECTED_SURFACES)


def test_contract_api_does_not_access_adapter_storage(factory, monkeypatch):
    adapter = factory()
    client = TestClient(create_app(adapter_factory=lambda: adapter))
    monkeypatch.setattr(adapter, "inspect_memory_retention", unavailable)
    monkeypatch.setattr(adapter._journal, "read_snapshot", unavailable)
    monkeypatch.setattr(adapter._journal, "read_retention_view", unavailable)
    response = client.get("/api/contracts")
    assert response.status_code == 200
    assert response.json()["retention"] == retention_contract()
    assert response.json()["production_ready"] is False
    assert response.json()["stable"] == []


def test_cli_contract_never_opens_or_creates_storage(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dml_journal, "JournalStateStore", unavailable)
    monkeypatch.setattr(DMLAdapter, "__init__", unavailable)
    assert dml_journal.main(["retention-contract"]) == 0
    assert json.loads(capsys.readouterr().out) == retention_contract()
    assert list(tmp_path.iterdir()) == []


def test_cli_contract_rejects_database_argument_without_creating_it(tmp_path, capsys):
    missing = tmp_path / "must-not-be-created.sqlite3"
    with pytest.raises(SystemExit) as error:
        dml_journal.main(["retention-contract", str(missing)])
    assert error.value.code == 2
    capsys.readouterr()
    assert not missing.exists()
