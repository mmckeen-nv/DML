"""Independent negative oracles for the frozen candidate profile boundary."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from daystrom_dml.config import load_config
from daystrom_dml.dml_adapter import DMLAdapter
from daystrom_dml.embeddings import RandomEmbedder, SentenceTransformerEmbedder
from daystrom_dml.journal import JournalStateStore
from daystrom_dml.provider_server import create_app

PROFILE = "dml-receipted-local-v1"
TOKEN = "profile-adversarial-startup-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class NativeProbe:
    """Explicit test double; this is no claim of embedding-provider qualification."""

    model_name = "independent-profile-test-provider"

    def __init__(self):
        self.calls = 0

    def embed(self, text):
        self.calls += 1
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def unavailable(*_args, **_kwargs):
    raise AssertionError("Excluded component was contacted")


@pytest.fixture
def admission(tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("DML_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    config = {
        "production_profile": PROFILE,
        "storage_dir": str(tmp_path / "authority"),
        "model_name": "dummy", "llm_backend": "dummy",
        "embedding_model": None, "strict_embedding_required": True,
        "strict_llm_required": False,
        "persistence": {"enable": False, "interval_sec": 0, "journal": True,
                        "receipts": True, "outbox": False,
                        "receipt_embedding_identity": "operator-fixed-v1"},
        "rag_store": {"enable": False},
        "dpm": {"enable": False, "mode": "disabled", "include_in_context": False,
                "include_in_preamble": False},
        "skip_rag_state_import": True, "survival_ledger_enabled": False,
        "background_processing_enabled": False,
    }
    path = tmp_path / "candidate.json"
    instances = []

    def build(*, change=None, embedder=None):
        selected = deepcopy(config)
        if change:
            change(selected)
        path.write_text(json.dumps(selected), encoding="utf-8")
        instance = DMLAdapter(config_path=path, embedder=embedder or NativeProbe(),
                              start_aging_loop=False)
        instances.append(instance)
        return instance

    yield config, path, build
    for instance in reversed(instances):
        instance.close(persist=False)


@pytest.mark.parametrize("change", [
    lambda c: c.update(production_profile="dml-receipted-local-v2"),
    lambda c: c.update(production_profile=True),
    lambda c: c.update(production_profile=[]),
    lambda c: c.update(ann_min_items=1),
    lambda c: c.update(enable_quality_on_retrieval=True),
    lambda c: c.update(strict_embedding_required="true"),
    lambda c: c.update(capacity=True),
    lambda c: c.update(gpu_acceleration=0),
    lambda c: c.update(agentic_mode={"enabled": True}),
    lambda c: c.update(dml={"agentic_mode": {"enabled": True}}),
    lambda c: c.update({"dml.agentic_mode.enabled": True}),
    lambda c: c["persistence"].update(receipts="true"),
    lambda c: c["persistence"].update(**{"reciepts": True}),
    lambda c: c["rag_store"].update(**{"enabel": False}),
    lambda c: c["dpm"].update(mode="active-read"),
    lambda c: c["dpm"].update(include_in_context=True),
    lambda c: c["persistence"].update(receipt_embedding_identity=None),
])
def test_invalid_effective_config_rejects_before_components_or_directory(admission, monkeypatch, change):
    import daystrom_dml.dml_adapter as adapter_module

    config, _, build = admission
    for name in ("GPTRunner", "create_embedder", "PersonalityMatrix", "JournalStateStore", "MemoryStore"):
        monkeypatch.setattr(adapter_module, name, unavailable)
    with pytest.raises(ValueError):
        build(change=change)
    assert not Path(config["storage_dir"]).exists()


@pytest.mark.parametrize("name,value", [
    ("DML_PERSISTENCE__RECEIPTS", "yes"),
    ("DML_ANN_MIN_ITEMS", "01"),
    ("DML_CAPACITY", "1.0"),
    ("DML_PERSISTENCE___RECEIPTS", "true"),
    ("DML_UNKNOWN_CAPABILITY", "private-config-sentinel"),
])
def test_noncanonical_or_unknown_environment_is_rejected(admission, monkeypatch, name, value):
    config, path, _ = admission
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError) as rejected:
        load_config(path)
    assert "private-config-sentinel" not in str(rejected.value)


def test_explicit_overrides_win_over_environment_without_coercing_yaml(admission, monkeypatch):
    config, path, _ = admission
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("DML_PERSISTENCE__RECEIPTS", "false")
    monkeypatch.setenv("DML_API_TOKEN", "private-auth-sentinel")
    with pytest.raises(ValueError):
        load_config(path)
    resolved = load_config(path, overrides={"persistence": {"receipts": True}})
    assert resolved.persistence.receipts is True
    assert "private-auth-sentinel" not in json.dumps(resolved.as_dict())


@pytest.mark.parametrize("variant", ["random", "random-subclass", "fallback"])
def test_known_test_or_fallback_embedders_reject_before_directory(admission, variant):
    config, _, build = admission
    if variant == "fallback":
        embedder = object.__new__(SentenceTransformerEmbedder)
        embedder.model_name = "failed-model"
        embedder._model = None
        embedder._dim = 4
    elif variant == "random-subclass":
        class WrappedRandom(RandomEmbedder):
            pass
        embedder = WrappedRandom(4)
    else:
        embedder = RandomEmbedder(4)
    with pytest.raises(ValueError):
        build(embedder=embedder)
    assert not Path(config["storage_dir"]).exists()


@pytest.mark.parametrize("version,outbox", [(1, False), (3, False), (2, True)])
def test_wrong_existing_authority_is_not_adopted_or_given_new_identity(admission, monkeypatch, version, outbox):
    import daystrom_dml.dml_adapter as adapter_module

    config, _, build = admission
    path = Path(config["storage_dir"]) / "dml_state.sqlite3"
    journal = JournalStateStore(path, receipt_mode=version >= 2, outbox_mode=version == 3)
    journal.save({"items": [], "lineage": []})
    before = path.read_bytes()
    journal.identity_path.unlink()
    monkeypatch.setattr(adapter_module, "create_embedder", unavailable)
    monkeypatch.setattr(adapter_module, "JournalStateStore", unavailable)
    with pytest.raises(ValueError):
        build(change=lambda c: c["persistence"].update(outbox=outbox))
    assert path.read_bytes() == before
    assert not journal.identity_path.exists()


def test_malformed_legacy_rag_never_enters_profile_startup_or_refresh(admission, monkeypatch):
    config, _, build = admission
    directory = Path(config["storage_dir"])
    directory.mkdir()
    legacy = directory / "rag_store.json"
    legacy.write_text("private-invalid-legacy-rag", encoding="utf-8")
    instance = build()
    monkeypatch.setattr(instance.rag_store, "import_state", unavailable)
    receipt = instance.ingest_memory_receipted("accepted memory", tenant_id="owner", idempotency_key="first")
    legacy.write_text("private-modified-invalid-legacy-rag", encoding="utf-8")
    report = instance.retrieve_context("accepted", tenant_id="owner")
    assert [int(item["id"]) for item in report["items"]] == [receipt["result"]["memory"]["id"]]
    assert legacy.read_text(encoding="utf-8") == "private-modified-invalid-legacy-rag"


@pytest.mark.parametrize("operation", [
    lambda a: a.ingest("forbidden"),
    lambda a: a.ingest_fast("forbidden"),
    lambda a: a.run_generation("forbidden"),
    lambda a: a.personality_graph(),
    lambda a: a.record_personality_preference("forbidden"),
    lambda a: a.create_checkpoint(),
    lambda a: a.run_maintenance(),
    lambda a: a.sync_projection(object()),
    lambda a: a.deliver_outbox(object()),
    lambda a: a.start_projection_worker(object()),
])
def test_excluded_public_capabilities_reject_without_provider_or_authority_effects(admission, operation):
    _, _, build = admission
    instance = build()
    before = instance._journal.verified_snapshot()
    calls = instance.embedder.calls
    with pytest.raises(ValueError):
        operation(instance)
    assert instance._journal.verified_snapshot() == before
    assert instance.embedder.calls == calls


def test_restart_receipt_retry_preserves_identity_and_detached_candidate_status(admission, monkeypatch):
    _, _, build = admission
    first = build()
    receipt = first.ingest_memory_receipted("accepted memory", tenant_id="owner", idempotency_key="first")
    before = first._journal.verified_snapshot()
    first.close(persist=False)
    second = build()
    monkeypatch.setattr(second.embedder, "embed", unavailable)
    assert second.ingest_memory_receipted("accepted memory", tenant_id="owner", idempotency_key="first") == receipt
    assert second._journal.verified_snapshot() == before
    status = second.production_profile_status()
    assert status["profile_id"] == PROFILE
    assert status["production_ready"] is False and status["validated"] is True
    assert status["status"] == "candidate"
    status.clear()
    assert second.production_profile_status()["profile_id"] == PROFILE


@pytest.mark.parametrize("arguments", [
    {"tenant_id": None}, {"client_id": False}, {"session_id": "é" * 129},
    {"top_k": True}, {"top_k": 1.0}, {"as_of": 10**400},
    {"include_quarantined": "false"}, {"phase": "invented-phase"},
    {"dpm_thread_id": "excluded-personality-graph"},
])
def test_python_scoped_read_rejects_bad_options_before_embedding(admission, monkeypatch, arguments):
    _, _, build = admission
    instance = build()
    before = instance._journal.verified_snapshot()
    monkeypatch.setattr(instance.embedder, "embed", unavailable)
    with pytest.raises(ValueError):
        instance.retrieve_context("accepted query", **{"tenant_id": "owner", **arguments})
    assert instance._journal.verified_snapshot() == before


def test_profile_app_admission_requires_auth_before_factory(admission, monkeypatch):
    config, path, _ = admission
    path.write_text(json.dumps(config), encoding="utf-8")
    for value in (None, "", " ", " padded ", "embedded\nnewline", "embedded\ttab", "non-ascii-é"):
        if value is None:
            monkeypatch.delenv("DML_API_TOKEN", raising=False)
        else:
            monkeypatch.setenv("DML_API_TOKEN", value)
        with pytest.raises(ValueError):
            create_app(config_path=str(path), adapter_factory=unavailable)


def test_nonencodable_credential_is_rejected_before_factory(admission, monkeypatch):
    config, path, _ = admission
    path.write_text(json.dumps(config), encoding="utf-8")
    # A process can inherit undecodable environment bytes on POSIX. Substitute
    # the mapping to exercise the boundary portably without OS encoding rules.
    monkeypatch.setattr(os, "environ", {"DML_API_TOKEN": "\udc80"})
    with pytest.raises(ValueError):
        create_app(config_path=str(path), adapter_factory=unavailable)


def test_provider_credentials_are_frozen_and_excluded_routes_never_initialize(admission, monkeypatch):
    import daystrom_dml.provider_server as server

    _, path, build = admission
    instance = build()
    monkeypatch.setenv("DML_API_TOKEN", TOKEN)
    monkeypatch.setattr(server, "CognitionController", unavailable)
    monkeypatch.setattr(server, "ProceduralLearningPolicy", unavailable)
    app = create_app(config_path=str(path), adapter_factory=lambda: instance)
    assert len(app.routes) == 9
    with TestClient(app) as client:
        monkeypatch.delenv("DML_API_TOKEN")
        assert client.get("/api/contracts").status_code == 401
        assert client.get("/api/contracts", headers=AUTH).status_code == 200
        monkeypatch.setenv("DML_API_TOKEN", "replacement-credential")
        replacement = {"Authorization": "Bearer replacement-credential"}
        assert client.get("/api/contracts", headers=replacement).status_code == 401
        assert client.get("/api/contracts", headers=AUTH).status_code == 200
        for route in ("/api/dcn/policy/export", "/api/frontier/prepare", "/api/embed", "/api/fetch/0", "/openapi.json"):
            response = client.post(route, json={"private": "request-sentinel"}, headers=AUTH)
            assert response.status_code == 404
        assert not hasattr(app.state, "dcn_controller")


@pytest.mark.parametrize("payload", [
    {"query": "private-request-sentinel"},
    {"query": "private-request-sentinel", "tenant_id": 7},
    {"query": "private-request-sentinel", "tenant_id": " "},
    {"query": "private-request-sentinel", "tenant_id": "owner", "client_id": False},
    {"query": "private-request-sentinel", "tenant_id": "owner", "top_k": True},
    {"query": "private-request-sentinel", "tenant_id": "owner", "include_quarantined": True},
])
def test_profile_recall_validation_is_strict_sanitized_and_preprovider(admission, monkeypatch, payload):
    _, path, build = admission
    instance = build()
    monkeypatch.setenv("DML_API_TOKEN", TOKEN)
    monkeypatch.setattr(instance.embedder, "embed", unavailable)
    with TestClient(create_app(config_path=str(path), adapter_factory=lambda: instance)) as client:
        response = client.post("/api/recall", json=payload, headers=AUTH)
    assert response.status_code == 422
    assert response.json() == {"detail": {"code": "profile_validation_failed"}}


def test_healthy_profile_response_does_not_disclose_memory_scope_or_storage(admission, monkeypatch):
    config, path, build = admission
    instance = build()
    instance.ingest_memory_receipted("private-memory-sentinel", tenant_id="private-owner-sentinel",
                                    idempotency_key="secret-key")
    monkeypatch.setenv("DML_API_TOKEN", TOKEN)
    with TestClient(create_app(config_path=str(path), adapter_factory=lambda: instance)) as client:
        health = client.get("/health")
        contract = client.get("/api/contracts", headers=AUTH)
    assert health.status_code == 200
    assert health.json()["production_ready"] is False
    for secret in ("private-memory-sentinel", "private-owner-sentinel", "secret-key", config["storage_dir"], TOKEN):
        assert secret not in health.text + contract.text


@pytest.mark.parametrize("invalid", [float("nan"), object(), "\ud800"])
def test_postcommit_serialization_failure_instructs_same_key_retry(admission, monkeypatch, invalid):
    _, path, build = admission
    instance = build()
    monkeypatch.setenv("DML_API_TOKEN", TOKEN)
    original = instance.ingest_memory_receipted

    def committed_but_invalid_response(**kwargs):
        result = original(**kwargs)
        return {**result, "unserializable": invalid}

    monkeypatch.setattr(instance, "ingest_memory_receipted", committed_but_invalid_response)
    app = create_app(config_path=str(path), adapter_factory=lambda: instance)
    payload = {"text": "accepted before serialization", "tenant_id": "owner", "idempotency_key": "same-key"}
    with TestClient(app) as client:
        response = client.post("/api/remember/receipt", json=payload, headers=AUTH)
        assert response.status_code == 503
        assert response.json() == {"detail": {"code": "receipt_outcome_unavailable", "retry_same_key": True}}
        assert instance._journal.revision == 1
        monkeypatch.setattr(instance, "ingest_memory_receipted", original)
        retry = client.post("/api/remember/receipt", json=payload, headers=AUTH)
        assert retry.status_code == 200
        assert retry.json()["revision"] == 1
        assert instance._journal.revision == 1
