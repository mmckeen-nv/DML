"""HTTP admission and real-journal behavior for the selected candidate profile."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from daystrom_dml.contracts.profile import PROFILE_ID, production_profile_contract
from daystrom_dml.dml_adapter import DMLAdapter
from daystrom_dml.provider_server import create_app
from daystrom_dml.services.receipt_lifecycle import memory_digest


TOKEN = "profile-test-credential"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
SCOPE = {"tenant_id": "owner", "client_id": "client", "session_id": "session", "instance_id": "instance"}


class NativeFixtureEmbedder:
    """Deterministic test provider; this does not qualify a live embedding model."""

    def __init__(self):
        self.calls = 0

    def embed(self, text):
        self.calls += 1
        return np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float32)


@pytest.fixture
def profile_config(tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("DML_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DML_API_TOKEN", TOKEN)
    config = {
        "production_profile": PROFILE_ID,
        "storage_dir": str(tmp_path / "authority"),
        "model_name": "dummy", "llm_backend": "dummy",
        "embedding_model": "fixture-native", "strict_embedding_required": True,
        "strict_llm_required": False,
        "persistence": {"enable": False, "interval_sec": 0, "journal": True,
                        "receipts": True, "outbox": False,
                        "receipt_embedding_identity": "fixture-native-v1"},
        "rag_store": {"enable": False}, "ann_min_items": 0,
        "checkpoint_interval_seconds": 0, "skip_rag_state_import": True,
        "survival_ledger_enabled": False, "background_processing_enabled": False,
        "enable_stm_controller": False, "enable_workflow_cache": False,
        "enable_quality_on_retrieval": False, "mirror_agentic_memory_to_rag": False,
        "gpu_acceleration": False,
        "dpm": {"enable": False, "mode": "disabled", "include_in_context": False,
                "include_in_preamble": False},
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


@pytest.fixture
def profile_app(profile_config):
    adapter = DMLAdapter(config_path=profile_config, embedder=NativeFixtureEmbedder(), start_aging_loop=False)
    app = create_app(config_path=str(profile_config), adapter_factory=lambda: adapter)
    yield app
    adapter.close(persist=False)


def _append(client, *, text="A scoped memory", key="append", scope=None):
    response = client.post("/api/remember/receipt", json={
        **(scope or SCOPE), "text": text, "idempotency_key": key,
        "meta": {"source_trust": "trusted"},
    }, headers=HEADERS)
    assert response.status_code == 200, response.text
    return response.json()


def test_registered_routes_and_contract_are_exact(profile_app):
    expected = {
        ("GET", "/health"), ("GET", "/api/contracts"),
        ("POST", "/api/recall"), ("POST", "/api/remember/receipt"),
        ("POST", "/api/memory/retention/inspect"),
        ("POST", "/api/memory/retire/receipt"),
        ("POST", "/api/memory/supersede/receipt"),
        ("POST", "/api/memory/update/receipt"),
        ("POST", "/api/memory/promote/receipt"),
    }
    actual = {(method, route.path) for route in profile_app.routes for method in route.methods}
    assert actual == expected
    assert {tuple(pair) for pair in production_profile_contract()["apis"]["http"]} == expected
    assert not hasattr(profile_app.state, "dcn_controller")
    with TestClient(profile_app) as client:
        health = client.get("/health").json()
        assert health == {"status": "ok", "profile_id": PROFILE_ID, "maturity": "candidate", "production_ready": False}
        contract = client.get("/api/contracts", headers=HEADERS).json()
        assert contract["profile_id"] == PROFILE_ID
        contract["apis"]["http"].clear()
        assert len(client.get("/api/contracts", headers=HEADERS).json()["apis"]["http"]) == 9


def test_real_http_lifecycle_receipts_replay_and_scope(profile_app):
    with TestClient(profile_app) as client:
        original_receipt = _append(client)
        original = original_receipt["result"]["memory"]
        foreign = _append(client, text="Foreign secret", key="foreign", scope={"tenant_id": "foreign"})
        recall = client.post("/api/recall", json={**SCOPE, "query": "memory", "top_k": 10}, headers=HEADERS)
        assert recall.status_code == 200
        assert {int(item["id"]) for item in recall.json()["items"]} == {original["id"]}
        assert str(foreign["result"]["memory"]["text"]) not in recall.text

        update = client.post("/api/memory/update/receipt", json={
            **SCOPE, "memory_id": original["id"], "expected_memory_digest": memory_digest(original),
            "reason": "Correction", "idempotency_key": "update", "text": "Corrected memory",
        }, headers=HEADERS)
        assert update.status_code == 200, update.text
        updated = update.json()["result"]["memory"]

        replacement = _append(client, text="Replacement", key="replacement")["result"]["memory"]
        supersede = client.post("/api/memory/supersede/receipt", json={
            **SCOPE, "memory_id": updated["id"], "expected_memory_digest": memory_digest(updated),
            "replacement_memory_id": replacement["id"], "expected_replacement_digest": memory_digest(replacement),
            "reason": "Superseded", "idempotency_key": "supersede",
        }, headers=HEADERS)
        assert supersede.status_code == 200, supersede.text

        promotion = client.post("/api/memory/promote/receipt", json={
            **SCOPE, "sources": [{"memory_id": replacement["id"], "expected_memory_digest": memory_digest(replacement)}],
            "text": "Derived memory", "reason": "Operator derivation", "idempotency_key": "promote",
        }, headers=HEADERS)
        assert promotion.status_code == 200, promotion.text

        retire = client.post("/api/memory/retire/receipt", json={
            **SCOPE, "memory_id": replacement["id"], "expected_memory_digest": memory_digest(replacement),
            "reason": "Retired", "idempotency_key": "retire",
        }, headers=HEADERS)
        assert retire.status_code == 200, retire.text
        inspection = client.post("/api/memory/retention/inspect", json={**SCOPE, "memory_id": original["id"]}, headers=HEADERS)
        assert inspection.status_code == 200
        assert inspection.json()["erasure_proven"] is False
        assert inspection.json()["known_reference_count"] > 0
        assert _append(client) == original_receipt

        final = client.post("/api/recall", json={**SCOPE, "query": "memory"}, headers=HEADERS)
        assert final.status_code == 200
        ids = {int(item["id"]) for item in final.json()["items"]}
        assert original["id"] not in ids and replacement["id"] not in ids
        assert promotion.json()["result"]["memory"]["id"] in ids


def test_auth_is_checked_before_factory_and_cannot_fail_open(profile_config, monkeypatch):
    monkeypatch.delenv("DML_API_TOKEN")
    calls = []
    with pytest.raises(ValueError, match="nonempty"):
        create_app(config_path=str(profile_config), adapter_factory=lambda: calls.append("called"))
    assert calls == []
    assert not (profile_config.parent / "authority").exists()


def test_credentials_are_frozen_across_environment_changes(profile_app, monkeypatch):
    monkeypatch.delenv("DML_API_TOKEN")
    monkeypatch.setenv("DML_ADMIN_TOKEN", "replacement-credential")
    with TestClient(profile_app) as client:
        assert client.get("/api/contracts").status_code == 401
        assert client.get("/api/contracts", headers={"Authorization": "Bearer replacement-credential"}).status_code == 401
        assert client.get("/api/contracts", headers=HEADERS).status_code == 200


def test_invalid_config_precedes_factory(profile_config):
    config = json.loads(profile_config.read_text(encoding="utf-8"))
    config["ann_min_items"] = 1
    profile_config.write_text(json.dumps(config), encoding="utf-8")
    calls = []
    with pytest.raises(ValueError):
        create_app(config_path=str(profile_config), adapter_factory=lambda: calls.append("called"))
    assert calls == []
    assert not (profile_config.parent / "authority").exists()


def test_legacy_factory_cannot_be_relabelled(profile_config):
    class LegacyAdapter:
        production_profile_id = None

    with pytest.raises(ValueError, match="does not match"):
        create_app(config_path=str(profile_config), adapter_factory=LegacyAdapter)


def test_factory_configuration_and_runtime_evidence_must_match(profile_app, profile_config, monkeypatch):
    adapter = profile_app.state.adapter
    with pytest.raises(ValueError, match="settings do not match"):
        create_app(config_path=str(profile_config), storage_dir=str(profile_config.parent / "other"),
                   adapter_factory=lambda: adapter)
    status = adapter.production_profile_status()
    status["authority"]["outbox_enabled"] = True
    monkeypatch.setattr(adapter, "production_profile_status", lambda: status)
    with pytest.raises(ValueError):
        create_app(config_path=str(profile_config), adapter_factory=lambda: adapter)


def test_normal_builder_uses_one_validated_configuration_snapshot(profile_config, monkeypatch):
    import daystrom_dml.dml_adapter as adapter_module

    def reread_forbidden(*_args, **_kwargs):
        raise AssertionError("Provider reread configuration after admission")

    monkeypatch.setattr(adapter_module, "load_config", reread_forbidden)
    monkeypatch.setattr(adapter_module, "create_embedder", lambda *_args, **_kwargs: NativeFixtureEmbedder())
    directory = profile_config.parent / "explicit-storage"
    app = create_app(config_path=str(profile_config), storage_dir=str(directory))
    with TestClient(app) as client:
        receipt = _append(client)
        assert receipt["revision"] == 1
        assert app.state.adapter.storage_dir == directory
    assert not (profile_config.parent / "authority").exists()
    assert (directory / "dml_state.sqlite3").is_file()


def test_legacy_server_refuses_profile_before_adapter(profile_config):
    environment = dict(os.environ)
    environment["DML_CONFIG_PATH"] = str(profile_config)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, "-c", "import daystrom_dml.server"],
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "only through dml-provider" in result.stderr
    assert not (profile_config.parent / "authority").exists()
