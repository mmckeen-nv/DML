"""Independent public-boundary oracles for the frozen local receipt profile.

The accepted configuration and expected operation history are handwritten here;
they are not generated from the profile validator or its advertised inventory.
"""
from __future__ import annotations

from contextlib import closing
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import pytest

import daystrom_dml.dml_adapter as adapter_module
import daystrom_dml.memory_store as memory_store_module
from daystrom_dml.dml_adapter import DMLAdapter
from daystrom_dml.embeddings import RandomEmbedder, SentenceTransformerEmbedder
from daystrom_dml.journal import IdempotencyConflict, JournalStateStore
from daystrom_dml.services.outbox_migration import upgrade_outbox_journal
from daystrom_dml.services.receipt_ingestion import ReceiptEmbeddingCompatibilityError
from daystrom_dml.services.receipt_lifecycle import ReceiptMemoryNotFound
from daystrom_dml.vector_backend import NumpyVectorBackend


PROFILE = "dml-receipted-local-v1"
SCOPE = {"tenant_id": "owner", "client_id": "client", "session_id": "session", "instance_id": "instance"}


class PinnedEmbedder:
    """A local test double with an operator-declared, non-random embedding space."""

    model_name = "profile-integration-embedding"

    def __init__(self, *, unavailable=False):
        self.calls = []
        self.unavailable = unavailable

    def embed(self, text):
        if self.unavailable:
            raise AssertionError("Historical replay contacted the embedding backend")
        self.calls.append(text)
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def supported_config(directory, *, outbox=False):
    return {
        "production_profile": PROFILE,
        "storage_dir": str(directory),
        "model_name": "dummy",
        "llm_backend": "dummy",
        "embedding_model": PinnedEmbedder.model_name,
        "strict_embedding_required": True,
        "persistence": {
            "enable": False, "interval_sec": 0, "journal": True, "receipts": True,
            "outbox": outbox, "receipt_embedding_identity": "operator-pinned-test-revision-v1",
            "snapshot_interval": 1,
        },
        "rag_store": {"enable": False},
        "dpm": {"enable": False, "mode": "disabled", "include_in_context": False,
                "include_in_preamble": False},
        "ann_min_items": 0,
        "checkpoint_interval_seconds": 0,
        "skip_rag_state_import": True,
        "survival_ledger_enabled": False,
        "background_processing_enabled": False,
        "enable_stm_controller": False,
        "enable_workflow_cache": False,
        "enable_quality_on_retrieval": False,
        "mirror_agentic_memory_to_rag": False,
        "gpu_acceleration": False,
        "metrics_enabled": False,
    }


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("DML_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def factory(tmp_path):
    instances = []

    def build(*, directory=None, outbox=False, embedder=None, config=None):
        instance = DMLAdapter(
            config_path=tmp_path / "absent-config.yaml",
            config_overrides=config or supported_config(directory or tmp_path / "authority", outbox=outbox),
            embedder=embedder if embedder is not None else PinnedEmbedder(),
            start_aging_loop=True,
        )
        instances.append(instance)
        return instance

    yield build
    for instance in reversed(instances):
        instance.close()


def full_record_digest(record):
    """Independent wire-format oracle, without importing the implementation hash."""
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sql_history(path):
    """Observe logical authority directly, independent of adapter hydration."""
    with closing(sqlite3.connect(f"{Path(path).as_uri()}?mode=ro", uri=True)) as connection:
        return {
            "schema": connection.execute("PRAGMA user_version").fetchone()[0],
            "revision": connection.execute("SELECT revision FROM state WHERE id=1").fetchone()[0],
            "records": [json.loads(row[0]) for row in connection.execute(
                "SELECT payload FROM records WHERE bucket='items' ORDER BY position")],
            "receipts": [json.loads(row[0]) for row in connection.execute(
                "SELECT payload FROM receipts ORDER BY revision")],
            "decisions": [json.loads(row[0]) for row in connection.execute(
                "SELECT payload FROM decisions ORDER BY revision")],
        }


def files_under(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*")
            if path.is_file() and not path.name.endswith(("-shm", ".lock"))
            and not (path.name.endswith("-wal") and path.stat().st_size == 0)}


def refuse_effect(*_args, **_kwargs):
    raise AssertionError("Rejected configuration reached a constructor with effects")


@pytest.mark.parametrize("path,value", [
    (("production_profile",), "production"),
    (("persistence", "journal"), False),
    (("persistence", "receipts"), "yes"),
    (("persistence", "receipts_typo"), True),
    (("capacity",), True),
    (("storage_dir",), None),
    (("rag_store", "enable"), True),
    (("dpm", "mode"), "active-read"),
    (("ann_min_items",), 1),
    (("agentic_mode",), {"enabled": "false"}),
    (("dml.agentic_mode.enabled",), True),
    (("skip_rag_state_import",), False),
])
def test_invalid_candidate_is_rejected_before_model_or_storage_effects(tmp_path, monkeypatch, path, value):
    directory = tmp_path / "must-remain-absent"
    config = supported_config(directory)
    target = config
    for member in path[:-1]:
        target = target.setdefault(member, {})
    target[path[-1]] = value
    for name in ("GPTRunner", "create_embedder", "PersonalityMatrix", "JournalStateStore", "CheckpointManager"):
        monkeypatch.setattr(adapter_module, name, refuse_effect)
    with pytest.raises(ValueError):
        DMLAdapter(config_path=tmp_path / "absent.yaml", config_overrides=config)
    assert not directory.exists()
    assert set(tmp_path.iterdir()) == set()


@pytest.mark.parametrize("kind", ["random", "random-subclass", "sentence-transformer-fallback"])
def test_runtime_fallback_embedder_is_refused_before_storage_creation(tmp_path, kind):
    class RandomSubclass(RandomEmbedder):
        pass

    if kind == "sentence-transformer-fallback":
        embedder = object.__new__(SentenceTransformerEmbedder)
        embedder.model_name = "unavailable-model"
        embedder._model = None
        embedder._dim = 4
    else:
        embedder = (RandomEmbedder if kind == "random" else RandomSubclass)(dim=4)
    directory = tmp_path / "authority"
    with pytest.raises((ValueError, RuntimeError)):
        DMLAdapter(config_path=tmp_path / "absent.yaml",
                   config_overrides=supported_config(directory), embedder=embedder)
    assert not directory.exists()


@pytest.mark.parametrize("schema,outbox", [(1, False), (2, True), (3, False), (4, False)])
def test_existing_schema_mismatch_is_rejected_before_effects(tmp_path, monkeypatch, schema, outbox):
    directory = tmp_path / "authority"
    path = directory / "dml_state.sqlite3"
    if schema == 4:
        source = JournalStateStore(tmp_path / "source" / "dml_state.sqlite3", receipt_mode=True)
        upgrade_outbox_journal(source.path, path)
    else:
        JournalStateStore(path, receipt_mode=schema != 1, outbox_mode=schema == 3)
    before = files_under(directory)
    for name in ("GPTRunner", "create_embedder", "PersonalityMatrix", "CheckpointManager"):
        monkeypatch.setattr(adapter_module, name, refuse_effect)
    with pytest.raises(ValueError):
        DMLAdapter(config_path=tmp_path / "absent.yaml",
                   config_overrides=supported_config(directory, outbox=outbox))
    assert files_under(directory) == before


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_supported_lifecycle_history_scope_and_restart(factory, tmp_path, schema):
    directory = tmp_path / "authority"
    if schema == 4:
        source = JournalStateStore(tmp_path / "migration-source" / "dml_state.sqlite3", receipt_mode=True)
        upgrade_outbox_journal(source.path, directory / "dml_state.sqlite3")
    instance = factory(directory=directory, outbox=schema in (3, 4))
    path = directory / "dml_state.sqlite3"
    initial_revision = sql_history(path)["revision"]
    calls = []

    def operation(method, *args, **kwargs):
        receipt = getattr(instance, method)(*args, **kwargs)
        calls.append((method, args, deepcopy(kwargs), deepcopy(receipt)))
        assert receipt["schema_version"] == 1
        assert receipt["revision"] == initial_revision + len(calls)
        return receipt["result"]["memory"]

    original = operation("ingest_memory_receipted", "The owner prefers blue notebooks",
        idempotency_key="source", meta={"source_trust": "trusted"}, **SCOPE)
    replacement = operation("ingest_memory_receipted", "The owner prefers green notebooks",
        idempotency_key="replacement", meta={"source_trust": "trusted"}, **SCOPE)
    updated = operation("update_memory_receipted", original["id"], text="The owner uses blue notebooks",
        expected_memory_digest=full_record_digest(original), reason="Owner corrected the wording",
        idempotency_key="update", **SCOPE)
    derived = operation("promote_memories_receipted",
        [{"memory_id": updated["id"], "expected_memory_digest": full_record_digest(updated)}],
        text="Notebook summary approved by the owner", reason="Explicit first-level derivation",
        idempotency_key="promotion", **SCOPE)
    superseded = operation("supersede_memory_receipted", updated["id"],
        replacement_memory_id=replacement["id"], expected_memory_digest=full_record_digest(updated),
        expected_replacement_digest=full_record_digest(replacement), reason="Owner selected replacement",
        idempotency_key="supersession", **SCOPE)
    retired = operation("retire_memory_receipted", replacement["id"],
        expected_memory_digest=full_record_digest(replacement), reason="Owner withdrew replacement",
        idempotency_key="retirement", **SCOPE)
    other_scope = {**SCOPE, "tenant_id": "other-owner"}
    other = operation("ingest_memory_receipted", "private-other-owner-only-text",
        idempotency_key="source", **other_scope)

    expected_source = deepcopy(updated)
    expected_source["meta"].update({
        "memory_state": "superseded", "superseded_by": replacement["id"],
        "supersession_decision": {
            "schema_version": "dml-supersession-decision-v1",
            "prior_memory_digest": full_record_digest(updated),
            "replacement_memory_id": replacement["id"],
            "replacement_memory_digest": full_record_digest(replacement),
            "reason": "Owner selected replacement",
        },
    })
    assert superseded == expected_source
    expected_replacement = deepcopy(replacement)
    expected_replacement["meta"].update({
        "memory_state": "deleted", "retirement_decision": {
            "schema_version": "dml-retirement-decision-v1",
            "prior_memory_digest": full_record_digest(replacement),
            "reason": "Owner withdrew replacement",
        },
    })
    assert retired == expected_replacement
    history = sql_history(path)
    assert history["schema"] == schema
    assert history["revision"] == initial_revision + 7
    assert history["records"] == [expected_source, expected_replacement, derived, other]
    assert history["receipts"] == [entry[3] for entry in calls]
    assert [entry["operation"] for entry in history["decisions"][-7:]] == [
        "append-receipt-v1", "append-receipt-v1", "update-receipt-v1", "promote-receipt-v1",
        "supersede-receipt-v1", "retire-receipt-v1", "append-receipt-v1",
    ]

    report = instance.retrieve_context("notebooks", **SCOPE)
    assert {int(item["id"]) for item in report["items"]} == {derived["id"]}
    assert "private-other-owner-only-text" not in report["raw_context"]
    for member in SCOPE:
        assert instance.retrieve_context("notebooks", **{**SCOPE, member: "elsewhere"})["items"] == []
    retention = instance.inspect_memory_retention(original["id"], **SCOPE)
    assert retention["source"]["revision"] == history["revision"]
    assert retention["source"]["journal_schema_version"] == schema
    assert retention["surfaces"]["receipts"]["direct_records"] >= 3
    assert retention["physical_erasure_supported"] is False
    assert retention["erasure_proven"] is False
    with pytest.raises(ReceiptMemoryNotFound):
        instance.inspect_memory_retention(original["id"], **other_scope)
    with pytest.raises(IdempotencyConflict):
        instance.ingest_memory_receipted("changed-content", idempotency_key="source", **SCOPE)
    assert sql_history(path) == history
    assert instance.durability_status() == {"status": "ok", "failures": {}}

    instance.close()
    unavailable = PinnedEmbedder(unavailable=True)
    restarted = factory(directory=directory, outbox=schema in (3, 4), embedder=unavailable)
    for method, args, kwargs, receipt in calls:
        assert getattr(restarted, method)(*args, **kwargs) == receipt
    assert unavailable.calls == []
    assert sql_history(path) == history
    assert restarted.inspect_memory_retention(original["id"], **SCOPE) == retention


def test_legacy_auxiliary_files_do_not_enter_candidate_startup_or_refresh(factory, tmp_path):
    directory = tmp_path / "authority"
    directory.mkdir()
    rag_path = directory / "rag_store.json"
    rag_path.write_text("invalid legacy RAG must remain outside the candidate", encoding="utf-8")
    # The old loader removes a duplicate storage-directory basename, then copies
    # this nested legacy source to authority/state.jsonl before opening a journal.
    legacy_copy = directory / directory.name / "state.jsonl"
    legacy_copy.parent.mkdir()
    legacy_copy.write_text("must never be copied", encoding="utf-8")
    config = supported_config(directory)
    config["persistence"]["path"] = f"{directory.name}/state.jsonl"
    instance = factory(config=config)
    receipt = instance.ingest_memory_receipted("The owner remembers a notebook", idempotency_key="one", **SCOPE)
    assert not (directory / "state.jsonl").exists()
    assert rag_path.read_text(encoding="utf-8").startswith("invalid legacy")
    assert instance.retrieve_context("notebook", **SCOPE)["items"]
    rag_path.write_text("a different invalid legacy RAG generation", encoding="utf-8")
    assert instance.retrieve_context("notebook after auxiliary change", **SCOPE)["items"]
    instance.close()
    restarted = factory(config=config)
    assert restarted.ingest_memory_receipted("The owner remembers a notebook", idempotency_key="one", **SCOPE) == receipt
    assert restarted.retrieve_context("notebook after restart", **SCOPE)["items"]
    assert not (directory / "state.jsonl").exists()


@pytest.mark.parametrize("method,args,kwargs", [
    ("ingest", ("legacy-write",), {}),
    ("ingest_memory", ("legacy-write",), {"tenant_id": "owner"}),
    ("query_database", ("legacy-read",), {}),
    ("run_generation", ("legacy-generation",), {}),
    ("build_preamble", ("legacy-preamble",), {}),
    ("personality_graph", (), {}),
    ("get_stm_state", (), {}),
    ("create_checkpoint", (), {}),
    ("run_maintenance", (), {}),
    ("deliver_outbox", (object(),), {}),
    ("sync_projection", (object(),), {}),
    ("start_projection_worker", (object(),), {}),
])
def test_excluded_public_surfaces_reject_without_authority_or_backend_work(factory, method, args, kwargs):
    instance = factory(outbox=True)
    path = instance.storage_dir / "dml_state.sqlite3"
    before = sql_history(path)
    instance.embedder.unavailable = True
    with pytest.raises(ValueError):
        getattr(instance, method)(*args, **kwargs)
    assert sql_history(path) == before
    assert instance.embedder.calls == []


def test_legacy_configuration_remains_explicitly_available(factory, tmp_path):
    config = supported_config(tmp_path / "legacy")
    config.pop("production_profile")
    config["capacity"] = "12"
    config["strict_embedding_required"] = False
    config["persistence"].update(journal=False, receipts=False)
    instance = factory(config=config, embedder=RandomEmbedder(dim=4))
    assert instance.store.capacity == 12
    status = instance.production_profile_status()
    assert status["profile_id"] is None
    assert status["status"] == "unselected"
    assert status["validated"] is False
    assert status["production_ready"] is False
    memory = instance.ingest_memory("A legacy public memory", tenant_id="owner")
    assert memory.text == "A legacy public memory"
    assert (instance.storage_dir / "dml_store.json").exists()


def test_profile_status_is_a_detached_candidate_claim(factory):
    instance = factory()
    receipt = instance.ingest_memory_receipted("A status fixture", idempotency_key="one", **SCOPE)
    status = instance.production_profile_status()
    assert instance.production_profile_id == PROFILE
    assert status["profile_id"] == PROFILE
    assert status["status"] == "candidate"
    assert status["production_ready"] is False
    assert status["validated"] is True
    assert status["authority"] == {"journal_schema_version": 2, "outbox_enabled": False}
    assert status["embedding_identity"]["revision"] == "operator-pinned-test-revision-v1"
    assert status["embedding_identity"]["mode"] == "native"
    expected = deepcopy(status)
    status["authority"]["journal_schema_version"] = 999
    status["embedding_identity"]["revision"] = "caller-edit"
    assert instance.production_profile_status() == expected
    assert instance.ingest_memory_receipted("A status fixture", idempotency_key="one", **SCOPE) == receipt


def test_selected_profile_never_starts_mutating_background_workers(factory, monkeypatch):
    monkeypatch.setattr(adapter_module.MemoryStore, "start_aging", refuse_effect)
    monkeypatch.setattr(DMLAdapter, "_start_persistence_loop", refuse_effect)
    monkeypatch.setattr(adapter_module.CheckpointManager, "start", refuse_effect)
    instance = factory()
    assert instance.store._aging_thread is None
    assert instance._persistence_thread is None
    assert instance.checkpoint_manager is None or instance.checkpoint_manager._thread is None
    assert instance.persistent_rag_store is None
    assert instance.agentic_router is None
    assert instance.stm_controller is None


def test_candidate_exact_search_does_not_consult_global_accelerator_selection(factory, monkeypatch):
    monkeypatch.setattr(memory_store_module, "get_vector_backend", refuse_effect)
    instance = factory()
    assert type(instance.store._vector_backend) is NumpyVectorBackend
    receipt = instance.ingest_memory_receipted("The owner uses a notebook", idempotency_key="one", **SCOPE)
    report = instance.retrieve_context("notebook", **SCOPE)
    assert [int(item["id"]) for item in report["items"]] == [receipt["result"]["memory"]["id"]]


def test_candidate_does_not_construct_personality_or_semantic_checkpoint_components(factory, monkeypatch):
    monkeypatch.setattr(adapter_module, "PersonalityMatrix", refuse_effect)
    monkeypatch.setattr(adapter_module, "CheckpointManager", refuse_effect)
    instance = factory()
    assert instance.personality_matrix is None
    assert instance.checkpoint_manager is None
    assert instance.production_profile_status()["validated"] is True


def test_committed_embedding_identity_mismatch_cannot_complete_candidate_startup(factory, tmp_path):
    directory = tmp_path / "authority"
    instance = factory(directory=directory)
    instance.ingest_memory_receipted("A notebook fact", idempotency_key="one", **SCOPE)
    instance.close()
    before = sql_history(directory / "dml_state.sqlite3")
    incompatible = supported_config(directory)
    incompatible["persistence"]["receipt_embedding_identity"] = "different-operator-pinned-revision"
    embedder = PinnedEmbedder(unavailable=True)
    with pytest.raises(ReceiptEmbeddingCompatibilityError):
        factory(config=incompatible, embedder=embedder)
    assert embedder.calls == []
    assert sql_history(directory / "dml_state.sqlite3") == before


@pytest.mark.parametrize("operation", ["append", "update", "promote", "retrieve"])
def test_live_fallback_replacement_blocks_new_embedding_work_but_preserves_history(factory, monkeypatch, operation):
    instance = factory()
    receipt = instance.ingest_memory_receipted("A trusted notebook fact", idempotency_key="one",
        meta={"source_trust": "trusted"}, **SCOPE)
    record = receipt["result"]["memory"]
    before = sql_history(instance.storage_dir / "dml_state.sqlite3")
    fallback = RandomEmbedder(dim=4)
    monkeypatch.setattr(fallback, "embed", refuse_effect)
    instance.embedder = fallback
    assert instance.ingest_memory_receipted("A trusted notebook fact", idempotency_key="one",
        meta={"source_trust": "trusted"}, **SCOPE) == receipt
    with pytest.raises((ValueError, ReceiptEmbeddingCompatibilityError)):
        if operation == "append":
            instance.ingest_memory_receipted("A new fact", idempotency_key="two", **SCOPE)
        elif operation == "update":
            instance.update_memory_receipted(record["id"], text="A corrected fact",
                expected_memory_digest=full_record_digest(record), reason="Owner correction",
                idempotency_key="two", **SCOPE)
        elif operation == "promote":
            instance.promote_memories_receipted([
                {"memory_id": record["id"], "expected_memory_digest": full_record_digest(record)}],
                text="Approved summary", reason="Owner derivation", idempotency_key="two", **SCOPE)
        else:
            instance.retrieve_context("notebook", **SCOPE)
    assert sql_history(instance.storage_dir / "dml_state.sqlite3") == before
