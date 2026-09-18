"""Direct admission oracles for the finite candidate production boundary."""
from copy import deepcopy
from pathlib import Path

import pytest

from daystrom_dml.contracts import profile
from daystrom_dml.contracts.production import production_status
from daystrom_dml.embeddings import RandomEmbedder, SentenceTransformerEmbedder
from daystrom_dml.settings import DMLSettings


def valid_config():
    # Literal operator configuration; deliberately not derived from the contract.
    return {
        "production_profile": "dml-receipted-local-v1",
        "model_name": "dummy", "llm_backend": "dummy",
        "strict_embedding_required": True,
        "persistence": {"enable": False, "interval_sec": 0, "journal": True,
                        "receipts": True, "receipt_embedding_identity": "operator-fixed-v1"},
        "rag_store": {"enable": False},
        "skip_rag_state_import": True,
        "survival_ledger_enabled": False,
        "background_processing_enabled": False,
        "dpm": {"enable": False, "mode": "disabled",
                "include_in_context": False, "include_in_preamble": False},
    }


def change(config, path, value):
    parts = path.split("/")
    container = config
    for part in parts[:-1]:
        container = container.setdefault(part, {})
    container[parts[-1]] = value


def test_raw_and_resolved_configuration_admitted_without_changing_inputs():
    config = valid_config()
    original = deepcopy(config)
    assert profile.validate_profile_config(config) == "dml-receipted-local-v1"
    assert config == original
    assert profile.validate_profile_config(DMLSettings(**config).as_dict()) == profile.PROFILE_ID


def test_default_mode_keeps_legacy_coercion_and_extras_outside_this_validator():
    assert profile.validate_profile_config({"capacity": "12", "anything": object()}) is None
    assert profile.validate_profile_config({"production_profile": None, "persistence": None}) is None


@pytest.mark.parametrize("selector", ["", " ", "candidate", "DML-receipted-local-v1", False, 0, [], {}])
def test_unknown_profile_never_silently_falls_back(selector):
    with pytest.raises(profile.ProductionProfileError, match="production_profile"):
        profile.validate_profile_config({"production_profile": selector})


@pytest.mark.parametrize("path,value", [
    ("capacity", True), ("capacity", 1.0), ("capacity", "12"), ("capacity", 0),
    ("capacity", -1), ("ann_min_items", False), ("ann_min_items", "0"),
    ("top_k", True), ("token_budget", 0), ("scope_cache_bytes", -1),
    ("beta_a", float("nan")), ("beta_a", float("inf")), ("beta_a", "0.08"),
    ("beta_a", True), ("beta_a", 10 ** 400),
    ("similarity_threshold", 1.1), ("storage_dir", None), ("storage_dir", {}),
    ("storage_dir", " "), ("persistence", None), ("persistence", []),
    ("persistence/receipts", "true"), ("persistence/receipts", "yes"),
    ("persistence/receipts", 1), ("persistence/snapshot_interval", 0),
    ("persistence/path", {}), ("rag_store", False), ("dpm/enable", "false"),
    ("dpm/include_in_context", 0), ("literal/max_snippets", 0),
    ("budgets/semantic_pct", -0.1), ("budgets/semantic_pct", 1.1),
    ("budgets/free_pct", 0.5), ("dpm/overlay_path", {}),
])
def test_raw_type_range_and_budget_rejections(path, value):
    config = valid_config()
    change(config, path, value)
    with pytest.raises(profile.ProductionProfileError):
        profile.validate_profile_config(config)


@pytest.mark.parametrize("path,value", [
    ("model_name", "real-model"), ("llm_backend", "auto"),
    ("strict_embedding_required", False), ("strict_llm_required", True),
    ("persistence/enable", True), ("persistence/interval_sec", 1),
    ("persistence/journal", False), ("persistence/receipts", False),
    ("rag_store/enable", True), ("ann_min_items", 1),
    ("checkpoint_interval_seconds", 1), ("skip_rag_state_import", False),
    ("survival_ledger_enabled", True), ("background_processing_enabled", True),
    ("enable_stm_controller", True), ("enable_workflow_cache", True),
    ("enable_quality_on_retrieval", True), ("mirror_agentic_memory_to_rag", True),
    ("gpu_acceleration", True), ("dpm/enable", True), ("dpm/mode", "observe-only"),
    ("dpm/include_in_context", True), ("dpm/include_in_preamble", True),
    ("agentic_mode/enabled", True), ("agentic_mode/router/enabled", True),
    ("router/enabled", True), ("dml/agentic_mode/enabled", True),
    ("dml/agentic_mode/router/enabled", True), ("dml/router/enabled", True),
    ("dml.agentic_mode.enabled", True), ("dml.router.enabled", True),
    ("agentic_mode/enabled", "false"), ("dml.agentic_mode.enabled", "false"),
])
def test_excluded_features_are_rejected_instead_of_silently_disabled(path, value):
    config = valid_config()
    change(config, path, value)
    with pytest.raises(profile.ProductionProfileError):
        profile.validate_profile_config(config)


@pytest.mark.parametrize("path", [
    "unknown", "k", "persistence/reciepts", "rag_store/enabel", "literal/extra",
    "budgets/extra", "dpm/extra", "agentic_mode/extra", "agentic_mode/router/profile",
    "router/profile", "dml/extra", "dml/agentic_mode/extra", "dml/router/profile",
])
def test_unknown_and_nested_misspelled_keys_are_rejected(path):
    config = valid_config()
    change(config, path, False)
    with pytest.raises(profile.ProductionProfileError, match="Unknown"):
        profile.validate_profile_config(config)


def test_inert_legacy_aliases_and_valid_declared_types_are_admitted():
    config = valid_config()
    config.update({"storage_dir": Path("local"), "beta_a": 1, "K": 2,
                   "agentic_mode": {"enabled": False, "router": {"enabled": False}},
                   "router": {"enabled": False}, "dml.agentic_mode.enabled": False,
                   "dml.router.enabled": False,
                   "dml": {"agentic_mode": {"enabled": False, "router": {"enabled": False}},
                           "router": {"enabled": False}}})
    assert profile.validate_profile_config(config) == profile.PROFILE_ID


@pytest.mark.parametrize("identity", [None, "", " ", 42, "é" * 513, "\ud800"])
def test_embedding_identity_is_explicit_bounded_utf8(identity):
    config = valid_config()
    config["persistence"]["receipt_embedding_identity"] = identity
    with pytest.raises(profile.ProductionProfileError):
        profile.validate_profile_config(config)


def test_invalid_raw_config_does_not_construct_settings(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Pydantic construction happened during admission")

    monkeypatch.setattr(DMLSettings, "__init__", forbidden)
    assert profile.validate_profile_config(valid_config()) == profile.PROFILE_ID
    config = valid_config()
    config["rag_store"]["extra"] = True
    with pytest.raises(profile.ProductionProfileError):
        profile.validate_profile_config(config)


class NativeEmbedder:
    model_name = "fixed-test-model"

    def embed(self, text):
        raise AssertionError("Admission must not invoke provider I/O")


def test_native_provider_admission_requires_no_model_access():
    profile.validate_profile_embedding(profile.PROFILE_ID, NativeEmbedder(), "operator-fixed-v1")
    profile.validate_profile_embedding(None, object(), None)


@pytest.mark.parametrize("provider", [RandomEmbedder(), object()])
def test_non_native_provider_is_rejected(provider):
    with pytest.raises(profile.ProductionProfileError):
        profile.validate_profile_embedding(profile.PROFILE_ID, provider, "operator-fixed-v1")


def test_model_provider_fallback_is_rejected_without_loading_model():
    fallback = object.__new__(SentenceTransformerEmbedder)
    fallback._model = None
    with pytest.raises(profile.ProductionProfileError, match="Fallback"):
        profile.validate_profile_embedding(profile.PROFILE_ID, fallback, "operator-fixed-v1")


@pytest.mark.parametrize("schema,outbox", [(2, False), (3, True), (4, True)])
def test_authority_selection_has_exact_admitted_combinations(schema, outbox):
    profile.validate_profile_authority(profile.PROFILE_ID, schema, outbox)


@pytest.mark.parametrize("schema,outbox", [
    (0, False), (1, False), (2, True), (3, False), (4, False), (5, True),
    (True, False), (2.0, False), ("2", False), (2, 0), (3, "true"),
])
def test_other_authority_combinations_never_silently_adopt_a_format(schema, outbox):
    with pytest.raises(profile.ProductionProfileError):
        profile.validate_profile_authority(profile.PROFILE_ID, schema, outbox)


@pytest.mark.parametrize("system", ["Linux", "Darwin", "Windows"])
@pytest.mark.parametrize("version", [(3, 10), (3, 11), (3, 12), (3, 13)])
def test_admitted_platforms_remain_candidate_targets(monkeypatch, system, version):
    monkeypatch.setattr(profile.platform, "system", lambda: system)
    monkeypatch.setattr(profile.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(profile.sys, "version_info", (*version, 0))
    profile.validate_profile_platform(profile.PROFILE_ID)


@pytest.mark.parametrize("system,implementation,version", [
    ("FreeBSD", "CPython", (3, 12)), ("Linux", "PyPy", (3, 12)),
    ("Linux", "CPython", (3, 9)), ("Linux", "CPython", (3, 14)),
])
def test_unadmitted_runtimes_fail_explicitly(monkeypatch, system, implementation, version):
    monkeypatch.setattr(profile.platform, "system", lambda: system)
    monkeypatch.setattr(profile.platform, "python_implementation", lambda: implementation)
    monkeypatch.setattr(profile.sys, "version_info", (*version, 0))
    with pytest.raises(profile.ProductionProfileError):
        profile.validate_profile_platform(profile.PROFILE_ID)


def test_manifest_and_status_are_detached_and_preserve_open_qualification_gates():
    manifest = profile.production_profile_contract()
    assert manifest["profile_id"] == "dml-receipted-local-v1"
    assert manifest["maturity"] == "candidate"
    assert manifest["production_ready"] is False
    assert manifest["qualification_pending"] is True
    assert len(manifest["apis"]["http"]) == 9
    assert manifest["authority"]["journal_schema_versions"] == [2, 3, 4]
    assert manifest["platform"]["power_loss_qualified"] is False
    manifest["configuration"]["required_values"]["persistence.journal"] = False
    manifest["apis"]["python"].append("invented")
    assert profile.production_profile_contract()["configuration"]["required_values"]["persistence.journal"] is True
    assert "invented" not in profile.production_profile_contract()["apis"]["python"]
    status = production_status()
    assert status["production_ready"] is False
    assert status["stable"] == []
    assert len(status["remaining_release_gates"]) == status["remaining_first_release_milestones"] == 9
    assert len(status["deferred_milestones"]) == status["remaining_deferred_milestones"] == 2
    status["supported_profiles"][0]["platform"]["power_loss_qualified"] = True
    assert production_status()["supported_profiles"][0]["platform"]["power_loss_qualified"] is False
