"""Profile admission precedes coercion and preserves configuration precedence."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path

import pytest
import yaml

from daystrom_dml.config import load_config
from daystrom_dml.contracts.profile import PROFILE_ID, ProductionProfileError
from daystrom_dml.settings import DMLSettings


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    for key in tuple(os.environ):
        if key.startswith("DML_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def configuration(tmp_path):
    return {
        "production_profile": PROFILE_ID,
        "storage_dir": str(tmp_path / "uncreated-store"),
        "model_name": "dummy",
        "llm_backend": "dummy",
        "embedding_model": None,
        "strict_embedding_required": True,
        "persistence": {
            "enable": False,
            "interval_sec": 0,
            "journal": True,
            "receipts": True,
            "receipt_embedding_identity": "operator-pinned-revision-v1",
        },
        "rag_store": {"enable": False},
        "dpm": {"include_in_context": False, "include_in_preamble": False},
        "skip_rag_state_import": True,
        "survival_ledger_enabled": False,
        "background_processing_enabled": False,
    }


def write_config(tmp_path, configuration):
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
    return path


def set_value(payload, path, value):
    current = payload
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def test_valid_profile_has_typed_settings_without_creating_store(tmp_path, configuration):
    settings = load_config(write_config(tmp_path, configuration))
    assert settings.production_profile == PROFILE_ID
    assert settings.persistence.receipts is True
    assert isinstance(settings.storage_dir, Path)
    assert not settings.storage_dir.exists()


def test_profile_selector_defaults_to_legacy(tmp_path):
    settings = load_config(tmp_path / "missing.yaml", overrides={
        "capacity": "12", "storage_dir": 123, "unknown_legacy_key": "kept",
        "persistence": {"receipts": "false", "ignored_legacy_key": True},
    })
    assert settings.production_profile is None
    assert settings.capacity == 12
    assert settings.storage_dir == Path("123")
    assert settings.as_dict()["unknown_legacy_key"] == "kept"
    assert settings.persistence.receipts is False


@pytest.mark.parametrize("selector", ["", False, True, 0, 1, [], {}, "unknown", "DML-RECEIPTED-LOCAL-V1"])
def test_unknown_selector_fails_closed_before_pydantic(tmp_path, monkeypatch, selector):
    monkeypatch.setattr(DMLSettings, "model_validate", lambda *_: pytest.fail("Pydantic ran"))
    with pytest.raises(ProductionProfileError, match="production_profile"):
        load_config(tmp_path / "missing.yaml", overrides={"production_profile": selector})


@pytest.mark.parametrize(("path", "value"), [
    (("capacity",), "12"),
    (("capacity",), True),
    (("capacity",), 12.0),
    (("metrics_enabled",), "false"),
    (("metrics_enabled",), 0),
    (("storage_dir",), False),
    (("storage_dir",), 123),
    (("storage_dir",), None),
    (("storage_dir",), []),
    (("storage_dir",), ""),
    (("persistence", "path"), False),
    (("rag_store", "meta_path"), 123),
    (("dpm", "overlay_path"), False),
    (("persistence", "receipts"), "true"),
    (("persistence", "interval_sec"), "0"),
    (("persistence", "interval_sec"), False),
    (("budgets", "semantic_pct"), "0.7"),
    (("budgets", "semantic_pct"), True),
    (("budgets", "semantic_pct"), float("nan")),
    (("beta_a",), float("inf")),
    (("skip_rag_state_import",), "true"),
    (("agentic_mode", "enabled"), "false"),
    (("capacity_typo",), 100),
    (("persistence", "reciepts"), True),
    (("rag_store", "enabel"), False),
    (("literal", "max_snipets"), 3),
    (("budgets", "semantic_percent"), 0.7),
    (("dpm", "include_in_contxt"), False),
    (("agentic_mode", "router", "enabeld"), False),
])
@pytest.mark.parametrize("source", ["yaml", "direct"])
def test_raw_invalid_values_never_reach_pydantic(tmp_path, monkeypatch, configuration, path, value, source):
    set_value(configuration, path, value)
    monkeypatch.setattr(DMLSettings, "model_validate", lambda *_: pytest.fail("Pydantic ran"))
    with pytest.raises(ProductionProfileError):
        if source == "yaml":
            load_config(write_config(tmp_path, configuration))
        else:
            load_config(tmp_path / "missing.yaml", overrides=configuration)
    assert not (tmp_path / "uncreated-store").exists()


@pytest.mark.parametrize(("name", "value", "field", "expected"), [
    ("DML_METRICS_ENABLED", "false", ("metrics_enabled",), False),
    ("DML_CAPACITY", "17", ("capacity",), 17),
    ("DML_K", "5", ("K",), 5),
    ("DML_BETA_A", "0.125", ("beta_a",), 0.125),
    ("DML_BETA_A", "1e-3", ("beta_a",), 0.001),
    ("DML_PERSISTENCE__RECEIPTS", "true", ("persistence", "receipts"), True),
    ("DML_PERSISTENCE_INTERVAL_SEC", "0", ("persistence", "interval_sec"), 0),
    ("DML_RAG_STORE_DIM", "7", ("rag_store", "dim"), 7),
    ("DML_RAG_STORE__DIM", "7", ("rag_store", "dim"), 7),
    ("DML_DPM_OVERLAY_PATH", "null", ("dpm", "overlay_path"), None),
    ("DML_EMBEDDING_DEVICE", "null", ("embedding_device",), None),
    ("DML_AGENTIC_MODE__ROUTER__ENABLED", "false", ("agentic_mode", "router", "enabled"), False),
    ("DML_BACKGROUND_PROCESSING_ENABLED", "false", ("background_processing_enabled",), False),
])
def test_canonical_environment_values_are_typed(tmp_path, monkeypatch, configuration, name, value, field, expected):
    monkeypatch.setenv(name, value)
    result = load_config(write_config(tmp_path, configuration)).as_dict()
    for part in field:
        result = result[part]
    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize(("name", "value"), [
    ("DML_METRICS_ENABLED", "False"),
    ("DML_METRICS_ENABLED", "0"),
    ("DML_METRICS_ENABLED", "yes"),
    ("DML_METRICS_ENABLED", " false "),
    ("DML_CAPACITY", "01"),
    ("DML_CAPACITY", "+1"),
    ("DML_CAPACITY", "1.0"),
    ("DML_CAPACITY", " 1 "),
    ("DML_CAPACITY", "null"),
    ("DML_BETA_A", "NaN"),
    ("DML_BETA_A", "Infinity"),
    ("DML_BETA_A", "1e999"),
    ("DML_BETA_A", ".5"),
    ("DML_BETA_A", "1_000"),
    ("DML_PERSISTENCE_INTERVAL_SEC", "-0"),
    ("DML_PERSISTENCE__RECIEPTS", "true"),
    ("DML_API_TOKNE", "secret-must-not-appear"),
    ("DML_SKIP_VENV_REEXE", "1"),
    ("DML_STORAGE_DIR__", "some-path"),
    ("DML_PERSISTENCE____PATH", "some-path"),
])
def test_invalid_environment_values_and_names_rejected(tmp_path, monkeypatch, configuration, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ProductionProfileError) as caught:
        load_config(write_config(tmp_path, configuration))
    assert "secret-must-not-appear" not in str(caught.value)


def test_yaml_dotenv_process_and_explicit_precedence(tmp_path, monkeypatch, configuration):
    configuration["capacity"] = 11
    path = write_config(tmp_path, configuration)
    assert load_config(path).capacity == 11
    (tmp_path / ".env").write_text("DML_CAPACITY=22\n", encoding="utf-8")
    assert load_config(path).capacity == 22
    monkeypatch.setenv("DML_CAPACITY", "33")
    assert load_config(path).capacity == 33
    assert load_config(path, overrides={"capacity": 44}).capacity == 44
    monkeypatch.setenv("DML_CAPACITY", "not-an-integer")
    assert load_config(path, overrides={"capacity": 44}).capacity == 44


def test_process_environment_wins_across_equivalent_aliases(tmp_path, monkeypatch, configuration):
    path = write_config(tmp_path, configuration)
    (tmp_path / ".env").write_text(
        "DML_RAG_STORE_DIM=11\nDML_RAG_STORE__DIM=22\n", encoding="utf-8",
    )
    monkeypatch.setenv("DML_RAG_STORE_DIM", "33")
    assert load_config(path).rag_store.dim == 33


@pytest.mark.parametrize("source", ["dotenv", "process"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("parent_collision", [False, True])
def test_same_source_environment_conflicts_reject_before_pydantic(
    tmp_path, monkeypatch, configuration, source, reverse, parent_collision,
):
    path = write_config(tmp_path, configuration)
    values = [
        ("DML_PERSISTENCE" if parent_collision else "DML_PERSISTENCE_RECEIPTS", "false"),
        ("DML_PERSISTENCE__RECEIPTS", "true"),
    ]
    if reverse:
        values.reverse()
    if source == "dotenv":
        (tmp_path / ".env").write_text("\n".join(f"{key}={value}" for key, value in values), encoding="utf-8")
    else:
        for key, value in values:
            monkeypatch.setenv(key, value)
    monkeypatch.setattr(DMLSettings, "model_validate", lambda *_: pytest.fail("Pydantic ran"))
    with pytest.raises(ProductionProfileError):
        load_config(path)


@pytest.mark.parametrize("source", ["dotenv", "process"])
def test_same_source_identical_canonical_environment_aliases_are_allowed(tmp_path, monkeypatch, configuration, source):
    path = write_config(tmp_path, configuration)
    values = {
        "DML_PERSISTENCE_RECEIPTS": "true", "DML_PERSISTENCE__RECEIPTS": "true",
        "DML_BUDGETS_SEMANTIC_PCT": "0.7", "DML_BUDGETS__SEMANTIC_PCT": "7e-1",
    }
    if source == "dotenv":
        (tmp_path / ".env").write_text("\n".join(f"{key}={value}" for key, value in values.items()), encoding="utf-8")
    else:
        for key, value in values.items():
            monkeypatch.setenv(key, value)
    settings = load_config(path)
    assert settings.persistence.receipts is True
    assert settings.budgets.semantic_pct == 0.7


def test_profile_selector_uses_merged_precedence(tmp_path, monkeypatch, configuration):
    configuration["production_profile"] = None
    path = write_config(tmp_path, configuration)
    monkeypatch.setenv("DML_PRODUCTION_PROFILE", PROFILE_ID)
    assert load_config(path).production_profile == PROFILE_ID
    assert load_config(path, overrides={"production_profile": None}).production_profile is None
    monkeypatch.setenv("DML_PRODUCTION_PROFILE", "unknown")
    assert load_config(path, overrides={"production_profile": PROFILE_ID}).production_profile == PROFILE_ID
    with pytest.raises(ProductionProfileError):
        load_config(path)


@pytest.mark.parametrize("selector", ["", "false", "0", "unknown"])
def test_invalid_environment_selector_is_not_an_opt_out(tmp_path, monkeypatch, selector):
    monkeypatch.setenv("DML_PRODUCTION_PROFILE", selector)
    with pytest.raises(ProductionProfileError, match="production_profile"):
        load_config(tmp_path / "missing.yaml")


def test_explicit_null_environment_selector_is_legacy(tmp_path, monkeypatch):
    monkeypatch.setenv("DML_PRODUCTION_PROFILE", "null")
    assert load_config(tmp_path / "missing.yaml").production_profile is None


@pytest.mark.parametrize("name", [
    "DML_API_TOKEN", "DML_ADMIN_TOKEN", "DML_SKIP_VENV_REEXEC", "DML_PROVIDER_URL",
    "DML_API_BASE", "DML_API_KEY", "DML_MAX_UPLOAD_BYTES", "DML_AUDIT_ACTOR",
    "DML_HOST", "DML_PORT", "DML_LOG_LEVEL", "DML_STORE", "DML_VISUALIZER_URL",
])
def test_known_operational_environment_does_not_pollute_profile(tmp_path, monkeypatch, configuration, name):
    monkeypatch.setenv(name, "not-configuration-secret")
    settings = load_config(write_config(tmp_path, configuration))
    assert name.removeprefix("DML_").lower() not in settings.as_dict()
    assert "not-configuration-secret" not in repr(settings.as_dict())


@pytest.mark.parametrize("source", ["yaml", "direct"])
@pytest.mark.parametrize("key", ["api_token", "admin_token", "skip_venv_reexec", "host"])
def test_operational_names_in_configuration_still_reject(tmp_path, configuration, key, source):
    configuration[key] = "secret-must-not-appear"
    with pytest.raises(ProductionProfileError) as caught:
        if source == "yaml":
            load_config(write_config(tmp_path, configuration))
        else:
            load_config(tmp_path / "missing.yaml", overrides=configuration)
    assert "secret-must-not-appear" not in str(caught.value)


def test_legacy_environment_behavior_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("DML_API_TOKEN", "legacy-extra")
    monkeypatch.setenv("DML_CAPACITY", "0017")
    settings = load_config(tmp_path / "missing.yaml")
    assert settings.capacity == 17
    assert settings.as_dict()["api_token"] == "legacy-extra"


def test_resolved_settings_are_validated_again(tmp_path, monkeypatch, configuration):
    resolved = DMLSettings(**deepcopy(configuration))
    resolved.rag_store.enable = True
    monkeypatch.setattr(DMLSettings, "model_validate", lambda *_: resolved)
    with pytest.raises(ProductionProfileError, match="rag_store.enable"):
        load_config(tmp_path / "missing.yaml", overrides=configuration)


def test_explicit_python_paths_are_accepted(tmp_path, configuration):
    configuration["storage_dir"] = tmp_path / "store"
    configuration["persistence"]["path"] = tmp_path / "state.jsonl"
    settings = load_config(tmp_path / "missing.yaml", overrides=configuration)
    assert settings.storage_dir == tmp_path / "store"
    assert not settings.storage_dir.exists()
