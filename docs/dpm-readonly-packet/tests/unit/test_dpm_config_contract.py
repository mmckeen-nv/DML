import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "specs" / "config" / "dpm-config.schema.json"
EXAMPLES_DIR = ROOT / "examples" / "dpm"


def load_json(path: Path):
    return json.loads(path.read_text())


def validate_config(config: dict) -> None:
    assert config["version"] == 1
    assert config["plugin"] == "dpm"
    assert config["mode"] in {"disabled", "observe-only", "active-read", "active-write"}

    read = config["read"]
    assert set(read.keys()) == {
        "allow_thread",
        "allow_project",
        "allow_relationship",
        "allow_preference_graph",
    }
    assert all(isinstance(value, bool) for value in read.values())

    write = config["write"]
    assert isinstance(write["enabled"], bool)
    assert write["require_explicit_runtime_support"] is True
    assert all(
        scope in {"thread", "project", "relationship", "preference-graph"}
        for scope in write["allowed_scopes"]
    )

    audit = config["audit"]
    assert isinstance(audit["enabled"], bool)
    assert isinstance(audit["include_excluded_sources"], bool)
    assert audit["max_sources"] >= 1
    assert audit["max_overlay_chars"] >= 1

    safety = config["safety"]
    assert safety == {
        "explicit_user_override": True,
        "fail_closed_on_invalid_mode": True,
        "cross_scope_fallback_requires_compatibility": True,
    }

    if config["mode"] == "active-write":
        assert write["enabled"] is True
        assert len(write["allowed_scopes"]) >= 1
    else:
        assert write["allowed_scopes"] == []


def test_schema_declares_canonical_lifecycle_modes():
    schema = load_json(SCHEMA_PATH)
    assert schema["properties"]["mode"]["enum"] == [
        "disabled",
        "observe-only",
        "active-read",
        "active-write",
    ]


ACTIVE_WRITE_EXAMPLE = {
    "version": 1,
    "plugin": "dpm",
    "mode": "active-write",
    "read": {
        "allow_thread": True,
        "allow_project": True,
        "allow_relationship": True,
        "allow_preference_graph": True,
    },
    "write": {
        "enabled": True,
        "require_explicit_runtime_support": True,
        "allowed_scopes": ["thread"],
    },
    "audit": {
        "enabled": True,
        "include_excluded_sources": True,
        "max_sources": 20,
        "max_overlay_chars": 2000,
    },
    "safety": {
        "explicit_user_override": True,
        "fail_closed_on_invalid_mode": True,
        "cross_scope_fallback_requires_compatibility": True,
    },
}


def test_example_configs_cover_reviewable_packet_modes_and_schema_declares_active_write():
    expected = {
        "config.disabled.json": "disabled",
        "config.observe-only.json": "observe-only",
        "config.active-read.json": "active-read",
    }
    for filename, mode in expected.items():
        config = load_json(EXAMPLES_DIR / filename)
        validate_config(config)
        assert config["mode"] == mode

    validate_config(ACTIVE_WRITE_EXAMPLE)
    assert ACTIVE_WRITE_EXAMPLE["mode"] == "active-write"


def test_non_write_modes_keep_allowed_scopes_empty():
    for filename in [
        "config.disabled.json",
        "config.observe-only.json",
        "config.active-read.json",
    ]:
        config = load_json(EXAMPLES_DIR / filename)
        assert config["write"]["allowed_scopes"] == []


def test_active_write_contract_requires_explicit_opt_in():
    config = ACTIVE_WRITE_EXAMPLE
    assert config["write"]["enabled"] is True
    assert config["write"]["require_explicit_runtime_support"] is True
    assert config["write"]["allowed_scopes"] == ["thread"]
