import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLUGIN_CONTRACT = ROOT / "specs" / "dpm-plugin-contract.md"
LIFECYCLE_CONTRACT = ROOT / "specs" / "config" / "dpm-lifecycle-config-contract.md"
SCHEMA_PATH = ROOT / "specs" / "config" / "dpm-config.schema.json"
EXAMPLES_DIR = ROOT / "examples" / "dpm"
PROMOTION_MANIFEST = ROOT / "notes" / "read-lane-promotion-manifest.md"
BOUNDARY_CONTRACT = ROOT / "notes" / "packet-2-boundary-contract.md"


def load_json(path: Path):
    return json.loads(path.read_text())


def load_text(path: Path) -> str:
    return path.read_text()


def derive_overlay_policy(config: dict) -> dict:
    mode = config["mode"]
    audit = config["audit"]
    return {
        "mode": mode,
        "retrieval_allowed": mode in {"observe-only", "active-read", "active-write"},
        "overlay_allowed": mode in {"active-read", "active-write"},
        "overlay_max_chars": audit["max_overlay_chars"],
        "audit_required_for_non_disabled": mode != "disabled" and audit["enabled"] is True,
        "writes_allowed": mode == "active-write"
        and config["write"]["enabled"] is True
        and config["write"]["require_explicit_runtime_support"] is True,
        "allowed_write_scopes": list(config["write"]["allowed_scopes"]),
    }


def synthetic_active_write_config() -> dict:
    config = load_json(EXAMPLES_DIR / "config.active-read.json")
    config["mode"] = "active-write"
    config["write"] = {
        "enabled": True,
        "require_explicit_runtime_support": True,
        "allowed_scopes": ["thread"],
    }
    return config


def validate_overlay_contract(config: dict, overlay: str) -> None:
    policy = derive_overlay_policy(config)
    if not policy["overlay_allowed"]:
        assert overlay == ""
        return

    assert len(overlay) <= policy["overlay_max_chars"]
    if overlay:
        assert policy["audit_required_for_non_disabled"] is True


def test_plugin_contract_declares_required_runtime_object_parts():
    doc = load_text(PLUGIN_CONTRACT)

    for required_line in [
        "- mode: current lifecycle mode",
        "- sources: ordered records actually used for the result",
        "- overlay: compact continuity guidance produced from those records",
        "- audit: machine-readable explanation of why each source was included or excluded",
        "- override_state: whether explicit user instruction constrained or nullified plugin guidance",
    ]:
        assert required_line in doc


def test_plugin_contract_declares_canonical_retrieval_precedence():
    doc = load_text(PLUGIN_CONTRACT)

    expected_order = [
        "1. explicit current-turn user instructions",
        "2. thread-local plugin continuity",
        "3. project-scoped plugin continuity",
        "4. relationship memory",
        "5. weighted preference graph",
    ]
    positions = [doc.index(line) for line in expected_order]
    assert positions == sorted(positions)


def test_lifecycle_contract_and_schema_align_on_write_scope_gate():
    doc = load_text(LIFECYCLE_CONTRACT)
    schema = load_json(SCHEMA_PATH)

    assert "- `allowed_scopes` must be empty unless `mode` is `active-write`" in doc
    else_clause = schema["allOf"][0]["else"]["properties"]["write"]["properties"]["allowed_scopes"]
    then_clause = schema["allOf"][0]["then"]["properties"]["write"]["properties"]["allowed_scopes"]
    assert else_clause == {"maxItems": 0}
    assert then_clause == {"minItems": 1}


def test_config_examples_derive_deterministic_lifecycle_overlay_policy():
    expected = {
        "config.disabled.json": {
            "retrieval_allowed": False,
            "overlay_allowed": False,
            "writes_allowed": False,
            "allowed_write_scopes": [],
        },
        "config.observe-only.json": {
            "retrieval_allowed": True,
            "overlay_allowed": False,
            "writes_allowed": False,
            "allowed_write_scopes": [],
        },
        "config.active-read.json": {
            "retrieval_allowed": True,
            "overlay_allowed": True,
            "writes_allowed": False,
            "allowed_write_scopes": [],
        },
        "synthetic-active-write": {
            "retrieval_allowed": True,
            "overlay_allowed": True,
            "writes_allowed": True,
            "allowed_write_scopes": ["thread"],
        },
    }

    for filename, expected_policy in expected.items():
        config = synthetic_active_write_config() if filename == "synthetic-active-write" else load_json(EXAMPLES_DIR / filename)
        policy = derive_overlay_policy(config)
        assert policy["overlay_max_chars"] == config["audit"]["max_overlay_chars"]
        assert policy["audit_required_for_non_disabled"] is (config["mode"] != "disabled")
        for key, value in expected_policy.items():
            assert policy[key] == value


def test_overlay_contract_rejects_overlay_when_mode_forbids_it():
    for filename in ["config.disabled.json", "config.observe-only.json"]:
        config = load_json(EXAMPLES_DIR / filename)
        validate_overlay_contract(config, "")


def test_overlay_contract_enforces_configured_char_limit():
    config = load_json(EXAMPLES_DIR / "config.active-read.json")
    validate_overlay_contract(config, "x" * config["audit"]["max_overlay_chars"])


def test_overlay_contract_accepts_active_write_within_bound():
    config = synthetic_active_write_config()
    validate_overlay_contract(config, "thread continuity guidance")


def test_read_lane_manifest_includes_boundary_contract_and_read_only_exclusions():
    manifest = load_text(PROMOTION_MANIFEST)
    boundary = load_text(BOUNDARY_CONTRACT)

    required_packet_entries = [
        "- `notes/packet-2-boundary-contract.md`",
        "- `notes/sprint-freeze-readonly-plugin-state.md`",
        "- `examples/dpm/config.active-read.json`",
        "- `examples/dpm/config.observe-only.json`",
        "- `examples/dpm/config.disabled.json`",
    ]
    for entry in required_packet_entries:
        assert entry in manifest

    forbidden_enabled_target = "- `examples/dpm/config.active-write.json` as an enabled target"
    assert forbidden_enabled_target in manifest
    assert forbidden_enabled_target in boundary

    for excluded_path in ["- `runtime/`", "- `out/`", "- `scratch/`"]:
        assert excluded_path in manifest
        assert excluded_path in boundary

    assert "- the packet remains read-only and additive" in manifest
    assert "- no active-write semantics are introduced during promotion" in manifest
    assert "- `notes/packet-2-boundary-contract.md`" in boundary
    assert "- excluded runtime/output/cache/scaffolding paths are described consistently in notes and manifest" in boundary
