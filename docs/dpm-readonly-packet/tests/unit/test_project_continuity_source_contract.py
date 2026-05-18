import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
CONTRACT = ROOT / "specs" / "project-continuity-source-contract.md"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text())


def is_iso_utc(value: str) -> bool:
    return isinstance(value, str) and value.endswith("Z") and "T" in value


def validate_project_source(payload: dict) -> None:
    assert payload["schema_version"] == "dpm.project-source.v1"
    assert payload["source_id"].startswith("project:")
    assert payload["project_id"].startswith("project:")
    assert payload["scope"] == "project"
    assert isinstance(payload["label"], str) and payload["label"]
    assert isinstance(payload["summary"], str) and payload["summary"]
    assert isinstance(payload["directives"], list)
    assert isinstance(payload["constraints"], list)
    assert isinstance(payload["priority"], int)
    assert payload["priority"] >= 3
    assert 0.0 <= payload["confidence"] <= 1.0
    assert is_iso_utc(payload["updated_at"])
    assert isinstance(payload["audit_hint"], str) and payload["audit_hint"]

    boundedness = payload["boundedness"]
    assert isinstance(boundedness["summary_max_chars"], int)
    assert 1 <= boundedness["summary_max_chars"] <= 160
    assert isinstance(boundedness["directive_max_items"], int)
    assert 1 <= boundedness["directive_max_items"] <= 4
    assert isinstance(boundedness["constraint_max_items"], int)
    assert 1 <= boundedness["constraint_max_items"] <= 4

    assert len(payload["summary"]) <= boundedness["summary_max_chars"]
    assert len(payload["directives"]) <= boundedness["directive_max_items"]
    assert len(payload["constraints"]) <= boundedness["constraint_max_items"]


def test_project_continuity_source_contract_doc_exists_and_declares_packet_a_rules():
    doc = CONTRACT.read_text()

    for required_line in [
        "# Project Continuity Source Contract",
        "1. explicit current-turn instruction",
        "2. thread continuity source",
        "3. project continuity source",
        "project source must never erase, broaden, or contradict thread-specific constraints",
        "at most one project source record may materially shape the result for this packet",
        "`boundedness_violation`",
        "`conflicts_with_thread_scope`",
    ]:
        assert required_line in doc


def test_valid_project_source_fixture_matches_contract():
    payload = load_fixture("project_source.valid.fixture.json")
    validate_project_source(payload)


def test_thread_scope_keeps_precedence_when_project_source_is_compatible():
    payload = load_fixture("project_source.thread_compatible.fixture.json")
    thread_source = payload["thread_source"]
    project_source = payload["project_source"]
    expected = payload["expected"]

    validate_project_source(project_source)
    assert thread_source["scope"] == "thread"
    assert thread_source["priority"] < project_source["priority"]
    assert expected["thread_scope_primary"] is True
    assert expected["project_included"] is True
    assert expected["project_effect"] == "compatible_refinement"
    assert "formal" in thread_source["summary"].lower()
    assert "concise" in project_source["summary"].lower()


def test_thread_scope_excludes_conflicting_project_guidance():
    payload = load_fixture("project_source.thread_conflict.fixture.json")
    thread_source = payload["thread_source"]
    project_source = payload["project_source"]
    expected = payload["expected"]

    validate_project_source(project_source)
    assert thread_source["scope"] == "thread"
    assert expected["thread_scope_primary"] is True
    assert expected["project_included"] is False
    assert expected["project_effect"] == "excluded_for_conflict"
    assert "conflicts_with_thread_scope" in expected["excluded_reasons"]
    assert "casual shorthand" in project_source["summary"].lower()
    assert any("avoid casual shorthand" in item.lower() for item in thread_source["constraints"])


def test_boundedness_violation_fixture_fails_contract_limits():
    payload = load_fixture("project_source.boundedness_violation.fixture.json")

    try:
        validate_project_source(payload)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected boundedness violation fixture to fail validation")

    assert len(payload["summary"]) > payload["boundedness"]["summary_max_chars"]
    assert len(payload["directives"]) > payload["boundedness"]["directive_max_items"]
