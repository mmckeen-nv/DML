import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
CONTRACT = ROOT / "specs" / "relationship-continuity-source-contract.md"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text())


def is_iso_utc(value: str) -> bool:
    return isinstance(value, str) and value.endswith("Z") and "T" in value


def validate_relationship_source(payload: dict) -> None:
    assert payload["schema_version"] == "dpm.relationship-source.v1"
    assert payload["source_id"].startswith("relationship:")
    assert payload["relationship_id"].startswith("relationship:")
    assert payload["scope"] == "relationship"
    assert isinstance(payload["label"], str) and payload["label"]
    assert isinstance(payload["summary"], str) and payload["summary"]
    assert isinstance(payload["directives"], list)
    assert isinstance(payload["constraints"], list)
    assert isinstance(payload["exclusion_rules"], list) and payload["exclusion_rules"]
    assert isinstance(payload["priority"], int)
    assert payload["priority"] >= 4
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
    assert isinstance(boundedness["audit_note_max_items"], int)
    assert 1 <= boundedness["audit_note_max_items"] <= 3

    assert len(payload["summary"]) <= boundedness["summary_max_chars"]
    assert len(payload["directives"]) <= boundedness["directive_max_items"]
    assert len(payload["constraints"]) <= boundedness["constraint_max_items"]


def test_relationship_continuity_source_contract_doc_exists_and_declares_packet_4_rules():
    doc = CONTRACT.read_text()

    for required_line in [
        "# Relationship Continuity Source Contract",
        "1. explicit current-turn instruction",
        "2. thread continuity source",
        "3. project continuity source",
        "4. relationship continuity source",
        "relationship source must never erase, broaden, or contradict thread-specific constraints",
        "at most one relationship source record may materially shape the result for this packet",
        "`boundedness_violation`",
        "`conflicts_with_thread_scope`",
        "`conflicts_with_project_scope`",
        "at most `boundedness.audit_note_max_items` summary notes",
    ]:
        assert required_line in doc


def test_valid_relationship_source_fixture_matches_contract():
    payload = load_fixture("relationship_source.valid.fixture.json")
    validate_relationship_source(payload)



def test_thread_scope_keeps_precedence_when_relationship_source_is_compatible():
    payload = load_fixture("relationship_source.thread_compatible.fixture.json")
    thread_source = payload["thread_source"]
    relationship_source = payload["relationship_source"]
    expected = payload["expected"]

    validate_relationship_source(relationship_source)
    assert thread_source["scope"] == "thread"
    assert thread_source["priority"] < relationship_source["priority"]
    assert expected["thread_scope_primary"] is True
    assert expected["relationship_included"] is True
    assert expected["relationship_effect"] == "compatible_refinement"
    assert expected["audit"]["consulted_relationship_id"] == relationship_source["relationship_id"]
    assert expected["audit"]["relationship_outcome"] == "included"
    assert len(expected["audit"]["notes"]) <= relationship_source["boundedness"]["audit_note_max_items"]
    assert "formal" in thread_source["summary"].lower()
    assert "concise" in relationship_source["summary"].lower()



def test_thread_scope_excludes_conflicting_relationship_guidance():
    payload = load_fixture("relationship_source.thread_conflict.fixture.json")
    thread_source = payload["thread_source"]
    relationship_source = payload["relationship_source"]
    expected = payload["expected"]

    validate_relationship_source(relationship_source)
    assert thread_source["scope"] == "thread"
    assert expected["thread_scope_primary"] is True
    assert expected["relationship_included"] is False
    assert expected["relationship_effect"] == "excluded_for_conflict"
    assert "conflicts_with_thread_scope" in expected["excluded_reasons"]
    assert expected["audit"]["consulted_relationship_id"] == relationship_source["relationship_id"]
    assert expected["audit"]["relationship_outcome"] == "excluded"
    assert len(expected["audit"]["notes"]) <= relationship_source["boundedness"]["audit_note_max_items"]
    assert "casual warmth" in relationship_source["summary"].lower()
    assert any("avoid casual warmth" in item.lower() for item in thread_source["constraints"])



def test_boundedness_violation_fixture_fails_contract_limits():
    payload = load_fixture("relationship_source.boundedness_violation.fixture.json")

    try:
        validate_relationship_source(payload)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected boundedness violation fixture to fail validation")

    assert len(payload["summary"]) > payload["boundedness"]["summary_max_chars"]
    assert len(payload["directives"]) > payload["boundedness"]["directive_max_items"]
    assert len(payload["constraints"]) > payload["boundedness"]["constraint_max_items"]
