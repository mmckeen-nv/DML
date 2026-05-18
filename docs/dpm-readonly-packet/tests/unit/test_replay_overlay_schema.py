import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SCHEMA_DOC = Path(__file__).resolve().parents[2] / "specs" / "replay-overlay-schema.md"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text())


def is_iso_utc(value: str) -> bool:
    return isinstance(value, str) and value.endswith("Z") and "T" in value


def validate_overlay(payload: dict) -> None:
    assert payload["schema_version"] == "dpm.replay-overlay.v1"
    assert payload["mode"] in {"observe-only", "active-read", "active-write"}
    assert is_iso_utc(payload["generated_at"])

    scope = payload["scope"]
    assert scope["primary"] in {"thread", "project", "relationship"}
    if scope["primary"] == "thread":
        assert isinstance(scope["thread_id"], str)
    if scope["primary"] == "project":
        assert isinstance(scope["project_id"], str)
    if scope["primary"] == "relationship":
        assert isinstance(scope["relationship_id"], str)

    order = payload["retrieval_order_applied"]
    canonical = [
        "explicit_current_turn",
        "thread",
        "project",
        "relationship",
        "preference_graph",
    ]
    assert order == [step for step in canonical if step in order]

    overlay = payload["overlay"]
    assert isinstance(overlay["persona_summary"], str)
    assert isinstance(overlay["style_directives"], list)
    assert isinstance(overlay["do_not_do"], list)
    assert isinstance(overlay["open_questions"], list)
    assert isinstance(overlay["max_chars"], int)
    assert overlay["max_chars"] >= 1
    assert len(overlay["rendered_text"]) <= overlay["max_chars"]

    constraints = payload["effective_constraints"]
    assert constraints["explicit_instruction_precedence"] == "always_override"
    assert constraints["narrowest_scope_wins"] is True
    assert constraints["cross_scope_fallback_requires_compatibility"] is True
    if payload["mode"] != "active-write":
        assert constraints["writes_allowed"] is False

    source_ids = []
    for source in payload["sources"]:
        source_ids.append(source["source_id"])
        assert source["scope"] in {"thread", "project", "relationship", "global"}
        assert isinstance(source["included"], bool)
        assert isinstance(source["priority"], int)
        assert 0.0 <= source["confidence"] <= 1.0
        assert is_iso_utc(source["updated_at"])
        assert isinstance(source["summary"], str)
    assert source_ids == [source["source_id"] for source in payload["sources"]]

    audit = payload["audit"]
    for source_id in audit["included_source_ids"]:
        assert source_id in source_ids
    for excluded in audit["excluded_sources"]:
        assert set(excluded.keys()) == {"source_id", "reason"}
    assert isinstance(audit["conflicts_detected"], list)
    assert isinstance(audit["notes"], list)

    override_state = payload["override_state"]
    assert isinstance(override_state["has_explicit_instruction"], bool)
    assert isinstance(override_state["override_applied"], bool)
    assert isinstance(override_state["suppressed_source_ids"], list)
    assert isinstance(override_state["effective_for_turn"], list)
    if override_state["override_applied"]:
        assert override_state["has_explicit_instruction"] is True
        assert isinstance(override_state["instruction_source_id"], str)
    else:
        assert override_state["instruction_source_id"] is None


def test_replay_overlay_schema_doc_exists():
    assert SCHEMA_DOC.exists()


def test_relationship_overlay_fixture_matches_schema_contract():
    payload = load_fixture("replay_overlay.relationship.fixture.json")
    validate_overlay(payload)

    assert payload["scope"]["primary"] == "relationship"
    assert payload["retrieval_order_applied"] == ["relationship", "preference_graph"]
    assert payload["override_state"]["override_applied"] is False
    assert payload["audit"]["conflicts_detected"] == []


def test_thread_override_overlay_fixture_preserves_override_and_conflict_audit():
    payload = load_fixture("replay_overlay.thread_override.fixture.json")
    validate_overlay(payload)

    assert payload["scope"]["primary"] == "thread"
    assert payload["retrieval_order_applied"][0] == "explicit_current_turn"
    assert payload["override_state"]["override_applied"] is True
    assert payload["override_state"]["instruction_source_id"] == "turn:current"
    assert "relationship:humor-default" in payload["override_state"]["suppressed_source_ids"]
    assert payload["audit"]["conflicts_detected"]
    assert "turn:current" in payload["audit"]["included_source_ids"]
    assert any(
        item["reason"] == "overridden_by_explicit_instruction"
        for item in payload["audit"]["excluded_sources"]
    )



def test_thread_overlay_preserves_thread_over_project_source_ordering():
    payload = load_fixture("replay_overlay.thread_override.fixture.json")

    source_ids = [source["source_id"] for source in payload["sources"]]
    assert source_ids.index("thread:self-architecture-portrait") < source_ids.index("project:dpm")

    priorities = {source["source_id"]: source["priority"] for source in payload["sources"]}
    assert priorities["thread:self-architecture-portrait"] < priorities["project:dpm"]

    assert payload["retrieval_order_applied"].index("thread") < payload["retrieval_order_applied"].index("project")



def test_thread_overlay_suppresses_conflicting_lower_precedence_guidance_when_explicit_instruction_exists():
    payload = load_fixture("replay_overlay.thread_override.fixture.json")

    assert payload["override_state"]["has_explicit_instruction"] is True
    assert payload["override_state"]["override_applied"] is True
    assert payload["override_state"]["instruction_source_id"] == "turn:current"
    assert "relationship:humor-default" in payload["override_state"]["suppressed_source_ids"]
    assert any(
        excluded == {
            "source_id": "relationship:humor-default",
            "reason": "overridden_by_explicit_instruction",
        }
        for excluded in payload["audit"]["excluded_sources"]
    )
    assert "suppress humor or broader defaults" in payload["overlay"]["rendered_text"]



def test_thread_overlay_audit_output_is_bounded_and_source_aligned():
    payload = load_fixture("replay_overlay.thread_override.fixture.json")

    audit = payload["audit"]
    assert len(payload["overlay"]["rendered_text"]) <= payload["overlay"]["max_chars"]
    assert len(audit["included_source_ids"]) == len(payload["sources"])
    assert len(audit["excluded_sources"]) <= 2
    assert len(audit["conflicts_detected"]) <= 2
    assert len(audit["notes"]) <= 2
