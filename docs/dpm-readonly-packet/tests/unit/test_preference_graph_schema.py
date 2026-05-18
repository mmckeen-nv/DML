import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SCHEMA_DOC = Path(__file__).resolve().parents[2] / "specs" / "preference-graph-schema.md"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text())


def test_preference_graph_schema_doc_exists():
    assert SCHEMA_DOC.exists()


def test_relationship_fixture_matches_schema_contract():
    payload = load_fixture("preference_graph.relationship.fixture.json")

    assert payload["schema_version"] == "dpm.preference-graph.v1"
    assert payload["default_policy"]["explicit_instruction_precedence"] == "always_override"
    assert payload["nodes"]
    assert payload["edges"]
    assert {node["id"] for node in payload["nodes"]} == {
        "pref.directness",
        "pref.concision",
        "value.privacy_caution",
    }
    for node in payload["nodes"]:
        assert 0.0 <= node["weight"] <= 1.0
        assert 0.0 <= node["confidence"] <= 1.0
        assert node["scope"] in {"thread", "project", "relationship", "global"}
    node_ids = {node["id"] for node in payload["nodes"]}
    for edge in payload["edges"]:
        assert edge["from"] in node_ids
        assert edge["to"] in node_ids
        assert 0.0 <= edge["weight"] <= 1.0
        assert 0.0 <= edge["confidence"] <= 1.0


def test_thread_override_fixture_preserves_conflict_and_override_examples():
    payload = load_fixture("preference_graph.thread_override.fixture.json")

    assert payload["schema_version"] == "dpm.preference-graph.v1"
    node_ids = {node["id"] for node in payload["nodes"]}
    assert "instruction.override.be_brief" in node_ids
    override_node = next(node for node in payload["nodes"] if node["id"] == "instruction.override.be_brief")
    humor_node = next(node for node in payload["nodes"] if node["id"] == "pref.humor")

    assert override_node["kind"] == "instruction_override"
    assert override_node["scope"] == "thread"
    assert override_node["confidence"] == 1.0
    assert humor_node["state"] == "conflicted"

    relations = {edge["relation"] for edge in payload["edges"]}
    assert "suppressed_by" in relations
    assert "conflicts_with" in relations

    edge_ids = {edge["id"] for edge in payload["edges"]}
    assert "edge.override.suppresses.humor" in edge_ids
    assert payload["audit"]["conflicts_detected"]
