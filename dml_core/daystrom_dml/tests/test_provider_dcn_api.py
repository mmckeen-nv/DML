from fastapi.testclient import TestClient

from daystrom_dml.provider_server import create_app
from daystrom_dml.tests.test_provider_server import DummyAdapter


def test_dcn_policy_endpoint_exposes_deterministic_v0():
    client = TestClient(create_app(adapter_factory=DummyAdapter))

    payload = client.get("/api/dcn/policy").json()

    assert payload["status"] == "ok"
    assert payload["component"] == "daystrom-cognition-network"
    assert payload["policy_version"] == "dcn-policy-v0"
    assert "feedback" in payload["capabilities"]
    assert "eval_smoke" in payload["capabilities"]
    assert "raw_transcript" in payload["writeback_forbidden_classes"]


def test_dcn_observe_returns_intent_retrieval_writeback_and_frontier_plans():
    client = TestClient(create_app(adapter_factory=DummyAdapter))

    payload = client.post(
        "/api/dcn/observe",
        json={"content": "continue the DML hardening work", "scope": {"tenant_id": "openclaw", "session_id": "s1"}},
    ).json()

    plan = payload["plan"]
    assert payload["status"] == "ok"
    assert plan["intent"]["needs_memory"] is True
    assert plan["retrieval_plan"]["mode"] in {"resume", "hybrid"}
    assert plan["writeback_plan"]["mode"] == "durable_signal_only"
    assert plan["frontier_plan"]["mode"] == "dml_context"


def test_dcn_plan_context_greeting_does_not_retrieve_memory():
    client = TestClient(create_app(adapter_factory=DummyAdapter))

    payload = client.post("/api/dcn/plan-context", json={"content": "hello again"}).json()

    plan = payload["plan"]
    assert plan["retrieval_plan"]["mode"] == "none"
    assert plan["intent"]["needs_memory"] is False


def test_dcn_cognitive_packet_calls_dml_adapter_when_memory_needed():
    client = TestClient(create_app(adapter_factory=DummyAdapter))

    payload = client.post(
        "/api/dcn/cognitive-packet",
        json={"content": "continue provider memory work", "scope": {"tenant_id": "openclaw", "session_id": "s1"}},
    ).json()

    packet = payload["packet"]
    assert packet["packet_version"] == "daystrom-cognitive-packet-v1"
    assert packet["dcn_plan"]["retrieval_plan"]["mode"] in {"resume", "hybrid"}
    assert "Provider memory text" in packet["dml_context"]["raw_context"]
    assert packet["dpm_overlay"] == {}
    assert "=== DML Context ===" in packet["assembled_context"]


def test_dcn_feedback_records_and_audit_lists_entry():
    client = TestClient(create_app(adapter_factory=DummyAdapter))

    result = client.post(
        "/api/dcn/feedback",
        json={"decision_id": "decision-1", "outcome": "verified", "signals": {"tests_passed": True}},
    ).json()
    audit = client.get("/api/dcn/audit").json()

    assert result["status"] == "ok"
    assert result["accepted"] is True
    assert audit["count"] == 1
    assert audit["entries"][0]["decision_id"] == "decision-1"
    assert audit["entries"][0]["signals"] == {"tests_passed": True}


def test_dcn_eval_smoke_endpoint_runs_offline_safe_fixture_suite():
    client = TestClient(create_app(adapter_factory=DummyAdapter))

    payload = client.get("/api/dcn/eval/smoke").json()
    rendered = str(payload).lower()

    assert payload["status"] == "ok"
    assert payload["component"] == "daystrom-cognition-network"
    assert payload["mode"] == "offline_fixture_smoke"
    report = payload["report"]
    assert report["passed"] is True
    assert report["suite_id"] == "provider-dcn-eval-smoke"
    assert report["summary"]["case_count"] == 3
    assert report["summary"]["passed_count"] == 3
    assert report["summary"]["max_pollution_score"] == 0.0
    assert report["summary"]["blocked_polluting_items"] == 1
    assert "provider memory text" not in rendered
    assert "raw_transcript" not in rendered
    assert "tool_calls" not in rendered
    assert "prompt_scaffold" not in rendered
    assert "sk-" not in rendered


def test_existing_frontier_prepare_shape_still_works_with_dcn_routes_present():
    client = TestClient(create_app(adapter_factory=DummyAdapter))

    payload = client.post(
        "/api/frontier/prepare",
        json={"prompt": "What should the agent remember?", "tenant_id": "openclaw", "session_id": "s1"},
    ).json()

    assert payload["mode"] == "frontier_with_dml_context"
    assert "frontier_prompt" in payload
    assert "Provider memory text" in payload["dml_context"]
    assert payload["telemetry"]["frontier_input_tokens"] > 0
