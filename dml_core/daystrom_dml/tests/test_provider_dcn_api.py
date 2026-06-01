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
    assert report["summary"]["case_count"] == 9
    assert report["summary"]["passed_count"] == 9
    assert report["summary"]["max_pollution_score"] == 0.0
    assert report["summary"]["blocked_polluting_items"] == 3
    artifact = payload["artifact"]
    assert artifact["schema_version"] == "dcn-eval-artifact-v1"
    assert artifact["summary"] == report["summary"]
    assert artifact["artifact_hash"]
    assert artifact["readiness"]["ready"] is True
    assert artifact["readiness"]["failed_gates"] == []
    assert artifact["readiness"]["gate_count"] == 15
    assert "code_change" in artifact["coverage"]["task_types"]
    assert "debugging_requires_verification" in artifact["coverage"]["case_ids"]
    assert "side_effect_merge_requires_confirmation" in artifact["coverage"]["case_ids"]
    assert artifact["coverage"]["confirmation_required_cases"] >= 1
    assert all(flag is False for flag in artifact["redaction_policy"].values())
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


def test_dcn_policy_export_import_roundtrip_applies_explicit_overlay():
    client = TestClient(create_app(adapter_factory=DummyAdapter))

    before = client.post("/api/dcn/plan-context", json={"content": "what is the status?"}).json()["plan"]
    exported = client.post("/api/dcn/policy/export").json()
    snapshot = exported["snapshot"]
    snapshot["mutable_overlay"] = {
        "answer": {
            "task_type": "answer",
            "memory_mode_preference": "hybrid",
            "version": 1,
            "updated_at": 0.0,
            "provenance": {"source": "test"},
        }
    }
    imported = client.post("/api/dcn/policy/import", json={"snapshot": snapshot}).json()
    after = client.post("/api/dcn/plan-context", json={"content": "what is the status?"}).json()["plan"]

    assert exported["status"] == "ok"
    assert snapshot["schema_version"] == "dcn-procedural-learning-v1"
    assert imported["status"] == "ok"
    assert imported["imported"] is True
    assert before["retrieval_plan"]["mode"] == "none"
    assert after["retrieval_plan"]["mode"] == "hybrid"
    assert after["policy_version"] == "dcn-policy-v0+procedural-v1"


def test_dcn_policy_import_rejects_wrong_schema_without_mutating_policy():
    client = TestClient(create_app(adapter_factory=DummyAdapter))

    response = client.post(
        "/api/dcn/policy/import",
        json={"snapshot": {"schema_version": "wrong", "base_policy_ref": "dcn-policy-v0", "mutable_overlay": {}}},
    )
    plan = client.post("/api/dcn/plan-context", json={"content": "what is the status?"}).json()["plan"]

    assert response.status_code == 400
    assert "unsupported procedural learning schema_version" in response.json()["detail"]
    assert plan["retrieval_plan"]["mode"] == "none"
    assert plan["policy_version"] == "dcn-policy-v0"


def test_dcn_policy_checkpoint_and_rollback_endpoints_restore_overlay():
    client = TestClient(create_app(adapter_factory=DummyAdapter))

    snapshot = client.post("/api/dcn/policy/export").json()["snapshot"]
    snapshot["mutable_overlay"] = {
        "answer": {
            "task_type": "answer",
            "memory_mode_preference": "hybrid",
            "version": 1,
            "updated_at": 0.0,
            "provenance": {"source": "test"},
        }
    }
    imported = client.post("/api/dcn/policy/import", json={"snapshot": snapshot}).json()
    checkpoint = client.post("/api/dcn/policy/checkpoint", json={"label": "before-strict"}).json()
    snapshot["mutable_overlay"]["answer"]["verification_requirement"] = "strict"
    client.post("/api/dcn/policy/import", json={"snapshot": snapshot})

    learned = client.post("/api/dcn/plan-context", json={"content": "what is the status?"}).json()["plan"]
    rollback = client.post("/api/dcn/policy/rollback", json={"checkpoint_id": checkpoint["checkpoint_id"]}).json()
    restored = client.post("/api/dcn/plan-context", json={"content": "what is the status?"}).json()["plan"]
    checkpoints = client.get("/api/dcn/policy/checkpoints").json()

    assert imported["status"] == "ok"
    assert checkpoint["label"] == "before-strict"
    assert "real_tool_output" in learned["tool_plan"]["verification_required"]
    assert rollback["status"] == "ok"
    assert rollback["rolled_back"] is True
    assert restored["retrieval_plan"]["mode"] == "hybrid"
    assert "real_tool_output" not in restored["tool_plan"]["verification_required"]
    assert checkpoints["count"] >= 2
    assert all("mutable_overlay" not in checkpoint for checkpoint in checkpoints["checkpoints"])


def test_dcn_policy_rollback_rejects_unknown_checkpoint_without_mutating_overlay():
    client = TestClient(create_app(adapter_factory=DummyAdapter))
    snapshot = client.post("/api/dcn/policy/export").json()["snapshot"]
    snapshot["mutable_overlay"] = {"answer": {"task_type": "answer", "memory_mode_preference": "hybrid"}}
    client.post("/api/dcn/policy/import", json={"snapshot": snapshot})

    response = client.post("/api/dcn/policy/rollback", json={"checkpoint_id": "missing"})
    plan = client.post("/api/dcn/plan-context", json={"content": "what is the status?"}).json()["plan"]

    assert response.status_code == 400
    assert "unknown checkpoint_id" in response.json()["detail"]
    assert plan["retrieval_plan"]["mode"] == "hybrid"


def test_dcn_active_learn_promotion_requires_checkpoint_and_hygiene_evidence():
    client = TestClient(create_app(adapter_factory=DummyAdapter))

    missing_checkpoint = client.post(
        "/api/dcn/mode/promote",
        json={"target_mode": "active_learn", "hygiene_evidence": {"passed": True}},
    )
    checkpoint = client.post("/api/dcn/policy/checkpoint", json={"label": "before-active-learn"}).json()
    missing_hygiene = client.post(
        "/api/dcn/mode/promote",
        json={"target_mode": "active_learn", "checkpoint_id": checkpoint["checkpoint_id"]},
    )
    policy = client.get("/api/dcn/policy").json()

    assert missing_checkpoint.status_code == 400
    assert missing_checkpoint.json()["detail"]["reason"] == "checkpoint_required"
    assert missing_hygiene.status_code == 400
    assert missing_hygiene.json()["detail"]["reason"] == "hygiene_evidence_required"
    assert policy["runtime_mode"] == "deterministic_v0"
    assert policy["last_promotion"] is None


def test_dcn_active_learn_promotion_records_rollbackable_audit_without_raw_evidence():
    client = TestClient(create_app(adapter_factory=DummyAdapter))
    checkpoint = client.post("/api/dcn/policy/checkpoint", json={"label": "before-active-learn"}).json()

    promoted = client.post(
        "/api/dcn/mode/promote",
        json={
            "target_mode": "active_learn",
            "checkpoint_id": checkpoint["checkpoint_id"],
            "hygiene_evidence": {"passed": True, "artifact_hash": "abc123", "raw_transcript": "secret transcript", "token": "secret-token"},
            "operator": "pytest",
            "reason": "enable active learn after smoke gates",
        },
    ).json()
    policy = client.get("/api/dcn/policy").json()
    promotions = client.get("/api/dcn/mode/promotions").json()
    rendered = str(promoted).lower()

    assert promoted["status"] == "ok"
    assert promoted["promoted"] is True
    assert promoted["runtime_mode"] == "active_learn"
    audit = promoted["audit"]
    assert audit["previous_mode"] == "deterministic_v0"
    assert audit["target_mode"] == "active_learn"
    assert audit["checkpoint_id"] == checkpoint["checkpoint_id"]
    assert audit["rollback_command"].endswith(checkpoint["checkpoint_id"])
    assert audit["eval"]["passed"] is True
    assert audit["eval"]["artifact_hash"]
    assert audit["eval"]["readiness"]["ready"] is True
    assert audit["eval"]["readiness"]["failed_gates"] == []
    assert "debugging_requires_verification" in audit["eval"]["coverage"]["case_ids"]
    assert audit["eval"]["summary"]["max_pollution_score"] == 0.0
    assert audit["hygiene"]["passed"] is True
    assert audit["hygiene"]["artifact_hash"] == "abc123"
    assert "secret transcript" not in rendered
    assert "secret-token" not in rendered
    assert policy["runtime_mode"] == "active_learn"
    assert policy["last_promotion"]["promotion_id"] == audit["promotion_id"]
    assert promotions["count"] == 1
    assert promotions["entries"][0]["promotion_id"] == audit["promotion_id"]


def test_dcn_active_learn_promotion_rejects_unknown_checkpoint_without_mode_change():
    client = TestClient(create_app(adapter_factory=DummyAdapter))

    response = client.post(
        "/api/dcn/mode/promote",
        json={"target_mode": "active_learn", "checkpoint_id": "missing", "hygiene_evidence": {"passed": True}},
    )
    policy = client.get("/api/dcn/policy").json()

    assert response.status_code == 400
    assert response.json()["detail"]["reason"] == "unknown_checkpoint"
    assert policy["runtime_mode"] == "deterministic_v0"
