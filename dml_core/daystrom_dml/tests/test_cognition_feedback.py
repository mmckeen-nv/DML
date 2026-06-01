from daystrom_dml.cognition.audit import DCNAuditStore, has_forbidden_audit_content, sanitize_audit_payload
from daystrom_dml.cognition.controller import CognitionController
from daystrom_dml.cognition.schema import CognitionFeedback, CognitionPlan
from daystrom_dml.tests.test_cognition_controller import FakeAdapter


def test_every_cognition_plan_has_decision_id_and_roundtrips():
    plan = CognitionPlan()

    restored = CognitionPlan.from_dict(plan.to_dict())

    assert plan.decision_id
    assert restored.decision_id == plan.decision_id


def test_audit_store_appends_and_reads_tail(tmp_path):
    store = DCNAuditStore(tmp_path / "dcn_audit.jsonl")

    store.append("feedback", {"decision_id": "d1", "outcome": "verified"})
    store.append("feedback", {"decision_id": "d2", "outcome": "accepted"})

    records = store.tail(limit=1)
    assert len(records) == 1
    assert records[0]["event_type"] == "feedback"
    assert records[0]["payload"]["decision_id"] == "d2"


def test_audit_redacts_secrets_raw_context_and_tool_logs():
    payload = {
        "decision_id": "d1",
        "api_key": "secret-value",
        "nested": {"authorization": "bearer token", "ok": "safe"},
        "raw_context": "full memory context",
        "tool_log": "terminal output",
    }

    clean = sanitize_audit_payload(payload)

    assert clean["api_key"] == "[REDACTED]"
    assert clean["nested"]["authorization"] == "[REDACTED]"
    assert clean["nested"]["ok"] == "safe"
    assert clean["raw_context"] == "[REDACTED]"
    assert clean["tool_log"] == "[REDACTED]"


def test_controller_writes_sanitized_packet_and_feedback_audit(tmp_path):
    store = DCNAuditStore(tmp_path / "dcn_audit.jsonl")
    controller = CognitionController(adapter=FakeAdapter(), audit_store=store, clock=lambda: 1.0)

    packet = controller.cognitive_packet("continue the DCN work")
    controller.feedback(
        CognitionFeedback(
            decision_id=packet.dcn_plan.decision_id,
            outcome="verified",
            signals={"api_key": "secret-value", "safe": True},
        )
    )
    records = controller.audit_tail(limit=10)

    assert [record["event_type"] for record in records] == ["cognitive_packet", "feedback"]
    assert records[0]["payload"]["decision_id"] == packet.dcn_plan.decision_id
    assert records[1]["payload"]["signals"]["api_key"] == "[REDACTED]"
    assert not has_forbidden_audit_content(records, ["secret-value", "terminal output", "full memory context"])
