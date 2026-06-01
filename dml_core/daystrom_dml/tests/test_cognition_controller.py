from daystrom_dml.api_contracts import DaystromScope
from daystrom_dml.cognition.controller import CognitionController
from daystrom_dml.cognition.schema import CognitionEvent, CognitionFeedback


class FakeAdapter:
    def __init__(self):
        self.calls = []

    def retrieve_context(self, **kwargs):
        self.calls.append(kwargs)
        return {"raw_context": "remembered context", "items": [{"id": "m1"}]}


class FakeDPM:
    def __init__(self):
        self.calls = []

    def render_overlay(self, **kwargs):
        self.calls.append(kwargs)
        return {"overlay_text": "bounded personality"}


def test_controller_does_not_call_dml_for_simple_greeting():
    adapter = FakeAdapter()
    dpm = FakeDPM()
    controller = CognitionController(adapter=adapter, dpm=dpm, clock=lambda: 123.0)

    packet = controller.cognitive_packet("hello again", scope=DaystromScope(session_id="s1"))

    assert packet.dcn_plan.retrieval_plan.mode == "none"
    assert adapter.calls == []
    assert dpm.calls[0]["current_instruction"] == "hello again"
    assert packet.dml_context == {}
    assert "bounded personality" in packet.assembled_context


def test_controller_calls_dml_only_when_policy_requests_retrieval():
    adapter = FakeAdapter()
    dpm = FakeDPM()
    controller = CognitionController(adapter=adapter, dpm=dpm, clock=lambda: 456.0)

    packet = controller.cognitive_packet(CognitionEvent(content="continue the DCN work"), scope=DaystromScope(tenant_id="tenant", session_id="s2"))

    assert packet.dcn_plan.retrieval_plan.mode in {"resume", "hybrid"}
    assert len(adapter.calls) == 1
    assert adapter.calls[0]["tenant_id"] == "tenant"
    assert adapter.calls[0]["session_id"] == "s2"
    assert packet.dml_context["raw_context"] == "remembered context"
    assert packet.dpm_overlay["overlay_text"] == "bounded personality"
    assert "=== DPM Overlay ===" in packet.assembled_context
    assert "=== DML Context ===" in packet.assembled_context


def test_current_instruction_is_passed_to_dpm_for_conflict_suppression():
    dpm = FakeDPM()
    controller = CognitionController(dpm=dpm)

    controller.cognitive_packet("remember I prefer concise updates")

    assert dpm.calls[0]["current_instruction"] == "remember I prefer concise updates"
    assert dpm.calls[0]["budget_tokens"] == 300


def test_feedback_records_inspectable_result():
    controller = CognitionController(clock=lambda: 789.0)

    result = controller.feedback(CognitionFeedback(decision_id="p1", outcome="verified", signals={"test_passed": True}))

    assert result["accepted"] is True
    assert result["stored"] == 1
    assert result["feedback"]["recorded_at"] == 789.0
