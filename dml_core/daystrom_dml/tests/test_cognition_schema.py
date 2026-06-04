import json

import pytest

from daystrom_dml.api_contracts import ContractError, DaystromScope, RiskInfo
from daystrom_dml.cognition.schema import (
    FORBIDDEN_WRITEBACK_CLASSES,
    CognitivePacket,
    CognitionEvent,
    CognitionPlan,
    RetrievalPlan,
    WritebackPlan,
)


def test_cognition_plan_roundtrips_through_dict_and_json():
    plan = CognitionPlan(
        retrieval_plan=RetrievalPlan(mode="hybrid", queries=["continue DCN work"]),
        risk=RiskInfo(level="high", reasons=["deploy"], requires_confirmation=True),
        reason_codes=["memory_request", "high_risk"],
    )

    restored = CognitionPlan.from_dict(plan.to_dict())
    restored_json = CognitionPlan.from_json(plan.to_json())

    assert restored == plan
    assert restored_json == plan
    assert json.loads(plan.to_json())["retrieval_plan"]["mode"] == "hybrid"
    assert json.loads(plan.to_json())["risk"]["level"] == "high"


def test_default_user_message_event_and_plan_are_valid():
    event = CognitionEvent(content="hello again")
    plan = CognitionPlan()

    assert event.type == "user_message"
    assert plan.retrieval_plan.mode == "none"
    assert plan.writeback_plan.forbidden_classes == FORBIDDEN_WRITEBACK_CLASSES


def test_writeback_forbidden_classes_are_rejected_on_init_and_from_dict():
    with pytest.raises(ContractError):
        WritebackPlan(mode="durable_signal_only", candidate_classes=["raw_transcript"])

    with pytest.raises(ContractError):
        WritebackPlan.from_dict({"mode": "durable_signal_only", "candidate_classes": ["secret"]})


def test_cognitive_packet_keeps_dcn_dml_dpm_context_separate():
    packet = CognitivePacket(
        scope=DaystromScope(session_id="s1"),
        dcn_plan=CognitionPlan(retrieval_plan=RetrievalPlan(mode="semantic", queries=["q"])),
        dml_context={"raw_context": "memory"},
        dpm_overlay={"overlay_text": "personality"},
        assembled_context="assembled",
    )

    data = packet.to_dict()
    restored = CognitivePacket.from_dict(data)

    assert data["packet_version"] == "daystrom-cognitive-packet-v1"
    assert data["dcn_plan"]["retrieval_plan"]["mode"] == "semantic"
    assert data["dml_context"] == {"raw_context": "memory"}
    assert data["dpm_overlay"] == {"overlay_text": "personality"}
    assert restored.scope.session_id == "s1"


def test_cognitive_packet_rejects_unknown_packet_version():
    with pytest.raises(ContractError, match="packet_version"):
        CognitivePacket(packet_version="daystrom-cognitive-packet-v2")

    payload = CognitivePacket().to_dict()
    payload["packet_version"] = "daystrom-cognitive-packet-v2"
    with pytest.raises(ContractError, match="packet_version"):
        CognitivePacket.from_dict(payload)
