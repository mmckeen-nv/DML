import json

import pytest

from daystrom_dml.api_contracts import ContractError, DaystromScope
from daystrom_dml.context import (
    ACTIVE_ADMISSION_MODE,
    OBSERVE_ONLY_MODE,
    ContextAuthority,
    ContextPriority,
    ContextSegment,
)
from daystrom_dml.context.admission import admit_context_segments
from daystrom_dml.context.controller import ContextController


def seg(
    segment_id,
    *,
    authority=ContextAuthority.REFERENCE,
    priority=ContextPriority.REFERENCE,
    tokens=1,
    content=None,
    role="system",
    scope=None,
):
    return ContextSegment(
        segment_id=segment_id,
        kind="test",
        content=segment_id if content is None else content,
        authority=authority,
        priority=priority,
        scope=scope or DaystromScope(session_id="s1"),
        source={"role": role},
        estimated_tokens=tokens,
    )


def test_admission_pins_immutable_and_current_instruction_and_preserves_original_order():
    scope = DaystromScope(session_id="s1")
    packet = admit_context_segments(
        scope=scope,
        segments=[
            seg("optional", authority=ContextAuthority.REFERENCE, tokens=10, scope=scope),
            seg("system", authority=ContextAuthority.IMMUTABLE, tokens=5, content={"policy": "keep"}, scope=scope),
            seg("current", authority=ContextAuthority.CURRENT_INSTRUCTION, tokens=5, content=["do it"], scope=scope),
        ],
        model_id="model",
        runtime_id="runtime",
        model_limit_tokens=30,
        output_reserved_tokens=10,
    )

    assert [segment.segment_id for segment in packet.segments] == ["optional", "system", "current"]
    assert packet.budget.admitted_input_tokens == 20
    assert packet.rendered_messages == [
        {"role": "user", "content": "optional"},
        {"role": "system", "content": "{'policy': 'keep'}"},
        {"role": "user", "content": "['do it']"},
    ]
    assert packet.decisions["by_segment"]["system"]["reason_code"] == "pinned_immutable"
    assert packet.decisions["by_segment"]["current"]["reason_code"] == "pinned_current_instruction"


def test_admission_fails_closed_when_pinned_segments_exceed_available_input():
    scope = DaystromScope(session_id="s1")

    with pytest.raises(ContractError, match="pinned segments exceed available input"):
        admit_context_segments(
            scope=scope,
            segments=[
                seg("system", authority=ContextAuthority.IMMUTABLE, tokens=8, scope=scope),
                seg("current", authority=ContextAuthority.CURRENT_INSTRUCTION, tokens=5, scope=scope),
            ],
            model_id="model",
            runtime_id="runtime",
            model_limit_tokens=20,
            output_reserved_tokens=5,
            runtime_reserved_tokens=3,
        )


def test_optional_segments_use_authority_priority_and_stable_tie_order_for_hard_budget():
    scope = DaystromScope(session_id="s1")
    packet = admit_context_segments(
        scope=scope,
        segments=[
            seg("ref-critical-a", authority=ContextAuthority.REFERENCE, priority=ContextPriority.CRITICAL, tokens=4, scope=scope),
            seg("trusted-working", authority=ContextAuthority.TRUSTED_CONTROL, priority=ContextPriority.WORKING, tokens=4, scope=scope),
            seg("untrusted-critical", authority=ContextAuthority.UNTRUSTED_DATA, priority=ContextPriority.CRITICAL, tokens=4, scope=scope),
            seg("trusted-critical-a", authority=ContextAuthority.TRUSTED_CONTROL, priority=ContextPriority.CRITICAL, tokens=4, scope=scope),
            seg("trusted-critical-b", authority=ContextAuthority.TRUSTED_CONTROL, priority=ContextPriority.CRITICAL, tokens=4, scope=scope),
        ],
        model_id="model",
        runtime_id="runtime",
        model_limit_tokens=12,
    )

    assert [segment.segment_id for segment in packet.segments] == [
        "trusted-working",
        "trusted-critical-a",
        "trusted-critical-b",
    ]
    assert packet.budget.admitted_input_tokens == 12
    assert packet.decisions["by_segment"]["ref-critical-a"]["reason_code"] == "omitted_budget_exhausted"
    assert packet.decisions["by_segment"]["untrusted-critical"]["reason_code"] == "omitted_budget_exhausted"


def test_output_and_runtime_reservations_are_applied_before_input_admission():
    scope = DaystromScope(session_id="s1")
    packet = admit_context_segments(
        scope=scope,
        segments=[
            seg("current", authority=ContextAuthority.CURRENT_INSTRUCTION, tokens=6, scope=scope),
            seg("optional", authority=ContextAuthority.TRUSTED_CONTROL, tokens=1, scope=scope),
        ],
        model_id="model",
        runtime_id="runtime",
        model_limit_tokens=10,
        output_reserved_tokens=2,
        runtime_reserved_tokens=2,
    )

    assert [segment.segment_id for segment in packet.segments] == ["current"]
    assert packet.budget.available_input_tokens == 6
    assert packet.decisions["by_segment"]["optional"]["reason_code"] == "omitted_budget_exhausted"


def test_page_out_success_and_failure_are_contained_and_recorded():
    scope = DaystromScope(session_id="s1")

    def page_out(page_scope, segment):
        assert page_scope == scope
        if segment.segment_id == "fail":
            raise RuntimeError("failed on SECRET_EXCEPTION_PAYLOAD")
        return {"page_id": segment.segment_id, "payload": "SECRET_HANDLE_PAYLOAD"}

    packet = admit_context_segments(
        scope=scope,
        segments=[
            seg("keep", authority=ContextAuthority.TRUSTED_CONTROL, tokens=2, scope=scope),
            seg("ok", authority=ContextAuthority.REFERENCE, tokens=2, scope=scope),
            seg("fail", authority=ContextAuthority.UNTRUSTED_DATA, tokens=2, scope=scope),
        ],
        model_id="model",
        runtime_id="runtime",
        model_limit_tokens=2,
        page_out=page_out,
    )

    assert [segment.segment_id for segment in packet.segments] == ["keep"]
    assert packet.decisions["by_segment"]["ok"]["page_out"] == {"status": "stored", "handle": {"page_id": "ok"}}
    assert packet.decisions["by_segment"]["fail"]["page_out"] == {
        "status": "failed",
        "error_type": "RuntimeError",
    }
    serialized = json.dumps(packet.to_dict(), sort_keys=True)
    assert "SECRET_HANDLE_PAYLOAD" not in serialized
    assert "SECRET_EXCEPTION_PAYLOAD" not in serialized


def test_all_authority_classes_render_without_weaker_system_role_elevation():
    scope = DaystromScope(session_id="s1")
    packet = admit_context_segments(
        scope=scope,
        segments=[
            seg("immutable", authority=ContextAuthority.IMMUTABLE, scope=scope),
            seg("current", authority=ContextAuthority.CURRENT_INSTRUCTION, scope=scope),
            seg("trusted", authority=ContextAuthority.TRUSTED_CONTROL, scope=scope),
            seg("reference", authority=ContextAuthority.REFERENCE, scope=scope),
            seg("untrusted", authority=ContextAuthority.UNTRUSTED_DATA, scope=scope),
        ],
        model_id="model",
        runtime_id="runtime",
        model_limit_tokens=10,
    )

    assert [message["role"] for message in packet.rendered_messages] == ["system", "user", "user", "user", "user"]
    assert [message["content"] for message in packet.rendered_messages] == [
        "immutable",
        "current",
        "trusted",
        "reference",
        "untrusted",
    ]


def test_controller_defaults_are_observe_only_and_unknown_modes_are_rejected():
    default_controller = ContextController()

    assert default_controller.mode == OBSERVE_ONLY_MODE
    with pytest.raises(ContractError, match="active_admission"):
        default_controller.admit(
            segments=[],
            model_id="model",
            runtime_id="runtime",
            model_limit_tokens=1,
        )

    with pytest.raises(ContractError, match="mode"):
        ContextController(mode="active")


def test_controller_active_mode_returns_json_roundtrippable_context_packet():
    scope = DaystromScope(session_id="s1")
    controller = ContextController(mode=ACTIVE_ADMISSION_MODE)

    packet = controller.admit(
        scope=scope,
        segments=[seg("current", authority=ContextAuthority.CURRENT_INSTRUCTION, tokens=3, scope=scope)],
        model_id="model",
        runtime_id="runtime",
        model_limit_tokens=8,
        output_reserved_tokens=2,
    )
    data = json.loads(json.dumps(packet.to_dict(), sort_keys=True))

    assert type(packet).from_dict(data) == packet
    assert packet.manifest.decisions == packet.decisions
    assert packet.capabilities.model_id == "model"
    assert packet.capabilities.backend_id == "runtime"


def test_admission_copies_input_segments_and_packet_serialization_revalidates_integrity():
    scope = DaystromScope(session_id="s1")
    original = seg("current", authority=ContextAuthority.CURRENT_INSTRUCTION, tokens=1, scope=scope)
    packet = admit_context_segments(
        scope=scope,
        segments=[original],
        model_id="model",
        runtime_id="runtime",
        model_limit_tokens=8,
    )

    original.estimated_tokens = 99
    assert packet.segments[0].estimated_tokens == 1
    assert packet.to_dict()["segments"][0]["estimated_tokens"] == 1

    packet.segments[0].estimated_tokens = 99
    with pytest.raises(ContractError, match="segment token total cannot exceed available input budget"):
        packet.to_dict()


def test_packet_serialization_rejects_content_and_rendered_message_drift():
    scope = DaystromScope(session_id="s1")
    packet = admit_context_segments(
        scope=scope,
        segments=[seg("current", authority=ContextAuthority.CURRENT_INSTRUCTION, tokens=1, scope=scope)],
        model_id="model",
        runtime_id="runtime",
        model_limit_tokens=8,
    )
    packet.segments[0].content = "tampered segment"
    with pytest.raises(ContractError, match="packet_content_digest"):
        packet.to_dict()

    packet = admit_context_segments(
        scope=scope,
        segments=[seg("current", authority=ContextAuthority.CURRENT_INSTRUCTION, tokens=1, scope=scope)],
        model_id="model",
        runtime_id="runtime",
        model_limit_tokens=8,
    )
    packet.rendered_messages[0]["content"] = "tampered prompt"
    with pytest.raises(ContractError, match="packet_content_digest"):
        packet.to_dict()


def test_endpoint_identity_is_digest_bound_without_persisting_url_or_credentials():
    scope = DaystromScope(session_id="s1")
    endpoint = "https://user:pass@example.test/v1/chat/completions?api_key=secret"

    packet = admit_context_segments(
        scope=scope,
        segments=[seg("current", authority=ContextAuthority.CURRENT_INSTRUCTION, scope=scope)],
        model_id="m",
        runtime_id="r",
        endpoint_url=endpoint,
        model_limit_tokens=10,
    )

    digest = packet.capabilities.metadata["endpoint_url_digest"]
    rotated = admit_context_segments(
        scope=scope,
        segments=[seg("current", authority=ContextAuthority.CURRENT_INSTRUCTION, scope=scope)],
        model_id="m",
        runtime_id="r",
        endpoint_url="https://example.test/v1/chat/completions?api_key=other-secret",
        model_limit_tokens=10,
    )
    serialized = json.dumps(packet.to_dict(), sort_keys=True)
    assert isinstance(digest, str) and len(digest) == 64
    assert rotated.capabilities.metadata["endpoint_url_digest"] == digest
    assert endpoint not in serialized
    assert "user:pass" not in serialized
    assert "api_key=secret" not in serialized


def test_invalid_endpoint_identity_fails_before_page_out_side_effects():
    scope = DaystromScope(session_id="s1")
    paged: list[str] = []

    with pytest.raises(ContractError, match="endpoint_url"):
        admit_context_segments(
            scope=scope,
            segments=[seg("omitted", tokens=9, scope=scope)],
            model_id="m",
            runtime_id="r",
            endpoint_url="",
            model_limit_tokens=1,
            page_out=lambda _, item: paged.append(item.segment_id),
        )

    assert paged == []
