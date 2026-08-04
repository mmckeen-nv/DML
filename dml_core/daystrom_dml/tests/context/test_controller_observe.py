import copy

import pytest

from daystrom_dml.api_contracts import ContractError, DaystromScope
from daystrom_dml.cognition.schema import CognitivePacket, CognitionPlan, RetrievalPlan
from daystrom_dml.context.adapters.api_messages import APIMessageAdapter
from daystrom_dml.context.controller import ContextController


class RetrievalWouldFail:
    def retrieve_context(self, *args, **kwargs):
        raise AssertionError("observe-only must not retrieve")


def test_context_controller_observe_reports_census_pressure_and_actions_without_mutation():
    adapter = APIMessageAdapter(token_estimator=lambda value: 8 if isinstance(value, list) else len(str(value).split()))
    controller = ContextController(runtime_adapter=adapter, retrieval_adapter=RetrievalWouldFail(), clock=lambda: 123.0)
    messages = [{"role": "user", "content": "please continue"}]
    messages_before = copy.deepcopy(messages)
    dcn_plan = CognitionPlan(retrieval_plan=RetrievalPlan(mode="semantic", queries=["continue work"], budget_tokens=100))

    observation = controller.observe(
        scope=DaystromScope(session_id="s1", tenant_id="tenant"),
        model_limits={"max_input_tokens": 20},
        current_prompt="please continue",
        current_messages=messages,
        dcn_plan=dcn_plan.to_dict(),
        dml_context={"items": [{"content": "memory one"}, {"text": "memory two"}]},
        dpm_overlay={"overlay_text": "tone overlay"},
        output_reservation=6,
    )

    assert messages == messages_before
    assert observation["mode"] == "observe_only"
    assert observation["scope"]["session_id"] == "s1"
    assert observation["segment_census"]["total_segments"] == 5
    assert observation["segment_census"]["by_kind"]["prepared_message"] == 1
    assert observation["token_estimate"]["available_input_tokens"] == 14
    assert observation["token_estimate"]["reserved_runtime_tokens"] == 0
    assert observation["pressure_state"]["state"] == "ok"
    assert observation["capabilities"]["api_messages"] is True
    assert observation["proposed_actions"] == []
    assert "dml_context_payload" not in observation["telemetry"]
    assert observation["audit"]["reason_codes"] == ["observe_only_no_mutation", "retrieval_not_performed"]


def test_context_controller_observe_flags_pressure_without_changing_prompt():
    controller = ContextController(runtime_adapter=APIMessageAdapter(token_estimator=lambda value: 28))
    prompt = "unchanged prompt"

    observation = controller.observe(
        scope={"tenant_id": "tenant"},
        model_limits={"max_input_tokens": 30},
        current_prompt=prompt,
        dcn_packet=CognitivePacket(assembled_context="packet context", packet_id="p1").to_dict(),
        output_reservation=8,
    )

    assert prompt == "unchanged prompt"
    assert observation["token_estimate"]["input_tokens"] == 28
    assert observation["pressure_state"]["state"] == "over_limit"
    assert observation["proposed_actions"][0]["reason_code"] == "context_over_limit"
    assert observation["audit"]["packet_id"] == "p1"


def test_context_controller_observe_protects_output_and_runtime_reservations():
    controller = ContextController(
        runtime_adapter=APIMessageAdapter(token_estimator=lambda value: 70),
        clock=lambda: 321.0,
    )

    observation = controller.observe(
        scope={"tenant_id": "tenant"},
        model_limits={"context_window_tokens": 100, "runtime_reserved_tokens": 10},
        current_prompt="unchanged prompt",
        output_reservation=20,
    )

    assert observation["token_estimate"]["max_input_tokens"] == 100
    assert observation["token_estimate"]["reserved_output_tokens"] == 20
    assert observation["token_estimate"]["reserved_runtime_tokens"] == 10
    assert observation["token_estimate"]["available_input_tokens"] == 70
    assert observation["pressure_state"] == {"state": "critical", "ratio": 1.0}
    assert observation["telemetry"]["reserved_output_tokens"] == 20
    assert observation["telemetry"]["reserved_runtime_tokens"] == 10


def test_context_controller_observe_argument_runtime_reservation_overrides_limits():
    controller = ContextController(runtime_adapter=APIMessageAdapter(token_estimator=lambda value: 40))

    observation = controller.observe(
        model_limits={"context_window_tokens": 100, "runtime_reserved_tokens": 30},
        current_prompt="prompt",
        output_reservation=20,
        runtime_reserved_tokens=5,
    )

    assert observation["token_estimate"]["available_input_tokens"] == 75
    assert observation["token_estimate"]["reserved_runtime_tokens"] == 5


@pytest.mark.parametrize(
    ("output_reservation", "runtime_reserved_tokens", "limits"),
    [
        (-1, None, {}),
        (0, -1, {}),
        (0, None, {"output_reservation_tokens": -1}),
        (0, None, {"runtime_reserved_tokens": -1}),
    ],
)
def test_context_controller_observe_rejects_negative_reservations(output_reservation, runtime_reserved_tokens, limits):
    controller = ContextController()

    with pytest.raises(ContractError):
        controller.observe(
            model_limits=limits,
            current_prompt="prompt",
            output_reservation=output_reservation,
            runtime_reserved_tokens=runtime_reserved_tokens,
        )
