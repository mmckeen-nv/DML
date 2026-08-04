import copy

from daystrom_dml.api_contracts import DaystromScope
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
