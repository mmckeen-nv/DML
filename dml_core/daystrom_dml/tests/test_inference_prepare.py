import pytest

from daystrom_dml.api_contracts import ContractError, DaystromScope
from daystrom_dml.cognition.schema import CognitivePacket, CognitionPlan, RetrievalPlan
from daystrom_dml.frontier_pipeline import FrontierCompressionPipeline
from daystrom_dml.inference.prepare import DIPPreparationPipeline, InferencePreparationPipeline
from daystrom_dml.inference.schema import DIPPrepareRequest, DIPPrepareResult
from daystrom_dml.tests.test_provider_server import DummyAdapter


def test_dip_prepare_accepts_dcn_cognitive_packet():
    packet = CognitivePacket(
        dcn_plan=CognitionPlan(retrieval_plan=RetrievalPlan(mode="semantic", queries=["q"])),
        dml_context={"raw_context": "memory context"},
        dpm_overlay={"overlay_text": "bounded personality"},
        assembled_context="assembled request",
        packet_id="packet-1",
    )
    pipeline = InferencePreparationPipeline()

    result = pipeline.prepare(DIPPrepareRequest(cognitive_packet=packet, frontier_max_tokens=256))

    assert result.inference_enabled is False
    assert result.dcn_packet_id == "packet-1"
    assert result.dcn_policy_version == "dcn-policy-v0"
    assert result.dml_context_used is True
    assert "DCN plan" in result.frontier_prompt
    assert "memory context" in result.frontier_prompt


def test_dip_prepare_without_adapter_is_prepare_only_and_warns():
    result = InferencePreparationPipeline().prepare({"prompt": "hello", "frontier_max_tokens": 64})

    assert result.inference_enabled is False
    assert result.mode == "frontier_full"
    assert result.warnings == ["no_dml_adapter_configured"]
    assert "hello" in result.frontier_prompt


def test_dip_prepare_wraps_existing_frontier_pipeline_behavior_with_adapter():
    result = InferencePreparationPipeline(adapter=DummyAdapter()).prepare(
        DIPPrepareRequest(prompt="What should the agent remember?", scope=DaystromScope(session_id="s1"), top_k=4)
    )

    assert result.inference_enabled is False
    assert result.mode == "frontier_with_dml_context"
    assert result.dml_context_used is True
    assert "Provider memory text" in result.frontier_prompt
    assert result.telemetry["frontier_input_tokens"] > 0


def test_existing_frontier_compression_pipeline_still_works():
    result = FrontierCompressionPipeline(DummyAdapter()).prepare("What should the agent remember?", session_id="s1")

    assert result["mode"] == "frontier_with_dml_context"
    assert "Provider memory text" in result["dml_context"]


def test_dip_result_roundtrips_through_dict():
    result = DIPPrepareResult(
        prompt="p",
        frontier_prompt="fp",
        inference_enabled=False,
        mode="prepare_only",
        context_observation={"mode": "observe_only"},
        context_packet={"segments": []},
    )

    restored = DIPPrepareResult.from_dict(result.to_dict())

    assert restored.prompt == "p"
    assert restored.frontier_prompt == "fp"
    assert restored.inference_enabled is False
    assert restored.context_observation == {"mode": "observe_only"}
    assert restored.context_packet == {"segments": []}


def test_dip_prepare_can_attach_context_observation_without_changing_frontier_prompt():
    class ObserveController:
        def observe(self, **kwargs):
            assert kwargs["current_prompt"] == baseline.frontier_prompt
            assert kwargs["current_messages"] == [{"role": "user", "content": baseline.frontier_prompt}]
            return {
                "mode": "observe_only",
                "capabilities": {"api_messages": True},
                "pressure_state": {"state": "ok"},
                "token_estimate": {"input_tokens": 10},
                "audit": {"reason_codes": ["observe_only_no_mutation"]},
            }

    baseline = InferencePreparationPipeline().prepare({"prompt": "hello", "frontier_max_tokens": 64})
    result = InferencePreparationPipeline(context_controller=ObserveController()).prepare(
        {"prompt": "hello", "frontier_max_tokens": 64}
    )

    assert result.prompt == baseline.prompt
    assert result.frontier_prompt == baseline.frontier_prompt
    assert result.context_observation["mode"] == "observe_only"
    assert result.telemetry["context_pressure_state"] == "ok"
    assert result.telemetry["context_capabilities"] == {"api_messages": True}


def test_dip_prepare_observe_failure_is_fail_contained_and_preserves_prompt():
    class FailingController:
        def observe(self, **kwargs):
            raise RuntimeError("observe failed")

    baseline = InferencePreparationPipeline().prepare({"prompt": "hello", "frontier_max_tokens": 64})
    result = InferencePreparationPipeline(context_controller=FailingController()).prepare(
        {"prompt": "hello", "frontier_max_tokens": 64}
    )

    assert result.prompt == baseline.prompt
    assert result.frontier_prompt == baseline.frontier_prompt
    assert result.context_observation == {}
    assert result.warnings == ["no_dml_adapter_configured", "context_observation_failed"]
    assert result.telemetry["context_observation_warning"] == "context_observation_failed"
    assert result.telemetry["context_observation_error_type"] == "RuntimeError"


def test_dip_prepare_non_dict_observation_is_fail_contained_and_preserves_prompt():
    class BadController:
        def observe(self, **kwargs):
            return ["not", "a", "dict"]

    baseline = InferencePreparationPipeline().prepare({"prompt": "hello", "frontier_max_tokens": 64})
    result = InferencePreparationPipeline(context_controller=BadController()).prepare(
        {"prompt": "hello", "frontier_max_tokens": 64}
    )

    assert result.prompt == baseline.prompt
    assert result.frontier_prompt == baseline.frontier_prompt
    assert result.context_observation == {}
    assert result.warnings == ["no_dml_adapter_configured", "context_observation_invalid"]
    assert result.telemetry["context_observation_warning"] == "context_observation_invalid"


def test_dip_prepare_explicit_model_context_budget_fit_uses_total_context_and_reservations():
    result = InferencePreparationPipeline().prepare(
        DIPPrepareRequest(
            prompt="hello",
            frontier_max_tokens=64,
            model_context_tokens=128,
            runtime_reserved_tokens=8,
        )
    )

    assert result.token_budget.limit_tokens == 128
    assert result.token_budget.used_tokens == result.telemetry["frontier_input_tokens"]
    assert result.token_budget.reserved_tokens == 72
    assert result.token_budget.remaining_tokens == 128 - result.token_budget.used_tokens - 72


def test_dip_prepare_explicit_model_context_budget_overflow_fails_closed():
    with pytest.raises(ContractError, match="model_context_tokens cannot fit"):
        InferencePreparationPipeline().prepare(
            DIPPrepareRequest(
                prompt="hello",
                frontier_max_tokens=64,
                model_context_tokens=10,
                runtime_reserved_tokens=8,
            )
        )


def test_dip_prepare_without_explicit_model_context_derives_synthetic_total_limit():
    result = InferencePreparationPipeline().prepare({"prompt": "hello", "frontier_max_tokens": 64})

    assert result.token_budget.limit_tokens == result.token_budget.used_tokens + result.frontier_max_tokens
    assert result.token_budget.reserved_tokens == result.frontier_max_tokens
    assert result.token_budget.remaining_tokens == 0


def test_dip_prepare_passes_total_context_and_reservations_to_observe():
    class ObserveController:
        def observe(self, **kwargs):
            observed.update(kwargs)
            return {"pressure_state": {"state": "ok"}, "capabilities": {}}

    observed = {}
    result = InferencePreparationPipeline(context_controller=ObserveController()).prepare(
        DIPPrepareRequest(
            prompt="hello",
            frontier_max_tokens=64,
            model_context_tokens=128,
            runtime_reserved_tokens=8,
        )
    )

    assert observed["model_limits"] == {"context_window_tokens": 128}
    assert observed["output_reservation"] == 64
    assert observed["runtime_reserved_tokens"] == 8
    assert result.telemetry["context_pressure_state"] == "ok"


def test_dip_packet_observation_does_not_duplicate_embedded_components():
    class ObserveController:
        def observe(self, **kwargs):
            observed.update(kwargs)
            return {"pressure_state": {"state": "ok"}, "capabilities": {}}

    packet = CognitivePacket(
        dcn_plan=CognitionPlan(retrieval_plan=RetrievalPlan(mode="semantic", queries=["unique-query"])),
        dml_context={"raw_context": "unique-memory"},
        dpm_overlay={"overlay_text": "unique-overlay"},
        assembled_context="unique-request",
    )
    observed = {}

    result = InferencePreparationPipeline(context_controller=ObserveController()).prepare(
        DIPPrepareRequest(cognitive_packet=packet, frontier_max_tokens=64)
    )

    assert observed["current_messages"] == [{"role": "user", "content": result.frontier_prompt}]
    assert "dcn_plan" not in observed
    assert "dcn_packet" not in observed
    assert "dml_context" not in observed
    assert "dpm_overlay" not in observed
    assert result.frontier_prompt.count("unique-memory") == 1
    assert result.frontier_prompt.count("unique-overlay") == 1


def test_dip_observe_only_overflow_preserves_prompt_and_reports_pressure():
    class ObserveController:
        def observe(self, **kwargs):
            observed.update(kwargs)
            return {"pressure_state": {"state": "over_limit"}, "capabilities": {}}

    observed = {}
    baseline = InferencePreparationPipeline().prepare(DIPPrepareRequest(prompt="hello", frontier_max_tokens=64))
    result = InferencePreparationPipeline(context_controller=ObserveController()).prepare(
        DIPPrepareRequest(prompt="hello", frontier_max_tokens=64, model_context_tokens=10, runtime_reserved_tokens=8)
    )

    assert result.frontier_prompt == baseline.frontier_prompt
    assert observed["model_limits"] == {"context_window_tokens": 10}
    assert result.telemetry["context_pressure_state"] == "over_limit"
    assert result.telemetry["context_budget_overflow_observed"] is True
    assert "context_budget_overflow_observed" in result.warnings


def test_dip_preparation_alias_points_to_pipeline():
    assert DIPPreparationPipeline is InferencePreparationPipeline
