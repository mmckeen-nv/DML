from daystrom_dml.api_contracts import DaystromScope
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
    result = DIPPrepareResult(prompt="p", frontier_prompt="fp", inference_enabled=False, mode="prepare_only")

    restored = DIPPrepareResult.from_dict(result.to_dict())

    assert restored.prompt == "p"
    assert restored.frontier_prompt == "fp"
    assert restored.inference_enabled is False


def test_dip_preparation_alias_points_to_pipeline():
    assert DIPPreparationPipeline is InferencePreparationPipeline
