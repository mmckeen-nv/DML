from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from daystrom_dml.context.adapters.vllm import VLLMCooperativeKVTrace, VLLMNativeTransitionResult
from daystrom_dml.context.execution import RuntimeExecutionError
from daystrom_dml.context.native_transition import NativeContextRuntimeStep, NativeContextTransitionPlan
from scripts.dcm_context_exhaustion_ab import run_context_exhaustion_ab
from scripts.dcm_native_context_transition_canary import LiveTransitionCanaryFailure


def _digest(char: str) -> str:
    return "sha256:" + char * 64


def _plan(parent: str, child: str, stable: int, current: int, marker: str, *, served: int = 512) -> NativeContextTransitionPlan:
    return NativeContextTransitionPlan(
        model_id="model",
        runtime_id="runtime",
        parent_packet_digest=marker * 64,
        current_packet_digest=("c" if marker == "a" else "d") * 64,
        parent_manifest_digest=(marker + "1") * 32,
        current_manifest_digest=(marker + "2") * 32,
        model_native_limit=1024,
        served_limit=served,
        current_tokens=current,
        stable_prefix_span_ids=[f"stable-{marker}"],
        stable_prefix_tokens=stable,
        suffix_span_ids=[f"suffix-{marker}"],
        suffix_tokens=current - stable,
        page_out=[],
        page_in=[],
        steps=[
            NativeContextRuntimeStep("restore_parent_prefix", [f"stable-{marker}"], 0, stable, "restore", checkpoint_id=f"parent-{marker}", checkpoint_digest=parent, binding_digest=_digest("e")),
            NativeContextRuntimeStep("prefill_suffix", [f"suffix-{marker}"], stable, current - stable, "prefill"),
            NativeContextRuntimeStep("checkpoint_current_generation", [f"stable-{marker}", f"suffix-{marker}"], 0, current, "save", checkpoint_digest=child),
        ],
        current_checkpoint_digest=child,
        served_overflow_tokens=max(0, current - served),
        served_limit_shortfall=1024 - served,
        feasible=True,
    )


def _trace(operation: str, digest: str, tokens: int, *, latency: float = 10.0, cpu: int = 0, gpu: int = 0, route: str = "not_applicable", output: str = "same", ready: bool = False, blocks: int = 0, purged: int = 0) -> VLLMCooperativeKVTrace:
    return VLLMCooperativeKVTrace(
        output_text=output,
        latency_ms=latency,
        prompt_tokens=tokens,
        completion_tokens=8,
        cached_tokens=0,
        checkpoint_digest=digest,
        operation=operation,
        reason_code="checkpoint_ready" if ready else "purge_complete" if operation == "purge" else f"{operation}_authorized",
        matched_tokens=cpu + gpu,
        gpu_apc_matched_tokens=gpu,
        cpu_offload_matched_tokens=cpu,
        cache_route=route,
        saved_tokens=tokens,
        purged_blocks=purged,
        purged_bytes=purged * 4096,
        shared_blocks=0,
        checkpoint_ready=ready,
        stored_blocks=blocks,
        expected_blocks=blocks,
        temperature=0.0,
        seed=17,
        max_tokens=8,
    )


class FakeAdapter:
    runtime_version = "0.20.0"
    model_id = "model"

    def __init__(self, plans: list[NativeContextTransitionPlan], *, cold_cpu: int = 0, cold_output: str = "same") -> None:
        self.plans = plans
        self.cold_cpu = cold_cpu
        self.cold_output = cold_output
        self.prompt_tokens = {"root": plans[0].stable_prefix_tokens, **{f"p{i}": plan.current_tokens for i, plan in enumerate(plans, 1)}}
        self.purged: list[str] = []

    def capabilities(self) -> Any:
        return SimpleNamespace(metadata={"endpoint_origin_digest": _digest("f")})

    def complete_with_checkpoint(self, prompt: str, **kwargs: Any) -> VLLMCooperativeKVTrace:
        is_root = prompt == "root"
        return _trace(
            "save",
            kwargs["checkpoint_digest"],
            self.prompt_tokens[prompt],
            latency=20.0,
            cpu=0 if is_root else self.cold_cpu,
            route="not_applicable" if is_root or self.cold_cpu == 0 else "cpu_fallback",
            output="same" if is_root else self.cold_output,
        )

    def wait_for_checkpoint_ready(self, **kwargs: Any) -> VLLMCooperativeKVTrace:
        return _trace("status", kwargs["checkpoint_digest"], 1, ready=True, blocks=4)

    def execute_native_transition(self, prompt: str, **kwargs: Any) -> VLLMNativeTransitionResult:
        plan = kwargs["plan"]
        execution = _trace("transition", plan.steps[0].checkpoint_digest or "", plan.current_tokens, latency=10.0, cpu=plan.stable_prefix_tokens, route="cpu_fallback")
        readiness = self.wait_for_checkpoint_ready(checkpoint_digest=plan.current_checkpoint_digest)
        return VLLMNativeTransitionResult(execution=execution, readiness=readiness)

    def purge_checkpoint(self, **kwargs: Any) -> VLLMCooperativeKVTrace:
        digest = kwargs["checkpoint_digest"]
        self.purged.append(digest)
        purged = 4 if len(self.purged) == 5 else 0
        return _trace("purge", digest, 1, purged=purged)


def _fixtures() -> tuple[list[NativeContextTransitionPlan], list[str]]:
    root, middle, final = _digest("1"), _digest("2"), _digest("3")
    return [
        _plan(root, middle, 96, 160, "a"),
        _plan(middle, final, 160, 224, "b"),
    ], ["root", "p1", "p2"]


def test_paired_ladder_proves_cold_and_managed_routes_and_latency() -> None:
    plans, prompts = _fixtures()
    adapter = FakeAdapter(plans)
    resets: list[bool] = []

    report = run_context_exhaustion_ab(
        adapter=adapter,  # type: ignore[arg-type]
        plans=plans,
        prompts=prompts,
        source_ref="test-source",
        ttl_seconds=300,
        order_seed=1701,
        reset_local_gpu_prefix_cache=lambda: resets.append(True),
    )

    assert report["result"] == "pass"
    assert report["logical_limit_extended"] is False
    assert report["summary"] == {
        "paired_rungs": 2,
        "enabled_wins": 2,
        "disabled_wins": 0,
        "ties": 0,
        "largest_tested_tokens": 224,
        "remaining_served_headroom_tokens": 288,
    }
    assert [rung["enabled"]["recompute_avoidance_tokens"] for rung in report["rungs"]] == [96, 160]
    assert all(rung["disabled"]["recompute_avoidance_tokens"] == 0 for rung in report["rungs"])
    assert all(rung["output_equivalent"] for rung in report["rungs"])
    assert len(resets) == 5
    assert report["physical_zeroization_complete"] is True
    assert len(adapter.purged) == 5


def test_ladder_rejects_served_limit_overflow_before_mutation() -> None:
    plans, prompts = _fixtures()
    plans[1] = _plan(_digest("2"), _digest("3"), 160, 600, "b", served=512)
    adapter = FakeAdapter(plans)

    with pytest.raises(RuntimeExecutionError, match="served logical limit"):
        run_context_exhaustion_ab(
            adapter=adapter,  # type: ignore[arg-type]
            plans=plans,
            prompts=prompts,
            source_ref="test-source",
            ttl_seconds=300,
            order_seed=1,
            reset_local_gpu_prefix_cache=lambda: None,
        )

    assert adapter.purged == []


@pytest.mark.parametrize("cold_cpu,cold_output", [(64, "same"), (0, "different")])
def test_ladder_fails_closed_on_contamination_or_output_divergence(cold_cpu: int, cold_output: str) -> None:
    plans, prompts = _fixtures()
    adapter = FakeAdapter(plans, cold_cpu=cold_cpu, cold_output=cold_output)

    with pytest.raises(LiveTransitionCanaryFailure) as captured:
        run_context_exhaustion_ab(
            adapter=adapter,  # type: ignore[arg-type]
            plans=plans,
            prompts=prompts,
            source_ref="test-source",
            ttl_seconds=300,
            order_seed=1,
            reset_local_gpu_prefix_cache=lambda: None,
        )

    assert captured.value.report["cleanup_complete"] is True
