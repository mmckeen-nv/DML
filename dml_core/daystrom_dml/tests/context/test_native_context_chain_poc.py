from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from daystrom_dml.context.adapters.vllm import (
    VLLMCooperativeKVTrace,
    VLLMNativeTransitionResult,
)
from daystrom_dml.context.execution import RuntimeExecutionError
from daystrom_dml.context.native_transition import (
    NativeContextRuntimeStep,
    NativeContextTransitionPlan,
)
from scripts.dcm_native_context_chain_poc import run_chain_poc
from scripts.dcm_native_context_transition_canary import LiveTransitionCanaryFailure


def _digest(char: str) -> str:
    return "sha256:" + char * 64


def _plan(*, parent: str, child: str, stable: int, current: int, marker: str) -> NativeContextTransitionPlan:
    suffix = current - stable
    return NativeContextTransitionPlan(
        model_id="model",
        runtime_id="runtime",
        parent_packet_digest=marker * 64,
        current_packet_digest=("c" if marker == "a" else "d") * 64,
        parent_manifest_digest=(marker + "1") * 32,
        current_manifest_digest=(marker + "2") * 32,
        model_native_limit=1024,
        served_limit=512,
        current_tokens=current,
        stable_prefix_span_ids=[f"stable-{marker}"],
        stable_prefix_tokens=stable,
        suffix_span_ids=[f"suffix-{marker}"],
        suffix_tokens=suffix,
        page_out=[],
        page_in=[],
        steps=[
            NativeContextRuntimeStep(
                operation="restore_parent_prefix",
                span_ids=[f"stable-{marker}"],
                token_start=0,
                token_count=stable,
                reason_code="restore",
                checkpoint_id=f"checkpoint-{marker}",
                checkpoint_digest=parent,
                binding_digest=_digest("e"),
            ),
            NativeContextRuntimeStep(
                operation="prefill_suffix",
                span_ids=[f"suffix-{marker}"],
                token_start=stable,
                token_count=suffix,
                reason_code="prefill",
            ),
            NativeContextRuntimeStep(
                operation="checkpoint_current_generation",
                span_ids=[f"stable-{marker}", f"suffix-{marker}"],
                token_start=0,
                token_count=current,
                reason_code="save",
                checkpoint_digest=child,
            ),
        ],
        current_checkpoint_digest=child,
        served_overflow_tokens=0,
        served_limit_shortfall=512,
        feasible=True,
    )


def _trace(
    operation: str,
    digest: str,
    *,
    prompt_tokens: int,
    output: str = "same",
    cpu: int = 0,
    route: str = "not_applicable",
    ready: bool = False,
    stored: int = 0,
    expected: int = 0,
    purged: int = 0,
) -> VLLMCooperativeKVTrace:
    return VLLMCooperativeKVTrace(
        output_text=output,
        latency_ms=1.0,
        prompt_tokens=prompt_tokens,
        completion_tokens=8,
        cached_tokens=0,
        checkpoint_digest=digest,
        operation=operation,
        reason_code=(
            "checkpoint_ready"
            if ready
            else "purge_complete"
            if operation == "purge"
            else f"{operation}_authorized"
        ),
        matched_tokens=cpu,
        gpu_apc_matched_tokens=0,
        cpu_offload_matched_tokens=cpu,
        cache_route=route,
        saved_tokens=prompt_tokens,
        purged_blocks=purged,
        purged_bytes=purged * 4096,
        shared_blocks=0,
        checkpoint_ready=ready,
        stored_blocks=stored,
        expected_blocks=expected,
        temperature=0.0,
        seed=17,
        max_tokens=8,
    )


class FakeAdapter:
    runtime_version = "0.20.0"
    model_id = "model"

    def __init__(
        self,
        plans: list[NativeContextTransitionPlan],
        *,
        transition_cpu: list[int] | None = None,
        cold_output: str = "same",
        purge_counts: list[int] | None = None,
    ) -> None:
        self.plans = plans
        self.transition_cpu = transition_cpu or [
            plan.stable_prefix_tokens for plan in plans
        ]
        self.cold_output = cold_output
        self.purge_counts = list(purge_counts or [0, 0, 0, 2])
        self.transition_index = 0
        self.purged: list[str] = []
        self.completions: list[dict[str, Any]] = []

    def capabilities(self) -> Any:
        return SimpleNamespace(metadata={"endpoint_origin_digest": _digest("f")})

    def complete_with_checkpoint(self, prompt: str, **kwargs: Any) -> VLLMCooperativeKVTrace:
        self.completions.append(dict(kwargs))
        tokens = self.plans[0].stable_prefix_tokens if len(self.completions) == 1 else self.plans[-1].current_tokens
        output = "same" if len(self.completions) == 1 else self.cold_output
        return _trace(
            "save", kwargs["checkpoint_digest"], prompt_tokens=tokens, output=output
        )

    def wait_for_checkpoint_ready(self, **kwargs: Any) -> VLLMCooperativeKVTrace:
        return _trace(
            "status",
            kwargs["checkpoint_digest"],
            prompt_tokens=1,
            ready=True,
            stored=2,
            expected=2,
        )

    def execute_native_transition(self, prompt: str, **kwargs: Any) -> VLLMNativeTransitionResult:
        plan = kwargs["plan"]
        parent = plan.steps[0].checkpoint_digest
        child = plan.current_checkpoint_digest
        assert parent is not None and child is not None
        cpu = self.transition_cpu[self.transition_index]
        self.transition_index += 1
        execution = _trace(
            "transition",
            parent,
            prompt_tokens=plan.current_tokens,
            cpu=cpu,
            route="cpu_fallback",
        )
        readiness = self.wait_for_checkpoint_ready(checkpoint_digest=child)
        return VLLMNativeTransitionResult(execution=execution, readiness=readiness)

    def purge_checkpoint(self, **kwargs: Any) -> VLLMCooperativeKVTrace:
        digest = kwargs["checkpoint_digest"]
        self.purged.append(digest)
        purged = self.purge_counts[len(self.purged) - 1]
        return _trace("purge", digest, prompt_tokens=1, purged=purged)


def test_two_hop_chain_restores_each_parent_and_cleans_every_checkpoint() -> None:
    root, middle, final = _digest("1"), _digest("2"), _digest("3")
    plans = [
        _plan(parent=root, child=middle, stable=96, current=160, marker="a"),
        _plan(parent=middle, child=final, stable=160, current=224, marker="b"),
    ]
    adapter = FakeAdapter(plans)
    resets: list[bool] = []

    report = run_chain_poc(
        adapter=adapter,  # type: ignore[arg-type]
        plans=plans,
        prompts=["root", "middle", "final"],
        source_ref="test-source",
        ttl_seconds=300,
        reset_local_gpu_prefix_cache=lambda: resets.append(True),
    )

    assert report["result"] == "pass"
    assert report["output_equivalent"] is True
    assert [hop["cpu_offload_matched_tokens"] for hop in report["hops"]] == [96, 160]
    assert report["native_reuse_grew"] is True
    assert report["physical_zeroization_complete"] is True
    assert report["physically_purged_blocks"] == report["expected_physical_blocks"] == 2
    assert len(resets) == 4
    assert adapter.completions[0]["max_tokens"] >= 2
    assert adapter.purged == [report["cold_checkpoint_digest"], final, middle, root]
    assert all(event.get("output_text") is None for event in report["events"])


def test_chain_rejects_broken_parent_child_lineage_before_mutation() -> None:
    root, middle, wrong, final = (
        _digest("1"),
        _digest("2"),
        _digest("9"),
        _digest("3"),
    )
    plans = [
        _plan(parent=root, child=middle, stable=96, current=160, marker="a"),
        _plan(parent=wrong, child=final, stable=160, current=224, marker="b"),
    ]
    adapter = FakeAdapter(plans)

    with pytest.raises(RuntimeExecutionError, match="checkpoint lineage"):
        run_chain_poc(
            adapter=adapter,  # type: ignore[arg-type]
            plans=plans,
            prompts=["root", "middle", "final"],
            source_ref="test-source",
            ttl_seconds=300,
            reset_local_gpu_prefix_cache=lambda: None,
        )

    assert adapter.completions == []
    assert adapter.purged == []


def test_chain_fails_closed_when_native_reuse_does_not_grow() -> None:
    root, middle, final = _digest("1"), _digest("2"), _digest("3")
    plans = [
        _plan(parent=root, child=middle, stable=96, current=160, marker="a"),
        _plan(parent=middle, child=final, stable=160, current=224, marker="b"),
    ]
    adapter = FakeAdapter(plans, transition_cpu=[96, 96])

    with pytest.raises(LiveTransitionCanaryFailure) as captured:
        run_chain_poc(
            adapter=adapter,  # type: ignore[arg-type]
            plans=plans,
            prompts=["root", "middle", "final"],
            source_ref="test-source",
            ttl_seconds=300,
            reset_local_gpu_prefix_cache=lambda: None,
        )

    assert captured.value.report["result"] == "fail"
    assert captured.value.report["error_type"] == "RuntimeExecutionError"
    assert adapter.purged == [final, middle, root]


def test_chain_fails_closed_when_final_output_differs_from_cold_control() -> None:
    root, middle, final = _digest("1"), _digest("2"), _digest("3")
    plans = [
        _plan(parent=root, child=middle, stable=96, current=160, marker="a"),
        _plan(parent=middle, child=final, stable=160, current=224, marker="b"),
    ]
    adapter = FakeAdapter(plans, cold_output="different")

    with pytest.raises(LiveTransitionCanaryFailure) as captured:
        run_chain_poc(
            adapter=adapter,  # type: ignore[arg-type]
            plans=plans,
            prompts=["root", "middle", "final"],
            source_ref="test-source",
            ttl_seconds=300,
            reset_local_gpu_prefix_cache=lambda: None,
        )

    assert captured.value.report["failure_stage"] == "equivalence"
    assert captured.value.report["physical_zeroization_complete"] is True
    assert len(adapter.purged) == 4


def test_chain_requires_complete_physical_zeroization_accounting() -> None:
    root, middle, final = _digest("1"), _digest("2"), _digest("3")
    plans = [
        _plan(parent=root, child=middle, stable=96, current=160, marker="a"),
        _plan(parent=middle, child=final, stable=160, current=224, marker="b"),
    ]
    adapter = FakeAdapter(plans, purge_counts=[0, 0, 0, 0])

    with pytest.raises(LiveTransitionCanaryFailure) as captured:
        run_chain_poc(
            adapter=adapter,  # type: ignore[arg-type]
            plans=plans,
            prompts=["root", "middle", "final"],
            source_ref="test-source",
            ttl_seconds=300,
            reset_local_gpu_prefix_cache=lambda: None,
        )

    report = captured.value.report
    assert report["failure_stage"] == "cleanup_accounting"
    assert report["cleanup_complete"] is True
    assert report["physical_zeroization_complete"] is False
    assert report["physically_purged_blocks"] == 0
