from __future__ import annotations

import json
from pathlib import Path
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
from scripts.dcm_native_context_transition_canary import (
    LiveTransitionCanaryFailure,
    _load_plan,
    run_canary,
)


def _trace(
    operation: str,
    digest: str,
    *,
    prompt_tokens: int,
    output: str = "OK",
    ready: bool = False,
    stored: int = 0,
    expected: int = 0,
    cpu: int = 0,
    gpu: int = 0,
    route: str = "not_applicable",
    reason: str | None = None,
) -> VLLMCooperativeKVTrace:
    return VLLMCooperativeKVTrace(
        output_text=output,
        latency_ms=1.0,
        prompt_tokens=prompt_tokens,
        completion_tokens=1,
        cached_tokens=gpu,
        checkpoint_digest=digest,
        operation=operation,
        reason_code=reason or f"{operation}_authorized",
        matched_tokens=cpu,
        gpu_apc_matched_tokens=gpu,
        cpu_offload_matched_tokens=cpu,
        cache_route=route,
        saved_tokens=prompt_tokens,
        purged_blocks=2 if operation == "purge" else 0,
        purged_bytes=4096 if operation == "purge" else 0,
        shared_blocks=0,
        checkpoint_ready=ready,
        stored_blocks=stored,
        expected_blocks=expected,
        temperature=0.0,
        seed=17,
        max_tokens=8,
    )


def _plan() -> NativeContextTransitionPlan:
    parent = "sha256:" + "1" * 64
    child = "sha256:" + "2" * 64
    return NativeContextTransitionPlan(
        model_id="model",
        runtime_id="vllm-test",
        parent_packet_digest="a" * 64,
        current_packet_digest="b" * 64,
        parent_manifest_digest="c" * 64,
        current_manifest_digest="d" * 64,
        model_native_limit=1024,
        served_limit=512,
        current_tokens=160,
        stable_prefix_span_ids=["prefix"],
        stable_prefix_tokens=96,
        suffix_span_ids=["suffix"],
        suffix_tokens=64,
        page_out=[],
        page_in=[],
        steps=[
            NativeContextRuntimeStep(
                operation="restore_parent_prefix",
                span_ids=["prefix"],
                token_start=0,
                token_count=96,
                reason_code="restore",
                checkpoint_id="parent-id",
                checkpoint_digest=parent,
                binding_digest="sha256:" + "e" * 64,
            ),
            NativeContextRuntimeStep(
                operation="prefill_suffix",
                span_ids=["suffix"],
                token_start=96,
                token_count=64,
                reason_code="prefill",
            ),
            NativeContextRuntimeStep(
                operation="checkpoint_current_generation",
                span_ids=["prefix", "suffix"],
                token_start=0,
                token_count=160,
                reason_code="save",
                checkpoint_digest=child,
            ),
        ],
        current_checkpoint_digest=child,
        served_overflow_tokens=0,
        served_limit_shortfall=512,
        feasible=True,
    )


class FakeAdapter:
    runtime_version = "0.20.0"
    model_id = "model"

    def __init__(
        self, *, transition_route: str = "cpu_fallback", fail_purge: bool = False
    ) -> None:
        self.plan = _plan()
        self.parent = self.plan.steps[0].checkpoint_digest
        self.child = self.plan.current_checkpoint_digest
        assert self.parent is not None
        assert self.child is not None
        self.transition_route = transition_route
        self.fail_purge = fail_purge
        self.purged: list[str] = []
        self.completions: list[dict[str, Any]] = []

    def capabilities(self) -> Any:
        return SimpleNamespace(metadata={"endpoint_origin_digest": "sha256:" + "f" * 64})

    def complete_with_checkpoint(self, prompt: str, **kwargs: Any) -> VLLMCooperativeKVTrace:
        self.completions.append(dict(kwargs))
        operation = kwargs["operation"]
        digest = kwargs["checkpoint_digest"]
        tokens = 96 if digest == self.parent else 160
        return _trace(operation, digest, prompt_tokens=tokens, output="same")

    def wait_for_checkpoint_ready(self, **kwargs: Any) -> VLLMCooperativeKVTrace:
        return _trace(
            "status",
            kwargs["checkpoint_digest"],
            prompt_tokens=1,
            ready=True,
            stored=2,
            expected=2,
            reason="checkpoint_ready",
        )

    def execute_native_transition(self, prompt: str, **kwargs: Any) -> VLLMNativeTransitionResult:
        cpu = 96 if self.transition_route == "cpu_fallback" else 0
        gpu = 0 if self.transition_route == "cpu_fallback" else 96
        execution = _trace(
            "transition",
            self.parent,
            prompt_tokens=160,
            output="same",
            cpu=cpu,
            gpu=gpu,
            route=self.transition_route,
            reason="transition_authorized",
        )
        readiness = self.wait_for_checkpoint_ready(checkpoint_digest=self.child)
        return VLLMNativeTransitionResult(execution=execution, readiness=readiness)

    def purge_checkpoint(self, **kwargs: Any) -> VLLMCooperativeKVTrace:
        digest = kwargs["checkpoint_digest"]
        self.purged.append(digest)
        if self.fail_purge and len(self.purged) == 1:
            raise RuntimeExecutionError("bounded purge failure")
        return _trace(
            "purge",
            digest,
            prompt_tokens=1,
            reason="purge_complete",
        )


def test_live_canary_requires_cpu_only_transition_and_cleans_all_checkpoints() -> None:
    adapter = FakeAdapter()
    resets: list[bool] = []

    report = run_canary(
        adapter=adapter,  # type: ignore[arg-type]
        plan=adapter.plan,
        parent_prompt="parent",
        current_prompt="current",
        source_ref="reviewed-sha",
        ttl_seconds=300,
        reset_local_gpu_prefix_cache=lambda: resets.append(True),
    )

    assert report["result"] == "pass"
    assert report["output_equivalent"] is True
    # vLLM 0.20 eager CPU offload only considers previously confirmed blocks.
    # Keep a post-prefill decode step so the last full prompt block is scheduled
    # before the parent save request finishes (upstream TODO: flush on finish).
    assert adapter.completions[0]["max_tokens"] >= 2
    assert len(resets) == 2
    assert len(adapter.purged) == 3
    assert all(event.get("output_text") is None for event in report["events"])


def test_live_canary_rejects_gpu_only_reuse_and_still_attempts_cleanup() -> None:
    adapter = FakeAdapter(transition_route="gpu_apc")

    with pytest.raises(RuntimeExecutionError, match="live transition canary failed"):
        run_canary(
            adapter=adapter,  # type: ignore[arg-type]
            plan=adapter.plan,
            parent_prompt="parent",
            current_prompt="current",
            source_ref="reviewed-sha",
            ttl_seconds=300,
            reset_local_gpu_prefix_cache=lambda: None,
        )

    assert adapter.purged == [adapter.child, adapter.parent]


def test_live_canary_preserves_sanitized_cleanup_failure_evidence() -> None:
    adapter = FakeAdapter(fail_purge=True)

    with pytest.raises(LiveTransitionCanaryFailure) as captured:
        run_canary(
            adapter=adapter,  # type: ignore[arg-type]
            plan=adapter.plan,
            parent_prompt="parent",
            current_prompt="current",
            source_ref="reviewed-sha",
            ttl_seconds=300,
            reset_local_gpu_prefix_cache=lambda: None,
        )

    report = captured.value.report
    assert report["failure_stage"] == "cleanup"
    assert report["cleanup_complete"] is False
    assert report["cleanup"][0]["reason_code"] == "cleanup_failed"
    assert "bounded purge failure" not in json.dumps(report)
    assert len(adapter.purged) == 3


def test_load_plan_requires_explicit_passing_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "plan.json"
    artifact.write_text(json.dumps(_plan().to_dict()), encoding="utf-8")

    with pytest.raises(RuntimeExecutionError, match="not marked passing"):
        _load_plan(artifact)

    passing = {**_plan().to_dict(), "pass": True}
    artifact.write_text(json.dumps(passing), encoding="utf-8")
    assert _load_plan(artifact).plan_digest == _plan().plan_digest
