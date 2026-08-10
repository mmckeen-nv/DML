#!/usr/bin/env python3
"""Run a paired managed-restore versus cold-recomputation context ladder."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Sequence

_DML_CORE = Path(__file__).resolve().parents[1]
if str(_DML_CORE) not in sys.path:
    sys.path.insert(0, str(_DML_CORE))

from daystrom_dml.context.adapters.vllm import VLLMCooperativeExecutionAdapter  # noqa: E402
from daystrom_dml.context.execution import RuntimeExecutionError  # noqa: E402
from daystrom_dml.context.native_transition import NativeContextTransitionPlan  # noqa: E402
from daystrom_dml.context.probe import atomic_write_json  # noqa: E402
from scripts.dcm_native_context_chain_poc import _bind_chain  # noqa: E402
from scripts.dcm_native_context_transition_canary import (  # noqa: E402
    LiveTransitionCanaryFailure,
    _event,
    _load_plan,
    _nonce,
    _require_ready,
    _reset_local_gpu_prefix_cache,
)

_SCHEMA = "daystrom-vllm-context-exhaustion-ab-v1"


def _cold_digest(plan: NativeContextTransitionPlan, rung: int) -> str:
    material = f"{plan.plan_digest}|cold|{rung}|{secrets.token_hex(32)}"
    return "sha256:" + hashlib.sha256(material.encode()).hexdigest()


def _lane_summary(event: dict[str, Any]) -> dict[str, Any]:
    prompt_tokens = int(event["prompt_tokens"])
    reused = int(event["cpu_offload_matched_tokens"])
    return {
        **event,
        "prefill_tokens_charged": prompt_tokens - reused,
        "recompute_avoidance_tokens": reused,
        "recompute_avoidance_fraction": reused / prompt_tokens if prompt_tokens else 0.0,
    }


def run_context_exhaustion_ab(
    *,
    adapter: VLLMCooperativeExecutionAdapter,
    plans: Sequence[NativeContextTransitionPlan],
    prompts: Sequence[str],
    source_ref: str,
    ttl_seconds: float,
    order_seed: int,
    reset_local_gpu_prefix_cache: Any,
) -> dict[str, Any]:
    """Execute randomized paired rungs with route proof and complete cleanup."""

    if not source_ref.strip():
        raise RuntimeExecutionError("source_ref is required")
    if not 30 <= ttl_seconds <= 900:
        raise RuntimeExecutionError("ttl_seconds must be between 30 and 900")
    bound, root_digest, child_digests = _bind_chain(plans, prompts)
    if any(plan.current_tokens > plan.served_limit for plan in bound):
        raise RuntimeExecutionError("A/B ladder cannot exceed the served logical limit")
    if len({plan.served_limit for plan in bound}) != 1:
        raise RuntimeExecutionError("served limit drifted across A/B ladder")
    if len({plan.model_native_limit for plan in bound}) != 1:
        raise RuntimeExecutionError("model-native limit drifted across A/B ladder")

    expires_at = time.time() + ttl_seconds
    rng = random.Random(order_seed)
    created: list[str] = []
    cleanup: list[dict[str, Any]] = []
    rungs: list[dict[str, Any]] = []
    primary_error: Exception | None = None
    stage = "root_save"
    final_expected_blocks = 0
    root_setup: dict[str, Any] | None = None

    try:
        reset_local_gpu_prefix_cache()
        created.append(root_digest)
        root = adapter.complete_with_checkpoint(
            prompts[0],
            operation="save",
            checkpoint_digest=root_digest,
            expires_at=expires_at,
            nonce=_nonce("ab-root-save"),
            max_tokens=8,
            temperature=0.0,
            seed=17,
        )
        if root.prompt_tokens != bound[0].stable_prefix_tokens:
            raise RuntimeExecutionError("root prompt length did not match first A/B plan")
        root_ready = adapter.wait_for_checkpoint_ready(
            checkpoint_digest=root_digest,
            expires_at=expires_at,
            nonce=_nonce("ab-root-ready"),
        )
        _require_ready(root_ready, "A/B root")
        root_setup = {"save": _event(root), "readiness": _event(root_ready)}

        for index, (plan, prompt, child_digest) in enumerate(
            zip(bound, prompts[1:], child_digests), start=1
        ):
            cold_digest = _cold_digest(plan, index)
            created.extend([child_digest, cold_digest])
            order = ["enabled", "disabled"]
            rng.shuffle(order)
            enabled_event: dict[str, Any] | None = None
            enabled_ready_event: dict[str, Any] | None = None
            disabled_event: dict[str, Any] | None = None
            disabled_ready_event: dict[str, Any] | None = None

            for lane in order:
                stage = f"rung_{index}_{lane}"
                reset_local_gpu_prefix_cache()
                if lane == "enabled":
                    transitioned = adapter.execute_native_transition(
                        prompt,
                        plan=plan,
                        expires_at=expires_at,
                        nonce=_nonce(f"ab-enabled-{index}"),
                        max_tokens=8,
                        temperature=0.0,
                        seed=17,
                    )
                    trace = transitioned.execution
                    _require_ready(transitioned.readiness, f"A/B child {index}")
                    final_expected_blocks = transitioned.readiness.expected_blocks
                    if not (
                        trace.prompt_tokens == plan.current_tokens
                        and trace.cache_route == "cpu_fallback"
                        and trace.cpu_offload_matched_tokens > 0
                        and trace.gpu_apc_matched_tokens == 0
                    ):
                        raise RuntimeExecutionError("enabled A/B lane was contaminated or incomplete")
                    enabled_event = _event(trace)
                    enabled_ready_event = _event(transitioned.readiness)
                else:
                    trace = adapter.complete_with_checkpoint(
                        prompt,
                        operation="save",
                        checkpoint_digest=cold_digest,
                        expires_at=expires_at,
                        nonce=_nonce(f"ab-disabled-{index}"),
                        max_tokens=8,
                        temperature=0.0,
                        seed=17,
                    )
                    if not (
                        trace.prompt_tokens == plan.current_tokens
                        and trace.cpu_offload_matched_tokens == 0
                        and trace.gpu_apc_matched_tokens == 0
                        and trace.cache_route in {"miss", "not_applicable"}
                    ):
                        raise RuntimeExecutionError("disabled A/B lane was contaminated")
                    disabled_ready = adapter.wait_for_checkpoint_ready(
                        checkpoint_digest=cold_digest,
                        expires_at=expires_at,
                        nonce=_nonce(f"ab-disabled-ready-{index}"),
                    )
                    _require_ready(disabled_ready, f"A/B cold {index}")
                    disabled_event = _event(trace)
                    disabled_ready_event = _event(disabled_ready)

            if (
                enabled_event is None
                or enabled_ready_event is None
                or disabled_event is None
                or disabled_ready_event is None
            ):
                raise RuntimeExecutionError("A/B rung did not execute both lanes")
            if enabled_event["output_digest"] != disabled_event["output_digest"]:
                raise RuntimeExecutionError("enabled and disabled outputs diverged")
            enabled_latency = float(enabled_event["latency_ms"])
            disabled_latency = float(disabled_event["latency_ms"])
            rungs.append(
                {
                    "rung": index,
                    "order": order,
                    "plan_digest": plan.plan_digest,
                    "stable_prefix_tokens": plan.stable_prefix_tokens,
                    "current_tokens": plan.current_tokens,
                    "suffix_tokens": plan.suffix_tokens,
                    "served_headroom_tokens": plan.served_limit - plan.current_tokens,
                    "enabled": _lane_summary(enabled_event),
                    "enabled_readiness": enabled_ready_event,
                    "disabled": _lane_summary(disabled_event),
                    "disabled_readiness": disabled_ready_event,
                    "output_equivalent": True,
                    "latency_delta_ms": disabled_latency - enabled_latency,
                    "enabled_faster_percent": (
                        (disabled_latency - enabled_latency) / disabled_latency * 100.0
                        if disabled_latency > 0
                        else 0.0
                    ),
                }
            )
    except Exception as exc:
        primary_error = exc
    finally:
        for digest in reversed(created):
            try:
                purged = adapter.purge_checkpoint(
                    checkpoint_digest=digest,
                    expires_at=expires_at,
                    nonce=_nonce("ab-cleanup-purge"),
                )
                cleanup.append(
                    {
                        "checkpoint_digest": digest,
                        "reason_code": purged.reason_code,
                        "purged_blocks": purged.purged_blocks,
                        "purged_bytes": purged.purged_bytes,
                        "shared_blocks": purged.shared_blocks,
                    }
                )
            except Exception as cleanup_exc:
                cleanup.append(
                    {
                        "checkpoint_digest": digest,
                        "reason_code": "cleanup_failed",
                        "error_type": type(cleanup_exc).__name__,
                    }
                )

    cleanup_ok = bool(created) and len(cleanup) == len(created) and all(
        item["reason_code"] == "purge_complete" for item in cleanup
    )
    purged_blocks = sum(int(item.get("purged_blocks", 0)) for item in cleanup)
    purged_bytes = sum(int(item.get("purged_bytes", 0)) for item in cleanup)
    zeroized = final_expected_blocks > 0 and purged_blocks == final_expected_blocks and purged_bytes > 0
    if primary_error is None and cleanup_ok and not zeroized:
        stage = "cleanup_accounting"
        primary_error = RuntimeExecutionError("physical zeroization accounting mismatch")
    if primary_error is not None or not cleanup_ok:
        reason = primary_error or RuntimeExecutionError("checkpoint cleanup incomplete")
        raise LiveTransitionCanaryFailure(
            {
                "schema_version": "daystrom-vllm-context-exhaustion-ab-failure-v1",
                "result": "fail",
                "failure_stage": stage if primary_error is not None else "cleanup",
                "error_type": type(reason).__name__,
                "reason_digest": hashlib.sha256(str(reason).encode()).hexdigest(),
                "completed_rungs": rungs,
                "cleanup": cleanup,
                "cleanup_complete": cleanup_ok,
            }
        ) from reason

    deltas = [float(rung["latency_delta_ms"]) for rung in rungs]
    return {
        "schema_version": _SCHEMA,
        "source_ref": source_ref,
        "runtime_version": adapter.runtime_version,
        "model_id": adapter.model_id,
        "endpoint_origin_digest": adapter.capabilities().metadata["endpoint_origin_digest"],
        "served_limit": bound[0].served_limit,
        "model_native_limit": bound[0].model_native_limit,
        "logical_limit_extended": False,
        "comparison": {
            "enabled": "managed_cpu_checkpoint_restore",
            "disabled": "cold_full_history_recomputation",
            "concurrency": 1,
            "order_seed": order_seed,
            "max_tokens": 8,
            "temperature": 0.0,
            "seed": 17,
        },
        "root_setup": root_setup,
        "rungs": rungs,
        "summary": {
            "paired_rungs": len(rungs),
            "enabled_wins": sum(delta > 0 for delta in deltas),
            "disabled_wins": sum(delta < 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "largest_tested_tokens": max(plan.current_tokens for plan in bound),
            "remaining_served_headroom_tokens": bound[0].served_limit
            - max(plan.current_tokens for plan in bound),
        },
        "cleanup": cleanup,
        "expected_physical_blocks": final_expected_blocks,
        "physically_purged_blocks": purged_blocks,
        "physically_purged_bytes": purged_bytes,
        "physical_zeroization_complete": zeroized,
        "result": "pass",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="append", required=True, type=Path)
    parser.add_argument("--prompt-file", action="append", required=True, type=Path)
    parser.add_argument("--endpoint-url", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--runtime-version", required=True)
    parser.add_argument("--secret-path", required=True, type=Path)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--order-seed", type=int, default=1701)
    parser.add_argument("--ttl-seconds", type=float, default=900.0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plans = [_load_plan(path) for path in args.plan]
        prompts = [path.read_text(encoding="utf-8") for path in args.prompt_file]
        adapter = VLLMCooperativeExecutionAdapter(
            args.endpoint_url,
            model_id=args.model_id,
            runtime_id=args.runtime_id,
            runtime_version=args.runtime_version,
            secret_path=args.secret_path,
            timeout_seconds=args.timeout_seconds,
        )
        report = run_context_exhaustion_ab(
            adapter=adapter,
            plans=plans,
            prompts=prompts,
            source_ref=args.source_ref,
            ttl_seconds=args.ttl_seconds,
            order_seed=args.order_seed,
            reset_local_gpu_prefix_cache=lambda: _reset_local_gpu_prefix_cache(
                args.endpoint_url, args.timeout_seconds
            ),
        )
        status = 0
    except LiveTransitionCanaryFailure as exc:
        report = exc.report
        status = 1
    except Exception as exc:
        report = {
            "schema_version": "daystrom-vllm-context-exhaustion-ab-failure-v1",
            "result": "fail",
            "error_type": type(exc).__name__,
            "reason_digest": hashlib.sha256(str(exc).encode()).hexdigest(),
        }
        status = 1
    atomic_write_json(args.artifact, report)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return status


if __name__ == "__main__":
    sys.exit(main())
