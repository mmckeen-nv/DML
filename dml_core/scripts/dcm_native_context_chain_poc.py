#!/usr/bin/env python3
"""Prove a growing multi-hop native vLLM checkpoint chain end to end."""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Sequence

_DML_CORE = Path(__file__).resolve().parents[1]
if str(_DML_CORE) not in sys.path:
    sys.path.insert(0, str(_DML_CORE))

from daystrom_dml.context.adapters.vllm import (  # noqa: E402
    VLLMCooperativeExecutionAdapter,
)
from daystrom_dml.context.execution import RuntimeExecutionError  # noqa: E402
from daystrom_dml.context.native_transition import (  # noqa: E402
    NativeContextTransitionPlan,
)
from daystrom_dml.context.probe import atomic_write_json  # noqa: E402
from scripts.dcm_native_context_transition_canary import (  # noqa: E402
    LiveTransitionCanaryFailure,
    _event,
    _load_plan,
    _nonce,
    _require_ready,
    _reset_local_gpu_prefix_cache,
)

_SCHEMA = "daystrom-vllm-transition-chain-poc-v1"


def _bind_chain(
    plans: Sequence[NativeContextTransitionPlan], prompts: Sequence[str]
) -> tuple[list[NativeContextTransitionPlan], str, list[str]]:
    if len(plans) < 2:
        raise RuntimeExecutionError("chain PoC requires at least two transitions")
    if len(prompts) != len(plans) + 1:
        raise RuntimeExecutionError("chain PoC requires one prompt per generation")

    bound = [NativeContextTransitionPlan.from_dict(plan.to_dict()) for plan in plans]
    model_ids = {plan.model_id for plan in bound}
    runtime_ids = {plan.runtime_id for plan in bound}
    if len(model_ids) != 1 or len(runtime_ids) != 1:
        raise RuntimeExecutionError("chain model/runtime identity drifted")

    parent_digests: list[str] = []
    child_digests: list[str] = []
    for index, plan in enumerate(bound):
        if not plan.feasible:
            raise RuntimeExecutionError("chain transition plan is not feasible")
        restore_step, _, checkpoint_step = plan.steps
        parent = restore_step.checkpoint_digest
        child = checkpoint_step.checkpoint_digest
        if not parent or not child or parent == child:
            raise RuntimeExecutionError("chain checkpoint identities are invalid")
        if plan.current_checkpoint_digest != child:
            raise RuntimeExecutionError("chain current checkpoint identity drifted")
        if index > 0:
            previous = bound[index - 1]
            if parent != previous.current_checkpoint_digest:
                raise RuntimeExecutionError("chain checkpoint lineage is broken")
            if plan.stable_prefix_tokens != previous.current_tokens:
                raise RuntimeExecutionError("chain token lineage is broken")
        parent_digests.append(parent)
        child_digests.append(child)

    identities = [parent_digests[0], *child_digests]
    if len(set(identities)) != len(identities):
        raise RuntimeExecutionError("chain checkpoint identities are not unique")
    return bound, parent_digests[0], child_digests


def run_chain_poc(
    *,
    adapter: VLLMCooperativeExecutionAdapter,
    plans: Sequence[NativeContextTransitionPlan],
    prompts: Sequence[str],
    source_ref: str,
    ttl_seconds: float,
    reset_local_gpu_prefix_cache: Any,
) -> dict[str, Any]:
    """Save generation zero, execute two or more growing transitions, and purge."""

    if not source_ref.strip():
        raise RuntimeExecutionError("source_ref is required")
    if not 30 <= ttl_seconds <= 900:
        raise RuntimeExecutionError("ttl_seconds must be between 30 and 900")
    bound, root_digest, child_digests = _bind_chain(plans, prompts)
    expires_at = time.time() + ttl_seconds
    cold_digest = "sha256:" + hashlib.sha256(
        f"{'|'.join(plan.plan_digest for plan in bound)}|cold|{secrets.token_hex(32)}".encode()
    ).hexdigest()

    events: list[dict[str, Any]] = []
    hops: list[dict[str, Any]] = []
    created: list[str] = []
    cleanup: list[dict[str, Any]] = []
    primary_error: Exception | None = None
    final_output_digest: str | None = None
    final_expected_blocks = 0
    native_matches: list[int] = []
    reuse_growth_verified = False
    output_equivalent = False
    stage = "root_save"

    try:
        # Begin from a cold local GPU prefix state so root publication cannot be
        # satisfied by stale APC blocks that bypass eager CPU store scheduling.
        reset_local_gpu_prefix_cache()
        created.append(root_digest)
        root = adapter.complete_with_checkpoint(
            prompts[0],
            operation="save",
            checkpoint_digest=root_digest,
            expires_at=expires_at,
            nonce=_nonce("chain-root-save"),
            # vLLM 0.20 eager offload needs a post-prefill scheduler step to
            # publish a finishing request's last complete block.
            max_tokens=8,
            temperature=0.0,
            seed=17,
        )
        if root.prompt_tokens != bound[0].stable_prefix_tokens:
            raise RuntimeExecutionError("root prompt length did not match chain plan")
        events.append(_event(root))
        stage = "root_readiness"
        root_ready = adapter.wait_for_checkpoint_ready(
            checkpoint_digest=root_digest,
            expires_at=expires_at,
            nonce=_nonce("chain-root-ready"),
        )
        _require_ready(root_ready, "root")
        events.append(_event(root_ready))

        for index, (plan, prompt, child_digest) in enumerate(
            zip(bound, prompts[1:], child_digests), start=1
        ):
            stage = f"transition_{index}"
            reset_local_gpu_prefix_cache()
            created.append(child_digest)
            transition = adapter.execute_native_transition(
                prompt,
                plan=plan,
                expires_at=expires_at,
                nonce=_nonce(f"chain-transition-{index}"),
                max_tokens=8,
                temperature=0.0,
                seed=17,
            )
            execution = transition.execution
            if execution.prompt_tokens != plan.current_tokens:
                raise RuntimeExecutionError("transition prompt length drifted")
            if (
                execution.gpu_apc_matched_tokens != 0
                or execution.cpu_offload_matched_tokens <= 0
                or execution.cache_route != "cpu_fallback"
            ):
                raise RuntimeExecutionError(
                    "chain transition did not prove isolated managed CPU reuse"
                )
            _require_ready(transition.readiness, f"child-{index}")
            native_matches.append(execution.cpu_offload_matched_tokens)
            final_expected_blocks = transition.readiness.expected_blocks
            execution_event = _event(execution)
            final_output_digest = execution_event["output_digest"]
            events.extend([execution_event, _event(transition.readiness)])
            hops.append(
                {
                    "hop": index,
                    "plan_digest": plan.plan_digest,
                    "parent_checkpoint_digest": plan.steps[0].checkpoint_digest,
                    "child_checkpoint_digest": child_digest,
                    "stable_prefix_tokens": plan.stable_prefix_tokens,
                    "current_tokens": plan.current_tokens,
                    "suffix_tokens": plan.suffix_tokens,
                    "cpu_offload_matched_tokens": execution.cpu_offload_matched_tokens,
                    "gpu_apc_matched_tokens": execution.gpu_apc_matched_tokens,
                    "cache_route": execution.cache_route,
                    "child_stored_blocks": transition.readiness.stored_blocks,
                    "child_expected_blocks": transition.readiness.expected_blocks,
                }
            )

        reuse_growth_verified = len(native_matches) >= 2 and all(
            later > earlier for earlier, later in zip(native_matches, native_matches[1:])
        )
        if not reuse_growth_verified:
            raise RuntimeExecutionError("native checkpoint reuse did not grow across hops")

        stage = "cold_control"
        reset_local_gpu_prefix_cache()
        created.append(cold_digest)
        cold = adapter.complete_with_checkpoint(
            prompts[-1],
            operation="save",
            checkpoint_digest=cold_digest,
            expires_at=expires_at,
            nonce=_nonce("chain-cold-control"),
            max_tokens=8,
            temperature=0.0,
            seed=17,
        )
        if cold.prompt_tokens != bound[-1].current_tokens:
            raise RuntimeExecutionError("cold-control prompt length drifted")
        if (
            cold.gpu_apc_matched_tokens != 0
            or cold.cpu_offload_matched_tokens != 0
            or cold.cache_route not in {"miss", "not_applicable"}
        ):
            raise RuntimeExecutionError("cold control reported native prefix reuse")
        stage = "cold_readiness"
        cold_ready = adapter.wait_for_checkpoint_ready(
            checkpoint_digest=cold_digest,
            expires_at=expires_at,
            nonce=_nonce("chain-cold-ready"),
        )
        _require_ready(cold_ready, "cold-control")
        events.extend([_event(cold), _event(cold_ready)])
        stage = "equivalence"
        if final_output_digest is None:
            raise RuntimeExecutionError("chain produced no transition output")
        output_equivalent = final_output_digest == _event(cold)["output_digest"]
        if not output_equivalent:
            raise RuntimeExecutionError("final chained output differed from cold control")
    except Exception as exc:
        primary_error = exc
    finally:
        for digest in reversed(created):
            try:
                purged = adapter.purge_checkpoint(
                    checkpoint_digest=digest,
                    expires_at=expires_at,
                    nonce=_nonce("chain-cleanup-purge"),
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
    physically_purged_blocks = sum(int(item.get("purged_blocks", 0)) for item in cleanup)
    physically_purged_bytes = sum(int(item.get("purged_bytes", 0)) for item in cleanup)
    physical_zeroization_complete = (
        final_expected_blocks > 0
        and physically_purged_blocks == final_expected_blocks
        and physically_purged_bytes > 0
    )
    if primary_error is None and cleanup_ok and not physical_zeroization_complete:
        stage = "cleanup_accounting"
        primary_error = RuntimeExecutionError("physical zeroization accounting mismatch")
    if primary_error is not None or not cleanup_ok:
        reason = primary_error or RuntimeExecutionError("checkpoint cleanup incomplete")
        raise LiveTransitionCanaryFailure(
            {
                "schema_version": "daystrom-vllm-transition-chain-poc-failure-v1",
                "result": "fail",
                "failure_stage": stage if primary_error is not None else "cleanup",
                "error_type": type(reason).__name__,
                "reason_digest": hashlib.sha256(str(reason).encode()).hexdigest(),
                "plan_digests": [plan.plan_digest for plan in bound],
                "hops": hops,
                "events": events,
                "cleanup": cleanup,
                "cleanup_complete": cleanup_ok,
                "expected_physical_blocks": final_expected_blocks,
                "physically_purged_blocks": physically_purged_blocks,
                "physically_purged_bytes": physically_purged_bytes,
                "physical_zeroization_complete": physical_zeroization_complete,
            }
        ) from reason

    return {
        "schema_version": _SCHEMA,
        "source_ref": source_ref,
        "runtime_version": adapter.runtime_version,
        "model_id": adapter.model_id,
        "endpoint_origin_digest": adapter.capabilities().metadata[
            "endpoint_origin_digest"
        ],
        "plan_digests": [plan.plan_digest for plan in bound],
        "root_checkpoint_digest": root_digest,
        "final_checkpoint_digest": child_digests[-1],
        "cold_checkpoint_digest": cold_digest,
        "hops": hops,
        "events": events,
        "native_reuse_grew": reuse_growth_verified,
        "output_equivalent": output_equivalent,
        "cleanup": cleanup,
        "expected_physical_blocks": final_expected_blocks,
        "physically_purged_blocks": physically_purged_blocks,
        "physically_purged_bytes": physically_purged_bytes,
        "physical_zeroization_complete": physical_zeroization_complete,
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
    parser.add_argument("--ttl-seconds", type=float, default=300.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
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
        report = run_chain_poc(
            adapter=adapter,
            plans=plans,
            prompts=prompts,
            source_ref=args.source_ref,
            ttl_seconds=args.ttl_seconds,
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
            "schema_version": "daystrom-vllm-transition-chain-poc-failure-v1",
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
