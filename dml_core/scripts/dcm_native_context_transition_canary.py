#!/usr/bin/env python3
"""Run a bounded, payload-free live canary for one native vLLM transition."""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Sequence
from urllib import request

_DML_CORE = Path(__file__).resolve().parents[1]
if str(_DML_CORE) not in sys.path:
    sys.path.insert(0, str(_DML_CORE))

from daystrom_dml.context.adapters.vllm import (  # noqa: E402
    VLLMCooperativeExecutionAdapter,
    VLLMCooperativeKVTrace,
)
from daystrom_dml.context.execution import RuntimeExecutionError  # noqa: E402
from daystrom_dml.context.native_transition import (  # noqa: E402
    NativeContextTransitionPlan,
)
from daystrom_dml.context.probe import atomic_write_json  # noqa: E402

_SCHEMA = "daystrom-vllm-transition-canary-v1"


class LiveTransitionCanaryFailure(RuntimeExecutionError):
    """A failed live canary with payload-free event and cleanup evidence."""

    def __init__(self, report: dict[str, Any]) -> None:
        super().__init__("live transition canary failed")
        self.report = report


def _load_plan(path: Path) -> NativeContextTransitionPlan:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeExecutionError("plan artifact must be an object")
    value = dict(raw)
    reported_pass = value.pop("pass", None)
    if reported_pass is not True:
        raise RuntimeExecutionError("plan artifact is not marked passing")
    return NativeContextTransitionPlan.from_dict(value)


def _nonce(label: str) -> str:
    return hashlib.sha256(f"{label}|{secrets.token_hex(32)}".encode()).hexdigest()


def _reset_local_gpu_prefix_cache(endpoint_url: str, timeout_seconds: float) -> None:
    url = endpoint_url.rstrip("/") + "/reset_prefix_cache?reset_external=false"
    req = request.Request(url, method="POST")
    with request.urlopen(req, timeout=timeout_seconds) as response:
        if getattr(response, "status", 200) != 200:
            raise RuntimeExecutionError("local GPU prefix-cache reset failed")
        response.read(4096)


def _require_ready(trace: VLLMCooperativeKVTrace, label: str) -> None:
    if (
        not trace.checkpoint_ready
        or trace.expected_blocks <= 0
        or trace.stored_blocks != trace.expected_blocks
    ):
        raise RuntimeExecutionError(f"{label} checkpoint is not completely ready")


def _event(trace: VLLMCooperativeKVTrace) -> dict[str, Any]:
    return trace.to_telemetry()


def run_canary(
    *,
    adapter: VLLMCooperativeExecutionAdapter,
    plan: NativeContextTransitionPlan,
    parent_prompt: str,
    current_prompt: str,
    source_ref: str,
    ttl_seconds: float,
    reset_local_gpu_prefix_cache: Any,
) -> dict[str, Any]:
    """Execute save→CPU transition→cold control→purge with digest-only output."""

    if not source_ref.strip():
        raise RuntimeExecutionError("source_ref must be non-empty")
    if not 30 <= ttl_seconds <= 900:
        raise RuntimeExecutionError("ttl_seconds must be between 30 and 900")
    bound_plan = NativeContextTransitionPlan.from_dict(plan.to_dict())
    if not bound_plan.feasible:
        raise RuntimeExecutionError("transition plan is not feasible")
    restore_step, _, checkpoint_step = bound_plan.steps
    parent_digest = restore_step.checkpoint_digest
    child_digest = checkpoint_step.checkpoint_digest
    if not parent_digest or not child_digest or parent_digest == child_digest:
        raise RuntimeExecutionError("transition checkpoint identities are invalid")
    cold_digest = "sha256:" + hashlib.sha256(
        f"{bound_plan.plan_digest}|cold-control|{secrets.token_hex(32)}".encode()
    ).hexdigest()
    expires_at = time.time() + ttl_seconds
    events: list[dict[str, Any]] = []
    created: list[str] = []
    cleanup: list[dict[str, Any]] = []
    primary_error: Exception | None = None
    stage = "parent_save"

    try:
        created.append(parent_digest)
        parent = adapter.complete_with_checkpoint(
            parent_prompt,
            operation="save",
            checkpoint_digest=parent_digest,
            expires_at=expires_at,
            nonce=_nonce("parent-save"),
            max_tokens=1,
            temperature=0.0,
            seed=17,
        )
        if parent.prompt_tokens != bound_plan.stable_prefix_tokens:
            raise RuntimeExecutionError("parent prompt length did not match stable prefix")
        events.append(_event(parent))
        stage = "parent_readiness"
        parent_ready = adapter.wait_for_checkpoint_ready(
            checkpoint_digest=parent_digest,
            expires_at=expires_at,
            nonce=_nonce("parent-ready"),
        )
        _require_ready(parent_ready, "parent")
        events.append(_event(parent_ready))

        stage = "transition"
        reset_local_gpu_prefix_cache()
        created.append(child_digest)
        transition = adapter.execute_native_transition(
            current_prompt,
            plan=bound_plan,
            expires_at=expires_at,
            nonce=_nonce("transition"),
            max_tokens=8,
            temperature=0.0,
            seed=17,
        )
        if (
            transition.execution.gpu_apc_matched_tokens != 0
            or transition.execution.cpu_offload_matched_tokens <= 0
            or transition.execution.cache_route != "cpu_fallback"
        ):
            raise RuntimeExecutionError("transition did not prove isolated managed CPU reuse")
        _require_ready(transition.readiness, "child")
        events.extend(
            [_event(transition.execution), _event(transition.readiness)]
        )

        stage = "cold_control"
        reset_local_gpu_prefix_cache()
        created.append(cold_digest)
        cold = adapter.complete_with_checkpoint(
            current_prompt,
            operation="save",
            checkpoint_digest=cold_digest,
            expires_at=expires_at,
            nonce=_nonce("cold-control"),
            max_tokens=8,
            temperature=0.0,
            seed=17,
        )
        if cold.prompt_tokens != bound_plan.current_tokens:
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
            nonce=_nonce("cold-ready"),
        )
        _require_ready(cold_ready, "cold-control")
        events.extend([_event(cold), _event(cold_ready)])
        stage = "equivalence"
        if transition.execution.to_telemetry()["output_digest"] != cold.to_telemetry()[
            "output_digest"
        ]:
            raise RuntimeExecutionError("transition output differed from cold control")
    except Exception as exc:
        primary_error = exc
    finally:
        for digest in reversed(created):
            try:
                purged = adapter.purge_checkpoint(
                    checkpoint_digest=digest,
                    expires_at=expires_at,
                    nonce=_nonce("cleanup-purge"),
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
    if primary_error is not None or not cleanup_ok:
        reason = primary_error or RuntimeExecutionError("checkpoint cleanup incomplete")
        raise LiveTransitionCanaryFailure(
            {
                "schema_version": "daystrom-vllm-transition-canary-failure-v1",
                "result": "fail",
                "failure_stage": stage if primary_error is not None else "cleanup",
                "error_type": type(reason).__name__,
                "reason_digest": hashlib.sha256(str(reason).encode()).hexdigest(),
                "events": events,
                "cleanup": cleanup,
                "cleanup_complete": cleanup_ok,
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
        "plan_digest": bound_plan.plan_digest,
        "parent_checkpoint_digest": parent_digest,
        "child_checkpoint_digest": child_digest,
        "events": events,
        "output_equivalent": True,
        "cleanup": cleanup,
        "result": "pass",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--parent-prompt-file", required=True, type=Path)
    parser.add_argument("--current-prompt-file", required=True, type=Path)
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
        plan = _load_plan(args.plan)
        adapter = VLLMCooperativeExecutionAdapter(
            args.endpoint_url,
            model_id=args.model_id,
            runtime_id=args.runtime_id,
            runtime_version=args.runtime_version,
            secret_path=args.secret_path,
            timeout_seconds=args.timeout_seconds,
        )
        report = run_canary(
            adapter=adapter,
            plan=plan,
            parent_prompt=args.parent_prompt_file.read_text(encoding="utf-8"),
            current_prompt=args.current_prompt_file.read_text(encoding="utf-8"),
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
            "schema_version": "daystrom-vllm-transition-canary-failure-v1",
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
