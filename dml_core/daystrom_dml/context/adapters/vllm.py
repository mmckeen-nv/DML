"""OpenAI-compatible client for the experimental cooperative vLLM KV connector."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, cast
from urllib import error, request

from daystrom_dml.context.execution import RuntimeExecutionCapabilities, RuntimeExecutionError
from daystrom_dml.context.probe import endpoint_origin_identity_digest
from daystrom_dml.context.vllm_bridge.policy import (
    DAYSTROM_KV_SCHEMA_VERSION,
    DAYSTROM_KV_TRANSITION_SCHEMA_VERSION,
    build_kv_transfer_params,
    build_kv_transition_params,
)

if TYPE_CHECKING:
    from daystrom_dml.context.native_transition import NativeContextTransitionPlan


@dataclass(frozen=True)
class VLLMCooperativeKVTrace:
    """One request's response plus native, payload-free connector evidence."""

    output_text: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    checkpoint_digest: str
    operation: str
    reason_code: str
    matched_tokens: int
    gpu_apc_matched_tokens: int
    cpu_offload_matched_tokens: int
    cache_route: str
    saved_tokens: int
    purged_blocks: int
    purged_bytes: int
    shared_blocks: int
    checkpoint_ready: bool
    stored_blocks: int
    expected_blocks: int
    temperature: float = 0.0
    seed: int = 0
    max_tokens: int = 1
    child_checkpoint_digest: str = ""

    def to_telemetry(self) -> Dict[str, Any]:
        import hashlib

        return {
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "checkpoint_digest": self.checkpoint_digest,
            **(
                {"child_checkpoint_digest": self.child_checkpoint_digest}
                if self.child_checkpoint_digest
                else {}
            ),
            "operation": self.operation,
            "reason_code": self.reason_code,
            "matched_tokens": self.matched_tokens,
            "gpu_apc_matched_tokens": self.gpu_apc_matched_tokens,
            "cpu_offload_matched_tokens": self.cpu_offload_matched_tokens,
            "cache_route": self.cache_route,
            "saved_tokens": self.saved_tokens,
            "purged_blocks": self.purged_blocks,
            "purged_bytes": self.purged_bytes,
            "shared_blocks": self.shared_blocks,
            "checkpoint_ready": self.checkpoint_ready,
            "stored_blocks": self.stored_blocks,
            "expected_blocks": self.expected_blocks,
            "output_digest": "sha256:" + hashlib.sha256(self.output_text.encode()).hexdigest(),
            "output_digest_method": "sha256_utf8_text",
            "sampling": {
                "temperature": self.temperature,
                "seed": self.seed,
                "max_tokens": self.max_tokens,
            },
        }


@dataclass(frozen=True)
class VLLMNativeTransitionResult:
    """Compound generation evidence plus verified child checkpoint readiness."""

    execution: VLLMCooperativeKVTrace
    readiness: VLLMCooperativeKVTrace

    def to_telemetry(self) -> Dict[str, Any]:
        return {
            "execution": self.execution.to_telemetry(),
            "readiness": self.readiness.to_telemetry(),
        }


class VLLMCooperativeExecutionAdapter:
    """Version-pinned request client for ``DaystromCooperativeKVConnector``.

    This adapter deliberately reports erase, physical deletion, and stable slot
    affinity as unsupported.  It therefore cannot pass DML's full checkpoint
    lifecycle probe yet; it only exercises controller-authorized save/restore
    requests and validates the connector's native response evidence.
    """

    def __init__(
        self,
        endpoint_url: str,
        *,
        model_id: str,
        runtime_id: str,
        runtime_version: str,
        secret_path: str | Path,
        timeout_seconds: float = 120.0,
        max_response_bytes: int = 4 * 1024 * 1024,
        opener: Optional[Any] = None,
    ) -> None:
        if not endpoint_url or not model_id or not runtime_id:
            raise RuntimeExecutionError("endpoint_url, model_id, and runtime_id must be non-empty")
        if not runtime_version or runtime_version == "unknown":
            raise RuntimeExecutionError("an exact runtime_version is required")
        path = Path(secret_path).expanduser().resolve()
        if not path.is_file():
            raise RuntimeExecutionError("secret_path must identify an existing file")
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model_id = model_id
        self.runtime_id = runtime_id
        self.runtime_version = runtime_version
        self.secret_path = path
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes
        self.opener = opener or request.urlopen

    def capabilities(self) -> RuntimeExecutionCapabilities:
        return RuntimeExecutionCapabilities(
            runtime_id=self.runtime_id,
            adapter_id="vllm_daystrom_cooperative_v1",
            runtime_version=self.runtime_version,
            supports_prompt_cache=True,
            supports_kv_checkpoint=True,
            supports_kv_restore=True,
            supports_kv_erase=False,
            supports_kv_checkpoint_delete=False,
            supports_slot_affinity=False,
            supports_metrics=True,
            metadata={
                "endpoint_origin_digest": "sha256:"
                + endpoint_origin_identity_digest(self.endpoint_url),
                "protocol": DAYSTROM_KV_SCHEMA_VERSION,
                "physical_purge": False,
                "request_bound_selective_purge": True,
                "purge_requires_resident_unshared_blocks": True,
                "purge_eager_offload_only": True,
                "cache_hierarchy": ["gpu_apc", "cpu_offload"],
                "gpu_apc_controller_scoped": False,
            },
        )

    def checkpoint_status(
        self,
        *,
        checkpoint_digest: str,
        expires_at: float,
        nonce: str,
    ) -> VLLMCooperativeKVTrace:
        """Return one signed, payload-free logical and physical readiness query."""

        return self.complete_with_checkpoint(
            "Daystrom cooperative KV checkpoint status request.",
            operation="status",
            checkpoint_digest=checkpoint_digest,
            expires_at=expires_at,
            nonce=nonce,
            max_tokens=1,
            temperature=0.0,
            seed=0,
        )

    def wait_for_checkpoint_ready(
        self,
        *,
        checkpoint_digest: str,
        expires_at: float,
        nonce: str,
        max_attempts: int = 3,
        poll_interval_seconds: float = 0.05,
    ) -> VLLMCooperativeKVTrace:
        """Poll signed read-only status until ready or a fail-closed state."""

        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or not 1 <= max_attempts <= 10
        ):
            raise RuntimeExecutionError("max_attempts must be between 1 and 10")
        if (
            isinstance(poll_interval_seconds, bool)
            or not isinstance(poll_interval_seconds, (int, float))
            or not 0 <= poll_interval_seconds <= 5
        ):
            raise RuntimeExecutionError(
                "poll_interval_seconds must be between 0 and 5"
            )
        import hashlib

        last_reason = "checkpoint_pending"
        for attempt in range(max_attempts):
            status_nonce = hashlib.sha256(
                f"{nonce}|checkpoint-status|{attempt}".encode()
            ).hexdigest()
            status = self.checkpoint_status(
                checkpoint_digest=checkpoint_digest,
                expires_at=expires_at,
                nonce=status_nonce,
            )
            last_reason = status.reason_code
            if status.checkpoint_ready:
                return status
            if status.reason_code != "checkpoint_pending":
                raise RuntimeExecutionError(
                    f"checkpoint readiness failed: {status.reason_code}"
                )
            if attempt + 1 < max_attempts and poll_interval_seconds:
                time.sleep(float(poll_interval_seconds))
        raise RuntimeExecutionError(
            f"checkpoint readiness timed out: {last_reason}"
        )

    def purge_checkpoint(
        self,
        *,
        checkpoint_digest: str,
        expires_at: float,
        nonce: str,
    ) -> VLLMCooperativeKVTrace:
        """Request and verify connector-native selective physical purge.

        This is intentionally not exposed as DML's generic checkpoint-delete
        capability: it only applies to this request-bound cooperative connector,
        requires eager-mode inventory with every unshared target row resident,
        and fails closed if a worker does not acknowledge zeroization.
        """

        first = self.complete_with_checkpoint(
            "Daystrom selective KV purge control request.",
            operation="purge",
            checkpoint_digest=checkpoint_digest,
            expires_at=expires_at,
            nonce=nonce,
            max_tokens=1,
            temperature=0.0,
            seed=0,
        )
        if first.reason_code == "purge_complete":
            return first
        if first.reason_code != "purge_pending":
            raise RuntimeExecutionError("selective KV purge was not scheduled")

        # vLLM constructs each request's response metadata before applying that
        # same step's worker acknowledgement on the scheduler. Independently
        # signed status requests read committed state and can safely retry an
        # idempotent zero command after one worker-side failure.
        import hashlib

        last = first
        for attempt in range(2):
            status_nonce = hashlib.sha256(
                f"{nonce}|purge-status|{attempt}".encode()
            ).hexdigest()
            last = self.checkpoint_status(
                checkpoint_digest=checkpoint_digest,
                expires_at=expires_at,
                nonce=status_nonce,
            )
            if last.reason_code == "purge_complete":
                return last
        raise RuntimeExecutionError("selective KV purge did not complete")

    def execute_native_transition(
        self,
        prompt: str,
        *,
        plan: NativeContextTransitionPlan,
        expires_at: float,
        nonce: str,
        max_tokens: int = 1,
        temperature: float = 0.0,
        seed: int = 0,
        readiness_attempts: int = 3,
        readiness_poll_interval_seconds: float = 0.05,
    ) -> VLLMNativeTransitionResult:
        """Run one compound generation and require child readiness evidence."""

        from daystrom_dml.context.native_transition import NativeContextTransitionPlan

        if not isinstance(plan, NativeContextTransitionPlan):
            raise RuntimeExecutionError("plan must be a NativeContextTransitionPlan")
        try:
            bound_plan = NativeContextTransitionPlan.from_dict(plan.to_dict())
        except Exception as exc:
            raise RuntimeExecutionError("native transition plan integrity check failed") from exc
        if not bound_plan.feasible:
            raise RuntimeExecutionError("native transition plan is not feasible")
        if bound_plan.model_id != self.model_id or bound_plan.runtime_id != self.runtime_id:
            raise RuntimeExecutionError("native transition runtime identity mismatch")
        if [step.operation for step in bound_plan.steps] != [
            "restore_parent_prefix",
            "prefill_suffix",
            "checkpoint_current_generation",
        ]:
            raise RuntimeExecutionError("native transition requires restore, suffix, and checkpoint steps")
        restore_step, suffix_step, checkpoint_step = bound_plan.steps
        parent_digest = restore_step.checkpoint_digest
        child_digest = checkpoint_step.checkpoint_digest
        if not parent_digest or not child_digest:
            raise RuntimeExecutionError("native transition checkpoint identities are required")
        if child_digest != bound_plan.current_checkpoint_digest:
            raise RuntimeExecutionError("native transition child checkpoint drifted")
        if parent_digest == child_digest:
            raise RuntimeExecutionError("native transition checkpoint identities must differ")
        if (
            restore_step.token_count != bound_plan.stable_prefix_tokens
            or suffix_step.token_start != bound_plan.stable_prefix_tokens
            or suffix_step.token_count != bound_plan.suffix_tokens
            or checkpoint_step.token_count != bound_plan.current_tokens
        ):
            raise RuntimeExecutionError("native transition token boundaries drifted")
        execution = self.complete_with_checkpoint(
            prompt,
            operation="transition",
            checkpoint_digest=parent_digest,
            child_checkpoint_digest=child_digest,
            expires_at=expires_at,
            nonce=nonce,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            _native_transition_plan_validated=True,
        )
        if (
            execution.gpu_apc_matched_tokens <= 0
            and execution.cpu_offload_matched_tokens <= 0
        ):
            raise RuntimeExecutionError(
                "native transition produced no verified parent-prefix reuse"
            )
        # ``saved_tokens`` is scheduler progress reported when the request
        # finishes; it is not a physical child-checkpoint coverage counter.
        # Bind the actual prompt length to the compiled transition here, then
        # require complete physical block coverage from the separately signed
        # child-readiness response below.
        if execution.prompt_tokens != bound_plan.current_tokens:
            raise RuntimeExecutionError(
                "native transition prompt length did not match the planned context"
            )
        import hashlib

        readiness_nonce = hashlib.sha256(
            f"{nonce}|child-readiness|{child_digest}".encode()
        ).hexdigest()
        readiness = self.wait_for_checkpoint_ready(
            checkpoint_digest=child_digest,
            expires_at=expires_at,
            nonce=readiness_nonce,
            max_attempts=readiness_attempts,
            poll_interval_seconds=readiness_poll_interval_seconds,
        )
        return VLLMNativeTransitionResult(
            execution=execution,
            readiness=readiness,
        )

    def complete_with_checkpoint(
        self,
        prompt: str,
        *,
        operation: str,
        checkpoint_digest: str,
        expires_at: float,
        nonce: str,
        max_tokens: int = 1,
        temperature: float = 0.0,
        seed: int = 0,
        child_checkpoint_digest: str = "",
        _native_transition_plan_validated: bool = False,
    ) -> VLLMCooperativeKVTrace:
        if operation == "transition" and not _native_transition_plan_validated:
            raise RuntimeExecutionError(
                "transition requests require a validated native transition plan"
            )
        if not isinstance(prompt, str) or not prompt:
            raise RuntimeExecutionError("prompt must be a non-empty string")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise RuntimeExecutionError("max_tokens must be a positive integer")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or temperature < 0
        ):
            raise RuntimeExecutionError("temperature must be a non-negative number")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise RuntimeExecutionError("seed must be an integer")
        try:
            if operation == "transition":
                transfer = build_kv_transition_params(
                    parent_checkpoint_digest=checkpoint_digest,
                    child_checkpoint_digest=child_checkpoint_digest,
                    expires_at=expires_at,
                    nonce=nonce,
                    secret_path=self.secret_path,
                )
            else:
                if child_checkpoint_digest:
                    raise ValueError(
                        "child checkpoint digest is only valid for transition"
                    )
                transfer = build_kv_transfer_params(
                    operation=operation,
                    checkpoint_digest=checkpoint_digest,
                    expires_at=expires_at,
                    nonce=nonce,
                    secret_path=self.secret_path,
                )
        except (ValueError, OSError) as exc:
            raise RuntimeExecutionError("invalid cooperative KV authorization envelope") from exc
        started = time.perf_counter()
        payload = self._post(
            "/v1/completions",
            {
                "model": self.model_id,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "seed": seed,
                "kv_transfer_params": transfer,
            },
        )
        latency_ms = (time.perf_counter() - started) * 1000
        choices = payload.get("choices")
        usage = payload.get("usage")
        connector = payload.get("kv_transfer_params")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise RuntimeExecutionError("malformed vLLM completion response")
        if not isinstance(choices[0].get("text"), str) or not isinstance(usage, dict):
            raise RuntimeExecutionError("malformed vLLM completion response")
        daystrom = connector.get("daystrom") if isinstance(connector, dict) else None
        if not isinstance(daystrom, dict):
            raise RuntimeExecutionError("vLLM response lacks cooperative KV evidence")
        self._validate_connector_evidence(
            daystrom, operation, checkpoint_digest, child_checkpoint_digest
        )
        ready, stored_blocks, expected_blocks = _status_readiness(
            daystrom, operation
        )
        details = usage.get("prompt_tokens_details")
        cached = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
        return VLLMCooperativeKVTrace(
            output_text=cast(str, choices[0]["text"]),
            latency_ms=latency_ms,
            prompt_tokens=_counter(usage, "prompt_tokens"),
            completion_tokens=_counter(usage, "completion_tokens"),
            cached_tokens=_counter({"value": cached}, "value"),
            checkpoint_digest=checkpoint_digest,
            operation=operation,
            reason_code=cast(str, daystrom["reason_code"]),
            matched_tokens=_counter(daystrom, "matched_tokens"),
            gpu_apc_matched_tokens=_counter(daystrom, "gpu_apc_matched_tokens"),
            cpu_offload_matched_tokens=_counter(
                daystrom, "cpu_offload_matched_tokens"
            ),
            cache_route=cast(str, daystrom.get("cache_route", "legacy")),
            saved_tokens=_counter(daystrom, "saved_tokens"),
            purged_blocks=_counter(daystrom, "purged_blocks"),
            purged_bytes=_counter(daystrom, "purged_bytes"),
            shared_blocks=_counter(daystrom, "shared_blocks"),
            checkpoint_ready=ready,
            stored_blocks=stored_blocks,
            expected_blocks=expected_blocks,
            temperature=float(temperature),
            seed=seed,
            max_tokens=max_tokens,
            child_checkpoint_digest=child_checkpoint_digest,
        )

    @staticmethod
    def _validate_connector_evidence(
        evidence: Dict[str, Any],
        operation: str,
        checkpoint_digest: str,
        child_checkpoint_digest: str = "",
    ) -> None:
        expected_schema = (
            DAYSTROM_KV_TRANSITION_SCHEMA_VERSION
            if operation == "transition"
            else DAYSTROM_KV_SCHEMA_VERSION
        )
        if evidence.get("schema_version") != expected_schema:
            raise RuntimeExecutionError("cooperative KV schema mismatch")
        if evidence.get("operation") != operation:
            raise RuntimeExecutionError("cooperative KV operation mismatch")
        if evidence.get("checkpoint_digest") != checkpoint_digest:
            raise RuntimeExecutionError("cooperative KV checkpoint mismatch")
        if operation == "transition" and evidence.get(
            "child_checkpoint_digest"
        ) != child_checkpoint_digest:
            raise RuntimeExecutionError("cooperative KV child checkpoint mismatch")
        reason = evidence.get("reason_code")
        if operation == "transition":
            expected_reasons = {"transition_authorized"}
        elif operation == "purge":
            expected_reasons = {"purge_pending", "purge_complete"}
        elif operation == "status":
            expected_reasons = {
                "checkpoint_pending",
                "checkpoint_ready",
                "checkpoint_partial",
                "checkpoint_evicted",
                "checkpoint_below_granularity",
                "record_not_found",
                "record_expired",
                "purge_pending",
                "purge_complete",
                "purge_shared_ownership_changed",
            }
        else:
            expected_reasons = {f"{operation}_authorized"}
        if not isinstance(reason, str) or reason not in expected_reasons:
            raise RuntimeExecutionError("cooperative KV request was not authorized")
        _counter(evidence, "matched_tokens")
        _counter(evidence, "gpu_apc_matched_tokens")
        _counter(evidence, "cpu_offload_matched_tokens")
        cache_route = evidence.get("cache_route", "legacy")
        if not isinstance(cache_route, str) or cache_route not in {
            "gpu_apc",
            "cpu_fallback",
            "gpu_apc_and_cpu",
            "miss",
            "not_applicable",
            "legacy",
        }:
            raise RuntimeExecutionError("invalid runtime cache route")
        _counter(evidence, "saved_tokens")
        _counter(evidence, "purged_blocks")
        _counter(evidence, "purged_bytes")
        _counter(evidence, "shared_blocks")

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload, separators=(",", ":")).encode()
        req = request.Request(
            self.endpoint_url + path,
            data=body,
            headers={"content-type": "application/json", "accept": "application/json"},
            method="POST",
        )
        try:
            with self.opener(req, timeout=self.timeout_seconds) as response:
                raw = response.read(self.max_response_bytes + 1)
        except error.HTTPError as exc:
            raise RuntimeExecutionError(f"runtime returned HTTP {exc.code}") from exc
        except (error.URLError, TimeoutError) as exc:
            raise RuntimeExecutionError("runtime transport failed") from exc
        if len(raw) > self.max_response_bytes:
            raise RuntimeExecutionError("runtime response exceeded byte limit")
        try:
            parsed = json.loads(raw.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeExecutionError("runtime returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise RuntimeExecutionError("runtime response must be an object")
        return parsed


def _status_readiness(
    evidence: Dict[str, Any], operation: str
) -> tuple[bool, int, int]:
    if operation != "status":
        return False, 0, 0
    ready = evidence.get("checkpoint_ready")
    if not isinstance(ready, bool):
        raise RuntimeExecutionError("invalid checkpoint readiness boolean")
    try:
        stored = _counter(evidence, "stored_blocks")
        expected = _counter(evidence, "expected_blocks")
    except RuntimeExecutionError as exc:
        raise RuntimeExecutionError("invalid checkpoint readiness counters") from exc
    reason = evidence.get("reason_code")
    valid_shape = False
    if reason == "checkpoint_ready":
        valid_shape = ready and expected > 0 and stored == expected
    elif reason == "checkpoint_pending":
        valid_shape = not ready and expected > 0 and stored <= expected
    elif reason == "checkpoint_partial":
        valid_shape = not ready and expected > 0 and 0 < stored < expected
    elif reason == "checkpoint_evicted":
        valid_shape = not ready and expected > 0 and stored == 0
    elif reason in {
        "checkpoint_below_granularity",
        "record_not_found",
        "record_expired",
        "purge_pending",
        "purge_complete",
        "purge_shared_ownership_changed",
    }:
        valid_shape = not ready and stored == 0 and expected == 0
    if not valid_shape:
        raise RuntimeExecutionError("inconsistent checkpoint readiness evidence")
    return ready, stored, expected


def _counter(payload: Dict[str, Any], key: str) -> int:
    value = payload.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeExecutionError(f"invalid runtime counter: {key}")
    return value
