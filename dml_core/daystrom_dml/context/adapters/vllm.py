"""OpenAI-compatible client for the experimental cooperative vLLM KV connector."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, cast
from urllib import error, request

from daystrom_dml.context.execution import RuntimeExecutionCapabilities, RuntimeExecutionError
from daystrom_dml.context.probe import endpoint_origin_identity_digest
from daystrom_dml.context.vllm_bridge.policy import (
    DAYSTROM_KV_SCHEMA_VERSION,
    build_kv_transfer_params,
)


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
    saved_tokens: int

    def to_telemetry(self) -> Dict[str, Any]:
        import hashlib

        return {
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "checkpoint_digest": self.checkpoint_digest,
            "operation": self.operation,
            "reason_code": self.reason_code,
            "matched_tokens": self.matched_tokens,
            "saved_tokens": self.saved_tokens,
            "output_digest": "sha256:" + hashlib.sha256(self.output_text.encode()).hexdigest(),
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
            },
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
    ) -> VLLMCooperativeKVTrace:
        if not isinstance(prompt, str) or not prompt:
            raise RuntimeExecutionError("prompt must be a non-empty string")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise RuntimeExecutionError("max_tokens must be a positive integer")
        try:
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
        self._validate_connector_evidence(daystrom, operation, checkpoint_digest)
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
            saved_tokens=_counter(daystrom, "saved_tokens"),
        )

    @staticmethod
    def _validate_connector_evidence(
        evidence: Dict[str, Any], operation: str, checkpoint_digest: str
    ) -> None:
        if evidence.get("schema_version") != DAYSTROM_KV_SCHEMA_VERSION:
            raise RuntimeExecutionError("cooperative KV schema mismatch")
        if evidence.get("operation") != operation:
            raise RuntimeExecutionError("cooperative KV operation mismatch")
        if evidence.get("checkpoint_digest") != checkpoint_digest:
            raise RuntimeExecutionError("cooperative KV checkpoint mismatch")
        reason = evidence.get("reason_code")
        if not isinstance(reason, str) or reason != f"{operation}_authorized":
            raise RuntimeExecutionError("cooperative KV request was not authorized")
        _counter(evidence, "matched_tokens")
        _counter(evidence, "saved_tokens")

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


def _counter(payload: Dict[str, Any], key: str) -> int:
    value = payload.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeExecutionError(f"invalid runtime counter: {key}")
    return value
