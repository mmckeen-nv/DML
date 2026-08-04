"""Provider-neutral real-model A/B probe harness for DCM-managed context."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol
from urllib import error, parse, request

from daystrom_dml.api_contracts import ContractError, SerializableDataclass

DEFAULT_MAX_REQUEST_BYTES = 256 * 1024
DEFAULT_MAX_RESPONSE_BYTES = 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 30.0
SECRET_QUERY_KEYS = {"api_key", "apikey", "key", "token", "access_token", "secret", "password"}


class ProbeTransportError(RuntimeError):
    """Raised when a model endpoint response fails closed."""

    def __init__(self, error_type: str, message: str):
        super().__init__(f"{error_type}: {message}")
        self.error_type = error_type


@dataclass
class ProbeSettings(SerializableDataclass):
    """Sampling and transport settings shared by both A/B requests."""

    temperature: float = 0
    max_output_tokens: int = 256
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not isinstance(self.temperature, (int, float)) or isinstance(self.temperature, bool):
            raise ContractError("temperature must be numeric")
        if self.temperature < 0:
            raise ContractError("temperature must be non-negative")
        if not isinstance(self.max_output_tokens, int) or isinstance(self.max_output_tokens, bool):
            raise ContractError("max_output_tokens must be an integer")
        if self.max_output_tokens <= 0:
            raise ContractError("max_output_tokens must be positive")
        if not isinstance(self.timeout_seconds, (int, float)) or isinstance(self.timeout_seconds, bool):
            raise ContractError("timeout_seconds must be numeric")
        if self.timeout_seconds <= 0:
            raise ContractError("timeout_seconds must be positive")


@dataclass
class MessageManifest(SerializableDataclass):
    """Digest-only manifest for a message list."""

    label: str
    message_count: int
    content_digest: str
    byte_count: int
    roles: List[str] = field(default_factory=list)


@dataclass
class EndpointIdentity(SerializableDataclass):
    """Endpoint identity safe for logs and artifacts."""

    url: str
    host: str = ""
    path: str = ""


@dataclass
class ModelProbeRequest(SerializableDataclass):
    """JSON-friendly request contract that excludes prompt content."""

    run_id: str
    endpoint: EndpointIdentity
    model_id: str
    baseline_manifest: MessageManifest
    managed_manifest: MessageManifest
    managed_authority_manifest_digest: str
    settings: ProbeSettings = field(default_factory=ProbeSettings)
    request_version: str = "daystrom-model-probe-request-v1"

    def to_dict(self) -> Dict[str, Any]:
        return _contract_to_dict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]):
        if data is None or not isinstance(data, dict):
            raise ContractError("ModelProbeRequest.from_dict expected dict")
        payload = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        if isinstance(payload.get("endpoint"), dict):
            payload["endpoint"] = EndpointIdentity.from_dict(payload["endpoint"])
        if isinstance(payload.get("baseline_manifest"), dict):
            payload["baseline_manifest"] = MessageManifest.from_dict(payload["baseline_manifest"])
        if isinstance(payload.get("managed_manifest"), dict):
            payload["managed_manifest"] = MessageManifest.from_dict(payload["managed_manifest"])
        if isinstance(payload.get("settings"), dict):
            payload["settings"] = ProbeSettings.from_dict(payload["settings"])
        return cls(**payload)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ContractError("run_id must be non-empty")
        if isinstance(self.endpoint, dict):
            self.endpoint = EndpointIdentity.from_dict(self.endpoint)
        if isinstance(self.baseline_manifest, dict):
            self.baseline_manifest = MessageManifest.from_dict(self.baseline_manifest)
        if isinstance(self.managed_manifest, dict):
            self.managed_manifest = MessageManifest.from_dict(self.managed_manifest)
        if isinstance(self.settings, dict):
            self.settings = ProbeSettings.from_dict(self.settings)
        if not self.model_id:
            raise ContractError("model_id must be non-empty")
        if self.request_version != "daystrom-model-probe-request-v1":
            raise ContractError("unsupported request_version")


@dataclass
class ModelCallEvidence(SerializableDataclass):
    """Persistable evidence for one model call, excluding completion text."""

    status: str
    latency_ms: float = 0
    output_digest: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    error_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _contract_to_dict(self)


@dataclass
class ModelProbeResult(SerializableDataclass):
    """JSON-friendly result contract that excludes prompt and completion content."""

    run_id: str
    endpoint: EndpointIdentity
    model_id: str
    baseline_manifest: MessageManifest
    managed_manifest: MessageManifest
    managed_authority_manifest_digest: str
    baseline: ModelCallEvidence
    managed: ModelCallEvidence
    evaluator_outcomes: Dict[str, Any] = field(default_factory=dict)
    status: str = "completed"
    result_version: str = "daystrom-model-probe-result-v1"

    def to_dict(self) -> Dict[str, Any]:
        return _contract_to_dict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]):
        if data is None or not isinstance(data, dict):
            raise ContractError("ModelProbeResult.from_dict expected dict")
        payload = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        if isinstance(payload.get("endpoint"), dict):
            payload["endpoint"] = EndpointIdentity.from_dict(payload["endpoint"])
        if isinstance(payload.get("baseline_manifest"), dict):
            payload["baseline_manifest"] = MessageManifest.from_dict(payload["baseline_manifest"])
        if isinstance(payload.get("managed_manifest"), dict):
            payload["managed_manifest"] = MessageManifest.from_dict(payload["managed_manifest"])
        if isinstance(payload.get("baseline"), dict):
            payload["baseline"] = ModelCallEvidence.from_dict(payload["baseline"])
        if isinstance(payload.get("managed"), dict):
            payload["managed"] = ModelCallEvidence.from_dict(payload["managed"])
        return cls(**payload)

    def __post_init__(self) -> None:
        if isinstance(self.endpoint, dict):
            self.endpoint = EndpointIdentity.from_dict(self.endpoint)
        if isinstance(self.baseline_manifest, dict):
            self.baseline_manifest = MessageManifest.from_dict(self.baseline_manifest)
        if isinstance(self.managed_manifest, dict):
            self.managed_manifest = MessageManifest.from_dict(self.managed_manifest)
        if isinstance(self.baseline, dict):
            self.baseline = ModelCallEvidence.from_dict(self.baseline)
        if isinstance(self.managed, dict):
            self.managed = ModelCallEvidence.from_dict(self.managed)


@dataclass
class ModelClientResponse:
    content: str
    latency_ms: float
    usage: Dict[str, Any] = field(default_factory=dict)


class ModelClient(Protocol):
    def complete(
        self,
        endpoint_url: str,
        model_id: str,
        messages: List[Dict[str, Any]],
        settings: ProbeSettings,
        label: str = "",
    ) -> ModelClientResponse:
        ...


@dataclass
class EvaluationSpec(SerializableDataclass):
    required_facts: List[str] = field(default_factory=list)
    forbidden_markers: List[str] = field(default_factory=list)
    instruction_survival_marker: Optional[str] = None


class FakeModelClient:
    """Offline deterministic client for fixture and unit test use."""

    def __init__(
        self,
        outputs: Mapping[str, str],
        usage: Optional[Mapping[str, Dict[str, Any]]] = None,
        latencies: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.outputs = dict(outputs)
        self.usage = dict(usage or {})
        self.latencies = dict(latencies or {})
        self.calls: List[Dict[str, Any]] = []

    def complete(
        self,
        endpoint_url: str,
        model_id: str,
        messages: List[Dict[str, Any]],
        settings: ProbeSettings,
        label: str = "",
    ) -> ModelClientResponse:
        del endpoint_url, messages
        self.calls.append({"label": label, "model_id": model_id, "settings": settings.to_dict()})
        content = self.outputs.get(label, self.outputs.get("", ""))
        return ModelClientResponse(
            content=content,
            latency_ms=float(self.latencies.get(label, 0)),
            usage=dict(self.usage.get(label, {})),
        )


class OpenAICompatibleModelClient:
    """Minimal /v1/chat/completions client using Python stdlib HTTP."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        opener: Optional[Any] = None,
    ) -> None:
        self.api_key = api_key
        self.max_request_bytes = max_request_bytes
        self.max_response_bytes = max_response_bytes
        self.opener = opener or request.urlopen

    def complete(
        self,
        endpoint_url: str,
        model_id: str,
        messages: List[Dict[str, Any]],
        settings: ProbeSettings,
        label: str = "",
    ) -> ModelClientResponse:
        del label
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": settings.temperature,
            "max_tokens": settings.max_output_tokens,
        }
        body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        if len(body) > self.max_request_bytes:
            raise ProbeTransportError("oversized_request", "request JSON exceeds configured byte limit")
        headers = {"content-type": "application/json", "accept": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        req = request.Request(endpoint_url, data=body, headers=headers, method="POST")
        started = time.perf_counter()
        try:
            with self.opener(req, timeout=settings.timeout_seconds) as response:
                status = getattr(response, "status", response.getcode())
                raw = response.read(self.max_response_bytes + 1)
        except TimeoutError as exc:
            raise ProbeTransportError("timeout", "model request timed out") from exc
        except error.HTTPError as exc:
            raise ProbeTransportError("http_error", f"endpoint returned HTTP {exc.code}") from exc
        except error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError):
                raise ProbeTransportError("timeout", "model request timed out") from exc
            raise ProbeTransportError("transport_error", str(reason)) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        if status >= 400:
            raise ProbeTransportError("http_error", f"endpoint returned HTTP {status}")
        if len(raw) > self.max_response_bytes:
            raise ProbeTransportError("oversized_response", "response JSON exceeds configured byte limit")
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProbeTransportError("non_json", "endpoint returned non-JSON response") from exc
        content = _extract_chat_content(parsed)
        usage = parsed.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        return ModelClientResponse(content=content, latency_ms=latency_ms, usage=usage)


def redact_endpoint_url(endpoint_url: str) -> str:
    parsed = parse.urlsplit(endpoint_url)
    hostname = parsed.hostname or ""
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    query_items = parse.parse_qsl(parsed.query, keep_blank_values=True)
    redacted_query = parse.urlencode(
        [(key, "REDACTED" if key.lower() in SECRET_QUERY_KEYS else value) for key, value in query_items]
    )
    return parse.urlunsplit((parsed.scheme, netloc, parsed.path, redacted_query, ""))


def endpoint_identity_digest(endpoint_url: str) -> str:
    """Bind a redacted endpoint identity without deriving artifacts from credentials."""

    return hashlib.sha256(redact_endpoint_url(endpoint_url).encode("utf-8")).hexdigest()


def endpoint_identity(endpoint_url: str) -> EndpointIdentity:
    safe = redact_endpoint_url(endpoint_url)
    parsed = parse.urlsplit(safe)
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return EndpointIdentity(url=safe, host=host, path=parsed.path)


def manifest_messages(messages: List[Dict[str, Any]], label: str) -> MessageManifest:
    _validate_messages(messages)
    body = json.dumps(messages, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return MessageManifest(
        label=label,
        message_count=len(messages),
        content_digest=_sha256_bytes(body),
        byte_count=len(body),
        roles=[str(message.get("role", "")) for message in messages],
    )


def build_probe_request(
    *,
    run_id: Optional[str] = None,
    endpoint_url: str,
    model_id: str,
    baseline_messages: List[Dict[str, Any]],
    managed_messages: List[Dict[str, Any]],
    managed_authority_manifest: Mapping[str, Any] | str,
    settings: Optional[ProbeSettings] = None,
) -> ModelProbeRequest:
    request_contract = ModelProbeRequest(
        run_id=run_id or str(uuid.uuid4()),
        endpoint=endpoint_identity(endpoint_url),
        model_id=model_id,
        baseline_manifest=manifest_messages(baseline_messages, "baseline"),
        managed_manifest=manifest_messages(managed_messages, "managed"),
        managed_authority_manifest_digest=_manifest_digest(managed_authority_manifest),
        settings=settings or ProbeSettings(),
    )
    return attach_runtime_messages(request_contract, baseline_messages, managed_messages)


def run_ab_probe(
    request_contract: ModelProbeRequest,
    client: ModelClient,
    evaluation_spec: EvaluationSpec,
    *,
    endpoint_url: Optional[str] = None,
    baseline_messages: Optional[List[Dict[str, Any]]] = None,
    managed_messages: Optional[List[Dict[str, Any]]] = None,
) -> ModelProbeResult:
    if not request_contract.managed_authority_manifest_digest:
        raise ValueError("managed_authority_manifest_digest is required")
    if baseline_messages is None:
        baseline_messages = getattr(request_contract, "_baseline_messages", None)
    if managed_messages is None:
        managed_messages = getattr(request_contract, "_managed_messages", None)
    if baseline_messages is None or managed_messages is None:
        raise ValueError("baseline_messages and managed_messages are required at run time")
    resolved_endpoint = endpoint_url or request_contract.endpoint.url
    resolved_identity = endpoint_identity(resolved_endpoint)
    if resolved_identity != request_contract.endpoint:
        raise ValueError("endpoint_url must match the probe request endpoint identity")
    baseline_messages = _copy_messages(baseline_messages)
    managed_messages = _copy_messages(managed_messages)
    if manifest_messages(baseline_messages, "baseline") != request_contract.baseline_manifest:
        raise ValueError("baseline messages do not match the probe manifest")
    if manifest_messages(managed_messages, "managed") != request_contract.managed_manifest:
        raise ValueError("managed messages do not match the probe manifest")
    baseline_response = _call_model(client, resolved_endpoint, request_contract, baseline_messages, "baseline")
    managed_response = _call_model(client, resolved_endpoint, request_contract, managed_messages, "managed")
    outcomes = evaluate_responses(
        baseline_response,
        managed_response,
        evaluation_spec,
        baseline_text=str(getattr(baseline_response, "_content", "")),
        managed_text=str(getattr(managed_response, "_content", "")),
    )
    status = "completed" if baseline_response.status == managed_response.status == "ok" else "failed"
    return ModelProbeResult(
        run_id=request_contract.run_id,
        endpoint=request_contract.endpoint,
        model_id=request_contract.model_id,
        baseline_manifest=request_contract.baseline_manifest,
        managed_manifest=request_contract.managed_manifest,
        managed_authority_manifest_digest=request_contract.managed_authority_manifest_digest,
        baseline=baseline_response,
        managed=managed_response,
        evaluator_outcomes=outcomes,
        status=status,
    )


def evaluate_responses(
    baseline: ModelCallEvidence,
    managed: ModelCallEvidence,
    spec: EvaluationSpec,
    *,
    baseline_text: str = "",
    managed_text: str = "",
) -> Dict[str, Any]:
    outcomes: Dict[str, Any] = {
        "responses_equal": baseline.output_digest == managed.output_digest and bool(baseline.output_digest),
        "baseline_output_digest": baseline.output_digest,
        "managed_output_digest": managed.output_digest,
        "required_fact_retention": {},
        "forbidden_marker_leakage": {},
        "instruction_survival_marker": None,
        "latency_delta_ms": managed.latency_ms - baseline.latency_ms,
    }
    for fact in spec.required_facts:
        outcomes["required_fact_retention"][fact] = {"baseline": fact in baseline_text, "managed": fact in managed_text}
    for marker in spec.forbidden_markers:
        outcomes["forbidden_marker_leakage"][marker] = {
            "baseline": marker in baseline_text,
            "managed": marker in managed_text,
        }
    if spec.instruction_survival_marker:
        marker = spec.instruction_survival_marker
        outcomes["instruction_survival_marker"] = {"baseline": marker in baseline_text, "managed": marker in managed_text}
    token_delta = _usage_delta(baseline.usage, managed.usage)
    if token_delta:
        outcomes["token_use_delta"] = token_delta
    return outcomes


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def attach_runtime_messages(
    request_contract: ModelProbeRequest,
    baseline_messages: List[Dict[str, Any]],
    managed_messages: List[Dict[str, Any]],
) -> ModelProbeRequest:
    """Attach messages for immediate in-process execution without serializing them."""

    setattr(request_contract, "_baseline_messages", _copy_messages(baseline_messages))
    setattr(request_contract, "_managed_messages", _copy_messages(managed_messages))
    return request_contract


def _call_model(
    client: ModelClient,
    endpoint_url: str,
    request_contract: ModelProbeRequest,
    messages: List[Dict[str, Any]],
    label: str,
) -> ModelCallEvidence:
    try:
        response = client.complete(endpoint_url, request_contract.model_id, messages, request_contract.settings, label=label)
    except ProbeTransportError as exc:
        return ModelCallEvidence(status="error", error_type=exc.error_type)
    except Exception as exc:  # pragma: no cover - defensive fail closed
        return ModelCallEvidence(status="error", error_type=type(exc).__name__)
    digest = _sha256_text(response.content)
    evidence = ModelCallEvidence(status="ok", latency_ms=response.latency_ms, output_digest=digest, usage=response.usage)
    setattr(evidence, "_content", response.content)
    return evidence


def _extract_chat_content(parsed: Any) -> str:
    if not isinstance(parsed, dict):
        raise ProbeTransportError("invalid_response", "response JSON must be an object")
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProbeTransportError("missing_content", "response choices are missing")
    first = choices[0]
    if not isinstance(first, dict):
        raise ProbeTransportError("missing_content", "first choice is not an object")
    message = first.get("message")
    content = message.get("content") if isinstance(message, dict) else first.get("text")
    if not isinstance(content, str) or content == "":
        raise ProbeTransportError("missing_content", "response content is missing")
    return content


def _copy_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    _validate_messages(messages)
    try:
        copied = json.loads(json.dumps(messages, sort_keys=True, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ContractError("messages must be JSON-compatible") from exc
    if not isinstance(copied, list) or any(not isinstance(item, dict) for item in copied):
        raise ContractError("messages must contain objects")
    result: List[Dict[str, Any]] = [dict(item) for item in copied]
    _validate_messages(result)
    return result


def _validate_messages(messages: List[Dict[str, Any]]) -> None:
    if not isinstance(messages, list) or not messages:
        raise ContractError("messages must be a non-empty list")
    for message in messages:
        if not isinstance(message, dict):
            raise ContractError("messages must contain objects")
        if not isinstance(message.get("role"), str) or not message["role"]:
            raise ContractError("message role must be non-empty")
        if "content" not in message:
            raise ContractError("message content is required")


def _manifest_digest(manifest: Mapping[str, Any] | str) -> str:
    if isinstance(manifest, str):
        return manifest if len(manifest) == 64 else _sha256_text(manifest)
    body = json.dumps(manifest, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(body)


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _usage_delta(baseline: Dict[str, Any], managed: Dict[str, Any]) -> Dict[str, Any]:
    delta: Dict[str, Any] = {}
    for key in sorted(set(baseline) | set(managed)):
        left = baseline.get(key)
        right = managed.get(key)
        if isinstance(left, int) and isinstance(right, int) and not isinstance(left, bool) and not isinstance(right, bool):
            delta[key] = right - left
    return delta


def _contract_to_dict(value: SerializableDataclass) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for name in value.__dataclass_fields__:  # type: ignore[attr-defined]
        item = getattr(value, name)
        if isinstance(item, SerializableDataclass):
            result[name] = item.to_dict()
        elif isinstance(item, list):
            result[name] = [entry.to_dict() if isinstance(entry, SerializableDataclass) else entry for entry in item]
        elif isinstance(item, dict):
            result[name] = {
                key: entry.to_dict() if isinstance(entry, SerializableDataclass) else entry for key, entry in item.items()
            }
        else:
            result[name] = item
    return result
