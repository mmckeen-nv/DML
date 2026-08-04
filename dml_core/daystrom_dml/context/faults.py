"""Provider-neutral memory fault contracts and resolver.

The resolver descends from DML1 hot context to DML2 exact pages to optional
durable lookup, returning only evidence-shaped data. Memory text is never
promoted into trusted control authority by this layer.
"""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Sequence

from daystrom_dml.api_contracts import ContractError, DaystromScope, SerializableDataclass, enum_from_value


MEMORY_FAULT_SCHEMA_VERSION = "1.0"


class MemorySourceTier(str, Enum):
    DML1_HOT = "dml1_hot"
    DML2_EXACT = "dml2_exact"
    DURABLE = "durable"


class MemoryFaultStatus(str, Enum):
    HIT = "hit"
    MISS = "miss"
    DENIED = "denied"
    OVER_BUDGET = "over_budget"
    CORRUPT = "corrupt"
    ADAPTER_ERROR = "adapter_error"


class MemoryFaultReason(str, Enum):
    FOUND = "found"
    NOT_FOUND = "not_found"
    DURABLE_NOT_ALLOWED = "durable_not_allowed"
    PAYLOAD_OVER_BUDGET = "payload_over_budget"
    DIGEST_MISMATCH = "digest_mismatch"
    MALFORMED_ADAPTER_RESULT = "malformed_adapter_result"
    SCOPE_MISMATCH = "scope_mismatch"
    ADAPTER_EXCEPTION = "adapter_exception"
    INVALID_AUTHORITY = "invalid_authority"


class EvidenceAuthority(str, Enum):
    REFERENCE = "reference"
    UNTRUSTED_DATA = "untrusted_data"


@dataclass
class MemoryFaultProvenance(SerializableDataclass):
    adapter_id: str
    source_id: Optional[str] = None
    page_id: Optional[str] = None
    rank: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = MEMORY_FAULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MEMORY_FAULT_SCHEMA_VERSION:
            raise ContractError("schema_version is not supported")
        if not self.adapter_id:
            raise ContractError("adapter_id must be non-empty")
        if self.rank is not None:
            _require_non_negative_int("rank", self.rank)
        self.metadata = _json_dict(self.metadata, "metadata")

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]):
        return cls(**_payload_for(cls, data))


@dataclass
class MemoryFaultBudget(SerializableDataclass):
    max_items: int = 1
    max_payload_bytes: int = 0
    max_payload_tokens: int = 0

    def __post_init__(self) -> None:
        _require_non_negative_int("max_items", self.max_items)
        _require_non_negative_int("max_payload_bytes", self.max_payload_bytes)
        _require_non_negative_int("max_payload_tokens", self.max_payload_tokens)


@dataclass
class MemoryFaultRequest(SerializableDataclass):
    request_id: str
    scope: DaystromScope = field(default_factory=DaystromScope)
    query: Optional[str] = None
    key: Optional[str] = None
    budget: MemoryFaultBudget = field(default_factory=MemoryFaultBudget)
    include_payload: bool = False
    allow_durable: bool = False
    schema_version: str = MEMORY_FAULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MEMORY_FAULT_SCHEMA_VERSION:
            raise ContractError("schema_version is not supported")
        if not self.request_id:
            raise ContractError("request_id must be non-empty")
        if isinstance(self.scope, dict):
            self.scope = DaystromScope.from_dict(self.scope)
        if isinstance(self.budget, dict):
            self.budget = MemoryFaultBudget.from_dict(self.budget)
        if self.query is not None and not isinstance(self.query, str):
            raise ContractError("query must be a string when provided")
        if self.key is not None and not isinstance(self.key, str):
            raise ContractError("key must be a string when provided")
        if not isinstance(self.include_payload, bool):
            raise ContractError("include_payload must be boolean")
        if not isinstance(self.allow_durable, bool):
            raise ContractError("allow_durable must be boolean")

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]):
        payload = _payload_for(cls, data)
        if "scope" in payload:
            payload["scope"] = DaystromScope.from_dict(payload["scope"])
        if "budget" in payload:
            payload["budget"] = MemoryFaultBudget.from_dict(payload["budget"])
        return cls(**payload)


@dataclass
class EvidenceHandle(SerializableDataclass):
    handle_id: str
    tier: MemorySourceTier
    scope: DaystromScope
    digest: str
    authority: EvidenceAuthority
    size_bytes: int = 0
    media_type: str = "text/plain; charset=utf-8"
    payload_text: Optional[str] = None
    payload_bytes_b64: Optional[str] = None
    provenance: MemoryFaultProvenance = field(default_factory=lambda: MemoryFaultProvenance(adapter_id="unknown"))
    schema_version: str = MEMORY_FAULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MEMORY_FAULT_SCHEMA_VERSION:
            raise ContractError("schema_version is not supported")
        if not self.handle_id:
            raise ContractError("handle_id must be non-empty")
        self.tier = enum_from_value(MemorySourceTier, self.tier)
        self.authority = enum_from_value(EvidenceAuthority, self.authority)
        if isinstance(self.scope, dict):
            self.scope = DaystromScope.from_dict(self.scope)
        if not self.digest:
            raise ContractError("digest must be non-empty")
        _require_non_negative_int("size_bytes", self.size_bytes)
        if self.payload_text is not None and self.payload_bytes_b64 is not None:
            raise ContractError("evidence may contain only one payload representation")
        if isinstance(self.provenance, dict):
            self.provenance = MemoryFaultProvenance.from_dict(self.provenance)
        if not isinstance(self.provenance, MemoryFaultProvenance):
            raise ContractError("provenance must be MemoryFaultProvenance")

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]):
        payload = _payload_for(cls, data)
        if "scope" in payload:
            payload["scope"] = DaystromScope.from_dict(payload["scope"])
        if "provenance" in payload:
            payload["provenance"] = MemoryFaultProvenance.from_dict(payload["provenance"])
        return cls(**payload)

    def without_payload(self) -> "EvidenceHandle":
        return replace(self, payload_text=None, payload_bytes_b64=None)

    def payload_bytes(self) -> Optional[bytes]:
        if self.payload_text is not None:
            return self.payload_text.encode("utf-8")
        if self.payload_bytes_b64 is not None:
            try:
                return base64.b64decode(self.payload_bytes_b64.encode("ascii"), validate=True)
            except Exception as exc:
                raise ContractError("invalid evidence payload_bytes_b64") from exc
        return None


@dataclass
class MemoryFaultResult(SerializableDataclass):
    request_id: str
    scope: DaystromScope
    status: MemoryFaultStatus
    tier: Optional[MemorySourceTier] = None
    reason_codes: List[MemoryFaultReason] = field(default_factory=list)
    evidence: List[EvidenceHandle] = field(default_factory=list)
    telemetry: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = MEMORY_FAULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MEMORY_FAULT_SCHEMA_VERSION:
            raise ContractError("schema_version is not supported")
        if not self.request_id:
            raise ContractError("request_id must be non-empty")
        if isinstance(self.scope, dict):
            self.scope = DaystromScope.from_dict(self.scope)
        self.status = enum_from_value(MemoryFaultStatus, self.status)
        if self.tier is not None:
            self.tier = enum_from_value(MemorySourceTier, self.tier)
        self.reason_codes = [enum_from_value(MemoryFaultReason, value) for value in self.reason_codes]
        self.evidence = [item if isinstance(item, EvidenceHandle) else EvidenceHandle.from_dict(item) for item in self.evidence]
        self.telemetry = _json_dict(self.telemetry, "telemetry")

    @classmethod
    def hit(
        cls,
        request: MemoryFaultRequest,
        tier: MemorySourceTier,
        evidence: Sequence[EvidenceHandle],
        *,
        reason: MemoryFaultReason = MemoryFaultReason.FOUND,
    ) -> "MemoryFaultResult":
        return cls(
            request_id=request.request_id,
            scope=request.scope,
            status=MemoryFaultStatus.HIT,
            tier=tier,
            reason_codes=[reason],
            evidence=list(evidence),
        )

    @classmethod
    def empty(
        cls,
        request: MemoryFaultRequest,
        status: MemoryFaultStatus,
        tier: Optional[MemorySourceTier],
        reason_codes: Sequence[MemoryFaultReason],
        *,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> "MemoryFaultResult":
        return cls(
            request_id=request.request_id,
            scope=request.scope,
            status=status,
            tier=tier,
            reason_codes=list(reason_codes),
            evidence=[],
            telemetry=telemetry or {},
        )

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]):
        payload = _payload_for(cls, data)
        if "scope" in payload:
            payload["scope"] = DaystromScope.from_dict(payload["scope"])
        if "evidence" in payload:
            payload["evidence"] = [EvidenceHandle.from_dict(item) for item in payload["evidence"]]
        return cls(**payload)


class MemoryFaultLookup(Protocol):
    def lookup(self, request: MemoryFaultRequest) -> MemoryFaultResult:
        """Return evidence for one tier without side effects outside the adapter."""


class DurableMemoryFaultLookup(MemoryFaultLookup, Protocol):
    """Optional durable-memory lookup. Resolver calls it only when allowed."""


class MemoryFaultResolver:
    """Deterministic DML1 -> DML2 -> durable memory fault resolver."""

    def __init__(
        self,
        *,
        dml1_hot: Optional[MemoryFaultLookup] = None,
        dml2_exact: Optional[MemoryFaultLookup] = None,
        durable: Optional[DurableMemoryFaultLookup] = None,
    ) -> None:
        self._dml1_hot = dml1_hot
        self._dml2_exact = dml2_exact
        self._durable = durable

    def resolve(self, request: MemoryFaultRequest) -> MemoryFaultResult:
        attempts: List[Dict[str, Any]] = []
        for tier, adapter in (
            (MemorySourceTier.DML1_HOT, self._dml1_hot),
            (MemorySourceTier.DML2_EXACT, self._dml2_exact),
        ):
            if adapter is None:
                continue
            result = self._try_adapter(adapter, request, tier, attempts)
            if result.status is not MemoryFaultStatus.MISS:
                result.telemetry = {"attempts": attempts}
                return result

        if self._durable is not None:
            if not request.allow_durable:
                attempts.append(_attempt(MemorySourceTier.DURABLE, MemoryFaultStatus.DENIED, [MemoryFaultReason.DURABLE_NOT_ALLOWED]))
                return MemoryFaultResult.empty(
                    request,
                    MemoryFaultStatus.DENIED,
                    MemorySourceTier.DURABLE,
                    [MemoryFaultReason.DURABLE_NOT_ALLOWED],
                    telemetry={"attempts": attempts},
                )
            result = self._try_adapter(self._durable, request, MemorySourceTier.DURABLE, attempts)
            result.telemetry = {"attempts": attempts}
            return result

        return MemoryFaultResult.empty(
            request,
            MemoryFaultStatus.MISS,
            None,
            [MemoryFaultReason.NOT_FOUND],
            telemetry={"attempts": attempts},
        )

    def _try_adapter(
        self,
        adapter: MemoryFaultLookup,
        request: MemoryFaultRequest,
        tier: MemorySourceTier,
        attempts: List[Dict[str, Any]],
    ) -> MemoryFaultResult:
        try:
            raw = adapter.lookup(request)
        except Exception as exc:
            result = MemoryFaultResult.empty(
                request,
                MemoryFaultStatus.ADAPTER_ERROR,
                tier,
                [MemoryFaultReason.ADAPTER_EXCEPTION],
            )
            attempts.append(
                _attempt(
                    tier,
                    MemoryFaultStatus.ADAPTER_ERROR,
                    [MemoryFaultReason.ADAPTER_EXCEPTION],
                    error_type=type(exc).__name__,
                )
            )
            return result

        if not isinstance(raw, MemoryFaultResult):
            result = MemoryFaultResult.empty(
                request,
                MemoryFaultStatus.CORRUPT,
                tier,
                [MemoryFaultReason.MALFORMED_ADAPTER_RESULT],
            )
            attempts.append(_attempt(tier, result.status, result.reason_codes))
            return result

        result = self._normalize_result(request, tier, raw)
        attempts.append(_attempt(tier, result.status, result.reason_codes, evidence_count=len(result.evidence)))
        return result

    def _normalize_result(
        self,
        request: MemoryFaultRequest,
        expected_tier: MemorySourceTier,
        result: MemoryFaultResult,
    ) -> MemoryFaultResult:
        if result.status is not MemoryFaultStatus.HIT:
            return MemoryFaultResult.empty(request, result.status, expected_tier, result.reason_codes or [MemoryFaultReason.NOT_FOUND])

        normalized: List[EvidenceHandle] = []
        for item in result.evidence[: request.budget.max_items]:
            if item.tier is not expected_tier:
                return MemoryFaultResult.empty(
                    request,
                    MemoryFaultStatus.CORRUPT,
                    expected_tier,
                    [MemoryFaultReason.MALFORMED_ADAPTER_RESULT],
                )
            if not _same_scope(request.scope, item.scope):
                return MemoryFaultResult.empty(
                    request,
                    MemoryFaultStatus.CORRUPT,
                    expected_tier,
                    [MemoryFaultReason.SCOPE_MISMATCH],
                )
            if item.authority not in {EvidenceAuthority.REFERENCE, EvidenceAuthority.UNTRUSTED_DATA}:
                return MemoryFaultResult.empty(
                    request,
                    MemoryFaultStatus.CORRUPT,
                    expected_tier,
                    [MemoryFaultReason.INVALID_AUTHORITY],
                )
            try:
                _validate_digest(item)
            except ContractError:
                return MemoryFaultResult.empty(
                    request,
                    MemoryFaultStatus.CORRUPT,
                    expected_tier,
                    [MemoryFaultReason.DIGEST_MISMATCH],
                )
            normalized.append(_apply_payload_budget(item, request))

        if not normalized:
            return MemoryFaultResult.empty(request, MemoryFaultStatus.MISS, expected_tier, [MemoryFaultReason.NOT_FOUND])
        if request.include_payload and any(
            _had_payload(item) and not _had_payload(normalized[index]) for index, item in enumerate(result.evidence[: len(normalized)])
        ):
            return MemoryFaultResult(
                request_id=request.request_id,
                scope=request.scope,
                status=MemoryFaultStatus.OVER_BUDGET,
                tier=expected_tier,
                reason_codes=[MemoryFaultReason.PAYLOAD_OVER_BUDGET],
                evidence=normalized,
            )
        return MemoryFaultResult.hit(request, expected_tier, normalized)


def _apply_payload_budget(item: EvidenceHandle, request: MemoryFaultRequest) -> EvidenceHandle:
    if not request.include_payload:
        return item.without_payload()
    payload = item.payload_bytes()
    if payload is None:
        return item
    if len(payload) > request.budget.max_payload_bytes:
        return item.without_payload()
    if _estimate_tokens_from_bytes(payload) > request.budget.max_payload_tokens:
        return item.without_payload()
    return item


def _validate_digest(item: EvidenceHandle) -> None:
    payload = item.payload_bytes()
    if payload is None:
        if not item.digest.startswith("sha256:") or len(item.digest) != 71:
            raise ContractError("invalid digest")
        return
    expected = "sha256:" + hashlib.sha256(payload).hexdigest()
    if item.digest != expected:
        raise ContractError("digest mismatch")


def _had_payload(item: EvidenceHandle) -> bool:
    return item.payload_text is not None or item.payload_bytes_b64 is not None


def _attempt(
    tier: MemorySourceTier,
    status: MemoryFaultStatus,
    reasons: Sequence[MemoryFaultReason],
    *,
    evidence_count: int = 0,
    error_type: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "tier": tier.value,
        "status": status.value,
        "reason_codes": [reason.value for reason in reasons],
        "evidence_count": int(evidence_count),
    }
    if error_type:
        payload["error_type"] = error_type
    return payload


def _same_scope(left: DaystromScope, right: DaystromScope) -> bool:
    return _scope_tuple(left) == _scope_tuple(right)


def _scope_tuple(scope: DaystromScope) -> tuple[Optional[str], ...]:
    return (
        scope.tenant_id,
        scope.client_id,
        scope.session_id,
        scope.instance_id,
        scope.thread_id,
        scope.project_id,
        scope.relationship_id,
    )


def _estimate_tokens_from_bytes(payload: bytes) -> int:
    if not payload:
        return 0
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return max(1, len(payload) // 4)
    return max(1, len(text.split()))


def _require_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContractError(f"{name} must be an integer")
    if value < 0:
        raise ContractError(f"{name} must be non-negative")


def _payload_for(cls: type, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ContractError(f"{cls.__name__}.from_dict expected dict, got {type(data).__name__}")
    fields = getattr(cls, "__dataclass_fields__", {})
    return {key: value for key, value in data.items() if key in fields}


def _json_dict(value: Dict[str, Any], name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be a dictionary")
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        copied = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be JSON-compatible") from exc
    if not isinstance(copied, dict):
        raise ContractError(f"{name} must be a dictionary")
    return copied
