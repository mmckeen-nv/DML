"""Shared Daystrom Platform API contract types.

These dataclasses are intentionally dependency-light and JSON-friendly.  They
form the stable envelope shared by DML, DPM, DCN, and the prototype DIP without
letting those components import each other's internals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar


class ContractError(ValueError):
    """Raised when a Daystrom API contract object is invalid."""


class SerializableDataclass:
    """Small explicit serialization helper for contract dataclasses."""

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]):
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ContractError(f"{cls.__name__}.from_dict expected dict, got {type(data).__name__}")
        dataclass_fields = getattr(cls, "__dataclass_fields__", None)
        if dataclass_fields is None:
            raise ContractError(f"{cls.__name__} is not a dataclass contract")
        known_fields = set(dataclass_fields)
        filtered = {key: value for key, value in data.items() if key in known_fields}
        try:
            return cls(**filtered)
        except TypeError as exc:  # pragma: no cover - defensive normalization
            raise ContractError(f"Invalid {cls.__name__} payload: {exc}") from exc

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for name, value in self.__dict__.items():
            result[name] = _serialize(value)
        return result


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, SerializableDataclass):
        return value.to_dict()
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


class ComponentMode(str, Enum):
    """Runtime activation mode for a Daystrom component."""

    DISABLED = "disabled"
    OBSERVE_ONLY = "observe_only"
    ACTIVE_READ = "active_read"
    ACTIVE_WRITE = "active_write"
    ACTIVE_LEARN = "active_learn"


class ReasonCode(str, Enum):
    """Inspectable reason codes emitted by deterministic policy layers."""

    GENERAL = "general"
    GREETING = "greeting"
    MEMORY_REQUEST = "memory_request"
    RESUME_REQUEST = "resume_request"
    CONTINUATION_REQUEST = "continuation_request"
    CODE_TASK = "code_task"
    DEBUG_TASK = "debug_task"
    SETUP_TASK = "setup_task"
    TOOL_NEEDED = "tool_needed"
    VERIFICATION_NEEDED = "verification_needed"
    PREFERENCE_SIGNAL = "preference_signal"
    SIDE_EFFECT = "side_effect"
    COMPACTION_RESUME = "compaction_resume"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"


@dataclass
class DaystromScope(SerializableDataclass):
    """Tenant/client/session scope shared across Daystrom components."""

    tenant_id: str = "openclaw"
    client_id: Optional[str] = None
    session_id: Optional[str] = None
    instance_id: Optional[str] = None
    thread_id: Optional[str] = None
    project_id: Optional[str] = None
    relationship_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.tenant_id:
            raise ContractError("tenant_id must be non-empty")


@dataclass
class TokenBudget(SerializableDataclass):
    """Token budget envelope with non-negative limits."""

    limit_tokens: int = 0
    used_tokens: int = 0
    reserved_tokens: int = 0

    def __post_init__(self) -> None:
        for name in ("limit_tokens", "used_tokens", "reserved_tokens"):
            value = getattr(self, name)
            if value < 0:
                raise ContractError(f"{name} must be non-negative")
        if self.used_tokens > self.limit_tokens and self.limit_tokens > 0:
            raise ContractError("used_tokens cannot exceed limit_tokens")

    @property
    def remaining_tokens(self) -> int:
        if self.limit_tokens == 0:
            return 0
        return max(0, self.limit_tokens - self.used_tokens - self.reserved_tokens)


@dataclass
class AuditInfo(SerializableDataclass):
    """Inspectable audit metadata attached to contract responses."""

    reason: str = ""
    policy: str = ""
    reason_codes: List[str] = field(default_factory=list)
    trace_id: Optional[str] = None
    schema_version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.reason_codes = _coerce_reason_codes(self.reason_codes)


@dataclass
class RiskInfo(SerializableDataclass):
    """Risk and confirmation metadata for side-effecting actions."""

    level: str = "low"
    reasons: List[str] = field(default_factory=list)
    requires_confirmation: bool = False
    side_effect_classes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        allowed = {"low", "medium", "high", "blocked"}
        if self.level not in allowed:
            raise ContractError(f"risk level must be one of {sorted(allowed)}")


T = TypeVar("T", bound=Enum)


def enum_from_value(enum_type: Type[T], value: Any) -> T:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except Exception as exc:  # pragma: no cover - defensive branch
        raise ContractError(f"Invalid {enum_type.__name__}: {value!r}") from exc


def _coerce_reason_codes(values: Iterable[Any]) -> List[str]:
    codes: List[str] = []
    for value in values:
        if isinstance(value, ReasonCode):
            codes.append(value.value)
        else:
            codes.append(str(value))
    return codes
