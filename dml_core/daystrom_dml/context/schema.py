"""Provider-agnostic context segment contracts for Daystrom Context Manager."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from daystrom_dml.api_contracts import ContractError, DaystromScope, SerializableDataclass, enum_from_value


class ContextAuthority(str, Enum):
    """Authority class for context admission and prompt placement."""

    IMMUTABLE = "immutable"
    CURRENT_INSTRUCTION = "current_instruction"
    TRUSTED_CONTROL = "trusted_control"
    REFERENCE = "reference"
    UNTRUSTED_DATA = "untrusted_data"


class ContextPriority(str, Enum):
    """Deterministic priority class for context budget decisions."""

    CRITICAL = "critical"
    WORKING = "working"
    REFERENCE = "reference"
    DISPOSABLE = "disposable"


@dataclass
class ContextSegment(SerializableDataclass):
    """JSON-friendly context unit admitted into a DCM packet."""

    segment_id: str
    kind: str
    content: Any
    authority: ContextAuthority = ContextAuthority.REFERENCE
    priority: ContextPriority = ContextPriority.REFERENCE
    scope: DaystromScope = field(default_factory=DaystromScope)
    source: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    cache: Dict[str, Any] = field(default_factory=dict)
    retention: Dict[str, Any] = field(default_factory=dict)
    estimated_tokens: int = 0
    exact_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ContractError("segment_id must be non-empty")
        if not self.kind:
            raise ContractError("kind must be non-empty")
        self.authority = enum_from_value(ContextAuthority, self.authority)
        self.priority = enum_from_value(ContextPriority, self.priority)
        if isinstance(self.scope, dict):
            self.scope = DaystromScope.from_dict(self.scope)
        if not isinstance(self.estimated_tokens, int) or isinstance(self.estimated_tokens, bool):
            raise ContractError("estimated_tokens must be an integer")
        if self.estimated_tokens < 0:
            raise ContractError("estimated_tokens must be non-negative")
        if self.exact_tokens is not None:
            if not isinstance(self.exact_tokens, int) or isinstance(self.exact_tokens, bool):
                raise ContractError("exact_tokens must be an integer")
            if self.exact_tokens < 0:
                raise ContractError("exact_tokens must be non-negative")
        for name in ("source", "provenance", "cache", "retention"):
            if not isinstance(getattr(self, name), dict):
                raise ContractError(f"{name} must be a dictionary")

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]):
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ContractError(f"{cls.__name__}.from_dict expected dict, got {type(data).__name__}")
        payload = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        if "scope" in payload:
            payload["scope"] = DaystromScope.from_dict(payload["scope"])
        return cls(**payload)

    @property
    def effective_tokens(self) -> int:
        """Return exact token count when available, otherwise the estimate."""

        return self.exact_tokens if self.exact_tokens is not None else self.estimated_tokens
