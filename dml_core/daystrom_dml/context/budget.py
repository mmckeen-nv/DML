"""Deterministic context budget primitives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from daystrom_dml.api_contracts import ContractError, SerializableDataclass


@dataclass
class ContextBudget(SerializableDataclass):
    """Budget envelope protecting model, output, runtime, and input tokens."""

    model_limit_tokens: int = 0
    output_reserved_tokens: int = 0
    runtime_reserved_tokens: int = 0
    admitted_input_tokens: int = 0

    def __post_init__(self) -> None:
        for name in (
            "model_limit_tokens",
            "output_reserved_tokens",
            "runtime_reserved_tokens",
            "admitted_input_tokens",
        ):
            if getattr(self, name) < 0:
                raise ContractError(f"{name} must be non-negative")
        if self.output_reserved_tokens + self.runtime_reserved_tokens > self.model_limit_tokens:
            raise ContractError("reservations cannot exceed model_limit_tokens")
        if self.admitted_input_tokens > self.available_input_tokens:
            raise ContractError("admitted_input_tokens cannot exceed available_input_tokens")

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]):
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ContractError(f"{cls.__name__}.from_dict expected dict, got {type(data).__name__}")
        payload = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        return cls(**payload)

    @property
    def available_input_tokens(self) -> int:
        return self.model_limit_tokens - self.output_reserved_tokens - self.runtime_reserved_tokens

    @property
    def remaining_input_tokens(self) -> int:
        return max(0, self.available_input_tokens - self.admitted_input_tokens)

    @property
    def pressure(self) -> str:
        if self.available_input_tokens == 0 or self.remaining_input_tokens == 0:
            return "exhausted"
        used_ratio = self.admitted_input_tokens / self.available_input_tokens
        if used_ratio >= 0.9:
            return "critical"
        if used_ratio >= 0.75:
            return "warning"
        return "normal"

    def admit(self, tokens: int) -> bool:
        """Admit a non-negative token count without exceeding input allowance."""

        if tokens < 0:
            raise ContractError("tokens must be non-negative")
        if tokens > self.remaining_input_tokens:
            return False
        self.admitted_input_tokens += tokens
        return True
