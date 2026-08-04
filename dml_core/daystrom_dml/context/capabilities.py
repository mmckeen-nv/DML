"""Runtime capability declarations for DCM packets."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from daystrom_dml.api_contracts import ContractError, SerializableDataclass

RUNTIME_CAPABILITIES_V1 = "daystrom-runtime-capabilities-v1"


@dataclass
class RuntimeCapabilities(SerializableDataclass):
    """Provider-neutral API and hosted-runtime feature envelope."""

    model_id: str
    backend_id: str
    model_context_window: int = 0
    max_output_tokens: int = 0
    tokenizer_mode: str = "provider_estimated"
    prompt_cache_visibility: str = "opaque"
    supports_kv_checkpoint_restore: bool = False
    supports_kv_offload: bool = False
    supports_nvcache: bool = False
    supports_context_branching: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ContractError("model_id must be non-empty")
        if not self.backend_id:
            raise ContractError("backend_id must be non-empty")
        for name in ("model_context_window", "max_output_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ContractError(f"{name} must be an integer")
            if value < 0:
                raise ContractError(f"{name} must be non-negative")
        if not isinstance(self.metadata, dict):
            raise ContractError("metadata must be a dictionary")

    @classmethod
    def api_only(
        cls,
        *,
        model_id: str,
        backend_id: str,
        model_context_window: int = 0,
        max_output_tokens: int = 0,
        tokenizer_mode: str = "provider_estimated",
        prompt_cache_visibility: str = "opaque",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "RuntimeCapabilities":
        """Return conservative defaults for API-only providers."""

        return cls(
            model_id=model_id,
            backend_id=backend_id,
            model_context_window=model_context_window,
            max_output_tokens=max_output_tokens,
            tokenizer_mode=tokenizer_mode,
            prompt_cache_visibility=prompt_cache_visibility,
            supports_kv_checkpoint_restore=False,
            supports_kv_offload=False,
            supports_nvcache=False,
            supports_context_branching=False,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]):
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ContractError(f"{cls.__name__}.from_dict expected dict, got {type(data).__name__}")
        payload = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        return cls(**payload)
