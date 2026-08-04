"""Provider-neutral runtime adapter contracts for DCM orchestration."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Protocol, Union, runtime_checkable


MessageLike = Dict[str, Any]
SegmentLike = Dict[str, Any]
TokenInput = Union[str, Iterable[MessageLike]]


@runtime_checkable
class RuntimeContextAdapter(Protocol):
    """Protocol implemented by provider/runtime-specific DCM adapters."""

    def capabilities(self) -> Dict[str, Any]:
        """Return JSON-friendly adapter capabilities."""

    def estimate_tokens(self, value: TokenInput) -> int:
        """Estimate token count for text or rendered API messages."""

    def render_messages(self, segments: Iterable[SegmentLike]) -> Dict[str, Any]:
        """Render context segments into provider API messages plus side manifest."""

    def kv_get(self, scope: Any, key: str) -> Dict[str, Any]:
        """Read a scoped KV value when supported."""

    def kv_put(self, scope: Any, key: str, value: Any) -> Dict[str, Any]:
        """Write a scoped KV value when supported."""

    def kv_delete(self, scope: Any, key: str) -> Dict[str, Any]:
        """Delete a scoped KV value when supported."""


class BaseRuntimeContextAdapter:
    """Minimal ABC-style base with explicit unsupported optional methods."""

    def capabilities(self) -> Dict[str, Any]:
        return {
            "api_messages": False,
            "render_messages": False,
            "token_estimation": False,
            "kv": False,
            "retrieval": False,
            "writeback": False,
        }

    def estimate_tokens(self, value: TokenInput) -> int:
        if isinstance(value, str):
            return len(value.split())
        return sum(len(str(message.get("content", "")).split()) for message in value)

    def render_messages(self, segments: Iterable[SegmentLike]) -> Dict[str, Any]:
        raise NotImplementedError("render_messages_unsupported")

    def kv_get(self, scope: Any, key: str) -> Dict[str, Any]:
        return {"supported": False, "error": "kv_get_unsupported", "value": None}

    def kv_put(self, scope: Any, key: str, value: Any) -> Dict[str, Any]:
        return {"supported": False, "error": "kv_put_unsupported", "stored": False}

    def kv_delete(self, scope: Any, key: str) -> Dict[str, Any]:
        return {"supported": False, "error": "kv_delete_unsupported", "deleted": False}
