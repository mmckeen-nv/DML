"""API-message runtime adapter for observe-only DCM integration."""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterable, List, Optional

from daystrom_dml.context.adapters.base import BaseRuntimeContextAdapter, SegmentLike, TokenInput


TokenEstimator = Callable[[TokenInput], int]


class APIMessageAdapter(BaseRuntimeContextAdapter):
    """Render provider-agnostic context segments as chat-style API messages."""

    def __init__(self, token_estimator: Optional[TokenEstimator] = None) -> None:
        self._token_estimator = token_estimator

    def capabilities(self) -> Dict[str, Any]:
        return {
            "api_messages": True,
            "render_messages": True,
            "token_estimation": True,
            "kv": False,
            "retrieval": False,
            "writeback": False,
            "roles": ["system", "user", "assistant", "tool"],
            "authority_manifest": True,
        }

    def estimate_tokens(self, value: TokenInput) -> int:
        if self._token_estimator is not None:
            return int(self._token_estimator(value))
        if isinstance(value, str):
            return max(0, (len(value) + 3) // 4)
        serialized = json.dumps(list(value), sort_keys=True, separators=(",", ":"))
        return max(0, (len(serialized) + 3) // 4)

    def render_messages(self, segments: Iterable[SegmentLike]) -> Dict[str, Any]:
        messages: List[Dict[str, str]] = []
        manifest: List[Dict[str, Any]] = []
        for index, segment in enumerate(segments):
            role_requested = str(segment.get("role") or "user")
            content = str(segment.get("content", segment.get("text", "")))
            role_rendered = self._safe_role(role_requested, segment)
            messages.append({"role": role_rendered, "content": content})
            manifest.append(
                {
                    "index": index,
                    "id": segment.get("id"),
                    "kind": segment.get("kind"),
                    "source": segment.get("source"),
                    "role_requested": role_requested,
                    "role_rendered": role_rendered,
                    "metadata": dict(segment.get("metadata") or {}),
                }
            )
        return {"messages": messages, "manifest": manifest}

    @staticmethod
    def _safe_role(role: str, segment: SegmentLike) -> str:
        normalized = role if role in {"system", "user", "assistant", "tool"} else "user"
        metadata = dict(segment.get("metadata") or {})
        authority = str(metadata.get("authority") or segment.get("authority") or "").lower()
        source = str(segment.get("source") or "").lower()
        kind = str(segment.get("kind") or "").lower()
        untrusted = authority == "untrusted_data" or source in {
            "retrieval",
            "retrieved",
            "dml",
            "memory",
        }
        if untrusted or kind in {"retrieved", "dml", "dml_context", "memory"}:
            return "user"
        return normalized
