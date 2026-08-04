"""Observe-only Daystrom Context Manager orchestration boundary."""
from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional

from daystrom_dml.api_contracts import ContractError, DaystromScope
from daystrom_dml.context.adapters.api_messages import APIMessageAdapter
from daystrom_dml.context.adapters.base import RuntimeContextAdapter


class ContextController:
    """Provider-agnostic DCM controller in observe-only mode.

    The controller reports what DCM would need to manage context pressure. It
    does not retrieve, write back, or modify prepared prompts/messages.
    """

    def __init__(
        self,
        runtime_adapter: Optional[RuntimeContextAdapter] = None,
        *,
        retrieval_adapter: Any = None,
        clock: Any = None,
    ) -> None:
        self.runtime_adapter = runtime_adapter or APIMessageAdapter()
        self.retrieval_adapter = retrieval_adapter
        self.clock = clock or time.time

    def observe(
        self,
        *,
        scope: DaystromScope | Dict[str, Any] | None = None,
        model_limits: Optional[Dict[str, Any]] = None,
        backend_limits: Optional[Dict[str, Any]] = None,
        current_prompt: str = "",
        current_messages: Optional[List[Dict[str, Any]]] = None,
        dcn_plan: Optional[Dict[str, Any]] = None,
        dcn_packet: Optional[Dict[str, Any]] = None,
        dml_context: Optional[Dict[str, Any]] = None,
        dpm_overlay: Optional[Dict[str, Any]] = None,
        output_reservation: int = 0,
        runtime_reserved_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        scope_obj = scope if isinstance(scope, DaystromScope) else DaystromScope.from_dict(scope)
        limits = {**dict(backend_limits or {}), **dict(model_limits or {})}
        max_input = int(limits.get("max_input_tokens") or limits.get("context_window_tokens") or 0)
        output_reserved = _non_negative_reservation(
            "output_reservation_tokens",
            output_reservation or limits.get("output_reservation_tokens", 0),
        )
        runtime_reserved = _non_negative_reservation(
            "runtime_reserved_tokens",
            runtime_reserved_tokens if runtime_reserved_tokens is not None else limits.get("runtime_reserved_tokens", 0),
        )
        available_input = max(0, max_input - output_reserved - runtime_reserved) if max_input else 0

        segments = self._segments(
            current_prompt=current_prompt,
            current_messages=current_messages,
            dcn_plan=dcn_plan,
            dcn_packet=dcn_packet,
            dml_context=dml_context,
            dpm_overlay=dpm_overlay,
        )
        rendered = self.runtime_adapter.render_messages(segments)
        input_tokens = self.runtime_adapter.estimate_tokens(rendered["messages"])
        pressure = self._pressure_state(input_tokens=input_tokens, available_input=available_input)
        reason_codes = ["observe_only_no_mutation", "retrieval_not_performed"]
        proposed_actions = self._proposed_actions(pressure["state"])

        return {
            "context_version": "daystrom-context-observation-v1",
            "mode": "observe_only",
            "scope": scope_obj.to_dict(),
            "segment_census": self._segment_census(segments),
            "token_estimate": {
                "input_tokens": input_tokens,
                "max_input_tokens": max_input,
                "reserved_output_tokens": output_reserved,
                "reserved_runtime_tokens": runtime_reserved,
                "available_input_tokens": available_input,
            },
            "pressure_state": pressure,
            "capabilities": self.runtime_adapter.capabilities(),
            "proposed_actions": proposed_actions,
            "reason_codes": reason_codes + [action["reason_code"] for action in proposed_actions],
            "render_manifest": rendered["manifest"],
            "telemetry": {
                "observed_at": float(self.clock()),
                "segment_count": len(segments),
                "input_tokens": input_tokens,
                "reserved_output_tokens": output_reserved,
                "reserved_runtime_tokens": runtime_reserved,
                "pressure_state": pressure["state"],
                "capabilities": self._capability_telemetry(self.runtime_adapter.capabilities()),
            },
            "audit": {
                "reason": "dcm observe-only context observation",
                "reason_codes": reason_codes,
                "packet_id": (dcn_packet or {}).get("packet_id") if isinstance(dcn_packet, dict) else None,
            },
        }

    @staticmethod
    def _segments(
        *,
        current_prompt: str,
        current_messages: Optional[List[Dict[str, Any]]],
        dcn_plan: Optional[Dict[str, Any]],
        dcn_packet: Optional[Dict[str, Any]],
        dml_context: Optional[Dict[str, Any]],
        dpm_overlay: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        segments: List[Dict[str, Any]] = []
        for index, message in enumerate(current_messages or []):
            segments.append(
                {
                    "id": f"prepared_message:{index}",
                    "kind": "prepared_message",
                    "role": message.get("role", "user"),
                    "content": message.get("content", ""),
                    "metadata": {"authority": _prepared_message_authority(message.get("role"), index, len(current_messages or []))},
                }
            )
        if current_prompt and not current_messages:
            segments.append(
                {
                    "id": "prepared_prompt",
                    "kind": "prepared_prompt",
                    "role": "user",
                    "content": current_prompt,
                    "metadata": {"authority": "current_instruction"},
                }
            )
        if dcn_plan or dcn_packet:
            segments.append(
                {
                    "id": "dcn_plan",
                    "kind": "dcn_plan",
                    "role": "user",
                    "content": _summarize_keys(dcn_plan or (dcn_packet or {}).get("dcn_plan") or {}),
                    "metadata": {"authority": "trusted_control"},
                }
            )
        if dpm_overlay:
            segments.append(
                {
                    "id": "dpm_overlay",
                    "kind": "dpm_overlay",
                    "role": "user",
                    "content": str(dpm_overlay.get("overlay_text") or dpm_overlay.get("text") or ""),
                    "metadata": {"authority": "reference"},
                }
            )
        if dml_context:
            for index, item in enumerate(_iter_dml_items(dml_context)):
                segments.append(
                    {
                        "id": f"dml_context:{index}",
                        "kind": "dml_context",
                        "source": "dml",
                        "role": "user",
                        "content": item,
                        "metadata": {"authority": "untrusted_data"},
                    }
                )
        return segments

    @staticmethod
    def _segment_census(segments: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        by_kind: Dict[str, int] = {}
        total_chars = 0
        count = 0
        for segment in segments:
            count += 1
            kind = str(segment.get("kind") or "unknown")
            by_kind[kind] = by_kind.get(kind, 0) + 1
            total_chars += len(str(segment.get("content") or ""))
        return {"total_segments": count, "by_kind": by_kind, "total_content_chars": total_chars}

    @staticmethod
    def _pressure_state(*, input_tokens: int, available_input: int) -> Dict[str, Any]:
        if available_input <= 0:
            return {"state": "unknown", "ratio": None}
        ratio = input_tokens / max(1, available_input)
        if input_tokens > available_input:
            state = "over_limit"
        elif ratio >= 0.9:
            state = "critical"
        elif ratio >= 0.75:
            state = "high"
        else:
            state = "ok"
        return {"state": state, "ratio": round(ratio, 4)}

    @staticmethod
    def _proposed_actions(state: str) -> List[Dict[str, str]]:
        if state == "over_limit":
            return [{"action": "reduce_context", "reason_code": "context_over_limit"}]
        if state == "critical":
            return [{"action": "reserve_context", "reason_code": "context_pressure_critical"}]
        if state == "high":
            return [{"action": "monitor_context", "reason_code": "context_pressure_high"}]
        return []

    @staticmethod
    def _capability_telemetry(capabilities: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "api_messages": bool(capabilities.get("api_messages")),
            "token_estimation": bool(capabilities.get("token_estimation")),
            "kv": bool(capabilities.get("kv")),
            "retrieval": bool(capabilities.get("retrieval")),
            "writeback": bool(capabilities.get("writeback")),
        }


def _summarize_keys(payload: Dict[str, Any]) -> str:
    return ",".join(sorted(str(key) for key in payload.keys()))


def _iter_dml_items(payload: Dict[str, Any]) -> Iterable[str]:
    items = payload.get("items")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                yield str(item.get("content") or item.get("text") or "")
            else:
                yield str(item)
        return
    text = payload.get("raw_context") or payload.get("context")
    if text:
        yield str(text)


def _non_negative_reservation(name: str, value: Any) -> int:
    reservation = int(value or 0)
    if reservation < 0:
        raise ContractError(f"{name} must be non-negative")
    return reservation


def _prepared_message_authority(role: Any, index: int, total_messages: int) -> str:
    normalized_role = str(role or "user").lower()
    if normalized_role == "system":
        return "immutable"
    if normalized_role == "user" and index == total_messages - 1:
        return "current_instruction"
    return "trusted_control"
