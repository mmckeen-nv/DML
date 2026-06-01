"""Daystrom Inference Pipeline boundary schemas.

DIP is the unfinished/prototype inference-preparation layer.  These contracts
make that boundary explicit without turning DML, DPM, or DCN into inference
clients.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from daystrom_dml.api_contracts import ContractError, DaystromScope, SerializableDataclass, TokenBudget
from daystrom_dml.cognition.schema import CognitivePacket


@dataclass
class DIPPrepareRequest(SerializableDataclass):
    """Request to prepare frontier input from a DCN cognitive packet or prompt."""

    prompt: str = ""
    cognitive_packet: Optional[CognitivePacket] = None
    scope: DaystromScope = field(default_factory=DaystromScope)
    include_local_draft: bool = True
    local_max_tokens: int = 256
    frontier_max_tokens: int = 512
    top_k: int = 8
    direct_input_tokens_estimate: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "DIPPrepareRequest":
        data = data or {}
        if not isinstance(data, dict):
            raise ContractError(f"DIPPrepareRequest.from_dict expected dict, got {type(data).__name__}")
        packet_payload = data.get("cognitive_packet")
        return cls(
            prompt=str(data.get("prompt") or ""),
            cognitive_packet=CognitivePacket.from_dict(packet_payload) if isinstance(packet_payload, dict) else packet_payload,
            scope=DaystromScope.from_dict(data.get("scope")),
            include_local_draft=bool(data.get("include_local_draft", True)),
            local_max_tokens=int(data.get("local_max_tokens", 256)),
            frontier_max_tokens=int(data.get("frontier_max_tokens", 512)),
            top_k=int(data.get("top_k", 8)),
            direct_input_tokens_estimate=data.get("direct_input_tokens_estimate"),
        )

    def __post_init__(self) -> None:
        if self.local_max_tokens < 0 or self.frontier_max_tokens < 0 or self.top_k < 0:
            raise ContractError("DIP token and top_k limits must be non-negative")


@dataclass
class DIPPrepareResult(SerializableDataclass):
    """Prepared input for a frontier model, not the model's final response."""

    dip_version: str = "daystrom-inference-pipeline-prototype-v1"
    inference_enabled: bool = False
    mode: str = "prepare_only"
    prompt: str = ""
    frontier_prompt: str = ""
    frontier_max_tokens: int = 512
    token_budget: TokenBudget = field(default_factory=TokenBudget)
    dcn_packet_id: Optional[str] = None
    dcn_policy_version: Optional[str] = None
    dml_context_used: bool = False
    local_draft: str = ""
    telemetry: Dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "DIPPrepareResult":
        data = data or {}
        return cls(
            dip_version=data.get("dip_version", "daystrom-inference-pipeline-prototype-v1"),
            inference_enabled=bool(data.get("inference_enabled", False)),
            mode=data.get("mode", "prepare_only"),
            prompt=data.get("prompt", ""),
            frontier_prompt=data.get("frontier_prompt", ""),
            frontier_max_tokens=int(data.get("frontier_max_tokens", 512)),
            token_budget=TokenBudget.from_dict(data.get("token_budget")),
            dcn_packet_id=data.get("dcn_packet_id"),
            dcn_policy_version=data.get("dcn_policy_version"),
            dml_context_used=bool(data.get("dml_context_used", False)),
            local_draft=data.get("local_draft", ""),
            telemetry=dict(data.get("telemetry") or {}),
            warnings=list(data.get("warnings") or []),
        )
