"""Daystrom Cognition Network public schema contracts."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from daystrom_dml.api_contracts import (
    AuditInfo,
    ContractError,
    DaystromScope,
    RiskInfo,
    SerializableDataclass,
    TokenBudget,
)
from daystrom_dml.contracts import COGNITIVE_PACKET_V1

FORBIDDEN_WRITEBACK_CLASSES = ["raw_transcript", "tool_log", "secret", "prompt_scaffold"]


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, SerializableDataclass):
        return value.to_dict()
    if isinstance(value, dict):
        return value
    raise ContractError(f"Expected dict-compatible value, got {type(value).__name__}")


@dataclass
class CognitionEvent(SerializableDataclass):
    type: str = "user_message"
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitionConstraints(SerializableDataclass):
    max_total_context_tokens: int = 6000
    max_memory_tokens: int = 1800
    max_personality_tokens: int = 300
    allow_tools: bool = True
    allow_learning: bool = False

    def __post_init__(self) -> None:
        for name in ("max_total_context_tokens", "max_memory_tokens", "max_personality_tokens"):
            if getattr(self, name) < 0:
                raise ContractError(f"{name} must be non-negative")


@dataclass
class IntentAssessment(SerializableDataclass):
    task_type: str = "unknown"
    confidence: float = 0.0
    needs_memory: bool = False
    needs_personality: bool = True
    needs_tools: bool = False
    needs_verification: bool = False

    def __post_init__(self) -> None:
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ContractError("confidence must be between 0.0 and 1.0")


@dataclass
class RetrievalPlan(SerializableDataclass):
    mode: str = "none"
    queries: List[str] = field(default_factory=list)
    top_k: int = 6
    budget_tokens: int = 1800
    ground_truth_policy: str = "low_confidence"

    def __post_init__(self) -> None:
        allowed = {"none", "resume", "semantic", "continuity", "hybrid"}
        if self.mode not in allowed:
            raise ContractError(f"retrieval mode must be one of {sorted(allowed)}")
        if self.top_k < 0 or self.budget_tokens < 0:
            raise ContractError("retrieval top_k and budget_tokens must be non-negative")


@dataclass
class PersonalityPlan(SerializableDataclass):
    mode: str = "bounded_overlay"
    budget_tokens: int = 300
    suppress_if_conflicts_with_current_turn: bool = True

    def __post_init__(self) -> None:
        allowed = {"none", "bounded_overlay"}
        if self.mode not in allowed:
            raise ContractError(f"personality mode must be one of {sorted(allowed)}")
        if self.budget_tokens < 0:
            raise ContractError("personality budget_tokens must be non-negative")


@dataclass
class ToolPlan(SerializableDataclass):
    allowed: bool = True
    recommended_tools: List[str] = field(default_factory=list)
    verification_required: List[str] = field(default_factory=list)


@dataclass
class WritebackPlan(SerializableDataclass):
    mode: str = "none"
    forbidden_classes: List[str] = field(default_factory=lambda: list(FORBIDDEN_WRITEBACK_CLASSES))
    candidate_classes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        allowed = {"none", "durable_signal_only", "handoff", "preference_candidate"}
        if self.mode not in allowed:
            raise ContractError(f"writeback mode must be one of {sorted(allowed)}")
        forbidden = set(FORBIDDEN_WRITEBACK_CLASSES)
        self.forbidden_classes = list(dict.fromkeys(self.forbidden_classes or FORBIDDEN_WRITEBACK_CLASSES))
        for class_name in self.candidate_classes:
            if class_name in forbidden:
                raise ContractError(f"forbidden writeback class requested: {class_name}")


@dataclass
class FrontierPlan(SerializableDataclass):
    mode: str = "direct"
    max_input_tokens: int = 6000
    max_output_tokens: int = 1200

    def __post_init__(self) -> None:
        allowed = {"direct", "dml_context", "verify_local_draft", "full_frontier", "no_frontier"}
        if self.mode not in allowed:
            raise ContractError(f"frontier mode must be one of {sorted(allowed)}")
        if self.max_input_tokens < 0 or self.max_output_tokens < 0:
            raise ContractError("frontier token limits must be non-negative")


@dataclass
class CognitionPlan(SerializableDataclass):
    intent: IntentAssessment = field(default_factory=IntentAssessment)
    risk: RiskInfo = field(default_factory=RiskInfo)
    retrieval_plan: RetrievalPlan = field(default_factory=RetrievalPlan)
    personality_plan: PersonalityPlan = field(default_factory=PersonalityPlan)
    tool_plan: ToolPlan = field(default_factory=ToolPlan)
    writeback_plan: WritebackPlan = field(default_factory=WritebackPlan)
    frontier_plan: FrontierPlan = field(default_factory=FrontierPlan)
    reason_codes: List[str] = field(default_factory=list)
    policy_version: str = "dcn-policy-v0"
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "CognitionPlan":
        data = data or {}
        return cls(
            intent=IntentAssessment.from_dict(data.get("intent")),
            risk=RiskInfo.from_dict(data.get("risk")),
            retrieval_plan=RetrievalPlan.from_dict(data.get("retrieval_plan")),
            personality_plan=PersonalityPlan.from_dict(data.get("personality_plan")),
            tool_plan=ToolPlan.from_dict(data.get("tool_plan")),
            writeback_plan=WritebackPlan.from_dict(data.get("writeback_plan")),
            frontier_plan=FrontierPlan.from_dict(data.get("frontier_plan")),
            reason_codes=list(data.get("reason_codes") or []),
            policy_version=data.get("policy_version", "dcn-policy-v0"),
            decision_id=data.get("decision_id") or str(uuid.uuid4()),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "CognitionPlan":
        return cls.from_dict(json.loads(text))


@dataclass
class CognitivePacket(SerializableDataclass):
    packet_version: str = COGNITIVE_PACKET_V1
    scope: DaystromScope = field(default_factory=DaystromScope)
    dcn_plan: CognitionPlan = field(default_factory=CognitionPlan)
    dml_context: Dict[str, Any] = field(default_factory=dict)
    dpm_overlay: Dict[str, Any] = field(default_factory=dict)
    assembled_context: str = ""
    guardrails: List[str] = field(default_factory=list)
    telemetry: Dict[str, Any] = field(default_factory=dict)
    audit: AuditInfo = field(default_factory=AuditInfo)
    packet_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.packet_version != COGNITIVE_PACKET_V1:
            raise ContractError(
                f"packet_version must be {COGNITIVE_PACKET_V1!r}, got {self.packet_version!r}"
            )

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "CognitivePacket":
        data = data or {}
        return cls(
            packet_version=data.get("packet_version", COGNITIVE_PACKET_V1),
            scope=DaystromScope.from_dict(data.get("scope")),
            dcn_plan=CognitionPlan.from_dict(data.get("dcn_plan")),
            dml_context=dict(data.get("dml_context") or {}),
            dpm_overlay=dict(data.get("dpm_overlay") or {}),
            assembled_context=data.get("assembled_context", ""),
            guardrails=list(data.get("guardrails") or []),
            telemetry=dict(data.get("telemetry") or {}),
            audit=AuditInfo.from_dict(data.get("audit")),
            packet_id=data.get("packet_id") or str(uuid.uuid4()),
            created_at=float(data.get("created_at", time.time())),
        )


@dataclass
class CognitionFeedback(SerializableDataclass):
    decision_id: str = ""
    outcome: str = "accepted"
    signals: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    latency_ms: float = 0.0
    plan_fidelity: Optional[float] = None

    def __post_init__(self) -> None:
        if self.plan_fidelity is not None and not 0.0 <= self.plan_fidelity <= 1.0:
            raise ContractError("plan_fidelity must be between 0.0 and 1.0")
