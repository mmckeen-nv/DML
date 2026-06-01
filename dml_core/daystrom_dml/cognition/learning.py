"""Safe procedural learning overlay for DCN policy v1.

Phase 10 learning is deliberately narrow: it can tune routing/gating/procedural
fields, but it cannot learn identity, values, user preferences, safety
boundaries, autonomy permissions, or secret-handling rules.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from daystrom_dml.api_contracts import ContractError
from daystrom_dml.cognition.audit import sanitize_audit_payload
from daystrom_dml.cognition.schema import CognitionFeedback, CognitionPlan

SCHEMA_VERSION = "dcn-procedural-learning-v1"
BASE_POLICY_REF = "dcn-policy-v0"

ALLOWED_FIELDS = {
    "retrieval_query_template",
    "memory_mode_preference",
    "verification_requirement",
    "tool_recommendation",
    "context_budget_adjustment",
    "writeback_strictness",
}

FORBIDDEN_FIELDS = {
    "identity",
    "persona",
    "creature",
    "values",
    "ethics",
    "user_preference",
    "preference",
    "safety_boundary",
    "autonomy_permission",
    "permission",
    "secret_handling",
    "secrets",
    "api_key",
    "password",
    "token",
}

MEMORY_MODES = {"none", "resume", "semantic", "continuity", "hybrid"}
VERIFICATION_LEVELS = {"none", "standard", "strict"}
WRITEBACK_STRICTNESS = {"off", "strict", "stricter"}
MAX_BUDGET_DELTA = 600
MAX_TEMPLATE_CHARS = 240
MAX_TOOLS = 8


def _now() -> float:
    return time.time()


def _task_type(value: Any) -> str:
    text = str(value or "answer").strip().lower().replace(" ", "_")
    return text or "answer"


def _redacted_digest(payload: Any) -> str:
    data = json.dumps(sanitize_audit_payload(payload), sort_keys=True, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


@dataclass
class ProceduralProfile:
    task_type: str
    retrieval_query_template: Optional[str] = None
    memory_mode_preference: Optional[str] = None
    verification_requirement: Optional[str] = None
    tool_recommendations: List[str] = field(default_factory=list)
    context_budget_adjustment: int = 0
    writeback_strictness: Dict[str, str] = field(default_factory=dict)
    version: int = 0
    updated_at: float = field(default_factory=_now)
    provenance: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProceduralProfile":
        return cls(
            task_type=_task_type(data.get("task_type")),
            retrieval_query_template=data.get("retrieval_query_template"),
            memory_mode_preference=data.get("memory_mode_preference"),
            verification_requirement=data.get("verification_requirement"),
            tool_recommendations=list(data.get("tool_recommendations") or []),
            context_budget_adjustment=int(data.get("context_budget_adjustment") or 0),
            writeback_strictness=dict(data.get("writeback_strictness") or {}),
            version=int(data.get("version") or 0),
            updated_at=float(data.get("updated_at") or _now()),
            provenance=dict(data.get("provenance") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "retrieval_query_template": self.retrieval_query_template,
            "memory_mode_preference": self.memory_mode_preference,
            "verification_requirement": self.verification_requirement,
            "tool_recommendations": list(self.tool_recommendations),
            "context_budget_adjustment": self.context_budget_adjustment,
            "writeback_strictness": dict(self.writeback_strictness),
            "version": self.version,
            "updated_at": self.updated_at,
            "provenance": dict(self.provenance),
        }


class ProceduralLearningPolicy:
    """Versioned, reversible procedural overlay for deterministic DCN policy."""

    def __init__(self, *, clock: Any = None, audit_store: Any = None) -> None:
        self.clock = clock or _now
        self.audit_store = audit_store
        self.profiles: Dict[str, ProceduralProfile] = {}
        self.audit_log: List[Dict[str, Any]] = []
        self._checkpoints: Dict[str, Dict[str, Any]] = {}
        self.base_checkpoint_id = self.checkpoint("base")

    def learn_from_feedback(self, feedback: Any, *, task_type: Optional[str] = None, source: str = "feedback") -> Dict[str, Any]:
        feedback_obj = feedback if isinstance(feedback, CognitionFeedback) else CognitionFeedback.from_dict(feedback)
        signals = dict(feedback_obj.signals or {})
        task = _task_type(task_type or signals.get("task_type") or "answer")

        updates = []
        explicit = signals.get("learn")
        if isinstance(explicit, dict):
            updates.append(explicit)
        elif isinstance(explicit, list):
            updates.extend(item for item in explicit if isinstance(item, dict))
        else:
            updates.extend(self._updates_from_outcome(feedback_obj, task))

        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        for update in updates:
            result = self.apply_update(task, update, source=source, decision_id=feedback_obj.decision_id)
            (accepted if result.get("accepted") else rejected).append(result)
        return {"accepted": len(accepted), "rejected": len(rejected), "updates": accepted, "rejections": rejected}

    def apply_update(self, task_type: str, update: Dict[str, Any], *, source: str = "manual", decision_id: str = "") -> Dict[str, Any]:
        field_name = str(update.get("field") or "").strip().lower()
        value = update.get("value")
        task = _task_type(update.get("task_type") or task_type)
        if field_name in ALLOWED_FIELDS:
            pass
        elif self._is_forbidden_field(field_name):
            return self._reject(task, field_name, value, source, decision_id, "forbidden_field")
        else:
            return self._reject(task, field_name, value, source, decision_id, "unknown_or_unclassified_field")

        try:
            normalized_value = self._normalize_value(field_name, value)
        except Exception as exc:
            return self._reject(task, field_name, value, source, decision_id, exc.__class__.__name__)

        profile = self.profiles.get(task) or ProceduralProfile(task_type=task)
        if field_name == "tool_recommendation":
            profile.tool_recommendations = list(normalized_value)
        elif field_name == "writeback_strictness":
            profile.writeback_strictness.update(dict(normalized_value))
        else:
            setattr(profile, field_name, normalized_value)
        profile.version += 1
        profile.updated_at = float(self.clock())
        profile.provenance = {"source": source, "decision_id": decision_id, "field": field_name}
        self.profiles[task] = profile

        event = self._audit("accepted", task, field_name, source, decision_id, value, reason="accepted")
        return {"accepted": True, "task_type": task, "field": field_name, "audit": event}

    def apply_to_plan(self, plan: CognitionPlan) -> CognitionPlan:
        profile = self.profiles.get(_task_type(plan.intent.task_type))
        if profile is None:
            return plan
        learned = CognitionPlan.from_dict(plan.to_dict())
        if profile.retrieval_query_template and learned.retrieval_plan.queries:
            learned.retrieval_plan.queries = [profile.retrieval_query_template]
        if profile.memory_mode_preference:
            learned.retrieval_plan.mode = profile.memory_mode_preference
            learned.intent.needs_memory = profile.memory_mode_preference != "none"
        if profile.context_budget_adjustment:
            learned.retrieval_plan.budget_tokens = max(0, learned.retrieval_plan.budget_tokens + profile.context_budget_adjustment)
        if profile.tool_recommendations:
            merged = list(dict.fromkeys(list(learned.tool_plan.recommended_tools) + profile.tool_recommendations))
            learned.tool_plan.recommended_tools = merged[:MAX_TOOLS]
            learned.intent.needs_tools = bool(merged)
        if profile.verification_requirement:
            if profile.verification_requirement == "none":
                learned.tool_plan.verification_required = []
                learned.intent.needs_verification = False
            else:
                marker = "real_tool_output" if profile.verification_requirement == "strict" else "tests"
                learned.tool_plan.verification_required = list(dict.fromkeys(list(learned.tool_plan.verification_required) + [marker]))
                learned.intent.needs_verification = True
        if profile.writeback_strictness:
            if "default" in profile.writeback_strictness and profile.writeback_strictness["default"] in {"strict", "stricter"}:
                learned.writeback_plan.forbidden_classes = list(dict.fromkeys(learned.writeback_plan.forbidden_classes + ["synthetic_claim", "low_confidence"] ))
        learned.policy_version = "dcn-policy-v0+procedural-v1"
        learned.reason_codes = list(dict.fromkeys(list(learned.reason_codes) + ["procedural_learning_applied"] ))
        return learned

    def checkpoint(self, label: str = "checkpoint") -> str:
        checkpoint_id = str(uuid.uuid4())
        self._checkpoints[checkpoint_id] = {
            "checkpoint_id": checkpoint_id,
            "label": label,
            "created_at": float(self.clock()),
            "profiles": self._profiles_dict(),
        }
        return checkpoint_id

    def rollback(self, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        target = checkpoint_id or self.base_checkpoint_id
        if target not in self._checkpoints:
            raise ContractError(f"unknown checkpoint_id: {target}")
        self.profiles = {
            key: ProceduralProfile.from_dict(value)
            for key, value in self._checkpoints[target]["profiles"].items()
        }
        self._audit("rollback", "*", "mutable_overlay", "rollback", target, None, reason="rollback")
        return {"rolled_back": True, "checkpoint_id": target, "profiles": len(self.profiles)}

    def checkpoints(self) -> List[Dict[str, Any]]:
        """Return redacted checkpoint metadata without mutable overlay contents."""
        return [
            {
                "checkpoint_id": str(item.get("checkpoint_id") or checkpoint_id),
                "label": str(item.get("label") or "checkpoint"),
                "created_at": float(item.get("created_at") or 0.0),
                "profiles": len(item.get("profiles") or {}),
            }
            for checkpoint_id, item in sorted(
                self._checkpoints.items(),
                key=lambda pair: float(pair[1].get("created_at") or 0.0),
            )
        ]

    def policy_digest(self) -> str:
        """Return a stable redacted digest of the current mutable overlay."""
        return _redacted_digest({
            "schema_version": SCHEMA_VERSION,
            "base_policy_ref": BASE_POLICY_REF,
            "mutable_overlay": self._profiles_dict(),
        })

    def has_checkpoint(self, checkpoint_id: str) -> bool:
        return bool(checkpoint_id and checkpoint_id in self._checkpoints)

    def export_policy(self) -> Dict[str, Any]:
        profiles = self._profiles_dict()
        audit_digest = _redacted_digest(self.audit_log)
        return {
            "schema_version": SCHEMA_VERSION,
            "base_policy_ref": BASE_POLICY_REF,
            "checkpoint_id": self.checkpoint("export"),
            "created_at": float(self.clock()),
            "mutable_overlay": profiles,
            "audit_log_digest": audit_digest,
        }

    def import_policy(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        if snapshot.get("schema_version") != SCHEMA_VERSION:
            raise ContractError("unsupported procedural learning schema_version")
        if snapshot.get("base_policy_ref") != BASE_POLICY_REF:
            raise ContractError("unsupported base_policy_ref")
        overlay = snapshot.get("mutable_overlay") or {}
        if not isinstance(overlay, dict):
            raise ContractError("mutable_overlay must be a dict")
        profiles = {}
        for key, value in overlay.items():
            if not isinstance(value, dict):
                raise ContractError("mutable_overlay profiles must be dicts")
            profile = self._profile_from_import(_task_type(key), value)
            profiles[profile.task_type] = profile
        self.profiles = profiles
        checkpoint_id = self.checkpoint("import")
        self._audit("import", "*", "mutable_overlay", "import", str(snapshot.get("checkpoint_id") or ""), None, reason="import")
        return {"imported": True, "checkpoint_id": checkpoint_id, "profiles": len(self.profiles)}

    def audit_tail(self, limit: int = 50) -> List[Dict[str, Any]]:
        return list(self.audit_log)[-max(0, min(limit, 500)):]

    def _profiles_dict(self) -> Dict[str, Dict[str, Any]]:
        return {key: profile.to_dict() for key, profile in sorted(self.profiles.items())}

    def _profile_from_import(self, task: str, data: Dict[str, Any]) -> ProceduralProfile:
        allowed_profile_fields = {
            "task_type",
            "retrieval_query_template",
            "memory_mode_preference",
            "verification_requirement",
            "tool_recommendations",
            "context_budget_adjustment",
            "writeback_strictness",
            "version",
            "updated_at",
            "provenance",
        }
        unknown = set(data) - allowed_profile_fields
        if unknown:
            raise ContractError(f"unsupported mutable_overlay profile fields: {', '.join(sorted(unknown))}")
        profile = ProceduralProfile(task_type=_task_type(data.get("task_type") or task))
        if data.get("retrieval_query_template") is not None:
            profile.retrieval_query_template = self._normalize_value("retrieval_query_template", data.get("retrieval_query_template"))
        if data.get("memory_mode_preference") is not None:
            profile.memory_mode_preference = self._normalize_value("memory_mode_preference", data.get("memory_mode_preference"))
        if data.get("verification_requirement") is not None:
            profile.verification_requirement = self._normalize_value("verification_requirement", data.get("verification_requirement"))
        if data.get("tool_recommendations"):
            profile.tool_recommendations = self._normalize_value("tool_recommendation", data.get("tool_recommendations"))
        if data.get("context_budget_adjustment") is not None:
            profile.context_budget_adjustment = self._normalize_value("context_budget_adjustment", data.get("context_budget_adjustment"))
        if data.get("writeback_strictness"):
            profile.writeback_strictness = self._normalize_value("writeback_strictness", data.get("writeback_strictness"))
        profile.version = max(0, int(data.get("version") or 0))
        profile.updated_at = float(data.get("updated_at") or self.clock())
        provenance_raw = data.get("provenance")
        provenance: Dict[str, Any] = provenance_raw if isinstance(provenance_raw, dict) else {}
        profile.provenance = sanitize_audit_payload({
            "source": provenance.get("source") or "import",
            "decision_id": provenance.get("decision_id") or "",
            "field": provenance.get("field") or "mutable_overlay",
        })
        return profile

    def _updates_from_outcome(self, feedback: CognitionFeedback, task: str) -> List[Dict[str, Any]]:
        outcome = str(feedback.outcome or "").lower()
        signals = dict(feedback.signals or {})
        updates: List[Dict[str, Any]] = []
        if outcome in {"helpful_retrieval", "accepted", "verified"} and signals.get("retrieval_helpful"):
            updates.append({"field": "memory_mode_preference", "task_type": task, "value": signals.get("memory_mode") or "hybrid"})
        if signals.get("stale_context"):
            updates.append({"field": "memory_mode_preference", "task_type": task, "value": "none"})
            if signals.get("query_template"):
                updates.append({"field": "retrieval_query_template", "task_type": task, "value": signals.get("query_template")})
        if signals.get("tool_failed"):
            updates.append({"field": "verification_requirement", "task_type": task, "value": "strict"})
            alt = signals.get("alternate_tool")
            if alt:
                updates.append({"field": "tool_recommendation", "task_type": task, "value": [alt]})
        return updates

    def _normalize_value(self, field_name: str, value: Any) -> Any:
        if field_name == "retrieval_query_template":
            text = " ".join(str(value or "").split())
            if not text:
                raise ContractError("retrieval query template must be non-empty")
            return text[:MAX_TEMPLATE_CHARS]
        if field_name == "memory_mode_preference":
            mode = str(value or "").strip().lower()
            if mode not in MEMORY_MODES:
                raise ContractError("invalid memory mode preference")
            return mode
        if field_name == "verification_requirement":
            level = str(value or "").strip().lower()
            if level not in VERIFICATION_LEVELS:
                raise ContractError("invalid verification requirement")
            return level
        if field_name == "tool_recommendation":
            values = value if isinstance(value, list) else [value]
            tools = [str(item).strip() for item in values if str(item or "").strip()]
            if not tools:
                raise ContractError("tool recommendation must be non-empty")
            return list(dict.fromkeys(tools))[:MAX_TOOLS]
        if field_name == "context_budget_adjustment":
            delta = int(value)
            if abs(delta) > MAX_BUDGET_DELTA:
                raise ContractError("context budget adjustment exceeds drift ceiling")
            return delta
        if field_name == "writeback_strictness":
            if isinstance(value, str):
                value = {"default": value}
            if not isinstance(value, dict):
                raise ContractError("writeback strictness must be a dict or string")
            out = {}
            for key, strictness in value.items():
                level = str(strictness or "").strip().lower()
                if level not in WRITEBACK_STRICTNESS:
                    raise ContractError("invalid writeback strictness")
                out[str(key or "default")] = level
            return out
        raise ContractError("unknown field")

    def _is_forbidden_field(self, field_name: str) -> bool:
        if field_name in FORBIDDEN_FIELDS:
            return True
        return any(token in field_name for token in FORBIDDEN_FIELDS)

    def _reject(self, task: str, field_name: str, value: Any, source: str, decision_id: str, reason: str) -> Dict[str, Any]:
        event = self._audit("rejected", task, field_name, source, decision_id, value, reason=reason)
        return {"accepted": False, "task_type": task, "field": field_name, "reason": reason, "audit": event}

    def _audit(self, action: str, task: str, field_name: str, source: str, decision_id: str, value: Any, *, reason: str) -> Dict[str, Any]:
        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": float(self.clock()),
            "action": action,
            "task_type": task,
            "field": field_name,
            "source": source,
            "decision_id": decision_id,
            "reason": reason,
            "attempted_value": "[REDACTED]" if action == "rejected" else sanitize_audit_payload(value),
            "attempted_value_digest": _redacted_digest(value),
        }
        clean = sanitize_audit_payload(event)
        self.audit_log.append(clean)
        self.audit_log = self.audit_log[-500:]
        if self.audit_store is not None:
            self.audit_store.append("procedural_learning", clean)
        return clean
