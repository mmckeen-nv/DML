"""Daystrom Cognition Network controller facade."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from daystrom_dml.api_contracts import AuditInfo, DaystromScope
from daystrom_dml.cognition.audit import DCNAuditStore
from daystrom_dml.cognition.policy import DeterministicCognitionPolicy
from daystrom_dml.cognition.schema import (
    CognitivePacket,
    CognitionConstraints,
    CognitionEvent,
    CognitionFeedback,
    CognitionPlan,
)


class CognitionController:
    """Facade that assembles inspectable DCN cognitive packets.

    The controller owns orchestration only.  It does not persist memories, mutate
    personality, or execute frontier inference; those remain DML, DPM, and DIP
    concerns behind adapter-style interfaces.
    """

    def __init__(self, adapter: Any = None, dpm: Any = None, policy: Any = None, clock: Any = None, audit_store: Optional[DCNAuditStore] = None) -> None:
        self.adapter = adapter
        self.dpm = dpm
        self.policy = policy or DeterministicCognitionPolicy()
        self.clock = clock or time.time
        self.audit_store = audit_store
        self.feedback_log = []

    def observe(
        self,
        event: Any,
        scope: Optional[DaystromScope] = None,
        constraints: Optional[CognitionConstraints] = None,
    ) -> CognitionPlan:
        return self.plan_context(event, scope=scope, constraints=constraints)

    def plan_context(
        self,
        event: Any,
        scope: Optional[DaystromScope] = None,
        constraints: Optional[CognitionConstraints] = None,
    ) -> CognitionPlan:
        event_obj = self._event(event)
        scope_obj = scope or DaystromScope()
        constraints_obj = constraints or CognitionConstraints()
        return self.policy.plan(event_obj, scope=scope_obj, constraints=constraints_obj)

    def cognitive_packet(
        self,
        event: Any,
        scope: Optional[DaystromScope] = None,
        constraints: Optional[CognitionConstraints] = None,
    ) -> CognitivePacket:
        event_obj = self._event(event)
        scope_obj = scope or DaystromScope()
        constraints_obj = constraints or CognitionConstraints()
        plan = self.plan_context(event_obj, scope=scope_obj, constraints=constraints_obj)

        dml_context = self._retrieve_dml_context(plan, event_obj, scope_obj)
        dpm_overlay = self._render_dpm_overlay(plan, event_obj, scope_obj, constraints_obj)
        assembled_context = self._assemble_context(dpm_overlay, dml_context)

        packet = CognitivePacket(
            scope=scope_obj,
            dcn_plan=plan,
            dml_context=dml_context,
            dpm_overlay=dpm_overlay,
            assembled_context=assembled_context,
            guardrails=[
                "current_turn_instruction_overrides_dpm_and_dcn",
                "no_raw_transcripts_tool_logs_secrets_or_prompt_scaffolds_in_writeback",
            ],
            telemetry={"created_at": self.clock()},
            audit=AuditInfo(
                reason="dcn cognitive packet assembled",
                policy=plan.policy_version,
                reason_codes=plan.reason_codes,
            ),
        )
        if self.audit_store is not None:
            self.audit_store.append(
                "cognitive_packet",
                {
                    "decision_id": plan.decision_id,
                    "packet_id": packet.packet_id,
                    "policy_version": plan.policy_version,
                    "reason_codes": plan.reason_codes,
                    "retrieval_mode": plan.retrieval_plan.mode,
                    "writeback_mode": plan.writeback_plan.mode,
                },
            )
        return packet

    def feedback(self, feedback: Any) -> Dict[str, Any]:
        feedback_obj = feedback if isinstance(feedback, CognitionFeedback) else CognitionFeedback.from_dict(feedback)
        record = feedback_obj.to_dict()
        record["recorded_at"] = self.clock()
        self.feedback_log.append(record)
        if self.audit_store is not None:
            self.audit_store.append("feedback", record)
        return {"accepted": True, "stored": len(self.feedback_log), "feedback": record}

    def audit_tail(self, limit: int = 50) -> list[dict[str, Any]]:
        if self.audit_store is None:
            return list(self.feedback_log)[-max(0, min(limit, 500)):]
        return self.audit_store.tail(limit)

    def _retrieve_dml_context(self, plan: CognitionPlan, event: CognitionEvent, scope: DaystromScope) -> Dict[str, Any]:
        if plan.retrieval_plan.mode == "none" or not self.adapter:
            return {}
        query = plan.retrieval_plan.queries[0] if plan.retrieval_plan.queries else event.content
        if hasattr(self.adapter, "retrieve_context"):
            result = self.adapter.retrieve_context(
                query=query,
                tenant_id=scope.tenant_id,
                client_id=scope.client_id,
                session_id=scope.session_id,
                instance_id=scope.instance_id,
                top_k=plan.retrieval_plan.top_k,
                budget_tokens=plan.retrieval_plan.budget_tokens,
            )
        elif hasattr(self.adapter, "retrieve"):
            result = self.adapter.retrieve(query)
        else:
            result = {}
        return result if isinstance(result, dict) else {"raw_context": str(result)}

    def _render_dpm_overlay(
        self,
        plan: CognitionPlan,
        event: CognitionEvent,
        scope: DaystromScope,
        constraints: CognitionConstraints,
    ) -> Dict[str, Any]:
        if plan.personality_plan.mode == "none" or not self.dpm:
            return {}
        if hasattr(self.dpm, "render_overlay"):
            result = self.dpm.render_overlay(
                scope=scope,
                budget_tokens=constraints.max_personality_tokens,
                current_instruction=event.content,
            )
        elif hasattr(self.dpm, "overlay"):
            result = self.dpm.overlay(event.content)
        else:
            result = {}
        return result if isinstance(result, dict) else {"overlay_text": str(result)}

    @staticmethod
    def _assemble_context(dpm_overlay: Dict[str, Any], dml_context: Dict[str, Any]) -> str:
        parts = []
        overlay_text = dpm_overlay.get("overlay_text") or dpm_overlay.get("text")
        memory_text = dml_context.get("raw_context") or dml_context.get("context")
        if overlay_text:
            parts.append("=== DPM Overlay ===\n" + str(overlay_text).strip())
        if memory_text:
            parts.append("=== DML Context ===\n" + str(memory_text).strip())
        return "\n\n".join(parts)

    @staticmethod
    def _event(event: Any) -> CognitionEvent:
        if isinstance(event, CognitionEvent):
            return event
        if isinstance(event, str):
            return CognitionEvent(content=event)
        if isinstance(event, dict):
            return CognitionEvent.from_dict(event)
        raise TypeError(f"Unsupported cognition event type: {type(event).__name__}")
