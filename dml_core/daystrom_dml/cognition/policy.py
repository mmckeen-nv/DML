"""Deterministic Daystrom Cognition Network policy v0."""
from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional

from daystrom_dml.api_contracts import DaystromScope, ReasonCode, RiskInfo
from daystrom_dml.cognition.schema import (
    CognitionConstraints,
    CognitionEvent,
    CognitionPlan,
    FrontierPlan,
    IntentAssessment,
    PersonalityPlan,
    RetrievalPlan,
    ToolPlan,
    WritebackPlan,
)


class DeterministicCognitionPolicy:
    """Rules-first DCN policy.

    This intentionally does not call an LLM.  Its baseline is deterministic;
    optional Phase 10 procedural learning is an additive, reversible overlay for
    routing/gating fields only.
    """

    policy_version = "dcn-policy-v0"

    def __init__(self, learning: Any = None) -> None:
        self.learning = learning

    MEMORY_WORDS = ("remember", "recall", "memory", "what did we", "what was", "find the session")
    RESUME_WORDS = ("resume", "continue", "pick up", "where we left", "rehydrate", "compaction")
    CODE_WORDS = ("code", "build", "implement", "fix", "test", "run", "verify", "debug", "refactor", "compile")
    SETUP_WORDS = ("setup", "set up", "configure", "install", "troubleshoot", "endpoint", "api key")
    GREETING_RE = re.compile(r"^\s*(hi|hello|hey|yo|thanks|thank you|ok|okay|great|cool)(\s+again)?[!\.\s]*$", re.I)
    PREFERENCE_WORDS = ("prefer", "preference", "call me", "don't", "do not", "remember i", "i like", "i dislike")
    SIDE_EFFECT_WORDS = ("delete", "remove", "send", "publish", "deploy", "push", "merge", "restart", "kill", "overwrite")

    def plan(
        self,
        event: CognitionEvent,
        scope: Optional[DaystromScope] = None,
        constraints: Optional[CognitionConstraints] = None,
    ) -> CognitionPlan:
        scope = scope or DaystromScope()
        constraints = constraints or CognitionConstraints()
        # v0 is scope-aware at the contract boundary; future policy versions may
        # use scope for tenant/client-specific learned routing.

        text = event.content or ""
        lowered = text.lower()
        metadata = event.metadata or {}
        reason_codes: List[str] = []

        is_greeting = bool(self.GREETING_RE.match(text))
        metadata_resume = event.type in {"gateway_resume", "compaction_resume"} or bool(metadata.get("compaction_resume"))
        is_resume = self._has_any(lowered, self.RESUME_WORDS) or metadata_resume
        is_memory = self._has_any(lowered, self.MEMORY_WORDS)
        is_code = self._has_any(lowered, self.CODE_WORDS)
        is_setup = self._has_any(lowered, self.SETUP_WORDS)
        is_preference = self._has_any(lowered, self.PREFERENCE_WORDS)
        is_side_effect = self._has_any(lowered, self.SIDE_EFFECT_WORDS)

        task_type = "answer"
        confidence = 0.45
        if is_greeting:
            task_type = "answer"
            confidence = 0.9
            reason_codes.append(ReasonCode.GREETING.value)
        if is_memory:
            task_type = "recall"
            confidence = max(confidence, 0.8)
            reason_codes.append(ReasonCode.MEMORY_REQUEST.value)
        if is_resume:
            task_type = "planning" if not is_code else "code_change"
            confidence = max(confidence, 0.8)
            reason_codes.append(ReasonCode.RESUME_REQUEST.value)
        if is_code:
            task_type = "code_change"
            confidence = max(confidence, 0.82)
            reason_codes.append(ReasonCode.CODE_TASK.value)
        if "debug" in lowered or "failure" in lowered or "traceback" in lowered:
            task_type = "debugging"
            confidence = max(confidence, 0.85)
            reason_codes.append(ReasonCode.DEBUG_TASK.value)
        if is_setup:
            task_type = "admin" if not is_code else task_type
            confidence = max(confidence, 0.75)
            reason_codes.append(ReasonCode.SETUP_TASK.value)
        if is_preference:
            confidence = max(confidence, 0.72)
            reason_codes.append(ReasonCode.PREFERENCE_SIGNAL.value)
        if is_side_effect:
            reason_codes.append(ReasonCode.SIDE_EFFECT.value)

        needs_memory = (is_memory or is_resume or metadata.get("long_horizon") or metadata.get("needs_memory")) and (not is_greeting or metadata_resume)
        needs_tools = constraints.allow_tools and (is_code or is_setup or metadata.get("needs_tools"))
        needs_verification = bool(needs_tools and (is_code or "verify" in lowered or "test" in lowered or "run" in lowered))
        if needs_tools:
            reason_codes.append(ReasonCode.TOOL_NEEDED.value)
        if needs_verification:
            reason_codes.append(ReasonCode.VERIFICATION_NEEDED.value)

        retrieval_mode = "none"
        queries: List[str] = []
        if needs_memory:
            retrieval_mode = "resume" if (is_resume or event.type in {"gateway_resume", "compaction_resume"}) and not is_memory else "hybrid"
            queries = [self._query_from_text(text)]
        elif is_setup and not is_greeting:
            retrieval_mode = "semantic"
            queries = [self._query_from_text(text)]
            needs_memory = True

        recommended_tools: List[str] = []
        verification_required: List[str] = []
        if needs_tools:
            recommended_tools.extend(["terminal", "file"])
        if needs_verification:
            verification_required.extend(["tests", "real_tool_output"])

        writeback_mode = "none"
        candidate_classes: List[str] = []
        if is_preference:
            writeback_mode = "preference_candidate"
            candidate_classes.append("preference")
        elif is_resume or is_code or is_setup:
            writeback_mode = "durable_signal_only"
            candidate_classes.append("durable_signal")

        risk_level = "low"
        requires_confirmation = False
        side_effect_classes: List[str] = []
        if is_side_effect:
            risk_level = "medium"
            requires_confirmation = any(word in lowered for word in ("delete", "remove", "deploy", "push", "merge", "overwrite"))
            side_effect_classes = [word for word in self.SIDE_EFFECT_WORDS if word in lowered]
            reason_codes.append(ReasonCode.MEDIUM_RISK.value)
        else:
            reason_codes.append(ReasonCode.LOW_RISK.value)

        plan = CognitionPlan(
            intent=IntentAssessment(
                task_type=task_type,
                confidence=confidence,
                needs_memory=bool(needs_memory),
                needs_personality=True,
                needs_tools=bool(needs_tools),
                needs_verification=needs_verification,
            ),
            risk=RiskInfo(
                level=risk_level,
                reasons=side_effect_classes,
                requires_confirmation=requires_confirmation,
                side_effect_classes=side_effect_classes,
            ),
            retrieval_plan=RetrievalPlan(
                mode=retrieval_mode,
                queries=queries,
                top_k=6,
                budget_tokens=constraints.max_memory_tokens,
            ),
            personality_plan=PersonalityPlan(
                mode="bounded_overlay",
                budget_tokens=constraints.max_personality_tokens,
                suppress_if_conflicts_with_current_turn=True,
            ),
            tool_plan=ToolPlan(
                allowed=constraints.allow_tools,
                recommended_tools=list(dict.fromkeys(recommended_tools)),
                verification_required=verification_required,
            ),
            writeback_plan=WritebackPlan(mode=writeback_mode, candidate_classes=candidate_classes),
            frontier_plan=FrontierPlan(
                mode="dml_context" if retrieval_mode != "none" else "direct",
                max_input_tokens=constraints.max_total_context_tokens,
            ),
            reason_codes=list(dict.fromkeys(reason_codes or [ReasonCode.GENERAL.value])),
            policy_version=self.policy_version,
        )
        if self.learning is not None and hasattr(self.learning, "apply_to_plan"):
            plan = self.learning.apply_to_plan(plan)
        return plan

    @staticmethod
    def _has_any(text: str, needles: Iterable[str]) -> bool:
        return any(needle in text for needle in needles)

    @staticmethod
    def _query_from_text(text: str) -> str:
        clean = " ".join((text or "").split())
        return clean[:240]
