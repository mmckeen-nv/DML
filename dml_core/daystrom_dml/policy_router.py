"""Policy router for adaptive DML settings per task/phase."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .agent_schema import MemoryPhase

LOGGER = logging.getLogger(__name__)


class TaskType(Enum):
    """Task types for routing decisions."""
    DEVSOPS = "devops"
    CODING = "coding"
    RESEARCH = "research"
    CHAT = "chat"
    OTHER = "other"


@dataclass
class SettingsOverride:
    """Override settings for a specific step."""
    similarity_threshold: Optional[float] = None
    top_k: Optional[int] = None
    token_budget: Optional[int] = None
    semantic_pct: Optional[float] = None
    literal_pct: Optional[float] = None
    free_pct: Optional[float] = None
    recency_weight: Optional[float] = None
    commitment_threshold: Optional[float] = None
    allowed_kinds: Optional[List[str]] = None


@dataclass
class RouterDecision:
    """Decision made by the router."""
    task_type: TaskType
    phase: MemoryPhase
    selected_profile: str
    overrides: SettingsOverride
    reasons: List[str]


class PolicyRouter:
    """Routes task state to appropriate DML settings."""

    # Default profiles for each task type
    PROFILES = {
        TaskType.DEVSOPS.value: {
            "similarity_threshold": 0.4,
            "top_k": 6,
            "token_budget": 400,
            "semantic_pct": 0.6,
            "literal_pct": 0.4,
            "free_pct": 0.0,
            "recency_weight": 0.7,
            "commitment_threshold": 0.8,
            "allowed_kinds": ["observation", "action", "plan"],
        },
        TaskType.CODING.value: {
            "similarity_threshold": 0.35,
            "top_k": 8,
            "token_budget": 500,
            "semantic_pct": 0.5,
            "literal_pct": 0.3,
            "free_pct": 0.2,
            "recency_weight": 0.6,
            "commitment_threshold": 0.75,
            "allowed_kinds": ["action", "observation", "plan", "artifact_ref"],
        },
        TaskType.RESEARCH.value: {
            "similarity_threshold": 0.5,
            "top_k": 10,
            "token_budget": 600,
            "semantic_pct": 0.7,
            "literal_pct": 0.2,
            "free_pct": 0.1,
            "recency_weight": 0.4,
            "commitment_threshold": 0.7,
            "allowed_kinds": ["observation", "plan", "note"],
        },
        TaskType.CHAT.value: {
            "similarity_threshold": 0.3,
            "top_k": 4,
            "token_budget": 300,
            "semantic_pct": 0.6,
            "literal_pct": 0.4,
            "free_pct": 0.0,
            "recency_weight": 0.5,
            "commitment_threshold": 0.5,
            "allowed_kinds": ["note", "plan"],
        },
    }

    # Phase modifiers
    PHASE_MODIFIERS = {
        MemoryPhase.PLAN: {
            "similarity_threshold": 0.25,  # Lower for exploration
            "top_k": 12,  # More context for planning
            "recency_weight": 0.3,
        },
        MemoryPhase.BUILD: {
            "similarity_threshold": 0.4,
            "top_k": 8,
            "recency_weight": 0.5,
        },
        MemoryPhase.EXECUTE: {
            "similarity_threshold": 0.5,
            "top_k": 6,
            "recency_weight": 0.8,  # Recent actions important
            "allowed_kinds": ["action", "observation", "error"],
        },
        MemoryPhase.DEBUG: {
            "similarity_threshold": 0.6,  # High precision needed
            "top_k": 4,
            "recency_weight": 0.9,
            "allowed_kinds": ["error", "observation", "action"],
        },
        MemoryPhase.REFLECT: {
            "similarity_threshold": 0.35,
            "top_k": 8,
            "recency_weight": 0.4,
            "allowed_kinds": ["plan", "observation", "note"],
        },
    }

    def __init__(
        self,
        enabled: bool = False,
        profile: Optional[str] = None,
        log_level: str = "info",
    ):
        """
        Initialize policy router.

        Args:
            enabled: Whether router is enabled.
            profile: Force a specific profile (for debugging).
            log_level: Logging level.
        """
        self.enabled = enabled
        self.forced_profile = profile
        self.log_level = log_level

    def detect_task_type(self, meta: Optional[Dict] = None) -> TaskType:
        """
        Infer task type from context.

        Args:
            meta: Optional metadata with hints.

        Returns:
            Detected task type.
        """
        if not meta:
            return TaskType.OTHER

        # Check for tool hints
        tools = meta.get("tools", [])
        tools_lower = [t.lower() for t in tools]

        if "ssh" in tools_lower or "docker" in tools_lower or "kubectl" in tools_lower or "k8s" in tools_lower:
            return TaskType.DEVSOPS
        elif "git" in tools_lower or "python" in tools_lower or "code" in tools_lower:
            return TaskType.CODING
        elif "research" in tools_lower or "study" in tools_lower or "search" in tools_lower:
            return TaskType.RESEARCH
        else:
            return TaskType.CHAT

    def decide(
        self,
        meta: Optional[Dict] = None,
        phase: Optional[MemoryPhase] = None,
        token_pressure: float = 0.5,
        stuckness: int = 0,
        recency_need: bool = False,
    ) -> Optional[RouterDecision]:
        """
        Make routing decision for current state.

        Args:
            meta: Context metadata.
            phase: Current phase.
            token_pressure: How full context is (0-1).
            stuckness: Number of repeated similar actions.
            recency_need: Whether recent info is critical.

        Returns:
            RouterDecision or None if router disabled.
        """
        if not self.enabled:
            return None

        # Detect task type
        task_type = self.detect_task_type(meta)

        # Get base profile
        profile_name = self.forced_profile or task_type.value
        if profile_name not in self.PROFILES:
            LOGGER.warning(f"Unknown profile {profile_name}, using {task_type.value}")
            profile_name = task_type.value

        base_profile = self.PROFILES[profile_name].copy()

        # Apply phase modifiers
        if phase:
            if phase in self.PHASE_MODIFIERS:
                modifier = self.PHASE_MODIFIERS[phase]
                base_profile.update(modifier)

        # Apply state-based adjustments
        reasons = [f"Task: {task_type.value}"]

        if stuckness > 0:
            base_profile["similarity_threshold"] = min(0.7, base_profile["similarity_threshold"] + 0.1)
            reasons.append(f"Stuckness detected: {stuckness}")

        if token_pressure > 0.8:
            base_profile["top_k"] = max(3, base_profile["top_k"] - 2)
            base_profile["token_budget"] = max(200, int(base_profile["token_budget"] * 0.8))
            reasons.append("High token pressure")

        if recency_need:
            base_profile["recency_weight"] = min(1.0, base_profile["recency_weight"] + 0.2)
            reasons.append("Recency needed")

        # Extract overrides
        overrides = SettingsOverride(
            similarity_threshold=base_profile.get("similarity_threshold"),
            top_k=base_profile.get("top_k"),
            token_budget=base_profile.get("token_budget"),
            semantic_pct=base_profile.get("semantic_pct"),
            literal_pct=base_profile.get("literal_pct"),
            free_pct=base_profile.get("free_pct"),
            recency_weight=base_profile.get("recency_weight"),
            commitment_threshold=base_profile.get("commitment_threshold"),
            allowed_kinds=base_profile.get("allowed_kinds"),
        )

        # Log decision
        if self.log_level in ["debug", "info"]:
            LOGGER.info(
                f"Router decision: {task_type.value}/{phase.value if phase else 'N/A'} "
                f"profile={profile_name} reasons={reasons}"
            )

        return RouterDecision(
            task_type=task_type,
            phase=phase or MemoryPhase.EXECUTE,
            selected_profile=profile_name,
            overrides=overrides,
            reasons=reasons,
        )


# Default router instance
_default_router: Optional[PolicyRouter] = None


def get_default_router() -> PolicyRouter:
    """Get or create default router instance."""
    global _default_router
    if _default_router is None:
        _default_router = PolicyRouter()
    return _default_router