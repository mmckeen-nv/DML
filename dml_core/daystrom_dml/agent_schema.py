"""Agentic memory schema and validation utilities."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)


class MemoryKind(Enum):
    """Types of memory entries for agentic workflows."""
    ACTION = "action"
    OBSERVATION = "observation"
    PLAN = "plan"
    ARTIFACT_REF = "artifact_ref"
    ERROR = "error"
    NOTE = "note"


class MemoryPhase(Enum):
    """Phases of agent execution."""
    PLAN = "plan"
    BUILD = "build"
    EXECUTE = "execute"
    DEBUG = "debug"
    REFLECT = "reflect"


class MemoryOutcome(Enum):
    """Outcomes of agent actions."""
    SUCCESS = "success"
    FAIL = "fail"
    PARTIAL = "partial"


@dataclass
class MemoryProvenance:
    """Metadata tracking the origin of a memory."""
    task_id: str
    step_id: str
    episode_id: str
    timestamp: float
    code_ref: Optional[str] = None  # git hash or file path


@dataclass
class AgenticMemoryItem:
    """Extended memory item with agentic metadata."""
    id: int
    text: str
    embedding: Any  # numpy array or similar
    timestamp: float
    salience: float
    fidelity: float
    level: int
    meta: Dict[str, Any] = field(default_factory=dict)
    summary_of: List[int] = field(default_factory=list)

    # Agentic-specific fields
    kind: Optional[MemoryKind] = None
    phase: Optional[MemoryPhase] = None
    tool: Optional[str] = None
    outcome: Optional[MemoryOutcome] = None
    provenance: Optional[MemoryProvenance] = None
    promoted: bool = False  # Whether moved from scratch to durable


class AgenticMemorySchema:
    """Schema validator for agentic memory entries."""

    def __init__(self, strict: bool = True):
        """
        Initialize schema validator.

        Args:
            strict: If True, fail closed on invalid entries.
                    If False, warn and soft-fail.
        """
        self.strict = strict

    def validate(self, meta: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate agentic memory metadata.

        Args:
            meta: Metadata dictionary to validate.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Required fields
        if self.strict and "kind" not in meta:
            errors.append("Required field 'kind' missing")
        elif "kind" in meta:
            try:
                MemoryKind(meta["kind"])
            except ValueError:
                errors.append(f"Invalid 'kind': {meta['kind']}")

        if self.strict and "phase" in meta:
            try:
                MemoryPhase(meta["phase"])
            except ValueError:
                errors.append(f"Invalid 'phase': {meta['phase']}")

        if "provenance" in meta:
            provenance = meta["provenance"]
            if not isinstance(provenance, dict):
                errors.append("provenance must be a dict")
            else:
                required = {"task_id", "step_id", "episode_id", "timestamp"}
                missing = required - provenance.keys()
                if missing:
                    errors.append(f"Missing required provenance fields: {missing}")

        # Outcome validation
        if "outcome" in meta and self.strict:
            try:
                MemoryOutcome(meta["outcome"])
            except ValueError:
                errors.append(f"Invalid 'outcome': {meta['outcome']}")

        return len(errors) == 0, errors

    def make_action(self, text: str, meta: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a memory entry for an action."""
        entry = {
            "kind": "action",
            "phase": meta.get("phase") if meta else None,
            "tool": meta.get("tool") if meta else None,
            "outcome": meta.get("outcome") if meta else None,
            "text": text,
            "timestamp": meta.get("timestamp") if meta else __import__("time").time(),
        }
        if meta and "provenance" in meta:
            entry["provenance"] = meta["provenance"]

        return entry

    def make_observation(self, text: str, meta: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a memory entry for an observation."""
        entry = {
            "kind": "observation",
            "tool": meta.get("tool") if meta else None,
            "text": text,
            "timestamp": meta.get("timestamp") if meta else __import__("time").time(),
        }
        if meta and "provenance" in meta:
            entry["provenance"] = meta["provenance"]

        return entry

    def make_error(self, text: str, meta: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a memory entry for an error."""
        entry = {
            "kind": "error",
            "tool": meta.get("tool") if meta else None,
            "text": text,
            "timestamp": meta.get("timestamp") if meta else __import__("time").time(),
        }
        if meta and "provenance" in meta:
            entry["provenance"] = meta["provenance"]

        return entry

    def make_plan(self, text: str, meta: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a memory entry for a plan."""
        entry = {
            "kind": "plan",
            "text": text,
            "timestamp": meta.get("timestamp") if meta else __import__("time").time(),
        }
        if meta and "provenance" in meta:
            entry["provenance"] = meta["provenance"]

        return entry


def make_agentic_memory(
    text: str,
    kind: MemoryKind,
    meta: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Factory function to create agentic memory entries.

    Args:
        text: Memory content.
        kind: Type of memory (action, observation, etc.).
        meta: Optional additional metadata.

    Returns:
        Dictionary representing the memory entry.
    """
    if kind == MemoryKind.ACTION:
        return AgenticMemorySchema().make_action(text, meta)
    elif kind == MemoryKind.OBSERVATION:
        return AgenticMemorySchema().make_observation(text, meta)
    elif kind == MemoryKind.ERROR:
        return AgenticMemorySchema().make_error(text, meta)
    elif kind == MemoryKind.PLAN:
        return AgenticMemorySchema().make_plan(text, meta)
    else:
        return {
            "kind": kind.value,
            "text": text,
            "timestamp": __import__("time").time(),
        }