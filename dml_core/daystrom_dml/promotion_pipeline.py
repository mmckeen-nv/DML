"""Promotion pipeline for scratch→verified→durable memory."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .agent_schema import MemoryPhase, MemoryOutcome, AgenticMemorySchema

LOGGER = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """In-memory representation of a memory entry."""
    text: str
    embedding: Any
    timestamp: float
    meta: Dict[str, Any] = field(default_factory=dict)
    level: int = 0
    salience: float = 0.5
    fidelity: float = 0.5
    id: int = 0
    kind: Optional[str] = None
    phase: Optional[str] = None
    tool: Optional[str] = None
    outcome: Optional[str] = None
    promoted: bool = False
    is_durable: bool = False


class ScratchStore:
    """Temporary store for new entries during agentic mode."""

    def __init__(self):
        self.entries: List[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        """Add entry to scratch store."""
        entry.promoted = False
        entry.is_durable = False
        self.entries.append(entry)
        LOGGER.debug(f"Added {entry.kind or 'entry'} to scratch store")

    def get_by_provenance(self, episode_id: str) -> List[MemoryEntry]:
        """Get all entries for a specific episode."""
        return [e for e in self.entries if e.meta.get("provenance", {}).get("episode_id") == episode_id]

    def clear(self) -> None:
        """Clear all scratch entries."""
        self.entries.clear()


class VerifiedStore:
    """Intermediate store for verified entries ready for durability."""

    def __init__(self):
        self.entries: List[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        """Add verified entry."""
        entry.promoted = True
        entry.is_durable = False
        self.entries.append(entry)
        LOGGER.debug(f"Promoted {entry.kind or 'entry'} to verified store")

    def get_ready_for_durable(self, commitment_threshold: float = 0.75) -> List[MemoryEntry]:
        """Get entries ready for durable promotion."""
        ready = []
        for entry in self.entries:
            if self._is_ready(entry, commitment_threshold):
                ready.append(entry)
        return ready

    def _is_ready(self, entry: MemoryEntry, commitment_threshold: float) -> bool:
        """Check if entry is ready for durable promotion."""
        if not entry.promoted:
            return False

        outcome = entry.outcome
        if outcome == MemoryOutcome.SUCCESS:
            return True
        elif outcome == MemoryOutcome.PARTIAL:
            # Partial success needs higher threshold
            return entry.fidelity >= commitment_threshold
        else:
            # Failures are not promoted (unless strict mode)
            return False


class DurableStore:
    """Durable LTM store for proven memories."""
    def __init__(self):
        self.entries: List[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        """Promote to durable store."""
        entry.is_durable = True
        self.entries.append(entry)
        LOGGER.info(f"Promoted {entry.kind or 'entry'} to durable LTM")

    def get(self, limit: int = 10) -> List[MemoryEntry]:
        """Get recent durable entries."""
        return sorted(self.entries, key=lambda e: e.timestamp, reverse=True)[:limit]


class PromotionPipeline:
    """Scratch → Verified → Durable promotion pipeline."""

    def __init__(
        self,
        commitment_threshold: float = 0.75,
        allow_action_observation: bool = True,
        strict_mode: bool = True,
    ):
        """
        Initialize promotion pipeline.

        Args:
            commitment_threshold: Minimum fidelity/success for durable promotion.
            allow_action_observation: Allow action/observation to be durable.
            strict_mode: Fail closed on invalid entries.
        """
        self.scratch = ScratchStore()
        self.verified = VerifiedStore()
        self.durable = DurableStore()
        self.commitment_threshold = commitment_threshold
        self.allow_action_observation = allow_action_observation
        self.strict_mode = strict_mode

    def ingest_to_scratch(self, entry: MemoryEntry) -> None:
        """Ingest entry to scratch store."""
        self.scratch.add(entry)

    def promote_to_verified(self, entry: MemoryEntry) -> bool:
        """
        Promote entry from scratch to verified.

        Args:
            entry: Entry to promote.

        Returns:
            True if promoted, False otherwise.
        """
        # Validate
        is_valid, errors = AgenticMemorySchema().validate(entry.meta)
        if not is_valid:
            if self.strict_mode:
                LOGGER.error(f"Entry rejected: {errors}")
                return False
            else:
                LOGGER.warning(f"Entry warning: {errors}")

        # Check kind
        kind = entry.kind
        if kind in ["action", "observation"] and not self.allow_action_observation:
            LOGGER.debug("Skipping action/observation from verified")
            return False

        # Check outcome
        outcome = entry.outcome
        if outcome not in [MemoryOutcome.SUCCESS, MemoryOutcome.PARTIAL]:
            LOGGER.debug(f"Skipping {outcome} entry from verified")
            return False

        # Add to verified
        self.verified.add(entry)
        return True

    def promote_to_durable(self, entry: MemoryEntry) -> bool:
        """
        Promote entry from verified to durable.

        Args:
            entry: Entry to promote.

        Returns:
            True if promoted, False otherwise.
        """
        # Check if already durable
        if entry.is_durable:
            return True

        # Verify entry is in verified store
        if entry not in self.verified.entries:
            LOGGER.warning("Entry not in verified store, cannot promote to durable")
            return False

        # Check commitment threshold
        if entry.outcome == MemoryOutcome.PARTIAL and entry.fidelity < self.commitment_threshold:
            LOGGER.debug("Entry fidelity below commitment threshold")
            return False

        # Add to durable
        self.durable.add(entry)
        return True

    def auto_promote_all(self) -> Dict[str, int]:
        """
        Automatically promote verified entries to durable.

        Returns:
            Dict with counts of promoted entries.
        """
        counts = {"scratch": len(self.scratch.entries), "verified": len(self.verified.entries), "durable": len(self.durable.entries)}

        ready = self.verified.get_ready_for_durable(self.commitment_threshold)
        for entry in ready:
            self.promote_to_durable(entry)

        return {**counts, "promoted": len(self.durable.entries) - counts["durable"]}

    def get_all_durable(self, limit: int = 100) -> List[MemoryEntry]:
        """Get all durable entries."""
        return self.durable.get(limit)

    def clear_scratch(self) -> None:
        """Clear scratch store."""
        self.scratch.clear()