"""Policies for writing to long-term memory."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List

from .schema import Commitment


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class MemoryWrite:
    text: str
    meta: dict
    confidence: float
    source: str
    hypothesis: bool = False
    expires_at: datetime | None = None


@dataclass
class LTMWritePolicy:
    mode: str = "balanced"
    confidence_threshold: float = 0.75
    hypothesis_expiry_hours: int = 24

    def filter_writes(self, writes: Iterable[MemoryWrite]) -> List[MemoryWrite]:
        if self.mode == "off":
            return []
        results: List[MemoryWrite] = []
        for write in writes:
            if not self._eligible(write):
                continue
            results.append(write)
        return results

    def _eligible(self, write: MemoryWrite) -> bool:
        if self.mode not in {"strict", "balanced", "off"}:
            return False
        if write.source == "model":
            if not write.hypothesis:
                return False
            if write.confidence >= self.confidence_threshold:
                return False
            if write.expires_at is None:
                write.expires_at = _utc_now() + timedelta(hours=self.hypothesis_expiry_hours)
        else:
            if write.confidence < self.confidence_threshold:
                return False
        return True


def commitment_to_write(commitment: Commitment) -> MemoryWrite:
    meta = {
        "source": commitment.source,
        "confidence": commitment.confidence,
        "scope": commitment.scope,
        "kind": "memory",
        "commitment_id": commitment.id,
        "tags": commitment.tags,
        "hypothesis": commitment.hypothesis,
    }
    if commitment.expires_at is not None:
        meta["expires_at"] = commitment.expires_at.isoformat()
    return MemoryWrite(
        text=commitment.statement,
        meta=meta,
        confidence=commitment.confidence,
        source=commitment.source,
        hypothesis=commitment.hypothesis,
        expires_at=commitment.expires_at,
    )
