"""Structured short-term memory schema."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _dt_to_iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _dt_from_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass
class Commitment:
    id: str
    statement: str
    confidence: float
    source: str
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    tags: List[str] = field(default_factory=list)
    scope: str = "session"
    expires_at: Optional[datetime] = None
    hypothesis: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "confidence": float(self.confidence),
            "source": self.source,
            "created_at": _dt_to_iso(self.created_at),
            "updated_at": _dt_to_iso(self.updated_at),
            "tags": list(self.tags),
            "scope": self.scope,
            "expires_at": _dt_to_iso(self.expires_at),
            "hypothesis": bool(self.hypothesis),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Commitment":
        return cls(
            id=str(payload.get("id") or ""),
            statement=str(payload.get("statement") or ""),
            confidence=float(payload.get("confidence") or 0.0),
            source=str(payload.get("source") or "unknown"),
            created_at=_dt_from_iso(payload.get("created_at")) or _utc_now(),
            updated_at=_dt_from_iso(payload.get("updated_at")) or _utc_now(),
            tags=[str(tag) for tag in payload.get("tags") or []],
            scope=str(payload.get("scope") or "session"),
            expires_at=_dt_from_iso(payload.get("expires_at")),
            hypothesis=bool(payload.get("hypothesis", False)),
        )


@dataclass
class EntityRecord:
    name: str
    type: str = "unknown"
    attributes: Dict[str, Any] = field(default_factory=dict)
    relations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "attributes": dict(self.attributes),
            "relations": list(self.relations),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EntityRecord":
        return cls(
            name=str(payload.get("name") or ""),
            type=str(payload.get("type") or "unknown"),
            attributes=dict(payload.get("attributes") or {}),
            relations=list(payload.get("relations") or []),
        )


@dataclass
class Note:
    text: str
    source: str = "system"
    created_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "created_at": _dt_to_iso(self.created_at),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Note":
        return cls(
            text=str(payload.get("text") or ""),
            source=str(payload.get("source") or "system"),
            created_at=_dt_from_iso(payload.get("created_at")) or _utc_now(),
        )


@dataclass
class PlanState:
    steps: List[str] = field(default_factory=list)
    current_step: int = 0
    status: str = "idle"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": list(self.steps),
            "current_step": int(self.current_step),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlanState":
        return cls(
            steps=[str(step) for step in payload.get("steps") or []],
            current_step=int(payload.get("current_step") or 0),
            status=str(payload.get("status") or "idle"),
        )


@dataclass
class STMState:
    commitments: List[Commitment] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    entities: Dict[str, EntityRecord] = field(default_factory=dict)
    intermediate: List[Note] = field(default_factory=list)
    plan: PlanState = field(default_factory=PlanState)
    last_updated: datetime = field(default_factory=_utc_now)
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "commitments": [commitment.to_dict() for commitment in self.commitments],
            "goals": list(self.goals),
            "constraints": list(self.constraints),
            "entities": {key: value.to_dict() for key, value in self.entities.items()},
            "intermediate": [note.to_dict() for note in self.intermediate],
            "plan": self.plan.to_dict(),
            "last_updated": _dt_to_iso(self.last_updated),
            "version": int(self.version),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "STMState":
        commitments = [
            Commitment.from_dict(entry)
            for entry in payload.get("commitments") or []
            if isinstance(entry, dict)
        ]
        entities: Dict[str, EntityRecord] = {}
        for key, value in (payload.get("entities") or {}).items():
            if not isinstance(value, dict):
                continue
            entities[str(key)] = EntityRecord.from_dict(value)
        return cls(
            commitments=commitments,
            goals=[str(goal) for goal in payload.get("goals") or []],
            constraints=[str(item) for item in payload.get("constraints") or []],
            entities=entities,
            intermediate=[
                Note.from_dict(entry)
                for entry in payload.get("intermediate") or []
                if isinstance(entry, dict)
            ],
            plan=PlanState.from_dict(payload.get("plan") or {}),
            last_updated=_dt_from_iso(payload.get("last_updated")) or _utc_now(),
            version=int(payload.get("version") or 1),
        )
