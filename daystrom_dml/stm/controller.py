"""STM controller for retrieval, updates, and contradiction handling."""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from ..router import decide_mode
from .policy import LTMWritePolicy, MemoryWrite, commitment_to_write
from .schema import Commitment, EntityRecord, Note, PlanState, STMState

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class RetrievalPlan:
    use_stm: bool
    use_ltm: bool
    mode: str
    top_k: int


@dataclass
class Contradiction:
    existing: Commitment
    incoming: Commitment
    reason: str


@dataclass
class ExtractionResult:
    commitments: List[Commitment] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    entities: List[EntityRecord] = field(default_factory=list)
    plan: Optional[PlanState] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconcileResult:
    response: str
    asked_user: bool


class STMController:
    """Lightweight controller for STM and LTM interactions."""

    def __init__(
        self,
        *,
        policy: Optional[LTMWritePolicy] = None,
        stm_max_commitments: int = 8,
        stm_max_entities: int = 8,
        top_k: int = 6,
        extract_max_tokens: int = 256,
    ) -> None:
        self.policy = policy or LTMWritePolicy()
        self.stm_max_commitments = max(1, stm_max_commitments)
        self.stm_max_entities = max(1, stm_max_entities)
        self.top_k = max(1, top_k)
        self.extract_max_tokens = max(64, extract_max_tokens)

    def decide_retrieval(self, stm: STMState, user_msg: str) -> RetrievalPlan:
        mode = decide_mode(user_msg)
        use_stm = bool(stm.commitments or stm.goals or stm.constraints or stm.entities)
        return RetrievalPlan(
            use_stm=use_stm,
            use_ltm=True,
            mode=mode,
            top_k=self.top_k,
        )

    def extract_structured_updates(
        self,
        *,
        user_msg: str,
        model_msg: str,
        generator: Any,
    ) -> ExtractionResult:
        prompt = self._build_extraction_prompt(user_msg, model_msg)
        try:
            response = generator(
                prompt,
                max_new_tokens=self.extract_max_tokens,
                temperature=0.0,
                top_p=1.0,
                stop=["\n\n"],
            )
        except TypeError:
            response = generator(prompt, max_new_tokens=self.extract_max_tokens)
        payload = self._parse_json_block(response)
        if not payload:
            return ExtractionResult()
        return self._coerce_extraction(payload)

    def update_stm_from_turn(
        self,
        stm: STMState,
        *,
        user_msg: str,
        model_msg: str,
        extraction: ExtractionResult,
    ) -> List[Commitment]:
        new_commitments = self._merge_commitments(stm, extraction.commitments)
        stm.goals = self._merge_list(stm.goals, extraction.goals)
        stm.constraints = self._merge_list(stm.constraints, extraction.constraints)
        if extraction.entities:
            self._merge_entities(stm, extraction.entities)
        if extraction.plan is not None:
            stm.plan = extraction.plan
        summary_note = Note(text=f"User: {user_msg}\nAssistant: {model_msg}")
        stm.intermediate.append(summary_note)
        stm.last_updated = datetime.now(timezone.utc)
        stm.version += 1
        stm.commitments = self._prune_commitments(stm.commitments)
        stm.intermediate = stm.intermediate[-5:]
        return new_commitments

    def decide_ltm_writes(
        self,
        commitments: Iterable[Commitment],
    ) -> List[MemoryWrite]:
        writes = [commitment_to_write(commitment) for commitment in commitments]
        return self.policy.filter_writes(writes)

    def detect_contradictions(
        self, stm: STMState, new_commitments: Iterable[Commitment]
    ) -> List[Contradiction]:
        contradictions: List[Contradiction] = []
        for incoming in new_commitments:
            for existing in stm.commitments:
                if incoming.id == existing.id:
                    continue
                if _is_contradiction(existing.statement, incoming.statement):
                    contradictions.append(
                        Contradiction(
                            existing=existing,
                            incoming=incoming,
                            reason="conflicting commitment",
                        )
                    )
        return contradictions

    def slow_path_reconcile(
        self, *, contradictions: List[Contradiction], user_msg: str
    ) -> ReconcileResult:
        conflict_lines = [
            f"- Existing: {c.existing.statement}\n  Incoming: {c.incoming.statement}"
            for c in contradictions
        ]
        response = (
            "I found a potential contradiction with existing commitments. "
            "Please confirm which statement is correct before I update memory:\n"
            + "\n".join(conflict_lines)
        )
        return ReconcileResult(response=response, asked_user=True)

    def build_stm_summary(self, stm: STMState) -> str:
        lines: List[str] = []
        if stm.commitments:
            lines.append("Commitments:")
            for commitment in self._prune_commitments(stm.commitments):
                lines.append(
                    f"- ({commitment.confidence:.2f}) {commitment.statement} [{commitment.source}]"
                )
        if stm.goals:
            lines.append("Goals:")
            lines.extend([f"- {goal}" for goal in stm.goals[:5]])
        if stm.constraints:
            lines.append("Constraints:")
            lines.extend([f"- {constraint}" for constraint in stm.constraints[:5]])
        if stm.plan.steps:
            lines.append("Plan:")
            current_step = max(0, min(stm.plan.current_step, len(stm.plan.steps) - 1))
            lines.append(
                f"- Step {current_step + 1}/{len(stm.plan.steps)}: "
                f"{stm.plan.steps[current_step]} ({stm.plan.status})"
            )
        if stm.entities:
            lines.append("Entities:")
            for entity in list(stm.entities.values())[: self.stm_max_entities]:
                lines.append(f"- {entity.name} ({entity.type})")
        return "\n".join(lines).strip()

    def _merge_commitments(
        self, stm: STMState, incoming: Iterable[Commitment]
    ) -> List[Commitment]:
        new_commitments: List[Commitment] = []
        existing_map = {commitment.id: commitment for commitment in stm.commitments}
        for commitment in incoming:
            if not commitment.id:
                commitment.id = str(uuid.uuid4())
            existing = existing_map.get(commitment.id)
            if existing:
                existing.statement = commitment.statement
                existing.confidence = commitment.confidence
                existing.source = commitment.source
                existing.updated_at = datetime.now(timezone.utc)
                existing.tags = list({*existing.tags, *commitment.tags})
                existing.scope = commitment.scope
                existing.expires_at = commitment.expires_at
                existing.hypothesis = commitment.hypothesis
            else:
                stm.commitments.append(commitment)
                new_commitments.append(commitment)
        return new_commitments

    def _merge_entities(self, stm: STMState, entities: Iterable[EntityRecord]) -> None:
        for entity in entities:
            if not entity.name:
                continue
            existing = stm.entities.get(entity.name)
            if existing:
                existing.attributes.update(entity.attributes)
                existing.relations.extend(entity.relations)
            else:
                stm.entities[entity.name] = entity

    def _merge_list(self, existing: List[str], incoming: Iterable[str]) -> List[str]:
        merged = list(existing)
        for entry in incoming:
            cleaned = entry.strip()
            if not cleaned:
                continue
            if cleaned not in merged:
                merged.append(cleaned)
        return merged

    def _prune_commitments(self, commitments: List[Commitment]) -> List[Commitment]:
        sorted_commitments = sorted(
            commitments,
            key=lambda item: (item.confidence, item.updated_at),
            reverse=True,
        )
        return sorted_commitments[: self.stm_max_commitments]

    def _build_extraction_prompt(self, user_msg: str, model_msg: str) -> str:
        return (
            "Extract structured STM updates as strict JSON. "
            "Return only JSON with keys: commitments, goals, constraints, entities, plan. "
            "Commitments are objects with: statement, confidence (0-1), source (user/tool/model), "
            "tags (array), scope (global/session), hypothesis (bool). "
            "Entities are objects with: name, type, attributes, relations. "
            "Plan has: steps (array), current_step (int), status (string).\n\n"
            f"User message: {user_msg}\n\n"
            f"Assistant reply: {model_msg}"
        )

    def _parse_json_block(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}
        match = _JSON_BLOCK_RE.search(text)
        if not match:
            return {}
        block = match.group(0)
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            return {}

    def _coerce_extraction(self, payload: Dict[str, Any]) -> ExtractionResult:
        commitments: List[Commitment] = []
        for entry in payload.get("commitments") or []:
            if not isinstance(entry, dict):
                continue
            confidence = float(entry.get("confidence") or 0.0)
            commitment = Commitment(
                id=str(entry.get("id") or ""),
                statement=str(entry.get("statement") or ""),
                confidence=max(0.0, min(1.0, confidence)),
                source=str(entry.get("source") or "model"),
                tags=[str(tag) for tag in entry.get("tags") or []],
                scope=str(entry.get("scope") or "session"),
                hypothesis=bool(entry.get("hypothesis", False)),
            )
            if commitment.statement.strip():
                commitments.append(commitment)
        entities: List[EntityRecord] = []
        for entry in payload.get("entities") or []:
            if not isinstance(entry, dict):
                continue
            entities.append(EntityRecord.from_dict(entry))
        plan_payload = payload.get("plan")
        plan = PlanState.from_dict(plan_payload) if isinstance(plan_payload, dict) else None
        return ExtractionResult(
            commitments=commitments,
            goals=[str(item) for item in payload.get("goals") or []],
            constraints=[str(item) for item in payload.get("constraints") or []],
            entities=entities,
            plan=plan,
            raw=payload,
        )


def _normalize_statement(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return " ".join(cleaned.split())


def _is_contradiction(existing: str, incoming: str) -> bool:
    if not existing or not incoming:
        return False
    existing_norm = _normalize_statement(existing)
    incoming_norm = _normalize_statement(incoming)
    if existing_norm == incoming_norm:
        return False
    neg_tokens = {"not", "no", "never"}
    existing_words = existing_norm.split()
    incoming_words = incoming_norm.split()
    existing_core = " ".join([word for word in existing_words if word not in neg_tokens])
    incoming_core = " ".join([word for word in incoming_words if word not in neg_tokens])
    if existing_core and existing_core == incoming_core:
        existing_has_neg = any(word in neg_tokens for word in existing_words)
        incoming_has_neg = any(word in neg_tokens for word in incoming_words)
        return existing_has_neg != incoming_has_neg
    return False
