"""DML1 hot-context adapter over the existing STM implementation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from daystrom_dml.api_contracts import DaystromScope
from daystrom_dml.stm.controller import STMController
from daystrom_dml.stm.schema import Commitment, EntityRecord, STMState


TokenEstimator = Callable[[str], int]


@dataclass
class STMHotContextAdapter:
    """Render semantic STM state into compact DML1-compatible context data."""

    controller: Optional[STMController] = None
    top_commitments: int = 5
    max_entities: int = 5

    def __post_init__(self) -> None:
        if self.controller is None:
            self.controller = STMController(stm_max_commitments=self.top_commitments, stm_max_entities=self.max_entities)

    def snapshot(self, scope: DaystromScope, state: STMState) -> Dict[str, Any]:
        commitments = [_commitment_snapshot(item) for item in self._ordered_commitments(state)]
        entities = [_entity_snapshot(item) for item in self._ordered_entities(state)]
        return {
            "schema": "daystrom.dml1.hot_context.snapshot.v1",
            "scope": _scope_metadata(scope),
            "goals": list(state.goals),
            "constraints": list(state.constraints),
            "current_plan_step": _current_plan_step(state),
            "commitments": commitments,
            "entities": entities,
            "evidence_refs": _evidence_refs(commitments, entities),
            "version": int(state.version),
        }

    def render(
        self,
        scope: DaystromScope,
        state: STMState,
        budget_tokens: int,
        estimate_tokens: TokenEstimator = None,
    ) -> List[Dict[str, Any]]:
        if budget_tokens <= 0:
            return []
        estimator = estimate_tokens or _estimate_tokens
        snapshot = self.snapshot(scope, state)
        candidates = self._segment_candidates(snapshot)
        segments: List[Dict[str, Any]] = []
        used = 0
        for candidate in candidates:
            tokens = max(0, int(estimator(candidate["text"])))
            if tokens > budget_tokens - used:
                continue
            rank = len(segments)
            segments.append(
                {
                    "id": f"stm-hot-{rank}",
                    "rank": rank,
                    "kind": candidate["kind"],
                    "text": candidate["text"],
                    "tokens": tokens,
                    "metadata": {
                        "scope": snapshot["scope"],
                        "source": "stm",
                        "checkpoint_digest": self.checkpoint_digest(scope, state),
                        **candidate["metadata"],
                    },
                }
            )
            used += tokens
            if used >= budget_tokens:
                break
        return segments

    def checkpoint_digest(self, scope: DaystromScope, state: STMState) -> str:
        payload = self.snapshot(scope, state)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def _ordered_commitments(self, state: STMState) -> List[Commitment]:
        return sorted(
            state.commitments,
            key=lambda item: (-float(item.confidence), item.id, item.statement, item.source),
        )[: max(0, int(self.top_commitments))]

    def _ordered_entities(self, state: STMState) -> List[EntityRecord]:
        return [state.entities[key] for key in sorted(state.entities)][: max(0, int(self.max_entities))]

    def _segment_candidates(self, snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for index, goal in enumerate(snapshot["goals"]):
            candidates.append(_candidate("goal", f"Goal: {goal}", {"index": index}))
        for index, constraint in enumerate(snapshot["constraints"]):
            candidates.append(_candidate("constraint", f"Constraint: {constraint}", {"index": index}))
        current_step = snapshot["current_plan_step"]
        if current_step is not None:
            candidates.append(
                _candidate(
                    "plan_step",
                    f"Current plan step {current_step['index'] + 1}/{current_step['total']}: "
                    f"{current_step['text']} ({current_step['status']})",
                    {"index": current_step["index"], "total": current_step["total"]},
                )
            )
        for commitment in snapshot["commitments"]:
            candidates.append(
                _candidate(
                    "commitment",
                    f"Commitment ({commitment['confidence']:.2f}): {commitment['statement']}",
                    {"commitment_id": commitment["id"], "source_ref": commitment["source_ref"]},
                )
            )
        for entity in snapshot["entities"]:
            candidates.append(
                _candidate(
                    "entity",
                    f"Entity: {entity['name']} ({entity['type']})",
                    {"entity": entity["name"], "source_refs": entity["source_refs"]},
                )
            )
        if snapshot["evidence_refs"]:
            candidates.append(
                _candidate(
                    "evidence_refs",
                    "Evidence/source refs: " + ", ".join(snapshot["evidence_refs"]),
                    {"source_refs": snapshot["evidence_refs"]},
                )
            )
        return candidates


def _candidate(kind: str, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {"kind": kind, "text": text, "metadata": metadata}


def _commitment_snapshot(commitment: Commitment) -> Dict[str, Any]:
    return {
        "id": commitment.id,
        "statement": commitment.statement,
        "confidence": float(commitment.confidence),
        "source_ref": commitment.source,
        "tags": sorted(str(tag) for tag in commitment.tags),
        "scope": commitment.scope,
        "hypothesis": bool(commitment.hypothesis),
    }


def _entity_snapshot(entity: EntityRecord) -> Dict[str, Any]:
    return {
        "name": entity.name,
        "type": entity.type,
        "attributes": _stable_json_value(entity.attributes),
        "relations": _stable_json_value(entity.relations),
        "source_refs": _entity_source_refs(entity),
    }


def _current_plan_step(state: STMState) -> Optional[Dict[str, Any]]:
    if not state.plan.steps:
        return None
    index = max(0, min(int(state.plan.current_step), len(state.plan.steps) - 1))
    return {
        "index": index,
        "total": len(state.plan.steps),
        "status": state.plan.status,
        "text": state.plan.steps[index],
    }


def _evidence_refs(commitments: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> List[str]:
    refs = {str(item["source_ref"]) for item in commitments if item.get("source_ref")}
    for entity in entities:
        refs.update(str(ref) for ref in entity.get("source_refs", []) if ref)
    return sorted(refs)


def _entity_source_refs(entity: EntityRecord) -> List[str]:
    refs = set()
    for relation in entity.relations:
        if isinstance(relation, dict):
            for key in ("source_segment_id", "source_ref", "source", "segment_id"):
                value = relation.get(key)
                if value:
                    refs.add(str(value))
    return sorted(refs)


def _scope_metadata(scope: DaystromScope) -> Dict[str, Any]:
    return {
        "tenant_id": scope.tenant_id,
        "client_id": scope.client_id,
        "session_id": scope.session_id,
        "thread_id": scope.thread_id,
        "project_id": scope.project_id,
    }


def _stable_json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))
