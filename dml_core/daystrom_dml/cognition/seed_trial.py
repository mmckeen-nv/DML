"""Offline DCN seed-model trial candidate runner.

The seed trial runner is deliberately non-promoting: it consumes sanitized
feedback/proposal summaries, validates candidate procedural updates through the
same reversible learning policy used at runtime, and emits a metadata artifact
that separates accepted overlay candidates from rejected updates and unsupported
policy-pressure reports.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, Iterable, List, Optional

from daystrom_dml.cognition.audit import sanitize_audit_payload
from daystrom_dml.cognition.learning import (
    ALLOWED_FIELDS,
    BASE_POLICY_REF,
    FORBIDDEN_FIELDS,
    SCHEMA_VERSION,
    ProceduralLearningPolicy,
)
from daystrom_dml.cognition.schema import CognitionFeedback

ARTIFACT_SCHEMA_VERSION = "dcn-seed-trial-artifact-v1"
DEFAULT_SEED_MODEL = "llama3:8b"
DEFAULT_EMBEDDING_MODEL = "ollama:qwen3-embedding:0.6b"

_POLICY_PRESSURE_KEYS = {
    "task_type",
    "needed_capability",
    "field",
    "type",
    "bounds",
    "evidence",
    "proposed_schema_extension",
    "reason",
    "count",
}

_POLICY_PRESSURE_ALIASES = {
    "preferred_tool_sequence": "tool_sequence_policy",
    "prefer_tool_sequence": "tool_sequence_policy",
    "preferred_retrieval_mode": "retrieval_mode_policy",
    "prefer_retrieval_mode": "retrieval_mode_policy",
    "retrieval_mode_preference": "retrieval_mode_policy",
    "context_budget_adjustment": "context_budget_policy",
}


def run_seed_trial(payload: Dict[str, Any], *, clock: Any = None) -> Dict[str, Any]:
    """Validate an offline seed-trial batch without promotion or live mutation.

    Expected input shape is intentionally simple and file-friendly::

        {
          "seed_model": "llama3:8b",
          "embedding_model": "ollama:qwen3-embedding:0.6b",
          "feedback": [
            {"decision_id": "d1", "outcome": "verified", "signals": {...}}
          ],
          "candidate_updates": [
            {"task_type": "debugging", "field": "verification_requirement", "value": "strict"}
          ],
          "unsupported_policy_pressure": [...]
        }

    Feedback entries may also carry local ``candidate_updates`` or
    ``unsupported_policy_pressure`` lists. Raw notes/content are ignored.
    """
    now = clock or time.time
    learning = ProceduralLearningPolicy(clock=now)
    started_at = float(now())
    seed_model = str(payload.get("seed_model") or DEFAULT_SEED_MODEL)
    embedding_model = str(payload.get("embedding_model") or DEFAULT_EMBEDDING_MODEL)

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    pressures: List[Dict[str, Any]] = []
    feedback_summaries: List[Dict[str, Any]] = []

    for update in _iter_candidate_updates(payload):
        task = _task_type(update.get("task_type") or "answer")
        result = learning.apply_update(task, update, source="seed_trial", decision_id=str(update.get("decision_id") or ""))
        _append_result(result, accepted, rejected)

    for item in _feedback_items(payload):
        feedback = _feedback_from_item(item)
        task = _task_type(feedback.signals.get("task_type") or item.get("task_type") or "answer")
        result = learning.learn_from_feedback(feedback, task_type=task, source="seed_trial_feedback")
        accepted.extend(result.get("updates") or [])
        rejected.extend(result.get("rejections") or [])
        feedback_summaries.append(_feedback_summary(feedback, task))
        for update in _iter_candidate_updates(item):
            update = dict(update)
            update.setdefault("decision_id", feedback.decision_id)
            update.setdefault("task_type", task)
            _append_result(
                learning.apply_update(task, update, source="seed_trial_feedback_candidate", decision_id=feedback.decision_id),
                accepted,
                rejected,
            )
        pressures.extend(_policy_pressures(item, default_task=task))

    pressures.extend(_policy_pressures(payload, default_task="*"))
    snapshot = learning.export_policy()
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at": float(now()),
        "seed_model": seed_model,
        "embedding_model": embedding_model,
        "non_promoting": True,
        "policy_schema_version": SCHEMA_VERSION,
        "base_policy_ref": BASE_POLICY_REF,
        "allowed_fields": sorted(ALLOWED_FIELDS),
        "forbidden_field_count": len(FORBIDDEN_FIELDS),
        "summary": {
            "feedback_count": len(feedback_summaries),
            "accepted_update_count": len(accepted),
            "rejected_update_count": len(rejected),
            "unsupported_policy_pressure_count": len(pressures),
            "profile_count": len(snapshot.get("mutable_overlay") or {}),
        },
        "feedback": feedback_summaries,
        "accepted_updates": _compact_results(accepted),
        "rejected_updates": _compact_results(rejected),
        "unsupported_policy_pressure": pressures,
        "candidate_policy_snapshot": snapshot,
    }
    artifact["artifact_hash"] = _digest({k: v for k, v in artifact.items() if k != "artifact_hash"})
    artifact["run_seconds"] = max(0.0, float(now()) - started_at)
    return sanitize_audit_payload(artifact)


def _feedback_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = payload.get("feedback") or payload.get("feedback_items") or []
    if isinstance(items, dict):
        items = [items]
    return [item for item in items if isinstance(item, dict)]


def _feedback_from_item(item: Dict[str, Any]) -> CognitionFeedback:
    signals_raw = item.get("signals")
    signals: Dict[str, Any] = signals_raw if isinstance(signals_raw, dict) else {}
    return CognitionFeedback(
        decision_id=str(item.get("decision_id") or signals.get("decision_id") or ""),
        outcome=str(item.get("outcome") or signals.get("outcome") or ""),
        signals=sanitize_audit_payload(signals),
        notes="",
    )


def _feedback_summary(feedback: CognitionFeedback, task: str) -> Dict[str, Any]:
    signals = sanitize_audit_payload(dict(feedback.signals or {}))
    return {
        "decision_id_hash": _hash_text(feedback.decision_id),
        "outcome": feedback.outcome,
        "task_type": task,
        "signal_keys": sorted(str(key) for key in signals),
    }


def _iter_candidate_updates(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    raw = payload.get("candidate_updates") or payload.get("updates") or []
    if isinstance(raw, dict):
        raw = [raw]
    for update in raw:
        if isinstance(update, dict):
            yield sanitize_audit_payload(update)


def _policy_pressures(payload: Dict[str, Any], *, default_task: str) -> List[Dict[str, Any]]:
    raw = payload.get("unsupported_policy_pressure") or payload.get("policy_pressure") or []
    if isinstance(raw, dict):
        raw = [raw]
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        clean = sanitize_audit_payload({key: item.get(key) for key in _POLICY_PRESSURE_KEYS if key in item})
        if "needed_capability" not in clean and "field" not in clean:
            shorthand = _pressure_from_shorthand(item, default_task=default_task)
            if shorthand:
                out.append(shorthand)
                continue
        if not clean:
            continue
        clean.setdefault("task_type", default_task)
        clean["pressure_hash"] = _digest(clean)
        out.append(clean)
    return out


def _pressure_from_shorthand(item: Dict[str, Any], *, default_task: str) -> Optional[Dict[str, Any]]:
    clean_item = sanitize_audit_payload(item)
    for field_name, capability in _POLICY_PRESSURE_ALIASES.items():
        if field_name not in clean_item:
            continue
        pressure: Dict[str, Any] = {
            "task_type": _task_type(clean_item.get("task_type") or default_task),
            "needed_capability": capability,
            "field": field_name,
            "evidence": {"source_keys": [field_name]},
        }
        if clean_item.get("reason"):
            pressure["reason"] = str(clean_item.get("reason"))[:300]
        pressure["pressure_hash"] = _digest(pressure)
        return pressure
    return None


def _append_result(result: Dict[str, Any], accepted: List[Dict[str, Any]], rejected: List[Dict[str, Any]]) -> None:
    (accepted if result.get("accepted") else rejected).append(result)


def _compact_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact = []
    for result in results:
        audit_raw = result.get("audit")
        audit: Dict[str, Any] = audit_raw if isinstance(audit_raw, dict) else {}
        item = {
            "accepted": bool(result.get("accepted")),
            "task_type": str(result.get("task_type") or ""),
            "field": str(result.get("field") or ""),
            "audit_digest": _digest(audit),
        }
        if not result.get("accepted"):
            item["reason"] = str(result.get("reason") or audit.get("reason") or "rejected")
        compact.append(item)
    return compact


def _task_type(value: Any) -> str:
    text = str(value or "answer").strip().lower().replace(" ", "_")
    return text or "answer"


def _hash_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8", errors="ignore")).hexdigest()[:16]


def _digest(payload: Any) -> str:
    data = json.dumps(sanitize_audit_payload(payload), sort_keys=True, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]
