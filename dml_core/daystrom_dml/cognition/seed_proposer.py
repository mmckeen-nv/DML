"""Ollama-backed DCN seed-model candidate proposal.

This module keeps the seed model outside the authority boundary: the model may
propose procedural candidates and unsupported policy pressure, but the output is
still fed through ``seed_trial.run_seed_trial`` before it can become any policy
artifact.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from daystrom_dml.cognition.audit import sanitize_audit_payload
from daystrom_dml.cognition.learning import ALLOWED_FIELDS, FORBIDDEN_FIELDS
from daystrom_dml.cognition.seed_trial import DEFAULT_EMBEDDING_MODEL, DEFAULT_SEED_MODEL, run_seed_trial

PROPOSAL_SCHEMA_VERSION = "dcn-seed-proposal-v1"
LOOP_SCHEMA_VERSION = "dcn-seed-loop-artifact-v1"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
_ALLOWED_TASK_HINTS = ["answer", "admin", "code_change", "debugging"]
_POLICY_PRESSURE_ALIASES = {
    "preferred_tool_sequence": "tool_sequence_policy",
    "prefer_tool_sequence": "tool_sequence_policy",
    "preferred_retrieval_mode": "retrieval_mode_policy",
    "prefer_retrieval_mode": "retrieval_mode_policy",
    "retrieval_mode_preference": "retrieval_mode_policy",
    "context_budget_adjustment": "context_budget_policy",
}

GenerateFn = Callable[[str, str, str, float], str]


def propose_seed_updates(
    payload: Dict[str, Any],
    *,
    model: str = DEFAULT_SEED_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    timeout: float = 60.0,
    generate_fn: Optional[GenerateFn] = None,
    clock: Any = None,
) -> Dict[str, Any]:
    """Ask a seed model for sanitized candidate updates and policy pressure.

    The returned proposal is not trusted policy. It is a seed-trial input with
    provenance metadata and must be validated by ``run_seed_trial``.
    """
    now = clock or time.time
    started = float(now())
    sanitized_payload = _sanitize_seed_input(payload)
    prompt = _build_prompt(sanitized_payload, model=model)
    generator = generate_fn or _ollama_generate
    raw_text = generator(ollama_base_url, model, prompt, timeout)
    parsed = _parse_json_object(raw_text)
    proposal = _normalize_proposal(parsed, sanitized_payload, model=model)
    proposal["proposal_metadata"] = {
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "seed_model": model,
        "ollama_base_url_hash": _hash_text(ollama_base_url),
        "prompt_hash": _digest(prompt),
        "raw_response_hash": _digest(raw_text),
        "generated_at": float(now()),
        "run_seconds": max(0.0, float(now()) - started),
    }
    proposal["proposal_hash"] = _digest({k: v for k, v in proposal.items() if k != "proposal_hash"})
    return sanitize_audit_payload(proposal)


def run_seed_loop(
    payload: Dict[str, Any],
    *,
    model: str = DEFAULT_SEED_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    timeout: float = 60.0,
    generate_fn: Optional[GenerateFn] = None,
    clock: Any = None,
) -> Dict[str, Any]:
    """Run proposer then seed-trial validation without promotion."""
    now = clock or time.time
    proposal = propose_seed_updates(
        payload,
        model=model,
        ollama_base_url=ollama_base_url,
        timeout=timeout,
        generate_fn=generate_fn,
        clock=now,
    )
    trial = run_seed_trial(proposal, clock=now)
    artifact = {
        "schema_version": LOOP_SCHEMA_VERSION,
        "generated_at": float(now()),
        "non_promoting": True,
        "proposal_hash": proposal.get("proposal_hash"),
        "trial_artifact_hash": trial.get("artifact_hash"),
        "proposal_summary": {
            "candidate_update_count": len(proposal.get("candidate_updates") or []),
            "unsupported_policy_pressure_count": len(proposal.get("unsupported_policy_pressure") or []),
        },
        "trial_summary": trial.get("summary") or {},
        "proposal": proposal,
        "trial": trial,
    }
    artifact["artifact_hash"] = _digest({k: v for k, v in artifact.items() if k != "artifact_hash"})
    return sanitize_audit_payload(artifact)


def _sanitize_seed_input(payload: Dict[str, Any]) -> Dict[str, Any]:
    clean = sanitize_audit_payload(payload if isinstance(payload, dict) else {})
    keep: Dict[str, Any] = {}
    for key in (
        "seed_model",
        "embedding_model",
        "candidate_updates",
        "unsupported_policy_pressure",
        "policy_pressure",
    ):
        if key in clean:
            keep[key] = clean[key]
    if "feedback" in clean:
        keep["feedback"] = _sanitize_feedback_items(clean.get("feedback"))
    if "feedback_items" in clean:
        keep["feedback_items"] = _sanitize_feedback_items(clean.get("feedback_items"))
    keep.setdefault("seed_model", DEFAULT_SEED_MODEL)
    keep.setdefault("embedding_model", DEFAULT_EMBEDDING_MODEL)
    return keep


def _sanitize_feedback_items(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        raw = [raw]
    items: List[Dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        clean = sanitize_audit_payload(item)
        kept: Dict[str, Any] = {}
        for key in ("decision_id", "outcome", "task_type", "signals", "candidate_updates"):
            if key in clean:
                kept[key] = clean[key]
        if "unsupported_policy_pressure" in clean:
            kept["unsupported_policy_pressure"] = _sanitize_policy_pressure_items(clean.get("unsupported_policy_pressure"))
        if "policy_pressure" in clean:
            kept["policy_pressure"] = _sanitize_policy_pressure_items(clean.get("policy_pressure"))
        items.append(kept)
    return items


def _sanitize_policy_pressure_items(raw: Any) -> Any:
    if isinstance(raw, dict):
        items = [raw]
        single = True
    elif isinstance(raw, list):
        items = raw
        single = False
    else:
        return []
    cleaned_items: List[Dict[str, Any]] = []
    allowed = {
        "task_type",
        "needed_capability",
        "field",
        "type",
        "bounds",
        "evidence",
        "proposed_schema_extension",
        "reason",
        "count",
        *_POLICY_PRESSURE_ALIASES.keys(),
    }
    for item in items:
        if not isinstance(item, dict):
            continue
        clean = sanitize_audit_payload(item)
        kept = {key: clean[key] for key in allowed if key in clean}
        if kept:
            cleaned_items.append(kept)
    if single:
        return cleaned_items[0] if cleaned_items else {}
    return cleaned_items


def _build_prompt(payload: Dict[str, Any], *, model: str) -> str:
    prompt_payload = {
        "task": "Propose DCN procedural learning candidates from sanitized feedback.",
        "strict_output": {
            "candidate_updates": [
                {"task_type": "debugging", "field": "verification_requirement", "value": "strict", "reason": "short sanitized reason"}
            ],
            "unsupported_policy_pressure": [
                {
                    "task_type": "debugging",
                    "needed_capability": "tool_sequence_policy",
                    "proposed_schema_extension": {
                        "field": "preferred_tool_sequence",
                        "type": "list[str]",
                        "bounds": {"max_items": 6, "allowed_tools": ["terminal", "file", "search"]},
                    },
                    "reason": "short sanitized reason",
                }
            ],
        },
        "rules": [
            "Return JSON only. No markdown.",
            "Do not include raw prompts, raw transcripts, tool logs, memory context, credentials, or secrets.",
            "Only use allowed candidate fields exactly as listed.",
            "If a useful change needs a field outside the allowlist, put it in unsupported_policy_pressure instead.",
            "Do not propose identity, personality, values, user preference, permission, safety, autonomy, or secret-handling changes.",
        ],
        "allowed_fields": sorted(ALLOWED_FIELDS),
        "forbidden_fields": sorted(FORBIDDEN_FIELDS),
        "task_hints": _ALLOWED_TASK_HINTS,
        "seed_model": model,
        "sanitized_input": payload,
    }
    return json.dumps(prompt_payload, sort_keys=True, ensure_ascii=False)


def _ollama_generate(base_url: str, model: str, prompt: str, timeout: float) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_ctx": 4096},
        }
    ).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310 - local operator-configured Ollama URL
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Ollama seed proposal failed: {exc}") from exc
    text = payload.get("response") if isinstance(payload, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Ollama seed proposal returned no JSON response")
    return text


def _parse_json_object(text: str) -> Dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Seed proposal response must be a JSON object")
    return payload


def _normalize_proposal(parsed: Dict[str, Any], original: Dict[str, Any], *, model: str) -> Dict[str, Any]:
    candidate_updates, rejected_updates = _candidate_updates(parsed.get("candidate_updates") or parsed.get("updates") or [])
    model_pressures = _policy_pressure(parsed.get("unsupported_policy_pressure") or parsed.get("policy_pressure") or [])
    input_pressures = _policy_pressure(original.get("unsupported_policy_pressure") or original.get("policy_pressure") or [])
    proposal = {
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "seed_model": str(original.get("seed_model") or model),
        "embedding_model": str(original.get("embedding_model") or DEFAULT_EMBEDDING_MODEL),
        "feedback": original.get("feedback") or original.get("feedback_items") or [],
        "candidate_updates": candidate_updates,
        "unsupported_policy_pressure": input_pressures + model_pressures,
        "rejected_model_items": rejected_updates,
        "non_promoting": True,
    }
    return sanitize_audit_payload(proposal)


def _candidate_updates(raw: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if isinstance(raw, dict):
        raw = [raw]
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        clean = sanitize_audit_payload(item)
        field = str(clean.get("field") or "")
        task_type = str(clean.get("task_type") or "answer").strip().lower().replace(" ", "_") or "answer"
        if field not in ALLOWED_FIELDS:
            rejected.append({"field": field, "task_type": task_type, "reason": "field_not_allowed", "item_hash": _digest(clean)})
            continue
        update = {"task_type": task_type, "field": field, "value": clean.get("value")}
        if clean.get("reason"):
            update["reason"] = str(clean.get("reason"))[:240]
        if clean.get("decision_id"):
            update["decision_id"] = str(clean.get("decision_id"))[:120]
        accepted.append(update)
    return accepted, rejected


def _policy_pressure(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        raw = [raw]
    out: List[Dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        clean = sanitize_audit_payload(item)
        needed = str(clean.get("needed_capability") or clean.get("field") or "").strip()
        source_field = ""
        if not needed:
            for field_name, capability in _POLICY_PRESSURE_ALIASES.items():
                if field_name in clean:
                    needed = capability
                    source_field = field_name
                    break
        if not needed:
            continue
        pressure: Dict[str, Any] = {
            "task_type": str(clean.get("task_type") or "*").strip().lower().replace(" ", "_") or "*",
            "needed_capability": needed[:120],
            "reason": str(clean.get("reason") or "")[:300],
        }
        if source_field:
            pressure["field"] = source_field
            pressure["evidence"] = {"source_keys": [source_field]}
        extension = clean.get("proposed_schema_extension")
        if isinstance(extension, dict):
            pressure["proposed_schema_extension"] = extension
        out.append(pressure)
    return out


def _hash_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8", errors="ignore")).hexdigest()[:16]


def _digest(payload: Any) -> str:
    data = json.dumps(sanitize_audit_payload(payload), sort_keys=True, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]
