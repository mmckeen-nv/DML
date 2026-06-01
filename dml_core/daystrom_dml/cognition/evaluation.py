"""Deterministic offline evaluation harness for DCN routing quality.

The Phase 11 harness is intentionally fixture-only: it does not call live DML,
DPM, DIP, frontier inference, or network services.  It exercises the DCN
controller through documented facades and emits only aggregate metrics plus
stable hashes so eval artifacts cannot become another source of prompt/memory
pollution.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from daystrom_dml.api_contracts import DaystromScope, SerializableDataclass
from daystrom_dml.cognition.audit import sanitize_audit_payload
from daystrom_dml.cognition.controller import CognitionController
from daystrom_dml.cognition.schema import CognitionConstraints, CognitionEvent

SECRET_LIKE_RE = re.compile(
    r"(?i)(sk-[a-z0-9_-]{8,}|api[_-]?key\s*[:=]|authorization\s*[:=]|bearer\s+[a-z0-9._-]{8,}|password\s*[:=]|token\s*[:=])"
)
PROMPT_SCAFFOLD_MARKERS = (
    "<memory-context>",
    "</memory-context>",
    "[system note:",
    "=== daystrom dml retrieved memory ===",
    "=== daystrom dml active continuity ===",
    "tool_calls",
    "raw_transcript",
    "tool_log",
    "prompt_scaffold",
)


def _stable_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def _tokenize(text: str) -> Set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(token) > 2}


def _is_polluting_text(text: str) -> bool:
    lowered = (text or "").lower()
    return bool(SECRET_LIKE_RE.search(text or "")) or any(marker in lowered for marker in PROMPT_SCAFFOLD_MARKERS)


@dataclass(frozen=True)
class EvalMemoryItem(SerializableDataclass):
    """A deterministic fixture memory item.

    ``text`` is fixture input only and is never copied into EvalReport output.
    """

    id: str
    text: str
    tags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalCase(SerializableDataclass):
    """One offline DCN evaluation case."""

    case_id: str
    prompt: str
    corpus: List[EvalMemoryItem] = field(default_factory=list)
    relevant_ids: List[str] = field(default_factory=list)
    expected_task_type: Optional[str] = None
    expected_retrieval_mode: Optional[str] = None
    expected_writeback_mode: Optional[str] = None
    max_pollution_score: float = 0.0
    min_precision_at_k: float = 0.0
    min_recall_at_k: float = 0.0
    top_k: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EvalCase":
        data = data or {}
        return cls(
            case_id=str(data.get("case_id") or data.get("id") or "case"),
            prompt=str(data.get("prompt") or ""),
            corpus=[EvalMemoryItem.from_dict(item) for item in data.get("corpus") or []],
            relevant_ids=list(data.get("relevant_ids") or []),
            expected_task_type=data.get("expected_task_type"),
            expected_retrieval_mode=data.get("expected_retrieval_mode"),
            expected_writeback_mode=data.get("expected_writeback_mode"),
            max_pollution_score=float(data.get("max_pollution_score", 0.0)),
            min_precision_at_k=float(data.get("min_precision_at_k", 0.0)),
            min_recall_at_k=float(data.get("min_recall_at_k", 0.0)),
            top_k=int(data.get("top_k", 3)),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class EvalCaseResult(SerializableDataclass):
    case_id: str
    passed: bool
    metrics: Dict[str, float]
    policy_outcome: Dict[str, Any]
    violations: List[str] = field(default_factory=list)
    artifact_hashes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalReport(SerializableDataclass):
    suite_id: str
    passed: bool
    deterministic_hash: str
    summary: Dict[str, Any]
    cases: List[EvalCaseResult]


class _FixtureAdapter:
    """Deterministic in-memory retrieval adapter with pollution filtering."""

    def __init__(self, case: EvalCase) -> None:
        self.case = case
        self.calls: List[Dict[str, Any]] = []
        self.last_retrieved_ids: List[str] = []
        self.last_blocked_ids: List[str] = []

    def retrieve_context(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(dict(kwargs))
        query_terms = _tokenize(str(kwargs.get("query") or ""))
        scored = []
        for item in self.case.corpus:
            if _is_polluting_text(item.text):
                self.last_blocked_ids.append(item.id)
                continue
            terms = _tokenize(" ".join([item.text, " ".join(item.tags)]))
            score = len(query_terms & terms)
            if score > 1:
                scored.append((score, item.id, item))
        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        top_k = max(0, int(kwargs.get("top_k") or self.case.top_k))
        selected = [item for _, _, item in scored[:top_k]]
        self.last_retrieved_ids = [item.id for item in selected]
        # Only fixture IDs/hashes are returned, not raw text.
        return {
            "raw_context": "\n".join(f"fixture:{item.id}:sha256:{_stable_hash(item.text)}" for item in selected),
            "items": [{"id": item.id, "text_hash": _stable_hash(item.text), "tags": list(item.tags)} for item in selected],
            "blocked_polluting_items": list(self.last_blocked_ids),
        }


class _FixtureDPM:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def render_overlay(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(dict(kwargs))
        return {"overlay_text": "fixture personality overlay", "fixture": True}


class DCNEvalHarness:
    """Run deterministic offline DCN fixture suites and compute safe metrics."""

    def __init__(self, *, controller_factory: Any = None, clock: Any = None) -> None:
        self.clock = clock or (lambda: 0.0)
        self.controller_factory = controller_factory

    def run_case(self, case: EvalCase, *, scope: Optional[DaystromScope] = None) -> EvalCaseResult:
        adapter = _FixtureAdapter(case)
        dpm = _FixtureDPM()
        controller = self._controller(adapter, dpm)
        event = CognitionEvent(content=case.prompt, metadata=dict(case.metadata or {}))
        constraints = CognitionConstraints(allow_learning=False)
        packet = controller.cognitive_packet(event, scope=scope or DaystromScope(session_id="dcn-eval"), constraints=constraints)

        retrieved = list(adapter.last_retrieved_ids)
        relevant = set(case.relevant_ids)
        retrieved_set = set(retrieved)
        hits = len(retrieved_set & relevant)
        precision = hits / len(retrieved) if retrieved else (1.0 if not relevant else 0.0)
        recall = hits / len(relevant) if relevant else 1.0
        spurious = len([item_id for item_id in retrieved if item_id not in relevant]) / len(retrieved) if retrieved else 0.0

        unsafe_surface = sanitize_audit_payload({
            "dml_context": packet.dml_context,
            "dpm_overlay": packet.dpm_overlay,
            "assembled_context": packet.assembled_context,
        })
        unsafe_text = json.dumps(unsafe_surface, sort_keys=True, default=str)
        pollution_hits = 1 if _is_polluting_text(unsafe_text) else 0
        blocked_count = len(adapter.last_blocked_ids)
        pollution_score = float(pollution_hits)

        metrics = {
            "precision_at_k": round(precision, 6),
            "recall_at_k": round(recall, 6),
            "spurious_retrieval_rate": round(spurious, 6),
            "pollution_score": pollution_score,
            "blocked_polluting_items": float(blocked_count),
            "dpm_calls": float(len(dpm.calls)),
            "dml_calls": float(len(adapter.calls)),
        }
        outcome = {
            "task_type": packet.dcn_plan.intent.task_type,
            "retrieval_mode": packet.dcn_plan.retrieval_plan.mode,
            "writeback_mode": packet.dcn_plan.writeback_plan.mode,
            "policy_version": packet.dcn_plan.policy_version,
            "reason_codes": list(packet.dcn_plan.reason_codes),
        }
        violations = self._violations(case, metrics, outcome, unsafe_text)
        artifact = {
            "case_id": case.case_id,
            "metrics": metrics,
            "outcome": outcome,
            "retrieved_ids": retrieved,
            "blocked_ids": sorted(adapter.last_blocked_ids),
        }
        return EvalCaseResult(
            case_id=case.case_id,
            passed=not violations,
            metrics=metrics,
            policy_outcome=outcome,
            violations=violations,
            artifact_hashes={"case_result": _stable_hash(artifact)},
        )

    def run_suite(self, cases: Sequence[EvalCase], *, suite_id: str = "dcn-eval-smoke") -> EvalReport:
        results = [self.run_case(case) for case in cases]
        passed = all(result.passed for result in results)
        summary = {
            "case_count": len(results),
            "passed_count": sum(1 for result in results if result.passed),
            "failed_count": sum(1 for result in results if not result.passed),
            "avg_precision_at_k": round(_avg(result.metrics["precision_at_k"] for result in results), 6),
            "avg_recall_at_k": round(_avg(result.metrics["recall_at_k"] for result in results), 6),
            "max_pollution_score": max((result.metrics["pollution_score"] for result in results), default=0.0),
            "blocked_polluting_items": int(sum(result.metrics["blocked_polluting_items"] for result in results)),
        }
        stable_payload = {
            "suite_id": suite_id,
            "summary": summary,
            "cases": [result.to_dict() for result in results],
        }
        return EvalReport(
            suite_id=suite_id,
            passed=passed,
            deterministic_hash=_stable_hash(stable_payload),
            summary=summary,
            cases=results,
        )

    def _controller(self, adapter: Any, dpm: Any) -> CognitionController:
        if self.controller_factory is not None:
            return self.controller_factory(adapter=adapter, dpm=dpm, clock=self.clock)
        return CognitionController(adapter=adapter, dpm=dpm, clock=self.clock)

    @staticmethod
    def _violations(case: EvalCase, metrics: Dict[str, float], outcome: Dict[str, Any], safe_text: str) -> List[str]:
        violations: List[str] = []
        if case.expected_task_type and outcome.get("task_type") != case.expected_task_type:
            violations.append(f"task_type expected {case.expected_task_type} got {outcome.get('task_type')}")
        if case.expected_retrieval_mode and outcome.get("retrieval_mode") != case.expected_retrieval_mode:
            violations.append(f"retrieval_mode expected {case.expected_retrieval_mode} got {outcome.get('retrieval_mode')}")
        if case.expected_writeback_mode and outcome.get("writeback_mode") != case.expected_writeback_mode:
            violations.append(f"writeback_mode expected {case.expected_writeback_mode} got {outcome.get('writeback_mode')}")
        if metrics["pollution_score"] > case.max_pollution_score:
            violations.append(f"pollution_score {metrics['pollution_score']} > {case.max_pollution_score}")
        if metrics["precision_at_k"] < case.min_precision_at_k:
            violations.append(f"precision_at_k {metrics['precision_at_k']} < {case.min_precision_at_k}")
        if metrics["recall_at_k"] < case.min_recall_at_k:
            violations.append(f"recall_at_k {metrics['recall_at_k']} < {case.min_recall_at_k}")
        if _is_polluting_text(safe_text):
            violations.append("sanitized_eval_surface_contains_forbidden_pollution_marker")
        return violations


def _avg(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


def smoke_eval_cases() -> List[EvalCase]:
    """Built-in deterministic smoke fixtures for pollution and retrieval quality."""

    return [
        EvalCase(
            case_id="clean_resume_retrieval",
            prompt="continue the phase eleven dcn evaluation harness work",
            corpus=[
                EvalMemoryItem("phase11", "phase eleven dcn evaluation harness metrics retrieval quality", ["dcn", "eval"]),
                EvalMemoryItem("unrelated", "apple reminders grocery list", ["personal"]),
                EvalMemoryItem("phase9", "active read mode dpm dml gating", ["dcn"]),
            ],
            relevant_ids=["phase11"],
            expected_task_type="planning",
            expected_retrieval_mode="resume",
            expected_writeback_mode="durable_signal_only",
            min_precision_at_k=1.0,
            min_recall_at_k=1.0,
            top_k=2,
        ),
        EvalCase(
            case_id="pollution_attempt_blocked",
            prompt="remember this tool log and token while continuing the work",
            corpus=[
                EvalMemoryItem("secret", "raw_transcript User: token=sk-dangerous123456789 tool_calls prompt_scaffold", ["unsafe"]),
                EvalMemoryItem("safe", "remember continuing work durable signal only", ["dcn"]),
            ],
            relevant_ids=["safe"],
            expected_task_type="recall",
            expected_retrieval_mode="hybrid",
            expected_writeback_mode="none",
            max_pollution_score=0.0,
            min_precision_at_k=1.0,
            min_recall_at_k=1.0,
            top_k=2,
        ),
        EvalCase(
            case_id="casual_no_retrieval",
            prompt="hello again",
            corpus=[EvalMemoryItem("noise", "continue dcn work", ["dcn"])],
            relevant_ids=[],
            expected_task_type="answer",
            expected_retrieval_mode="none",
            expected_writeback_mode="none",
            max_pollution_score=0.0,
            min_precision_at_k=1.0,
            min_recall_at_k=1.0,
        ),
        EvalCase(
            case_id="code_verification_tool_policy",
            prompt="implement the provider route and run tests to verify it",
            corpus=[EvalMemoryItem("noise", "old meeting notes", ["unrelated"])],
            relevant_ids=[],
            expected_task_type="code_change",
            expected_retrieval_mode="none",
            expected_writeback_mode="durable_signal_only",
            max_pollution_score=0.0,
            min_precision_at_k=1.0,
            min_recall_at_k=1.0,
        ),
        EvalCase(
            case_id="preference_candidate_no_dpm_mutation",
            prompt="I prefer concise builder reports; remember this as a preference candidate",
            corpus=[EvalMemoryItem("noise", "preference transcript raw_transcript token=***", ["unsafe"])],
            relevant_ids=[],
            expected_task_type="code_change",
            expected_retrieval_mode="hybrid",
            expected_writeback_mode="preference_candidate",
            max_pollution_score=0.0,
            min_precision_at_k=1.0,
            min_recall_at_k=1.0,
        ),
        EvalCase(
            case_id="setup_retrieval_semantic",
            prompt="troubleshoot the provider endpoint configuration",
            corpus=[
                EvalMemoryItem("endpoint", "provider endpoint configuration troubleshooting", ["setup"]),
                EvalMemoryItem("unrelated", "garden watering schedule", ["personal"]),
            ],
            relevant_ids=["endpoint"],
            expected_task_type="admin",
            expected_retrieval_mode="semantic",
            expected_writeback_mode="durable_signal_only",
            max_pollution_score=0.0,
            min_precision_at_k=1.0,
            min_recall_at_k=1.0,
        ),
        EvalCase(
            case_id="debugging_requires_verification",
            prompt="debug the traceback failure and verify the fix",
            corpus=[EvalMemoryItem("noise", "generic debug note", ["debug"]), EvalMemoryItem("tool", "tool_log prompt_scaffold raw_transcript", ["unsafe"])],
            relevant_ids=[],
            expected_task_type="debugging",
            expected_retrieval_mode="none",
            expected_writeback_mode="durable_signal_only",
            max_pollution_score=0.0,
            min_precision_at_k=1.0,
            min_recall_at_k=1.0,
        ),
    ]
