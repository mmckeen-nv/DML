"""Provider-mode HTTP surface for Daystrom DML."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .api_contracts import DaystromScope
from .api_contracts import ContractError
from .cognition.audit import sanitize_audit_payload
from .cognition.controller import CognitionController
from .cognition.evaluation import DCNEvalHarness, smoke_eval_cases
from .cognition.learning import ProceduralLearningPolicy
from .cognition.policy import DeterministicCognitionPolicy
from .cognition.schema import CognitionConstraints, CognitionEvent, CognitionFeedback
from .dml_adapter import DMLAdapter
from .frontier_pipeline import FrontierCompressionPipeline, FrontierPipelineConfig


WEB_DIR = Path(__file__).with_name("provider_web")


class RememberRequest(BaseModel):
    text: str
    tenant_id: str = "openclaw"
    client_id: Optional[str] = None
    session_id: Optional[str] = None
    instance_id: Optional[str] = None
    kind: str = "note"
    meta: dict[str, Any] = Field(default_factory=dict)


class RecallRequest(BaseModel):
    query: str
    tenant_id: str = "openclaw"
    client_id: Optional[str] = None
    session_id: Optional[str] = None
    instance_id: Optional[str] = None
    top_k: int = 6


class ResumeRequest(BaseModel):
    query: str = "active continuity checkpoint compaction handoff resume next action"
    tenant_id: str = "openclaw"
    client_id: Optional[str] = None
    session_id: Optional[str] = None
    instance_id: Optional[str] = None
    top_k: int = 12


class FrontierPrepareRequest(BaseModel):
    prompt: str
    tenant_id: str = "openclaw"
    client_id: Optional[str] = None
    session_id: Optional[str] = None
    instance_id: Optional[str] = None
    top_k: int = 8
    include_local_draft: bool = True
    local_max_tokens: int = 256
    frontier_max_tokens: int = 512
    direct_input_tokens_estimate: Optional[int] = None


class DCNRequest(BaseModel):
    event: dict[str, Any] | str | None = None
    content: Optional[str] = None
    type: str = "user_message"
    metadata: dict[str, Any] = Field(default_factory=dict)
    scope: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)


class DCNFeedbackRequest(BaseModel):
    decision_id: str = ""
    outcome: str = "accepted"
    signals: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    latency_ms: float = 0.0
    plan_fidelity: Optional[float] = None


class DCNModePromotionRequest(BaseModel):
    target_mode: str = "active_learn"
    checkpoint_id: str = ""
    hygiene_evidence: dict[str, Any] = Field(default_factory=dict)
    operator: str = "operator"
    reason: str = ""


def _build_adapter(config_path: str | None, storage_dir: str | None) -> DMLAdapter:
    overrides: dict[str, Any] = {}
    if storage_dir:
        overrides["storage_dir"] = storage_dir
    return DMLAdapter(
        config_path=config_path,
        config_overrides=overrides or None,
        start_aging_loop=False,
    )


def _item_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or "")


def _embedding_from_text(text: str, dim: int = 384) -> list[float]:
    """Return a deterministic lightweight embedding for Ollama compatibility."""
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    values: list[float] = []
    for idx in range(dim):
        byte = digest[idx % len(digest)]
        values.append(round((byte / 127.5) - 1.0, 6))
    return values


def _message_text(messages: list[dict[str, Any]]) -> str:
    parts = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(str(part.get("text") or part) for part in content)
        parts.append(f"{role}: {content or ''}".strip())
    return "\n".join(parts)


def _jsonl(payload: dict[str, Any]):
    import json

    return json.dumps(payload, sort_keys=True, default=str) + "\n"


def _stable_digest(payload: Any) -> str:
    data = json.dumps(sanitize_audit_payload(payload), sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def _run_dcn_eval_smoke_report() -> Any:
    return DCNEvalHarness(clock=lambda: 0.0).run_suite(
        smoke_eval_cases(),
        suite_id="provider-dcn-eval-smoke",
    )


def _promotion_failure(*, target_mode: str, reason: str, checkpoint_id: str = "", extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return sanitize_audit_payload({
        "status": "failed",
        "promoted": False,
        "target_mode": target_mode,
        "checkpoint_id": checkpoint_id,
        "reason": reason,
        **(extra or {}),
    })


def _dcn_event(payload: DCNRequest) -> CognitionEvent:
    if isinstance(payload.event, dict):
        return CognitionEvent.from_dict(payload.event)
    if isinstance(payload.event, str):
        return CognitionEvent(content=payload.event, type=payload.type, metadata=dict(payload.metadata or {}))
    return CognitionEvent(content=payload.content or "", type=payload.type, metadata=dict(payload.metadata or {}))


def _dcn_scope(payload: DCNRequest) -> DaystromScope:
    return DaystromScope.from_dict(payload.scope)


def _dcn_constraints(payload: DCNRequest) -> CognitionConstraints:
    return CognitionConstraints.from_dict(payload.constraints)


def create_app(
    *,
    adapter_factory: Callable[[], DMLAdapter] | None = None,
    config_path: str | None = None,
    storage_dir: str | None = None,
) -> FastAPI:
    adapter = adapter_factory() if adapter_factory else _build_adapter(config_path, storage_dir)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            adapter.close()

    app = FastAPI(title="Daystrom DML Provider", lifespan=lifespan)
    app.state.adapter = adapter
    app.state.dcn_learning = ProceduralLearningPolicy()
    app.state.dcn_controller = CognitionController(
        adapter=app.state.adapter,
        policy=DeterministicCognitionPolicy(learning=app.state.dcn_learning),
    )
    app.state.started_at = time.time()
    app.state.dcn_runtime_mode = "deterministic_v0"
    app.state.dcn_promotion_audit = []

    if WEB_DIR.exists():
        app.mount("/assets", StaticFiles(directory=WEB_DIR), name="provider-assets")

    @app.get("/")
    def index(request: Request):
        accept = request.headers.get("accept", "")
        if "text/html" not in accept:
            return PlainTextResponse("Ollama is running")
        index_path = WEB_DIR / "index.html"
        if index_path.exists():
            return HTMLResponse(index_path.read_text(encoding="utf-8"))
        return HTMLResponse("<!doctype html><title>DML Provider</title><h1>DML Provider</h1>")

    @app.get("/api/version")
    def ollama_version() -> dict[str, str]:
        return {"version": "dml-ollama-compatible-v0.1"}

    @app.get("/api/ps")
    def ollama_ps() -> dict[str, list]:
        return {"models": []}

    @app.post("/api/pull")
    def ollama_pull(payload: dict[str, Any]) -> dict[str, Any]:
        return {"status": "success", "model": payload.get("model") or "daystrom-dml:memory"}

    @app.post("/api/copy")
    def ollama_copy(payload: dict[str, Any]) -> dict[str, Any]:
        return {"status": "success", "source": payload.get("source"), "destination": payload.get("destination")}

    @app.delete("/api/delete")
    def ollama_delete(payload: dict[str, Any]) -> dict[str, Any]:
        return {"status": "success", "model": payload.get("model")}

    @app.get("/health")
    def health() -> dict[str, Any]:
        adapter = app.state.adapter
        stats = adapter.stats()
        return {
            "status": "ok",
            "provider": "daystrom-dml",
            "uptime_seconds": round(time.time() - app.state.started_at, 2),
            "stats": stats,
        }

    @app.get("/api/stats")
    def stats() -> dict[str, Any]:
        return app.state.adapter.stats()

    @app.get("/api/dcn/policy")
    def dcn_policy() -> dict[str, Any]:
        controller = app.state.dcn_controller
        return {
            "status": "ok",
            "component": "daystrom-cognition-network",
            "policy_version": controller.policy.policy_version,
            "mode": "deterministic_v0",
            "runtime_mode": app.state.dcn_runtime_mode,
            "last_promotion": (app.state.dcn_promotion_audit[-1] if app.state.dcn_promotion_audit else None),
            "capabilities": [
                "observe",
                "plan_context",
                "cognitive_packet",
                "feedback",
                "policy_export",
                "policy_import",
                "policy_checkpoints",
                "policy_checkpoint",
                "policy_rollback",
                "mode_promote",
                "promotion_audit",
                "eval_smoke",
            ],
            "writeback_forbidden_classes": ["raw_transcript", "tool_log", "secret", "prompt_scaffold"],
        }

    @app.get("/api/dcn/audit")
    def dcn_audit(limit: int = 50) -> dict[str, Any]:
        controller = app.state.dcn_controller
        entries = controller.audit_tail(limit)
        return {"status": "ok", "count": len(entries), "entries": entries}

    @app.post("/api/dcn/policy/export")
    def dcn_policy_export() -> dict[str, Any]:
        """Export the explicit procedural-learning overlay snapshot.

        The deterministic v0 policy remains the immutable baseline. This exports
        only allowlisted routing/gating overlay fields and redacted audit digests;
        it does not include raw prompts, memory context, DPM state, or secrets.
        """
        snapshot = app.state.dcn_learning.export_policy()
        return {"status": "ok", "snapshot": snapshot}

    @app.post("/api/dcn/policy/import")
    def dcn_policy_import(payload: dict[str, Any]) -> dict[str, Any]:
        """Import an explicit procedural-learning overlay snapshot.

        Import remains bounded by ProceduralLearningPolicy validation: wrong
        schema/base refs fail, and forbidden identity/preference/safety fields
        are never accepted into the mutable overlay.
        """
        snapshot = payload.get("snapshot") if isinstance(payload.get("snapshot"), dict) else payload
        try:
            result = app.state.dcn_learning.import_policy(snapshot)
        except ContractError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "ok", **result}

    @app.get("/api/dcn/policy/checkpoints")
    def dcn_policy_checkpoints() -> dict[str, Any]:
        checkpoints = app.state.dcn_learning.checkpoints()
        return {"status": "ok", "count": len(checkpoints), "checkpoints": checkpoints}

    @app.post("/api/dcn/policy/checkpoint")
    def dcn_policy_checkpoint(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        label = str((payload or {}).get("label") or "operator")
        checkpoint_id = app.state.dcn_learning.checkpoint(label)
        return {"status": "ok", "checkpoint_id": checkpoint_id, "label": label}

    @app.post("/api/dcn/policy/rollback")
    def dcn_policy_rollback(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        checkpoint_id = str((payload or {}).get("checkpoint_id") or "") or None
        try:
            result = app.state.dcn_learning.rollback(checkpoint_id)
        except ContractError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "ok", **result}

    @app.get("/api/dcn/mode/promotions")
    def dcn_mode_promotions(limit: int = 20) -> dict[str, Any]:
        entries = list(app.state.dcn_promotion_audit)[-max(0, min(limit, 100)):]
        return {"status": "ok", "runtime_mode": app.state.dcn_runtime_mode, "count": len(entries), "entries": entries}

    @app.post("/api/dcn/mode/promote")
    def dcn_mode_promote(payload: DCNModePromotionRequest) -> dict[str, Any]:
        target_mode = str(payload.target_mode or "").strip().lower().replace("-", "_")
        checkpoint_id = str(payload.checkpoint_id or "").strip()
        previous_mode = str(app.state.dcn_runtime_mode)
        if target_mode != "active_learn":
            failure = _promotion_failure(target_mode=target_mode, reason="unsupported_target_mode", checkpoint_id=checkpoint_id)
            raise HTTPException(status_code=400, detail=failure)
        if not checkpoint_id:
            failure = _promotion_failure(target_mode=target_mode, reason="checkpoint_required")
            raise HTTPException(status_code=400, detail=failure)
        if not app.state.dcn_learning.has_checkpoint(checkpoint_id):
            failure = _promotion_failure(target_mode=target_mode, reason="unknown_checkpoint", checkpoint_id=checkpoint_id)
            raise HTTPException(status_code=400, detail=failure)

        eval_report = _run_dcn_eval_smoke_report()
        eval_artifact = eval_report.artifact()
        eval_summary = dict(eval_report.summary)
        eval_evidence = {
            "passed": bool(eval_report.passed),
            "suite_id": eval_report.suite_id,
            "deterministic_hash": eval_report.deterministic_hash,
            "artifact_hash": eval_artifact["artifact_hash"],
            "coverage": eval_artifact["coverage"],
            "readiness": eval_artifact["readiness"],
            "summary": eval_summary,
        }
        if not eval_report.passed or not eval_artifact["readiness"].get("ready"):
            failure = _promotion_failure(target_mode=target_mode, reason="eval_smoke_failed", checkpoint_id=checkpoint_id, extra={"eval": eval_evidence})
            raise HTTPException(status_code=400, detail=failure)

        hygiene = sanitize_audit_payload(dict(payload.hygiene_evidence or {}))
        if hygiene.get("passed") is not True:
            failure = _promotion_failure(target_mode=target_mode, reason="hygiene_evidence_required", checkpoint_id=checkpoint_id, extra={"eval": eval_evidence, "hygiene": hygiene})
            raise HTTPException(status_code=400, detail=failure)

        audit_record = sanitize_audit_payload({
            "promotion_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "event": "dcn.mode_promotion",
            "promoted": True,
            "previous_mode": previous_mode,
            "target_mode": target_mode,
            "checkpoint_id": checkpoint_id,
            "rollback_command": f"dml dcn policy rollback --checkpoint-id {checkpoint_id}",
            "policy_digest": app.state.dcn_learning.policy_digest(),
            "eval": eval_evidence,
            "hygiene": hygiene,
            "operator": payload.operator or "operator",
            "reason_digest": _stable_digest(payload.reason or ""),
        })
        app.state.dcn_runtime_mode = target_mode
        app.state.dcn_promotion_audit.append(audit_record)
        app.state.dcn_promotion_audit = app.state.dcn_promotion_audit[-100:]
        return {"status": "ok", "promoted": True, "runtime_mode": app.state.dcn_runtime_mode, "audit": audit_record}

    @app.post("/api/dcn/observe")
    def dcn_observe(payload: DCNRequest) -> dict[str, Any]:
        plan = app.state.dcn_controller.observe(
            _dcn_event(payload),
            scope=_dcn_scope(payload),
            constraints=_dcn_constraints(payload),
        )
        return {"status": "ok", "plan": plan.to_dict()}

    @app.post("/api/dcn/plan-context")
    def dcn_plan_context(payload: DCNRequest) -> dict[str, Any]:
        plan = app.state.dcn_controller.plan_context(
            _dcn_event(payload),
            scope=_dcn_scope(payload),
            constraints=_dcn_constraints(payload),
        )
        return {"status": "ok", "plan": plan.to_dict()}

    @app.post("/api/dcn/cognitive-packet")
    def dcn_cognitive_packet(payload: DCNRequest) -> dict[str, Any]:
        packet = app.state.dcn_controller.cognitive_packet(
            _dcn_event(payload),
            scope=_dcn_scope(payload),
            constraints=_dcn_constraints(payload),
        )
        return {"status": "ok", "packet": packet.to_dict()}

    @app.post("/api/dcn/feedback")
    def dcn_feedback(payload: DCNFeedbackRequest) -> dict[str, Any]:
        payload_dict = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        feedback = CognitionFeedback.from_dict(payload_dict)
        result = app.state.dcn_controller.feedback(feedback)
        return {"status": "ok", **result}

    @app.get("/api/dcn/eval/smoke")
    def dcn_eval_smoke() -> dict[str, Any]:
        """Run the built-in offline DCN eval smoke suite.

        This intentionally exposes only deterministic fixture metrics and hashes.
        It does not touch the provider's live adapter/store, DPM state, DIP, or
        frontier inference, so it is safe as a readiness/promotion probe.
        """
        report = _run_dcn_eval_smoke_report()
        artifact = report.artifact()
        ready = bool(artifact["readiness"].get("ready"))
        return {
            "status": "ok" if report.passed and ready else "failed",
            "component": "daystrom-cognition-network",
            "mode": "offline_fixture_smoke",
            "report": report.to_dict(),
            "artifact": artifact,
        }

    @app.get("/api/tags")
    def ollama_tags() -> dict[str, Any]:
        stats_payload = app.state.adapter.stats()
        return {
            "models": [
                {
                    "name": "daystrom-dml:memory",
                    "model": "daystrom-dml:memory",
                    "modified_at": "",
                    "size": int(stats_payload.get("count") or 0),
                    "details": {"family": "memory-provider", "parameter_size": "local", "quantization_level": "n/a"},
                }
            ]
        }

    @app.post("/api/show")
    def ollama_show(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "license": "MIT",
            "modelfile": "FROM daystrom-dml:memory",
            "parameters": "memory_provider true",
            "template": "{{ .Prompt }}",
            "details": {"family": "memory-provider", "format": "dml", "parameter_size": "local"},
        }

    @app.post("/api/remember")
    def remember(payload: RememberRequest) -> dict[str, Any]:
        meta = {
            "tenant_id": payload.tenant_id,
            "client_id": payload.client_id,
            "session_id": payload.session_id,
            "instance_id": payload.instance_id,
            "kind": payload.kind,
            **payload.meta,
        }
        app.state.adapter.ingest(payload.text, meta=meta)
        return {"status": "ok", "action": "remember", "tenant_id": payload.tenant_id, "session_id": payload.session_id}

    @app.post("/api/recall")
    def recall(payload: RecallRequest) -> dict[str, Any]:
        return app.state.adapter.retrieve_context(
            payload.query,
            tenant_id=payload.tenant_id,
            client_id=payload.client_id,
            session_id=payload.session_id,
            instance_id=payload.instance_id,
            top_k=payload.top_k,
        )

    @app.post("/api/resume")
    def resume(payload: ResumeRequest) -> dict[str, Any]:
        report = app.state.adapter.retrieve_context(
            payload.query,
            tenant_id=payload.tenant_id,
            client_id=payload.client_id,
            session_id=payload.session_id,
            instance_id=payload.instance_id,
            top_k=payload.top_k,
        )
        report["action"] = "resume"
        return report

    @app.post("/api/frontier/prepare")
    def frontier_prepare(payload: FrontierPrepareRequest) -> dict[str, Any]:
        adapter = app.state.adapter

        def _draft(prompt: str, max_tokens: int) -> str:
            runner = getattr(adapter, "runner", None)
            if runner is None or getattr(runner, "is_dummy", False):
                return ""
            return runner.generate(prompt, max_new_tokens=max_tokens)

        pipeline = FrontierCompressionPipeline(
            adapter,
            config=FrontierPipelineConfig(
                top_k=payload.top_k,
                local_max_tokens=payload.local_max_tokens,
                frontier_max_tokens=payload.frontier_max_tokens,
                include_local_draft=payload.include_local_draft,
            ),
            draft_generator=_draft,
        )
        return pipeline.prepare(
            payload.prompt,
            tenant_id=payload.tenant_id,
            client_id=payload.client_id,
            session_id=payload.session_id,
            instance_id=payload.instance_id,
            top_k=payload.top_k,
            local_max_tokens=payload.local_max_tokens,
            frontier_max_tokens=payload.frontier_max_tokens,
            include_local_draft=payload.include_local_draft,
            direct_input_tokens_estimate=payload.direct_input_tokens_estimate,
        )

    @app.post("/api/generate")
    def ollama_generate(payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt") or "")
        tenant_id = str(payload.get("tenant_id") or "openclaw")
        session_id = payload.get("session_id")
        report = app.state.adapter.retrieve_context(prompt, tenant_id=tenant_id, session_id=session_id, top_k=int(payload.get("top_k") or 6))
        result = {
            "model": payload.get("model") or "daystrom-dml:memory",
            "created_at": "",
            "response": report.get("raw_context") or "",
            "done": True,
            "context": [],
            "total_duration": int(float(report.get("latency_ms") or 0) * 1_000_000),
            "load_duration": 0,
            "prompt_eval_count": len(prompt.split()),
            "eval_count": int(report.get("context_tokens") or 0),
        }
        if payload.get("stream") is True:
            return StreamingResponse(iter([_jsonl(result)]), media_type="application/x-ndjson")
        return result

    @app.post("/api/chat")
    def ollama_chat(payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages") or []
        if not isinstance(messages, list):
            messages = []
        prompt = _message_text(messages)
        tenant_id = str(payload.get("tenant_id") or "openclaw")
        session_id = payload.get("session_id")
        report = app.state.adapter.retrieve_context(prompt, tenant_id=tenant_id, session_id=session_id, top_k=int(payload.get("top_k") or 6))
        result = {
            "model": payload.get("model") or "daystrom-dml:memory",
            "created_at": "",
            "message": {"role": "assistant", "content": report.get("raw_context") or ""},
            "done": True,
            "total_duration": int(float(report.get("latency_ms") or 0) * 1_000_000),
            "load_duration": 0,
            "prompt_eval_count": len(prompt.split()),
            "eval_count": int(report.get("context_tokens") or 0),
        }
        if payload.get("stream") is True:
            return StreamingResponse(iter([_jsonl(result)]), media_type="application/x-ndjson")
        return result

    @app.post("/api/embeddings")
    def ollama_embeddings(payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt") or payload.get("input") or "")
        return {"embedding": _embedding_from_text(prompt), "model": payload.get("model") or "daystrom-dml:memory"}

    @app.post("/api/embed")
    def ollama_embed(payload: dict[str, Any]) -> dict[str, Any]:
        raw_input = payload.get("input")
        if isinstance(raw_input, list):
            embeddings = [_embedding_from_text(str(item)) for item in raw_input]
        else:
            embeddings = [_embedding_from_text(str(raw_input or payload.get("prompt") or ""))]
        return {
            "model": payload.get("model") or "daystrom-dml:memory",
            "embeddings": embeddings,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": len(embeddings),
        }

    @app.get("/api/search")
    def search(q: str, tenant_id: str = "openclaw", session_id: str | None = None, top_k: int = 6) -> dict[str, Any]:
        report = app.state.adapter.retrieve_context(q, tenant_id=tenant_id, session_id=session_id, top_k=top_k)
        results = [
            {
                "id": _item_id(item),
                "title": (item.get("meta") or {}).get("source") or f"memory:{_item_id(item)}",
                "snippet": item.get("summary") or item.get("text") or "",
                "score": item.get("salience"),
                "metadata": item.get("meta") or {},
            }
            for item in report.get("items", [])
        ]
        return {"status": "ok", "query": q, "results": results}

    @app.get("/api/fetch/{memory_id}")
    def fetch(memory_id: str) -> dict[str, Any]:
        for item in app.state.adapter.store.items():
            if str(item.id) == memory_id:
                return {
                    "status": "ok",
                    "id": str(item.id),
                    "text": item.text,
                    "summary": item.cached_summary(max_len=400),
                    "metadata": item.meta or {},
                    "timestamp": float(item.timestamp),
                }
        raise HTTPException(status_code=404, detail="memory not found")

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Daystrom DML provider server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config-path")
    parser.add_argument("--storage-dir")
    args = parser.parse_args(argv)

    import uvicorn

    app = create_app(config_path=args.config_path, storage_dir=args.storage_dir)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
