"""Provider-mode HTTP surface for Daystrom DML."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .dml_adapter import DMLAdapter


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


def create_app(
    *,
    adapter_factory: Callable[[], DMLAdapter] | None = None,
    config_path: str | None = None,
    storage_dir: str | None = None,
) -> FastAPI:
    app = FastAPI(title="Daystrom DML Provider")
    app.state.adapter = adapter_factory() if adapter_factory else _build_adapter(config_path, storage_dir)
    app.state.started_at = time.time()

    if WEB_DIR.exists():
        app.mount("/assets", StaticFiles(directory=WEB_DIR), name="provider-assets")

    @app.on_event("shutdown")
    def _close_adapter() -> None:
        app.state.adapter.close()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        index_path = WEB_DIR / "index.html"
        if index_path.exists():
            return index_path.read_text(encoding="utf-8")
        return "<!doctype html><title>DML Provider</title><h1>DML Provider</h1>"

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

    @app.post("/api/generate")
    def ollama_generate(payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt") or "")
        tenant_id = str(payload.get("tenant_id") or "openclaw")
        session_id = payload.get("session_id")
        report = app.state.adapter.retrieve_context(prompt, tenant_id=tenant_id, session_id=session_id, top_k=int(payload.get("top_k") or 6))
        return {
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
