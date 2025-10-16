"""FastAPI service exposing the Daystrom Memory Lattice."""
from __future__ import annotations

import io
import os
import time
from pathlib import Path
from threading import Lock
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader

from . import utils
from .dml_adapter import DMLAdapter

try:  # requests is an optional dependency during some test scenarios
    import requests
except Exception:  # pragma: no cover - defensive fallback for minimal envs
    requests = None

WEB_DIR = Path(__file__).with_name("web")

app = FastAPI(title="Daystrom Memory Lattice")
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

ADAPTER_LOCK = Lock()
adapter = DMLAdapter(start_aging_loop=False)

CURRENT_NIM: Optional[dict] = None

NIM_DEFAULT_PORT = int(os.environ.get("NIM_PORT", "8000"))
NIM_HEALTH_TIMEOUT = int(os.environ.get("NIM_HEALTH_TIMEOUT", "300"))
NIM_HEALTH_INTERVAL = float(os.environ.get("NIM_HEALTH_INTERVAL", "5"))


class TextPayload(BaseModel):
    text: str
    meta: Optional[dict] = None


class QueryPayload(BaseModel):
    prompt: str


class ComparePayload(BaseModel):
    prompt: str
    top_k: Optional[int] = None
    max_new_tokens: Optional[int] = 512


class NimConfigurePayload(BaseModel):
    model_name: str
    api_key: str


class NimHealthPayload(BaseModel):
    wait_timeout: Optional[int] = None


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    index = WEB_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend bundle missing")
    return HTMLResponse(index.read_text(encoding="utf-8"))


@app.post("/ingest")
def ingest(payload: TextPayload) -> dict:
    adapter.ingest(payload.text, meta=payload.meta)
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    contents = await file.read()
    text = _extract_text(file.filename or "", contents, file.content_type)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Unable to extract any text from upload")
    chunks = utils.chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document produced no ingestible chunks")
    total_tokens = 0
    for chunk in chunks:
        tokens = utils.estimate_tokens(chunk)
        total_tokens += tokens
        adapter.ingest(chunk, meta={"doc_path": file.filename})
    return {
        "status": "ok",
        "chunks": len(chunks),
        "tokens": total_tokens,
    }


@app.post("/reinforce")
def reinforce(payload: TextPayload) -> dict:
    adapter.reinforce("", payload.text, meta=payload.meta)
    return {"status": "ok"}


@app.post("/query")
def query(payload: QueryPayload) -> dict:
    context = adapter.build_preamble(payload.prompt)
    augmented = f"{context}\n\n{payload.prompt}"
    response = adapter.runner.generate(augmented)
    adapter.reinforce(payload.prompt, response)
    return {
        "context": context,
        "response": response,
        "stats": adapter.stats(),
    }


@app.post("/rag/retrieve")
def rag_retrieve(payload: QueryPayload) -> dict:
    rag_top_k = adapter.config.get("top_k", 6)
    rag_report = adapter.rag_store.report(payload.prompt, top_k=rag_top_k)
    dml_report = adapter.retrieval_report(payload.prompt)
    return {
        "prompt": payload.prompt,
        "rag": rag_report,
        "dml": dml_report,
    }


@app.post("/rag/compare")
def rag_compare(payload: ComparePayload) -> dict:
    try:
        result = adapter.compare_responses(
            payload.prompt,
            top_k=payload.top_k,
            max_new_tokens=payload.max_new_tokens or 512,
        )
    except Exception as exc:
        if requests and isinstance(exc, requests.RequestException):
            raise HTTPException(status_code=503, detail="NIM not Running, Start a NIM.")
        raise
    prompt_tokens = utils.estimate_tokens(payload.prompt)
    return {
        **result,
        "prompt_tokens_est": prompt_tokens,
    }


@app.get("/stats")
def stats() -> dict:
    return adapter.stats()


@app.get("/nim/options")
def nim_options() -> dict:
    """Expose the currently configured NVIDIA NIM connection."""

    return {
        "current": CURRENT_NIM,
    }


@app.post("/nim/configure")
def nim_configure(payload: NimConfigurePayload) -> dict:
    """Configure the adapter to use a user-managed NIM endpoint."""

    api_key = payload.api_key.strip()
    model_name = payload.model_name.strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required")
    if not api_key:
        raise HTTPException(status_code=400, detail="NGC API key is required")
    api_base = f"http://localhost:{NIM_DEFAULT_PORT}"
    _apply_nim_configuration(model_name=model_name, api_key=api_key, api_base=api_base)
    summary = {
        "model_name": model_name,
        "api_base": api_base,
    }
    global CURRENT_NIM
    CURRENT_NIM = summary
    return {
        "status": "ok",
        "nim": summary,
        "message": "Configured connection to user-managed NIM.",
    }


@app.post("/nim/health")
def nim_health(payload: NimHealthPayload | None = None) -> dict:
    """Poll the configured NIM endpoint until it responds or the timeout elapses."""

    if CURRENT_NIM is None:
        raise HTTPException(status_code=400, detail="Configure the NIM connection before testing health.")
    if not requests:
        raise HTTPException(status_code=503, detail="Requests library unavailable; cannot perform health check.")
    wait_timeout = NIM_HEALTH_TIMEOUT
    if payload and payload.wait_timeout:
        wait_timeout = int(payload.wait_timeout)
    api_key = os.environ.get("NIM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    attempts: list[str] = []
    deadline = time.time() + max(wait_timeout, 1)
    while time.time() < deadline:
        healthy, reason = _nim_healthcheck(
            CURRENT_NIM["api_base"],
            api_key,
            CURRENT_NIM["model_name"],
        )
        if healthy:
            return {
                "status": "ok",
                "healthy": True,
                "attempts": attempts,
                "message": "NIM is healthy.",
            }
        attempts.append(reason or "Unknown error")
        time.sleep(NIM_HEALTH_INTERVAL)
    raise HTTPException(
        status_code=504,
        detail={
            "message": "Health check timed out. Please verify the NIM is running.",
            "attempts": attempts,
        },
    )


def _apply_nim_configuration(*, model_name: str, api_key: str, api_base: str) -> None:
    """Set environment variables and reload the adapter for the selected model."""

    os.environ["NIM_API_KEY"] = api_key
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["NIM_API_BASE"] = api_base
    os.environ["OPENAI_API_BASE"] = api_base
    os.environ["NIM_PORT"] = str(NIM_DEFAULT_PORT)
    _reload_adapter(config_overrides={"model_name": model_name})


def _reload_adapter(*, config_overrides: Optional[dict] = None) -> None:
    """Recreate the global adapter with the provided overrides."""

    global adapter
    with ADAPTER_LOCK:
        previous = adapter
        try:
            adapter = DMLAdapter(
                start_aging_loop=False,
                config_overrides=config_overrides,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            adapter = previous
            raise HTTPException(status_code=500, detail=f"Failed to initialise adapter: {exc}") from exc
        try:
            previous.close()
        except Exception:
            pass


def _nim_healthcheck(
    api_base: str,
    api_key: Optional[str],
    model_name: Optional[str],
) -> tuple[bool, Optional[str]]:
    """Perform a lightweight request to verify the NIM endpoint is responsive."""

    if not requests:
        return False, "Requests library unavailable; cannot perform health check."
    if not api_base:
        return False, "NIM API base URL is not configured."
    url = f"{api_base.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model_name or "model",
        "messages": [{"role": "user", "content": "Are you alive?"}],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        return False, str(exc)
    if response.status_code == 200:
        return True, None
    if response.status_code in {401, 403}:
        reason = "Authorization failed. Verify the NIM API key is valid."
        try:
            details = response.json()
        except ValueError:
            details = None
        if isinstance(details, dict):
            message = details.get("error") or details.get("message")
            if isinstance(message, str) and message.strip():
                reason = message.strip()
        return False, reason
    text = response.text[:200] if response.text else f"status {response.status_code}"
    return False, text


def _extract_text(filename: str, contents: bytes, content_type: str | None) -> str:
    suffix = (filename or "").lower()
    if suffix.endswith(".pdf") or (content_type and "pdf" in content_type):
        try:
            reader = PdfReader(io.BytesIO(contents))
        except Exception as exc:  # pragma: no cover - depends on external lib
            raise HTTPException(status_code=400, detail=f"Failed to read PDF: {exc}") from exc
        pages = []
        for page in reader.pages:
            try:
                extracted = page.extract_text() or ""
            except Exception:  # pragma: no cover - best effort for malformed PDFs
                extracted = ""
            pages.append(extracted)
        return "\n\n".join(pages)
    try:
        return contents.decode("utf-8")
    except UnicodeDecodeError:
        return contents.decode("latin-1", errors="ignore")
