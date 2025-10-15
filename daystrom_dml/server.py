"""FastAPI service exposing the Daystrom Memory Lattice."""
from __future__ import annotations

import io
import os
import shutil
import subprocess
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

WEB_DIR = Path(__file__).with_name("web")

app = FastAPI(title="Daystrom Memory Lattice")
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

ADAPTER_LOCK = Lock()
adapter = DMLAdapter(start_aging_loop=False)

NIM_OPTIONS = [
    {
        "id": "gpt-oss-20b",
        "label": "GPT-OSS 20B (OpenAI Compatible)",
        "image": "nvcr.io/nim/openai/gpt-oss-20b:latest",
        "model_name": "meta/llama3-70b-instruct",
        "default_api_base": "http://localhost:8000",
    },
    {
        "id": "llama3-8b",
        "label": "Llama 3 8B Instruct",
        "image": "nvcr.io/nim/openai/llama3-8b-instruct:latest",
        "model_name": "meta/llama3-8b-instruct",
        "default_api_base": "http://localhost:8000",
    },
    {
        "id": "mixtral-8x7b",
        "label": "Mixtral 8x7B Instruct",
        "image": "nvcr.io/nim/openai/mixtral-8x7b-instruct:latest",
        "model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "default_api_base": "http://localhost:8000",
    },
]

CURRENT_NIM: Optional[dict] = None


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
    nim_id: Optional[str] = None
    nim_image: Optional[str] = None
    api_key: str


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
    report = adapter.retrieval_report(payload.prompt)
    return report


@app.post("/rag/compare")
def rag_compare(payload: ComparePayload) -> dict:
    result = adapter.compare_responses(
        payload.prompt,
        top_k=payload.top_k,
        max_new_tokens=payload.max_new_tokens or 512,
    )
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
    """Expose the curated list of NVIDIA NIM container options."""

    return {
        "options": NIM_OPTIONS,
        "current": CURRENT_NIM,
    }


@app.post("/nim/configure")
def nim_configure(payload: NimConfigurePayload) -> dict:
    """Pull a NIM container image and reconfigure the adapter."""

    if not payload.api_key.strip():
        raise HTTPException(status_code=400, detail="NGC API key is required")
    option = None
    if payload.nim_id:
        option = _nim_option(payload.nim_id)
    elif payload.nim_image:
        option = _nim_option_by_image(payload.nim_image.strip())
    if not option:
        identifier = payload.nim_id or payload.nim_image or ""
        raise HTTPException(status_code=404, detail=f"Unknown NIM selection provided: {identifier}")
    try:
        pull_status, pull_logs = _pull_nim_image(option["image"], payload.api_key.strip())
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    _apply_nim_configuration(option, payload.api_key.strip())
    summary = {
        "id": option["id"],
        "label": option["label"],
        "model_name": option["model_name"],
        "api_base": option["default_api_base"],
        "image": option["image"],
    }
    global CURRENT_NIM
    CURRENT_NIM = summary
    return {
        "status": "ok",
        "nim": summary,
        "pull_status": pull_status,
        "logs": pull_logs,
    }


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


def _nim_option(nim_id: str) -> dict:
    for option in NIM_OPTIONS:
        if option["id"] == nim_id:
            return option
    raise HTTPException(status_code=404, detail=f"Unknown NIM identifier: {nim_id}")


def _nim_option_by_image(image: str) -> Optional[dict]:
    for option in NIM_OPTIONS:
        if option["image"] == image:
            return option
    return None


def _pull_nim_image(image: str, api_key: str) -> tuple[str, list[str]]:
    """Attempt to pull the requested NIM image via Docker."""

    docker_bin = shutil.which("docker")
    if not docker_bin:
        return "skipped", ["Docker binary not available on server; skipping image pull."]
    logs: list[str] = []
    login_cmd = [
        docker_bin,
        "login",
        "nvcr.io",
        "--username",
        "$oauthtoken",
        "--password-stdin",
    ]
    login_proc = subprocess.run(
        login_cmd,
        input=f"{api_key}\n",
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if login_proc.stdout:
        logs.append(login_proc.stdout.strip())
    if login_proc.stderr:
        logs.append(login_proc.stderr.strip())
    if login_proc.returncode != 0:
        raise RuntimeError("Docker login failed; verify the provided NGC API key is valid.")
    pull_proc = subprocess.run(
        [docker_bin, "pull", image],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    if pull_proc.stdout:
        logs.append(pull_proc.stdout.strip())
    if pull_proc.stderr:
        logs.append(pull_proc.stderr.strip())
    if pull_proc.returncode != 0:
        raise RuntimeError(f"Docker pull failed for image {image}.")
    return "ok", logs


def _apply_nim_configuration(option: dict, api_key: str) -> None:
    """Set environment variables and reload the adapter for the selected NIM."""

    os.environ["NIM_API_KEY"] = api_key
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["NIM_API_BASE"] = option["default_api_base"]
    os.environ["OPENAI_API_BASE"] = option["default_api_base"]
    _reload_adapter(config_overrides={"model_name": option["model_name"]})


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
