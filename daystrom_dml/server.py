"""FastAPI service exposing the Daystrom Memory Lattice."""
from __future__ import annotations

import io
import logging
import os
import shlex
import shutil
import subprocess
import time
from urllib.parse import urlparse, urlunparse
from pathlib import Path
from threading import Lock
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
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
LOGGER = logging.getLogger(__name__)

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

DEFAULT_NIM_ID = "gpt-oss-20b"
VISUALIZER_URL = os.environ.get("DML_VISUALIZER_URL", "http://localhost:8501")
NGC_KEY_FILE = Path(
    os.environ.get(
        "NGC_KEY_FILE",
        Path(__file__).resolve().parent.parent / "ngc_api_key.txt",
    )
)

CURRENT_NIM: Optional[dict] = None
CURRENT_NIM_RUNTIME: dict = {"container_id": None, "running": False, "healthy": False}

NIM_CONTAINER_NAME = os.environ.get("NIM_CONTAINER_NAME", "daystrom-dml-nim")
NIM_DEFAULT_PORT = int(os.environ.get("NIM_PORT", "8000"))
NIM_HEALTH_TIMEOUT = int(os.environ.get("NIM_HEALTH_TIMEOUT", "60"))
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
    nim_id: Optional[str] = None
    nim_image: Optional[str] = None
    api_key: str


class NimStartPayload(BaseModel):
    port: Optional[int] = None
    cache_dir: Optional[str] = None
    wait_timeout: Optional[int] = None


class NimStopPayload(BaseModel):
    timeout: Optional[int] = None


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    index = WEB_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend bundle missing")
    return HTMLResponse(index.read_text(encoding="utf-8"))


@app.get("/visualizer")
def visualizer_redirect() -> RedirectResponse:
    """Open the external visualiser in a new tab."""

    return RedirectResponse(url=VISUALIZER_URL)


@app.get("/visualizer/url")
def visualizer_url() -> dict:
    """Expose the configured visualiser target for the frontend."""

    return {"url": VISUALIZER_URL}


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
            raise HTTPException(status_code=503, detail="NIM backend is unreachable. Start the container and try again.")
        raise
    prompt_tokens = utils.estimate_tokens(payload.prompt)
    return {
        **result,
        "prompt_tokens_est": prompt_tokens,
    }


@app.get("/stats")
def stats() -> dict:
    return adapter.stats()


@app.get("/knowledge")
def knowledge() -> dict:
    """Expose summaries of the documents stored in RAG and the DML lattice."""

    return adapter.knowledge_report()


@app.get("/nim/options")
def nim_options() -> dict:
    """Expose the curated list of NVIDIA NIM container options."""

    return {
        "options": NIM_OPTIONS,
        "current": CURRENT_NIM,
        "default": _nim_summary(_nim_option(DEFAULT_NIM_ID)),
        "runtime": _runtime_status(),
    }


@app.post("/nim/configure")
def nim_configure(payload: NimConfigurePayload) -> dict:
    """Pull a NIM container image and reconfigure the adapter."""

    if not payload.api_key.strip():
        raise HTTPException(status_code=400, detail="NGC API key is required")
    nim_id = (payload.nim_id or "").strip()
    nim_image = (payload.nim_image or "").strip()
    if not nim_id and not nim_image:
        nim_id = DEFAULT_NIM_ID
    option = None
    if nim_id:
        option = _nim_option(nim_id)
    elif nim_image:
        option = _nim_option_by_image(nim_image)
    if not option:
        identifier = nim_id or nim_image or ""
        raise HTTPException(status_code=404, detail=f"Unknown NIM selection provided: {identifier}")
    try:
        pull_status, pull_logs = _pull_nim_image(option["image"], payload.api_key.strip())
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    _apply_nim_configuration(option, payload.api_key.strip())
    summary = _nim_summary(option)
    global CURRENT_NIM
    CURRENT_NIM = summary
    CURRENT_NIM_RUNTIME.update({"running": False, "healthy": False, "container_id": None})
    return {
        "status": "ok",
        "nim": summary,
        "pull_status": pull_status,
        "logs": pull_logs,
        "runtime": _runtime_status(),
    }


@app.post("/nim/start")
def nim_start(payload: NimStartPayload | None = None) -> dict:
    """Start the configured NIM container and wait for it to become healthy."""

    if CURRENT_NIM is None:
        raise HTTPException(status_code=400, detail="Configure a NIM before attempting to start it.")
    docker_bin = shutil.which("docker")
    runtime = _runtime_status()
    if not docker_bin:
        return {
            "status": "skipped",
            "message": "Docker binary not available on server; cannot start NIM.",
            "runtime": runtime,
        }
    api_key = os.environ.get("NIM_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("NGC_API_KEY")
    port = NIM_DEFAULT_PORT
    if payload and payload.port:
        port = int(payload.port)
    cache_dir = None
    if payload and payload.cache_dir:
        cache_dir = Path(payload.cache_dir).expanduser()
    else:
        cache_env = os.environ.get("LOCAL_NIM_CACHE")
        if cache_env:
            cache_dir = Path(cache_env).expanduser()
    if cache_dir:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:  # pragma: no cover - best effort for unusual permissions
            cache_dir = None
    if runtime.get("running"):
        healthy, reason = _nim_healthcheck(CURRENT_NIM["api_base"], api_key)
        CURRENT_NIM_RUNTIME.update({"healthy": healthy})
        runtime = _runtime_status()
        message = "NIM container already running and healthy." if healthy else (
            "NIM container is running but not responding yet." + (f" Reason: {reason}" if reason else "")
        )
        return {
            "status": "running" if healthy else "starting",
            "message": message,
            "runtime": runtime,
        }
    api_base = _configure_runtime_api_base(port)
    run_cmd = [
        docker_bin,
        "run",
        "-d",
        "--rm",
        "--gpus=all",
        "--name",
        NIM_CONTAINER_NAME,
        "-p",
        f"{port}:8000",
    ]
    extra_opts = os.environ.get("NIM_DOCKER_RUN_OPTS")
    if extra_opts:
        run_cmd.extend(shlex.split(extra_opts))
    if cache_dir:
        run_cmd.extend(["-v", f"{cache_dir}:/opt/nim/.cache"])
    if api_key:
        run_cmd.extend(["-e", f"NGC_API_KEY={api_key}"])
    run_cmd.append(CURRENT_NIM["image"])
    logs: list[str] = []
    try:
        run_proc = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except Exception as exc:  # pragma: no cover - subprocess errors are environment dependent
        raise HTTPException(status_code=500, detail=f"Failed to launch NIM container: {exc}") from exc
    if run_proc.stdout:
        logs.append(run_proc.stdout.strip())
    if run_proc.stderr:
        logs.append(run_proc.stderr.strip())
    if run_proc.returncode != 0:
        runtime = _runtime_status()
        return {
            "status": "error",
            "message": "Docker failed to start the NIM container.",
            "logs": logs,
            "runtime": runtime,
        }
    container_id = run_proc.stdout.strip()
    CURRENT_NIM_RUNTIME.update({"container_id": container_id or None, "running": True, "healthy": False})
    wait_timeout = NIM_HEALTH_TIMEOUT
    if payload and payload.wait_timeout:
        wait_timeout = int(payload.wait_timeout)
    healthy, health_logs = _wait_for_nim_health(
        api_base,
        api_key,
        timeout=wait_timeout,
    )
    CURRENT_NIM_RUNTIME.update({"healthy": healthy})
    runtime = _runtime_status()
    logs.extend(health_logs)
    status = "running" if healthy else "starting"
    message = "NIM container is ready." if healthy else "NIM container launched but health check timed out."
    return {
        "status": status,
        "message": message,
        "logs": logs,
        "runtime": runtime,
    }


@app.post("/nim/stop")
def nim_stop(payload: NimStopPayload | None = None) -> dict:
    """Stop the managed NIM container."""

    docker_bin = shutil.which("docker")
    runtime = _runtime_status()
    if not docker_bin:
        return {
            "status": "skipped",
            "message": "Docker binary not available on server; cannot stop NIM.",
            "runtime": runtime,
        }
    if not runtime.get("running"):
        return {
            "status": "not-running",
            "message": "No running NIM container detected.",
            "runtime": runtime,
        }
    timeout = 60
    if payload and payload.timeout:
        timeout = int(payload.timeout)
    stop_cmd = [docker_bin, "stop", NIM_CONTAINER_NAME]
    logs: list[str] = []
    stop_proc = subprocess.run(
        stop_cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if stop_proc.stdout:
        logs.append(stop_proc.stdout.strip())
    if stop_proc.stderr:
        logs.append(stop_proc.stderr.strip())
    if stop_proc.returncode != 0:
        runtime = _runtime_status()
        return {
            "status": "error",
            "message": "Docker failed to stop the NIM container.",
            "logs": logs,
            "runtime": runtime,
        }
    CURRENT_NIM_RUNTIME.update({"running": False, "healthy": False, "container_id": None})
    runtime = _runtime_status()
    return {
        "status": "stopped",
        "message": "NIM container stopped.",
        "logs": logs,
        "runtime": runtime,
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


def _nim_summary(option: dict) -> dict:
    return {
        "id": option["id"],
        "label": option["label"],
        "model_name": option["model_name"],
        "api_base": option["default_api_base"],
        "image": option["image"],
    }


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
    _save_ngc_key(api_key)
    _reload_adapter(config_overrides={"model_name": option["model_name"]})


def _save_ngc_key(api_key: str) -> None:
    """Persist the provided NGC API key for convenience."""

    try:
        NGC_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        NGC_KEY_FILE.write_text(api_key.strip() + "\n", encoding="utf-8")
    except Exception as exc:  # pragma: no cover - filesystem permissions vary
        LOGGER.warning("Failed to persist NGC API key: %s", exc)


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


def _runtime_status() -> dict:
    """Return the current runtime view of the managed NIM container."""

    docker_bin = shutil.which("docker")
    running = CURRENT_NIM_RUNTIME.get("running", False)
    healthy = CURRENT_NIM_RUNTIME.get("healthy", False)
    container_id = CURRENT_NIM_RUNTIME.get("container_id")
    if docker_bin:
        ps_proc = subprocess.run(
            [docker_bin, "ps", "-q", "--filter", f"name={NIM_CONTAINER_NAME}"],
            capture_output=True,
            text=True,
            check=False,
        )
        listed = ps_proc.stdout.strip().splitlines()
        if listed:
            container_id = listed[0]
            running = True
        else:
            running = False
            healthy = False
            container_id = None
    else:
        healthy = healthy if running else False
    CURRENT_NIM_RUNTIME.update(
        {
            "running": running,
            "healthy": healthy if running else False,
            "container_id": container_id,
        }
    )
    return {
        "running": CURRENT_NIM_RUNTIME["running"],
        "healthy": CURRENT_NIM_RUNTIME["healthy"],
        "container_id": CURRENT_NIM_RUNTIME["container_id"],
        "container_name": NIM_CONTAINER_NAME,
        "docker_available": docker_bin is not None,
    }


def _nim_healthcheck(api_base: str, api_key: Optional[str]) -> tuple[bool, Optional[str]]:
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
        "model": CURRENT_NIM["model_name"] if CURRENT_NIM else "model",
        "messages": [{"role": "user", "content": "Are you alive?"}],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        return False, str(exc)
    if response.status_code in {200, 401, 403}:
        return True, None
    text = response.text[:200] if response.text else f"status {response.status_code}"
    return False, text


def _wait_for_nim_health(
    api_base: str,
    api_key: Optional[str],
    *,
    timeout: int,
) -> tuple[bool, list[str]]:
    """Poll the NIM endpoint until it responds or the timeout elapses."""

    deadline = time.time() + max(timeout, 1)
    attempts: list[str] = []
    if not requests:
        attempts.append("Requests library unavailable; skipping health polling.")
        return False, attempts
    while time.time() < deadline:
        healthy, reason = _nim_healthcheck(api_base, api_key)
        if healthy:
            return True, attempts
        attempts.append(f"Health check failed: {reason or 'unknown error'}")
        time.sleep(NIM_HEALTH_INTERVAL)
    return False, attempts


def _configure_runtime_api_base(port: int) -> str:
    """Derive and apply the runtime API base for the configured NIM port."""

    if CURRENT_NIM is None:
        return f"http://localhost:{port}"
    existing_base = CURRENT_NIM.get("api_base") or f"http://localhost:{port}"
    updated_base = _nim_api_base_with_port(existing_base, port)
    CURRENT_NIM["api_base"] = updated_base
    os.environ["NIM_API_BASE"] = updated_base
    os.environ["OPENAI_API_BASE"] = updated_base
    os.environ["NIM_PORT"] = str(port)
    runner_backend = getattr(getattr(adapter, "runner", None), "_backend", None)
    if hasattr(runner_backend, "base_url"):
        runner_backend.base_url = updated_base.rstrip("/")
    return updated_base


def _nim_api_base_with_port(api_base: str, port: int) -> str:
    """Return the API base with the provided port applied to the netloc."""

    if not api_base:
        return f"http://localhost:{port}"
    parsed = urlparse(api_base)
    scheme = parsed.scheme or "http"
    if not parsed.netloc:
        return f"{scheme}://localhost:{port}"
    host = parsed.hostname or "localhost"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    userinfo = ""
    if parsed.username:
        userinfo = parsed.username
        if parsed.password:
            userinfo += f":{parsed.password}"
        userinfo += "@"
    netloc = f"{userinfo}{host}:{port}"
    rebuilt = parsed._replace(netloc=netloc)
    return urlunparse(rebuilt)
