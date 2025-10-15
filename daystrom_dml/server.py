"""FastAPI service exposing the Daystrom Memory Lattice."""
from __future__ import annotations

import io
from pathlib import Path
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
adapter = DMLAdapter(start_aging_loop=False)


class TextPayload(BaseModel):
    text: str
    meta: Optional[dict] = None


class QueryPayload(BaseModel):
    prompt: str


class ComparePayload(BaseModel):
    prompt: str
    top_k: Optional[int] = None
    max_new_tokens: Optional[int] = 512


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
