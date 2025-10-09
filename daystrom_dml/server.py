"""FastAPI service exposing the Daystrom Memory Lattice."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .dml_adapter import DMLAdapter

app = FastAPI(title="Daystrom Memory Lattice")
adapter = DMLAdapter(start_aging_loop=False)


class TextPayload(BaseModel):
    text: str


class QueryPayload(BaseModel):
    prompt: str


@app.post("/ingest")
def ingest(payload: TextPayload) -> dict:
    adapter.ingest(payload.text)
    return {"status": "ok"}


@app.post("/reinforce")
def reinforce(payload: TextPayload) -> dict:
    adapter.reinforce("", payload.text)
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


@app.get("/stats")
def stats() -> dict:
    return adapter.stats()
