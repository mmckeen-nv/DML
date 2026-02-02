"""Command line interface for the Daystrom Memory Lattice."""
from __future__ import annotations

import contextlib
import json
import logging
from typing import Iterator, Optional

import typer

from .dml_adapter import DMLAdapter

app = typer.Typer(help="Daystrom Memory Lattice control interface")


def _build_adapter(overrides: Optional[dict]) -> DMLAdapter:
    return DMLAdapter(config_overrides=overrides, start_aging_loop=False)


def _collect_overrides(
    *,
    model: Optional[str] = None,
    backend: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    enable_stm: bool = False,
    commitment_threshold: Optional[float] = None,
    ltm_write_policy: Optional[str] = None,
    stm_max_commitments: Optional[int] = None,
    ltm_top_k: Optional[int] = None,
) -> Optional[dict]:
    overrides: dict = {}
    if model:
        overrides["model_name"] = model
    if backend:
        overrides["llm_backend"] = backend
    if device:
        overrides["llm_device"] = device
    if dtype:
        overrides["llm_dtype"] = dtype
    if load_in_4bit:
        overrides["load_in_4bit"] = True
    if load_in_8bit:
        overrides["load_in_8bit"] = True
    if enable_stm:
        overrides["enable_stm_controller"] = True
    if commitment_threshold is not None:
        overrides["commitment_threshold"] = commitment_threshold
    if ltm_write_policy:
        overrides["ltm_write_policy"] = ltm_write_policy
    if stm_max_commitments is not None:
        overrides["stm_max_commitments"] = stm_max_commitments
    if ltm_top_k is not None:
        overrides["ltm_top_k"] = ltm_top_k
    return overrides or None


@contextlib.contextmanager
def _adapter_scope(overrides: Optional[dict]) -> Iterator[DMLAdapter]:
    adapter = _build_adapter(overrides)
    try:
        yield adapter
    finally:
        with contextlib.suppress(Exception):
            adapter.close()


@app.command()
def ingest(text: str) -> None:
    """Store a new memory fragment."""

    with _adapter_scope(None) as adapter:
        adapter.ingest(text)
    typer.echo("Ingested snippet.")


@app.command()
def query(prompt: str, model: Optional[str] = typer.Option(None)) -> None:
    """Retrieve context for a prompt."""

    overrides = _collect_overrides(model=model)
    with _adapter_scope(overrides) as adapter:
        context = adapter.build_preamble(prompt)
    typer.echo(context)


@app.command()
def reinforce(text: str) -> None:
    """Reinforce a conclusion."""

    with _adapter_scope(None) as adapter:
        adapter.reinforce("", text)
    typer.echo("Reinforced memory.")


@app.command()
def run(
    prompt: str,
    model: Optional[str] = typer.Option(None, help="Model name or HF checkpoint."),
    backend: Optional[str] = typer.Option(None, help="LLM backend: transformers, openai, nim, auto."),
    device: Optional[str] = typer.Option(None, help="Device: auto, cpu, cuda, mps."),
    dtype: Optional[str] = typer.Option(None, help="dtype: auto, float16, bfloat16, float32."),
    load_in_4bit: bool = typer.Option(False, help="Enable 4-bit quantization if bitsandbytes is installed."),
    load_in_8bit: bool = typer.Option(False, help="Enable 8-bit quantization if bitsandbytes is installed."),
    enable_stm: bool = typer.Option(False, help="Enable structured STM controller."),
    commitment_threshold: Optional[float] = typer.Option(None, help="Minimum confidence to write LTM."),
    ltm_write_policy: Optional[str] = typer.Option(None, help="LTM write policy: strict, balanced, off."),
    stm_max_commitments: Optional[int] = typer.Option(None, help="Max STM commitments to keep."),
    ltm_top_k: Optional[int] = typer.Option(None, help="Top-K LTM retrieval count."),
) -> None:
    """Run an augmented generation round-trip."""

    overrides = _collect_overrides(
        model=model,
        backend=backend,
        device=device,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        enable_stm=enable_stm,
        commitment_threshold=commitment_threshold,
        ltm_write_policy=ltm_write_policy,
        stm_max_commitments=stm_max_commitments,
        ltm_top_k=ltm_top_k,
    )
    with _adapter_scope(overrides) as adapter:
        response = adapter.run_generation(prompt)
    typer.echo(response)


@app.command()
def stats() -> None:
    """Print diagnostic information about the current lattice."""

    with _adapter_scope(None) as adapter:
        payload = adapter.stats()
    typer.echo(json.dumps(payload, indent=2))


@app.command()
def checkpoint() -> None:
    """Create an immediate persistence checkpoint."""

    with _adapter_scope(None) as adapter:
        path = adapter.create_checkpoint()
    typer.echo(f"Checkpoint written to {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app()
