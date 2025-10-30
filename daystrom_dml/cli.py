"""Command line interface for the Daystrom Memory Lattice."""
from __future__ import annotations

import json
import logging
from typing import Optional

import typer

from .dml_adapter import DMLAdapter

app = typer.Typer(help="Daystrom Memory Lattice control interface")


def _build_adapter(model: Optional[str]) -> DMLAdapter:
    overrides = {"model_name": model} if model else None
    return DMLAdapter(config_overrides=overrides, start_aging_loop=False)


@app.command()
def ingest(text: str) -> None:
    """Store a new memory fragment."""

    adapter = _build_adapter(None)
    adapter.ingest(text)
    typer.echo("Ingested snippet.")


@app.command()
def query(prompt: str, model: Optional[str] = typer.Option(None)) -> None:
    """Retrieve context for a prompt."""

    adapter = _build_adapter(model)
    context = adapter.build_preamble(prompt)
    typer.echo(context)


@app.command()
def reinforce(text: str) -> None:
    """Reinforce a conclusion."""

    adapter = _build_adapter(None)
    adapter.reinforce("", text)
    typer.echo("Reinforced memory.")


@app.command()
def run(prompt: str, model: Optional[str] = typer.Option(None)) -> None:
    """Run an augmented generation round-trip."""

    adapter = _build_adapter(model)
    response = adapter.run_generation(prompt)
    typer.echo(response)


@app.command()
def stats() -> None:
    """Print diagnostic information about the current lattice."""

    adapter = _build_adapter(None)
    typer.echo(json.dumps(adapter.stats(), indent=2))


@app.command()
def checkpoint() -> None:
    """Create an immediate persistence checkpoint."""

    adapter = _build_adapter(None)
    path = adapter.create_checkpoint()
    adapter.close()
    typer.echo(f"Checkpoint written to {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app()
