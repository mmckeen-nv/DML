"""Ollama-compatible entrypoint backed by the DML provider."""
from __future__ import annotations

from .provider_server import main


if __name__ == "__main__":
    main()
