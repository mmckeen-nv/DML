"""LLM backend interfaces."""
from __future__ import annotations

from typing import Iterable, Iterator, Protocol


class LLMBackend(Protocol):
    """Protocol implemented by language model backends."""

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: Iterable[str] | None = None,
    ) -> str:
        """Generate a completion for the provided prompt."""

    def stream_generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: Iterable[str] | None = None,
    ) -> Iterator[str]:
        """Stream completion tokens for the provided prompt."""
