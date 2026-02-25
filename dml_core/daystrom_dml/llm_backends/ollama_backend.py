"""Ollama backend for local LLM generation."""
from __future__ import annotations

import logging
from typing import Iterable, Optional

import requests

from .base import LLMBackend

LOGGER = logging.getLogger(__name__)


class OllamaBackend(LLMBackend):
    """Ollama backend for local LLM generation."""

    base_url: str = "http://localhost:11434"
    model_name: str
    temperature: float = 0.7
    top_p: float = 0.9

    def __init__(self, model_name: str, **kwargs):
        """Initialize Ollama backend.

        Args:
            model_name: Name of the model to use (e.g., "llama3.1:13b-instruct").
            **kwargs: Additional parameters (ignored).
        """
        self.model_name = model_name
        self.base_url = kwargs.get("base_url", self.base_url)
        self.temperature = kwargs.get("temperature", self.temperature)
        self.top_p = kwargs.get("top_p", self.top_p)
        LOGGER.info("Initialized Ollama backend for model %s at %s", model_name, self.base_url)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Iterable[str] | None = None,
    ) -> str:
        """Generate a completion using Ollama.

        Args:
            prompt: The prompt to generate from.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Sampling top-p.
            stop: Stop sequences.

        Returns:
            Generated text.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature or self.temperature,
                "top_p": top_p or self.top_p,
            },
        }
        if stop:
            payload["options"]["stop"] = list(stop)

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.RequestException as exc:
            LOGGER.error("Ollama generation failed: %s", exc)
            raise

    def stream_generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Iterable[str] | None = None,
    ):
        """Stream generation tokens from Ollama (not yet implemented)."""
        raise NotImplementedError("Streaming not yet supported for Ollama backend")