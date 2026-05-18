"""OpenAI Codex backend for code generation."""
from __future__ import annotations

from typing import Iterable, Optional

import requests

from .base import LLMBackend


class CodexBackend(LLMBackend):
    """OpenAI Codex backend for code generation."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4o",
        max_new_tokens: int = 2048,
        temperature: float = 0.2,
    ):
        """
        Initialize Codex backend.

        Args:
            api_key: OpenAI API key.
            base_url: OpenAI API base URL (default: https://api.openai.com/v1).
            model_name: Model to use (e.g., "gpt-4o", "gpt-4-turbo", "codex").
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Iterable[str] | None = None,
    ) -> str:
        """Generate code using Codex."""
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        headers["Authorization"] = f"Bearer {self.api_key}"

        # Codex-optimized prompt
        messages = [
            {
                "role": "system",
                "content": "You are an expert programmer. Provide clear, well-commented code. "
                          "Include explanations when helpful. Output only code when asked."
            },
            {"role": "user", "content": prompt}
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens or self.max_new_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": top_p if top_p is not None else 1.0,
        }

        if stop:
            payload["stop"] = list(stop)

        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]["message"]["content"]
        return choice.strip()

    def stream_generate(
        self,
        *,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Iterable[str] | None = None,
    ):
        """Stream code generation (not fully supported by OpenAI API)."""
        raise NotImplementedError("OpenAI API does not support streaming for this model combination.")