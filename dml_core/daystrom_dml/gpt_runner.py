"""Wrapper around HuggingFace models with graceful degradation."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional

import requests

LOGGER = logging.getLogger(__name__)


@dataclass
class GPTRunner:
    """Minimal wrapper providing ``generate`` and ``summarize``.

    The class attempts to instantiate a HuggingFace pipeline.  When the required
    dependencies or model weights are not available (as is often the case in
    offline tests) it falls back to a deterministic local backend.
    """

    model_name: str
    task: str = "text-generation"
    device: Optional[str] = None
    backend: str = "auto"
    dtype: str = "auto"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True
    temperature: float = 0.2
    top_p: float = 1.0

    def __post_init__(self) -> None:
        self._backend = None
        self._last_usage: Optional[dict] = None
        if str(self.model_name or "").strip().lower() == "dummy":
            LOGGER.warning("Using deterministic local completion backend.")
            self._backend = _DummyBackend()
            return
        remote_base = os.getenv("DML_API_BASE") or os.getenv("OPENAI_API_BASE")
        remote_base = remote_base or os.getenv("NIM_API_BASE")
        remote_key = os.getenv("DML_API_KEY") or os.getenv("OPENAI_API_KEY")
        remote_key = remote_key or os.getenv("NIM_API_KEY")
        backend_choice = (self.backend or "auto").lower()
        if backend_choice in {"openai", "nim", "remote"}:
            if not remote_base:
                LOGGER.warning(
                    "Remote backend requested but no API base was configured; falling back."
                )
            else:
                self._backend = _OpenAICompatibleBackend(
                    base_url=remote_base,
                    api_key=remote_key,
                    model_name=self.model_name,
                )
                LOGGER.info("Configured remote backend at %s", remote_base)
                return
        if backend_choice == "auto" and remote_base:
            self._backend = _OpenAICompatibleBackend(
                base_url=remote_base,
                api_key=remote_key,
                model_name=self.model_name,
            )
            LOGGER.info("Configured remote backend at %s", remote_base)
            return
        if backend_choice not in {"auto", "transformers", "local", "ollama"}:
            LOGGER.warning("Unknown backend choice %s; falling back to auto.", backend_choice)
            backend_choice = "auto"
        if backend_choice == "ollama":
            try:
                from .llm_backends.ollama_backend import OllamaBackend

                self._backend = OllamaBackend(
                    model_name=self.model_name,
                    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                LOGGER.info("Loaded ollama backend for %s", self.model_name)
                return
            except Exception as exc:
                LOGGER.warning("Ollama backend unavailable: %s", exc)
        if backend_choice in {"auto", "transformers", "local"}:
            try:
                from .llm_backends.transformers_backend import TransformersBackend

                self._backend = TransformersBackend(
                    model_name=self.model_name,
                    device=self.device or "auto",
                    dtype=self.dtype,
                    load_in_4bit=self.load_in_4bit,
                    load_in_8bit=self.load_in_8bit,
                    trust_remote_code=self.trust_remote_code,
                    use_fast_tokenizer=self.use_fast_tokenizer,
                )
                LOGGER.info("Loaded transformers backend for %s", self.model_name)
                return
            except Exception as exc:  # pragma: no cover - executed in offline tests
                LOGGER.warning("Transformers backend unavailable: %s", exc)
        LOGGER.warning("Using deterministic local completion backend.")
        self._backend = _DummyBackend()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Iterable[str] | None = None,
    ) -> str:
        self._last_usage = None
        approx_tokens = len(prompt.split())
        LOGGER.info(
            "Sending prompt to language model (model=%s, approx_tokens=%d)",
            self.model_name,
            approx_tokens,
        )
        LOGGER.debug("Prompt excerpt: %s", prompt[:400])
        if isinstance(self._backend, _DummyBackend):
            return self._backend.generate(prompt, max_new_tokens=max_new_tokens)
        if isinstance(self._backend, _OpenAICompatibleBackend):
            text, usage = self._backend.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                top_p=top_p if top_p is not None else self.top_p,
                stop=stop,
            )
            self._last_usage = usage
            return text
        if hasattr(self._backend, "generate"):
            try:
                return self._backend.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature is not None else self.temperature,
                    top_p=top_p if top_p is not None else self.top_p,
                    stop=stop,
                )
            except requests.RequestException as exc:
                if self._backend.__class__.__name__ != "OllamaBackend":
                    raise
                LOGGER.warning(
                    "Ollama generation failed for model %s; using deterministic local completion backend: %s",
                    self.model_name,
                    exc,
                )
                self._backend = _DummyBackend()
                return self._backend.generate(prompt, max_new_tokens=max_new_tokens)
        outputs = self._backend(prompt, max_new_tokens=max_new_tokens)
        if isinstance(outputs, list):
            return outputs[0]["generated_text"]
        return str(outputs)

    def summarize(self, text: str, max_len: int = 128) -> str:
        if isinstance(self._backend, _DummyBackend):
            return self._backend.summarize(text, max_len=max_len)
        if isinstance(self._backend, _OpenAICompatibleBackend):
            text, usage = self._backend.generate(
                (
                    "Summarise the following content in at most "
                    f"{max_len} characters. Return only the summary text; "
                    "do not preface it with phrases like 'Here is a summary'.\n"
                    f"{text}"
                ),
                max_new_tokens=max_len,
                system_prompt="You are a precise summariser that responds with plain text only.",
            )
            self._last_usage = usage
            return self._clean_summary_output(text, max_len=max_len)
        prompt = (
            "Summarise the following content in at most"
            f" {max_len} characters. Return only the summary text; do not "
            f"preface it with phrases like 'Here is a summary'.\n{text}\nSummary:"
        )
        output = self.generate(prompt, max_new_tokens=max_len)
        return self._clean_summary_output(output.split("Summary:")[-1], max_len=max_len)

    @staticmethod
    def _clean_summary_output(text: str, *, max_len: int) -> str:
        cleaned = (text or "").strip()
        cleaned = re.sub(
            r"(?is)^\s*(?:here is|here's|the content appears to be).*?(?:less|summary)\s*:\s*",
            "",
            cleaned,
        ).strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'\"', "'"}:
            cleaned = cleaned[1:-1].strip()
        if len(cleaned) > max_len:
            cleaned = cleaned[: max_len - 3].rstrip() + "..."
        return cleaned

    @property
    def is_dummy(self) -> bool:
        """Expose whether the runner is using the deterministic local backend."""

        return isinstance(self._backend, _DummyBackend)

    @property
    def last_usage(self) -> Optional[dict]:
        """Return the token usage payload from the most recent call."""

        return self._last_usage

class _DummyBackend:
    """Fallback backend used during tests."""

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        prompt_text = self._extract_user_prompt(prompt)
        snippets = self._extract_context_snippets(prompt)
        if snippets:
            body = " ".join(snippets[:2])
            text = body
        else:
            text = prompt_text or prompt.strip()
        return self._truncate(text, max_new_tokens)

    def summarize(self, text: str, max_len: int = 128) -> str:
        text = text.strip().replace("\n", " ")
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    @staticmethod
    def _extract_user_prompt(prompt: str) -> str:
        marker = "=== User Prompt ==="
        if marker not in prompt:
            return prompt.strip()
        return prompt.rsplit(marker, 1)[-1].strip()

    @staticmethod
    def _extract_context_snippets(prompt: str) -> list[str]:
        context = prompt.split("=== User Prompt ===", 1)[0]
        if "=== Private Grounding Notes ===" in context:
            context = context.split("=== Private Grounding Notes ===", 1)[1]
        snippets: list[str] = []
        for raw_line in context.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("==="):
                continue
            line = re.sub(r"^- L\d+ \(f=[^)]+\):\s*", "", line)
            line = re.sub(r"^Document \d+[^\\n]*", "", line).strip()
            if line.startswith(("Answer the user", "Treat the notes", "Do not mention", "Use only", "If the context")):
                continue
            if line.startswith(("Source:", "Prompt:", "Answer summary:")):
                continue
            if line:
                snippets.append(line)
        return snippets

    @staticmethod
    def _truncate(text: str, max_new_tokens: int) -> str:
        max_chars = max(24, int(max_new_tokens or 256) * 4)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars].rsplit(" ", 1)[0].strip()


class _OpenAICompatibleBackend:
    """Thin wrapper around OpenAI-compatible REST endpoints (incl. NVIDIA NIM)."""

    def __init__(self, *, base_url: str, api_key: Optional[str], model_name: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 1.0,
        stop: Iterable[str] | None = None,
    ) -> tuple[str, Optional[dict]]:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if stop:
            payload["stop"] = list(stop)
        LOGGER.info(
            "Dispatching completion request to NIM endpoint %s using model %s",
            url,
            self.model_name,
        )
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return "", data.get("usage")
        choice = choices[0] or {}
        message = choice.get("message") or {}
        content = message.get("content")
        if content is None:
            # Some OpenAI-compatible servers return ``null`` for empty content or use
            # the older ``text`` field.  Normalise both cases to an empty string so we
            # always return a ``str``.
            content = choice.get("text") or ""
        if not isinstance(content, str):
            content = str(content)
        LOGGER.info("Received response from NIM endpoint %s", url)
        return content.strip(), data.get("usage")
