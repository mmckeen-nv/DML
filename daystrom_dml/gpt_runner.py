"""Wrapper around HuggingFace models with graceful degradation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

LOGGER = logging.getLogger(__name__)


@dataclass
class GPTRunner:
    """Minimal wrapper providing ``generate`` and ``summarize``.

    The class attempts to instantiate a HuggingFace pipeline.  When the required
    dependencies or model weights are not available (as is often the case in
    offline tests) it falls back to a deterministic dummy backend.
    """

    model_name: str
    task: str = "text-generation"
    device: Optional[str] = None

    def __post_init__(self) -> None:
        self._backend = None
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._backend = pipeline(
                self.task,
                model=model,
                tokenizer=tokenizer,
                device=self.device,
            )
            LOGGER.info("Loaded HF model %s", self.model_name)
        except Exception as exc:  # pragma: no cover - executed in offline tests
            LOGGER.warning("Using DummyGPT backend: %s", exc)
            self._backend = _DummyBackend()

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        if isinstance(self._backend, _DummyBackend):
            return self._backend.generate(prompt, max_new_tokens=max_new_tokens)
        outputs = self._backend(prompt, max_new_tokens=max_new_tokens)
        if isinstance(outputs, list):
            return outputs[0]["generated_text"]
        return str(outputs)

    def summarize(self, text: str, max_len: int = 128) -> str:
        if isinstance(self._backend, _DummyBackend):
            return self._backend.summarize(text, max_len=max_len)
        prompt = (
            "Summarise the following content in at most"
            f" {max_len} characters:\n{text}\nSummary:"
        )
        output = self.generate(prompt, max_new_tokens=max_len)
        return output.split("Summary:")[-1].strip()

    @property
    def is_dummy(self) -> bool:
        """Expose whether the runner is using the dummy backend."""

        return isinstance(self._backend, _DummyBackend)


class _DummyBackend:
    """Fallback backend used during tests."""

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        text = prompt.strip()
        suffix = "\n[Dummy completion truncated]"
        if len(text) > max_new_tokens:
            text = text[: max_new_tokens]
        return text + suffix

    def summarize(self, text: str, max_len: int = 128) -> str:
        text = text.strip().replace("\n", " ")
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."
