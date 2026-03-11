"""Transformers backend for local generation."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from .base import LLMBackend

LOGGER = logging.getLogger(__name__)


@dataclass
class TransformersBackend(LLMBackend):
    """HuggingFace Transformers backend using AutoModelForCausalLM."""

    model_name: str
    device: str = "auto"
    dtype: str = "auto"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True

    def portable_load_options(self) -> dict[str, object]:
        """Return a backend-agnostic load contract for TorchForge parity work.

        The returned shape is intentionally serializable so orchestration layers can
        pass identical model intent to either Transformers or TorchForge loaders.
        """
        return _build_portable_load_options(
            model_name=self.model_name,
            device=self.device,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            use_fast_tokenizer=self.use_fast_tokenizer,
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
        )

    def __post_init__(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Transformers backend requires torch. Install torch to use it."
            ) from exc
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Transformers backend requires transformers. Install it to use this backend."
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=self.use_fast_tokenizer,
            trust_remote_code=self.trust_remote_code,
        )
        model_kwargs = dict(self._resolve_model_kwargs())
        LOGGER.info("Loading transformers model %s", self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            **model_kwargs,
        )
        self._model.eval()
        self._device = self._resolve_device()
        if self._device != "auto" and not self._uses_quantization(model_kwargs):
            self._model.to(self._device)
        LOGGER.info("Transformers backend ready on device=%s", self._device)

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: Iterable[str] | None = None,
    ) -> str:
        output = self._generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return _apply_stop_sequences(output, stop)

    def stream_generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: Iterable[str] | None = None,
    ) -> Iterator[str]:
        try:
            from transformers import TextIteratorStreamer
        except Exception:
            yield self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
            return

        inputs, input_len = self._tokenize_prompt(prompt)
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = self._generation_kwargs(
            inputs, max_new_tokens, temperature, top_p
        )
        generation_kwargs["streamer"] = streamer

        def _generate() -> None:
            with self._torch.inference_mode():
                self._model.generate(**generation_kwargs)

        thread = threading.Thread(target=_generate)
        thread.start()
        buffer = ""
        for chunk in streamer:
            buffer += chunk
            if stop:
                truncated = _apply_stop_sequences(buffer, stop)
                if truncated != buffer:
                    yield truncated
                    return
            yield chunk

    def _generate_text(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        inputs, input_len = self._tokenize_prompt(prompt)
        generation_kwargs = self._generation_kwargs(
            inputs, max_new_tokens, temperature, top_p
        )
        with self._torch.inference_mode():
            output_ids = self._model.generate(**generation_kwargs)
        decoded = self._tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )
        return decoded.strip()

    def _tokenize_prompt(self, prompt: str) -> tuple[dict, int]:
        formatted = self._format_prompt(prompt)
        inputs = self._tokenizer(formatted, return_tensors="pt")
        if self._device != "auto":
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
        input_len = int(inputs["input_ids"].shape[-1])
        return inputs, input_len

    def _format_prompt(self, prompt: str) -> str:
        if hasattr(self._tokenizer, "apply_chat_template") and getattr(
            self._tokenizer, "chat_template", None
        ):
            messages = [{"role": "user", "content": prompt}]
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def _generation_kwargs(
        self,
        inputs: dict,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict:
        do_sample = temperature > 0.0
        return {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
        }

    def _resolve_device(self) -> str:
        device = (self.device or "auto").lower()
        if device != "auto":
            return device
        if self._torch.cuda.is_available():
            return "cuda"
        if getattr(self._torch.backends, "mps", None) and self._torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self) -> Optional["torch.dtype"]:
        dtype = (self.dtype or "auto").lower()
        if dtype == "auto":
            device = self._resolve_device()
            if device in {"cuda", "mps"}:
                return self._torch.float16
            return self._torch.float32
        mapping = {
            "float16": self._torch.float16,
            "bfloat16": self._torch.bfloat16,
            "float32": self._torch.float32,
        }
        return mapping.get(dtype)

    def _resolve_model_kwargs(self) -> dict:
        model_kwargs: dict = {}
        dtype = self._resolve_dtype()
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if self.load_in_4bit or self.load_in_8bit:
            quant_config = self._build_quant_config()
            if quant_config is not None:
                model_kwargs["quantization_config"] = quant_config
                model_kwargs.setdefault("device_map", "auto")
        return model_kwargs

    def _build_quant_config(self) -> Optional[object]:
        try:
            import bitsandbytes  # noqa: F401
        except Exception:
            LOGGER.warning(
                "bitsandbytes not installed; ignoring 4-bit/8-bit quantization request."
            )
            return None
        try:
            from transformers import BitsAndBytesConfig
        except Exception:
            LOGGER.warning(
                "Transformers BitsAndBytesConfig unavailable; quantization disabled."
            )
            return None
        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
        )

    def _uses_quantization(self, model_kwargs: dict) -> bool:
        return bool(model_kwargs.get("quantization_config"))


def _build_portable_load_options(
    *,
    model_name: str,
    device: str,
    dtype: str,
    trust_remote_code: bool,
    use_fast_tokenizer: bool,
    load_in_4bit: bool,
    load_in_8bit: bool,
) -> dict[str, object]:
    options: dict[str, object] = {
        "loader": "transformers",
        "model_name": model_name,
        "device": (device or "auto").lower(),
        "dtype": (dtype or "auto").lower(),
        "trust_remote_code": bool(trust_remote_code),
        "use_fast_tokenizer": bool(use_fast_tokenizer),
        "load_in_4bit": bool(load_in_4bit),
        "load_in_8bit": bool(load_in_8bit),
    }
    if options["load_in_4bit"] or options["load_in_8bit"]:
        options["device_map"] = "auto"
    return options


def _apply_stop_sequences(text: str, stop: Iterable[str] | None) -> str:
    if not stop:
        return text
    truncated = text
    for token in stop:
        if not token:
            continue
        idx = truncated.find(token)
        if idx != -1:
            truncated = truncated[:idx]
    return truncated.strip()
