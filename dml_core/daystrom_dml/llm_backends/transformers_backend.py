"""Transformers backend for local generation."""
from __future__ import annotations

import logging
import re
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
        resolved_model_name, resolved_revision = _split_model_name_and_revision(
            (self.model_name or "").strip()
        )
        revision_kwargs: dict[str, object] = {}
        if resolved_revision is not None:
            revision_kwargs["revision"] = resolved_revision

        self._tokenizer = AutoTokenizer.from_pretrained(
            resolved_model_name,
            use_fast=self.use_fast_tokenizer,
            trust_remote_code=self.trust_remote_code,
            **revision_kwargs,
        )
        model_kwargs = dict(self._resolve_model_kwargs())
        LOGGER.info("Loading transformers model %s", resolved_model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            resolved_model_name,
            trust_remote_code=self.trust_remote_code,
            **revision_kwargs,
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
    if load_in_4bit and load_in_8bit:
        raise ValueError("load_in_4bit and load_in_8bit are mutually exclusive")

    normalized_model_name = (model_name or "").strip()
    if not normalized_model_name:
        raise ValueError("model_name is required")
    resolved_model_name, resolved_revision = _split_model_name_and_revision(
        normalized_model_name
    )

    options: dict[str, object] = {
        "loader": "transformers",
        "model_name": resolved_model_name,
        "device": _normalize_portable_device((device or "auto").strip().lower()),
        "dtype": _normalize_portable_dtype((dtype or "auto").strip().lower()),
        "trust_remote_code": bool(trust_remote_code),
        "use_fast_tokenizer": bool(use_fast_tokenizer),
        "load_in_4bit": bool(load_in_4bit),
        "load_in_8bit": bool(load_in_8bit),
    }
    if resolved_revision is not None:
        options["revision"] = resolved_revision
    if options["load_in_4bit"] or options["load_in_8bit"]:
        options["device_map"] = "auto"
    return options


def portable_to_torchforge_options(options: dict[str, object]) -> dict[str, object]:
    """Convert portable load options into a TorchForge-compatible load request.

    This keeps orchestration code backend-agnostic: loaders can emit the shared
    options contract once, then route to either Transformers or TorchForge.
    """
    loader = str(options.get("loader", "")).lower()
    if loader and loader != "transformers":
        raise ValueError(f"unsupported loader for portability bridge: {loader}")

    model_name = str(
        options.get("model_name")
        or options.get("model")
        or options.get("pretrained_model_name_or_path")
        or ""
    ).strip()
    if not model_name:
        raise ValueError("portable load options missing model_name")
    model, revision_from_model = _split_model_name_and_revision(model_name)
    revision_from_option = _resolve_revision_option(options)
    if (
        revision_from_model is not None
        and revision_from_option is not None
        and revision_from_model != revision_from_option
    ):
        raise ValueError("portable load options set conflicting model revision values")
    revision = revision_from_option or revision_from_model

    dtype_option = options.get("dtype")
    if dtype_option is None:
        dtype_option = options.get("torch_dtype")

    torchforge_options: dict[str, object] = {
        "model": model,
        "device": _normalize_portable_device(str(options.get("device") or "auto").strip().lower()),
        "dtype": _normalize_portable_dtype(str(dtype_option or "auto").strip().lower()),
        "trust_remote_code": _coerce_bool_option(
            options.get("trust_remote_code", False), option_name="trust_remote_code"
        ),
        "tokenizer_fast": _coerce_bool_option(
            _resolve_tokenizer_fast_option(options), option_name="use_fast_tokenizer"
        ),
    }
    if revision is not None:
        torchforge_options["revision"] = revision

    load_in_4bit = _coerce_bool_option(
        options.get("load_in_4bit", False), option_name="load_in_4bit"
    )
    load_in_8bit = _coerce_bool_option(
        options.get("load_in_8bit", False), option_name="load_in_8bit"
    )
    if load_in_4bit and load_in_8bit:
        raise ValueError("portable load options set both load_in_4bit and load_in_8bit")
    if load_in_4bit:
        torchforge_options["quantization"] = "4bit"
    elif load_in_8bit:
        torchforge_options["quantization"] = "8bit"

    if "device_map" in options and options.get("device_map") is not None:
        torchforge_options["device_map"] = options["device_map"]
    elif load_in_4bit or load_in_8bit:
        # Keep parity with transformers quantized loads, which rely on auto sharding.
        torchforge_options["device_map"] = "auto"

    if "local_files_only" in options:
        torchforge_options["local_files_only"] = _coerce_bool_option(
            options.get("local_files_only"), option_name="local_files_only"
        )

    for passthrough_key in (
        "cache_dir",
        "subfolder",
        "tokenizer_revision",
        "attn_implementation",
    ):
        normalized_value = _normalize_optional_string_option(options.get(passthrough_key))
        if normalized_value is not None:
            torchforge_options[passthrough_key] = normalized_value

    token = _resolve_auth_token_option(options)
    if token is not None:
        torchforge_options["token"] = token

    return torchforge_options


def _split_model_name_and_revision(model_name: str) -> tuple[str, str | None]:
    """Split HuggingFace-style model references (repo@revision)."""
    normalized_input = _normalize_hf_model_locator(model_name)
    if "@" not in normalized_input:
        return normalized_input, None

    model, revision = normalized_input.rsplit("@", 1)
    normalized_model = model.strip()
    normalized_revision = revision.strip()
    if not normalized_model:
        raise ValueError("portable load options missing model_name")
    if not normalized_revision:
        return normalized_model, None
    return normalized_model, normalized_revision


def _normalize_hf_model_locator(value: str) -> str:
    """Normalize common Hugging Face URI prefixes into repo ids."""
    normalized = (value or "").strip()
    for prefix in ("hf://", "huggingface://"):
        if normalized.lower().startswith(prefix):
            return normalized[len(prefix) :].lstrip("/")
    return normalized


def _normalize_optional_revision(value: object) -> str | None:
    if value is None:
        return None
    revision = str(value).strip()
    return revision or None


def _resolve_revision_option(options: dict[str, object]) -> str | None:
    revision = _normalize_optional_revision(options.get("revision"))
    model_revision = _normalize_optional_revision(options.get("model_revision"))
    if revision is not None and model_revision is not None and revision != model_revision:
        raise ValueError("portable load options set conflicting model revision values")
    return revision or model_revision


def _normalize_optional_string_option(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _resolve_tokenizer_fast_option(options: dict[str, object]) -> object:
    if "use_fast_tokenizer" in options:
        return options.get("use_fast_tokenizer")
    if "use_fast" in options:
        return options.get("use_fast")
    return True


def _resolve_auth_token_option(options: dict[str, object]) -> str | None:
    token = _normalize_optional_string_option(options.get("token"))
    if token is not None:
        return token
    return _normalize_optional_string_option(options.get("use_auth_token"))


def _coerce_bool_option(value: object, *, option_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    raise ValueError(f"portable load option {option_name} must be a boolean")


def _normalize_portable_dtype(dtype: str) -> str:
    aliases = {
        "fp16": "float16",
        "half": "float16",
        "torch.float16": "float16",
        "bf16": "bfloat16",
        "torch.bfloat16": "bfloat16",
        "fp32": "float32",
        "float": "float32",
        "torch.float": "float32",
        "torch.float32": "float32",
    }
    return aliases.get(dtype, dtype)


def _normalize_portable_device(device: str) -> str:
    aliases = {
        "gpu": "cuda",
        "cuda:0": "cuda",
        "cpu:0": "cpu",
    }
    normalized = aliases.get(device, device)
    if re.fullmatch(r"cuda:\d+", normalized):
        return "cuda"
    if re.fullmatch(r"cpu:\d+", normalized):
        return "cpu"
    return normalized


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
