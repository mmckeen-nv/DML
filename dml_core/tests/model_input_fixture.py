"""Generate a real, deterministic CPU model fixture without downloading weights.

Importing this helper is dependency-light. Optional dependency admission happens
before importing torch or transformers, including in environments with an
unqualified embedding extra installed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
from typing import Any

import pytest


REQUIRE_ENV = "REQUIRE_MODEL_INPUT_TESTS"
DEPENDENCY_PINS = {
    "torch": "2.8.0+cpu",
    "transformers": "4.56.2",
    "tokenizers": "0.22.0",
    "safetensors": "0.6.2",
    "jinja2": "3.1.6",
}


def require_model_input_dependencies() -> dict[str, str]:
    """Require exact real-runtime pins, failing the mandatory lane on mismatch."""
    actual = {}
    problems = []
    for name, expected in DEPENDENCY_PINS.items():
        try:
            found = metadata.version(name)
        except metadata.PackageNotFoundError:
            found = "missing"
        actual[name] = found
        if found != expected:
            problems.append(f"{name}={found} (required {expected})")
    if problems:
        message = "Real model-input fixture requires pinned optional dependencies: " + "; ".join(problems)
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail(message, pytrace=False)
        pytest.skip(message)
    return actual


@dataclass(frozen=True)
class TinySnapshot:
    path: Path
    tokenizer: Any
    model: Any
    manifest: dict


def create_snapshot(tmp_path: Path, *, context_window: int = 256, seed: int = 20260918) -> TinySnapshot:
    """Write the five-file snapshot consumed by the production compiler.

    The returned tokenizer and model are the independent fixture originals;
    consumers load their own verified copies from ``path``.
    """
    versions = require_model_input_dependencies()
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from daystrom_dml.contracts.model_input import SUPPORTED_CHAT_TEMPLATE

    path = Path(tmp_path)
    path.mkdir(parents=True, exist_ok=True)
    special = {
        "bos_token": "<|bos|>", "eos_token": "<|eos|>",
        "unk_token": "<|unk|>", "pad_token": "<|pad|>",
    }
    backend = Tokenizer(models.BPE(unk_token=special["unk_token"]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=384, min_frequency=1, special_tokens=list(special.values()),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
    )
    backend.train_from_iterator([
        "system user assistant tool content name function arguments tool_calls tool_call_id",
        "The owner prefers green notebooks. Use verified memory and current instructions.",
        "Read the notebook preference. Earlier notes preferred blue. Correct the previous summary.",
        '{"type":"function","function":{"name":"lookup_note","description":"Read an owner note",'
        '"parameters":{"type":"object","properties":{"subject":{"type":"string"}},"required":["subject"]}}}',
        "[INST] <<SYS>> </s> <s> <|im_start|> <|im_end|> \n === Retrieved context === \n",
        "A tool result can contain punctuation: [] {} : , = / \\\n Unicode café notes are exact.",
    ], trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, model_max_length=context_window, **special,
    )
    tokenizer.chat_template = SUPPORTED_CHAT_TEMPLATE
    config = GPT2Config(
        architectures=["GPT2LMHeadModel"],
        vocab_size=len(tokenizer), n_positions=context_window, n_ctx=context_window,
        n_embd=16, n_layer=1, n_head=2, n_inner=32,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, use_cache=False,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = GPT2LMHeadModel(config).to(device="cpu", dtype=torch.float32)
    model.eval()
    (path / "config.json").write_text(config.to_json_string(), encoding="utf-8")
    save_file({name: tensor.detach().cpu().clone().contiguous()
               for name, tensor in model.state_dict().items()}, str(path / "model.safetensors"))
    tokenizer.backend_tokenizer.save(str(path / "tokenizer.json"))
    (path / "chat_template.jinja").write_bytes(SUPPORTED_CHAT_TEMPLATE.encode("utf-8"))
    filenames = ("config.json", "model.safetensors", "tokenizer.json", "chat_template.jinja")
    manifest = {
        "schema_version": "dml-model-snapshot-v1",
        "model_id": "dml-integration-tiny-gpt2",
        "model_revision": f"deterministic-seed-{seed}-v1",
        "context_window": context_window,
        "runtime_versions": versions,
        "special_tokens": special,
        "files": {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in filenames},
    }
    (path / "snapshot.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False), encoding="utf-8",
    )
    return TinySnapshot(path=path, tokenizer=tokenizer, model=model, manifest=manifest)
