"""Tiny two-shard BF16 Qwen3; invented weights prove mechanics, not quality."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from model_input_fixture import TinySnapshot, require_model_input_dependencies


def create_qwen3_snapshot(tmp_path: Path, *, context_window: int = 4096, seed: int = 20260925) -> TinySnapshot:
    versions = require_model_input_dependencies()
    import torch
    from safetensors.torch import save_file
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    from daystrom_dml.services.qwen3_model_snapshot import (
        QWEN3_CHAT_TEMPLATE, REQUIRED_FILES, SHARD_FILES, SNAPSHOT_SCHEMA_VERSION, SPECIAL_TOKENS,
    )

    path = Path(tmp_path)
    path.mkdir(parents=True, exist_ok=True)
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=384, min_frequency=1,
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    backend.train_from_iterator([
        "system user assistant tool content tools name function arguments tool_calls tool_call_id",
        "Read the green notebook. Preserve café memory and every Unicode 雨 field.",
        '{"type":"function","function":{"name":"lookup_note","arguments":{"subject":"green"}}}',
        "<|im_start|> <|im_end|> <|endoftext|> \\u003c & \\\" </script> \n",
    ], trainer=trainer)
    backend.add_tokens([AddedToken(token, special=False, normalized=False) for token in ("<think>", "</think>")])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, model_max_length=context_window, **SPECIAL_TOKENS)
    tokenizer.chat_template = QWEN3_CHAT_TEMPLATE
    config_data = {"architectures": ["Qwen3ForCausalLM"], "attention_bias": False, "attention_dropout": 0.0,
        "bos_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
        "head_dim": 8, "hidden_act": "silu", "hidden_size": 16, "initializer_range": 0.02,
        "intermediate_size": 32, "max_position_embeddings": max(context_window, 4096),
        "max_window_layers": 2, "model_type": "qwen3", "num_attention_heads": 2, "num_hidden_layers": 2,
        "num_key_value_heads": 1, "rms_norm_eps": 1e-6, "rope_scaling": None, "rope_theta": 10000,
        "sliding_window": None, "tie_word_embeddings": True, "torch_dtype": "bfloat16",
        "transformers_version": versions["transformers"], "use_cache": True, "use_sliding_window": False,
        "vocab_size": len(tokenizer) + 8}
    config = Qwen3Config.from_dict(dict(config_data))
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.manual_seed(seed)
        model = Qwen3ForCausalLM(config).to(device="cpu", dtype=torch.bfloat16)
    model.model.rotary_emb = Qwen3RotaryEmbedding(config, device=torch.device("cpu"))
    model.eval()
    state = {name: value.detach().clone().contiguous() for name, value in model.state_dict().items()}
    mapping = {name: SHARD_FILES[1] if name == "lm_head.weight" else SHARD_FILES[0] for name in state}
    for shard in SHARD_FILES:
        save_file({name: value for name, value in state.items() if mapping[name] == shard}, str(path / shard))
    index = {"metadata": {"total_size": sum(value.numel() * value.element_size() for value in state.values())},
             "weight_map": mapping}
    (path / "model.safetensors.index.json").write_text(json.dumps(index, sort_keys=True))
    (path / "config.json").write_text(json.dumps(config_data, sort_keys=True))
    tokenizer.backend_tokenizer.save(str(path / "tokenizer.json"))
    (path / "chat_template.jinja").write_text(QWEN3_CHAT_TEMPLATE)
    manifest = {"schema_version": SNAPSHOT_SCHEMA_VERSION, "model_id": "dml-test-random-qwen3",
        "model_revision": f"deterministic-random-seed-{seed}-v1", "context_window": context_window,
        "runtime_versions": versions, "special_tokens": SPECIAL_TOKENS,
        "files": {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in REQUIRED_FILES}}
    (path / "snapshot.json").write_text(json.dumps(manifest, sort_keys=True))
    return TinySnapshot(path=path, tokenizer=tokenizer, model=model, manifest=manifest)
