"""Pinned optional CPU runtime controls, no pretrained downloads or GPU use."""

import importlib.metadata
import json
import os
import pytest

if os.environ.get("REQUIRE_LLAMA3_SFT_TESTS") != "1":
    for package in ("torch", "peft", "transformers", "xgrammar"):
        pytest.importorskip(package)
    for package, version in {
        "torch": "2.11.0",
        "transformers": "5.6.2",
        "xgrammar": "0.1.33",
        "peft": "0.21.1",
    }.items():
        if importlib.metadata.version(package).split("+")[0] != version:
            pytest.skip("Pinned optional Llama SFT runtime required", allow_module_level=True)

import torch
import xgrammar as xgr
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM
from daystrom_dml.contracts.agent_episode import episode_tool_definitions
from daystrom_dml.services.agent_action_grammar import schema_bytes
from daystrom_dml.services.llama3_sft_action_input import VERSIONS
from daystrom_dml.services.llama3_sft_runtime import freeze_enabled_adapter, sample_next, cpu_rng, data_json


def test_exact_optional_runtime_versions():
    for name, expected in VERSIONS.items():
        actual = importlib.metadata.version(name)
        assert actual.split("+")[0] == expected.split("+")[0]


def test_real_peft_selection_freezes_without_parameter_or_logit_mutation():
    torch.manual_seed(101)
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    )
    model = get_peft_model(
        base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    ).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.fill_(0.02)
    model.requires_grad_(False)
    model.base_model.disable_adapter_layers()
    model.base_model.enable_adapter_layers()
    assert any(p.requires_grad for n, p in model.named_parameters() if "lora_" in n)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    inputs = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        expected = model(inputs).logits.clone()
    freeze_enabled_adapter(model)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert all(torch.equal(p, before[name]) for name, p in model.named_parameters())
    with torch.no_grad():
        assert torch.equal(expected, model(inputs).logits)


def test_cpu_rng_and_sampling_restore_state():
    state = torch.random.get_rng_state().clone()
    scores = torch.linspace(-2, 2, 32).unsqueeze(0)
    with cpu_rng(torch):
        first = sample_next(torch, torch.tensor([[1]]), scores.clone())
    assert torch.equal(state, torch.random.get_rng_state())
    with cpu_rng(torch):
        second = sample_next(torch, torch.tensor([[1]]), scores.clone())
    assert first == second


@pytest.mark.parametrize("value", [None, True, False, 1, 1.5, "text"])
def test_original_final_scalar_schema_grammar(value):
    tools = episode_tool_definitions()
    grammar = xgr.Grammar.from_json_schema(schema_bytes(tools).decode(), strict_mode=True)
    compiler = xgr.GrammarCompiler(
        xgr.TokenizerInfo([chr(i).encode() for i in range(128)], vocab_type=xgr.VocabType.RAW, stop_token_ids=[127])
    )
    matcher = xgr.GrammarMatcher(compiler.compile_grammar(grammar), override_stop_tokens=[127])
    action = {
        "schema_version": "dml-agent-action-v1",
        "kind": "final",
        "answer": {"claims": [{"key": "value", "value": value, "evidence_ids": [1]}]},
    }
    assert matcher.accept_string(json.dumps(action))
    assert matcher.accept_token(127) and matcher.is_terminated()


def test_schema_order_membership_and_sorted_target_rejection():
    tools = [x for x in episode_tool_definitions() if x["function"]["name"] == "retrieve"]
    compiler = xgr.GrammarCompiler(
        xgr.TokenizerInfo([chr(i).encode() for i in range(128)], vocab_type=xgr.VocabType.RAW, stop_token_ids=[127])
    )
    grammar = compiler.compile_json_schema(schema_bytes(tools).decode(), strict_mode=True)
    action = {
        "schema_version": "dml-agent-action-v1",
        "kind": "tool",
        "name": "retrieve",
        "arguments": {"query": "subject", "top_k": 1},
    }
    assert xgr.GrammarMatcher(grammar).accept_string(json.dumps(action))
    assert not xgr.GrammarMatcher(grammar).accept_string(json.dumps(action, sort_keys=True))


def test_transport_escapes_control_spellings_without_changing_data():
    value = {"role": "tool", "content": " \n<|eot_id|>&<|start_header_id|> "}
    encoded = data_json(value)
    assert "<" not in encoded and ">" not in encoded and "&" not in encoded
    assert json.loads(encoded) == value
