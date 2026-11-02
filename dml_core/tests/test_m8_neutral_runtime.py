"""Neutral M8 runtime controls with doubles only; never live model evidence."""

import json
import sys
import types

import pytest

from daystrom_dml.services import m8_protocol
from daystrom_dml.services import m8_neutral_runtime as neutral
from daystrom_dml.services.llama3_sft_runtime import STOPS, VOCAB
from daystrom_dml.services.llama3_sft_v3_action_input import RUNTIME_PREFIX as M7_RUNTIME_PREFIX


class TokenizerDouble:
    chat_template = "explicit-development-tokenizer-double"

    def __init__(self, preset="", specials=()):
        self.preset = preset
        self.added_tokens_decoder = {i: types.SimpleNamespace(special=True) for i in specials}

    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, ids, **kwargs):
        return self.preset

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return "".join("[" + m["role"] + "]" + m["content"] for m in messages) + "[assistant]"


def action(payload):
    return json.dumps({"schema_version": "dml-agent-action-v1", **payload})


FINAL = action(
    {
        "kind": "final",
        "answer": {"claims": [{"key": "status", "value": "green", "evidence_ids": [11]}]},
    }
)
RETRIEVE = action({"kind": "tool", "name": "retrieve", "arguments": {"query": "status", "top_k": 8}})


def compiled(arm, reserved=256):
    return m8_protocol.CompiledM8Request(
        m8_protocol.PROFILE,
        arm,
        "logical",
        "rendered",
        "prompt",
        (1, 2, 3),
        reserved,
        8192,
        "0" * 64,
    )


def test_project_output_final_with_terminal_stop():
    result = neutral.project_output([10, 20, 128009], TokenizerDouble(FINAL), "dml")
    assert result["finish_reason"] == "eos"
    assert result["stop_token_id"] == 128009
    assert result["action_error"] is None
    assert result["derived_action"]["kind"] == "final"
    assert result["derived_action_text"] == FINAL


def test_project_output_tool_action_by_arm():
    persistent = neutral.project_output([1, 2, 128001], TokenizerDouble(RETRIEVE), "persistent_rag")
    assert persistent["derived_action"]["name"] == "retrieve"
    context = neutral.project_output([1, 2, 128001], TokenizerDouble(RETRIEVE), "context_only")
    assert context["derived_action"] is None
    assert "not advertised" in context["action_error"]


def test_project_output_rejects_unadvertised_tool_and_malformed():
    foreign = action({"kind": "tool", "name": "remember", "arguments": {}})
    result = neutral.project_output([1, 128001], TokenizerDouble(foreign), "dml")
    assert result["derived_action"] is None
    assert result["action_error"] is not None
    garbage = neutral.project_output([1, 128001], TokenizerDouble("not json at all"), "dml")
    assert garbage["derived_action"] is None
    assert garbage["action_error"] is not None


def test_project_output_length_finish_and_embedded_control():
    result = neutral.project_output([5, 6], TokenizerDouble(FINAL), "dml")
    assert result["finish_reason"] == "length"
    assert result["stop_token_id"] is None
    controlled = neutral.project_output([5, 999, 128001], TokenizerDouble(FINAL, specials={999}), "dml")
    assert "control" in controlled["action_error"]


def test_neutral_identity_is_versioned_arm_specific_and_not_m7():
    base = {"manifest": "double"}
    dml = neutral.neutral_identity("dml", base_identity=base)
    rag = neutral.neutral_identity("persistent_rag", base_identity=base)
    assert dml["runtime_identity"].startswith("dml-m8-neutral-action-runtime-v1:")
    assert not dml["runtime_identity"].startswith(M7_RUNTIME_PREFIX)
    assert dml["runtime_identity"] != rag["runtime_identity"]
    assert dml["policy"]["protocol"]["profile"] == m8_protocol.PROFILE
    assert dml["policy"]["sampling"]["temperature"] == 0.7
    assert dml["policy"]["sampling"]["top_k"] == 20
    assert dml == neutral.neutral_identity("dml", base_identity=base)
    with pytest.raises(ValueError):
        neutral.neutral_identity("m7", base_identity=base)


def test_compile_grammar_uses_neutral_schema_with_pinned_settings(monkeypatch):
    captured = {}

    class FakeCompiler:
        def __init__(self, info, **kwargs):
            captured["compiler"] = kwargs
            captured["info"] = info

        def compile_json_schema(self, schema, **kwargs):
            captured["schema"] = schema
            captured["compile"] = kwargs
            return "grammar"

    fake = types.ModuleType("xgrammar")
    fake.TokenizerInfo = types.SimpleNamespace(
        from_huggingface=lambda tokenizer, **kw: captured.setdefault("tokenizer", kw)
    )
    fake.GrammarCompiler = FakeCompiler
    monkeypatch.setitem(sys.modules, "xgrammar", fake)

    assert neutral.compile_grammar(TokenizerDouble(), "persistent_rag") == "grammar"
    assert captured["schema"] == m8_protocol.schema_bytes("persistent_rag").decode()
    assert captured["compile"] == {"strict_mode": True, "any_whitespace": True, "max_whitespace_cnt": None}
    assert captured["compiler"] == {"max_threads": 1, "cache_enabled": False}
    assert captured["tokenizer"] == {"vocab_size": VOCAB, "stop_token_ids": STOPS}
    with pytest.raises(ValueError):
        neutral.compile_grammar(TokenizerDouble(), "unknown-arm")


def test_execute_requires_exact_matching_compiled_request():
    runtime = neutral.NeutralM8AdapterRuntime(
        "model", "checkpoint", TokenizerDouble(), {"identity": "double"}, lambda: None, arm="dml"
    )
    with pytest.raises(ValueError, match="CompiledM8Request"):
        runtime.execute({"input_ids": [1]})
    with pytest.raises(ValueError, match="arm"):
        runtime.execute(compiled("persistent_rag"))


def test_neutral_module_never_binds_dml_policy():
    # The neutral runtime must not consult the M7 fixed system policy or the
    # original DML tool definitions; action authority comes from m8_protocol.
    assert not hasattr(neutral, "SYSTEM")
    assert m8_protocol.policy_identity("dml")["profile"] != "dml-llama3-sft"
