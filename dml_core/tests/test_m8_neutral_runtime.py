"""Neutral M8 runtime controls with doubles only; never live model evidence."""

import dataclasses
import hashlib
import json
import sys
import types

import pytest

from daystrom_dml.contracts.agent_episode import (
    LLAMA3_SFT_V3_CONSUMER_PROFILE,
    QWEN3_SFT_V1_CONSUMER_PROFILE,
)
from daystrom_dml.contracts.model_input import ModelInputError
from daystrom_dml.services import llama3_sft_runtime
from daystrom_dml.services import m8_protocol
from daystrom_dml.services import m8_neutral_runtime as neutral
from daystrom_dml.services import qwen3_sft_runtime
from daystrom_dml.services import llama3_sft_v3_action_input as llama3_snapshot
from daystrom_dml.services import qwen3_sft_v1_action_input as qwen3_snapshot
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


def test_sampler_seed_defaults_to_pinned_zero_and_validates():
    runtime = neutral.NeutralM8AdapterRuntime(
        "model", "checkpoint", TokenizerDouble(), {"identity": "double"}, lambda: None, arm="dml"
    )
    assert runtime._seed == 0
    seeded = neutral.NeutralM8AdapterRuntime(
        "model", "checkpoint", TokenizerDouble(), {"identity": "double"}, lambda: None, arm="dml", seed=7
    )
    assert seeded._seed == 7
    for bad in (-1, True, "3", 1.5):
        with pytest.raises(ValueError, match="seed"):
            neutral.NeutralM8AdapterRuntime(
                "model", "checkpoint", TokenizerDouble(), {"identity": "double"}, lambda: None, arm="dml", seed=bad
            )


def test_neutral_module_never_binds_dml_policy():
    # The neutral runtime must not consult the M7 fixed system policy or the
    # original DML tool definitions; action authority comes from m8_protocol.
    assert not hasattr(neutral, "SYSTEM")
    assert m8_protocol.policy_identity("dml")["profile"] != "dml-llama3-sft"


# --- consumer-profile selection (llama3 default byte-identical, qwen3 pair) --

LLAMA3_PROFILE = LLAMA3_SFT_V3_CONSUMER_PROFILE
QWEN3_PROFILE = QWEN3_SFT_V1_CONSUMER_PROFILE


class Qwen3TokenizerDouble(TokenizerDouble):
    """Qwen3.5 control layout: non-special added rows (<think>) are controls."""

    def __init__(self, preset="", specials=(), nonspecial=()):
        super().__init__(preset, specials)
        for i in nonspecial:
            self.added_tokens_decoder[i] = types.SimpleNamespace(special=False)


def test_profile_registry_binds_qualified_module_pairs():
    assert neutral.DEFAULT_CONSUMER_PROFILE == LLAMA3_PROFILE
    assert set(neutral.PROFILES) == {LLAMA3_PROFILE, QWEN3_PROFILE}
    llama3 = neutral.PROFILES[LLAMA3_PROFILE]
    qwen3 = neutral.PROFILES[QWEN3_PROFILE]
    assert llama3.snapshot is llama3_snapshot and llama3.runtime is llama3_sft_runtime
    assert qwen3.snapshot is qwen3_snapshot and qwen3.runtime is qwen3_sft_runtime
    # Every llama3-specific pin differs across the pair and comes from the modules.
    assert llama3.runtime.STOPS == [128001, 128009] and llama3.runtime.VOCAB == 128256
    assert qwen3.runtime.STOPS == [248044, 248046] and qwen3.runtime.VOCAB == 248320
    assert qwen3.template_sha256 == qwen3_snapshot.VENDOR_TEMPLATE_DIGEST
    assert qwen3.template_sha256 != m8_protocol.PINNED_TEMPLATE_SHA256
    assert qwen3.template_kwargs == (("enable_thinking", False),)
    assert qwen3.hardware == ("595.99.02", "NVIDIA RTX PRO 5000 72GB Blackwell", (12, 0))
    assert qwen3.platform_machine == "x86_64"
    assert llama3.hardware == ("590.48.01", "NVIDIA GB300", (10, 3))
    assert llama3.platform_machine == "aarch64"


def test_default_identity_is_byte_identical_to_single_profile_behavior():
    base = {"manifest": "double"}
    default = neutral.neutral_identity("dml", base_identity=base)
    explicit = neutral.neutral_identity("dml", base_identity=base, profile=neutral.PROFILES[LLAMA3_PROFILE])
    assert default == explicit
    # Pinned against the pre-profile implementation's exact bytes.
    assert default["runtime_identity"] == (
        "dml-m8-neutral-action-runtime-v1:0c0ef0008a2b4d7021a8648eec86a36cf4340fe1f228a012115379f89253a0d3"
    )
    assert default["policy"]["protocol"]["vendor_template_sha256"] == m8_protocol.PINNED_TEMPLATE_SHA256


def test_qwen3_identity_uses_qwen3_template_digest_and_differs():
    base = {"manifest": "double"}
    qwen3 = neutral.neutral_identity("dml", base_identity=base, profile=neutral.PROFILES[QWEN3_PROFILE])
    assert qwen3["policy"]["protocol"]["vendor_template_sha256"] == qwen3_snapshot.VENDOR_TEMPLATE_DIGEST
    assert qwen3["policy"]["sampling"] == qwen3_snapshot.sampling_policy_identity()
    assert qwen3["runtime_identity"].startswith(neutral.RUNTIME_PREFIX)
    assert qwen3["runtime_identity"] != neutral.neutral_identity("dml", base_identity=base)["runtime_identity"]


def test_compile_grammar_qwen3_profile_uses_qwen3_vocab_and_stops(monkeypatch):
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

    assert neutral.compile_grammar(Qwen3TokenizerDouble(), "dml", runtime_module=qwen3_sft_runtime) == "grammar"
    assert captured["tokenizer"] == {"vocab_size": 248320, "stop_token_ids": [248044, 248046]}
    assert captured["schema"] == m8_protocol.schema_bytes("dml").decode()
    assert captured["compile"] == {"strict_mode": True, "any_whitespace": True, "max_whitespace_cnt": None}


def test_project_output_qwen3_profile_uses_qwen3_stops_and_control_rows():
    # Qwen3 turn terminator ends the stream; llama3 stops do not.
    ended = neutral.project_output(
        [10, 20, 248046], Qwen3TokenizerDouble(FINAL), "dml", runtime_module=qwen3_sft_runtime
    )
    assert ended["finish_reason"] == "eos" and ended["stop_token_id"] == 248046
    assert ended["derived_action"]["kind"] == "final"
    llama3_stop = neutral.project_output(
        [10, 20, 128009], Qwen3TokenizerDouble(FINAL), "dml", runtime_module=qwen3_sft_runtime
    )
    assert llama3_stop["finish_reason"] == "length" and llama3_stop["stop_token_id"] is None
    # Non-special added rows (<think> et al.) are output controls under qwen3.
    think = neutral.project_output(
        [5, 248068, 248046], Qwen3TokenizerDouble(FINAL, nonspecial={248068}), "dml", runtime_module=qwen3_sft_runtime
    )
    assert "control" in think["action_error"]
    # The qwen3 vocab ceiling rejects llama3-only and reserved rows.
    over = neutral.project_output(
        [248320, 248046], Qwen3TokenizerDouble(FINAL), "dml", runtime_module=qwen3_sft_runtime
    )
    assert over["derived_action"] is None and "Invalid token row" in over["action_error"]


def test_qwen3_adapter_runtime_class_binds_qwen3_base_and_controls():
    cls = neutral.Qwen3NeutralM8AdapterRuntime
    assert issubclass(cls, qwen3_sft_runtime.EnabledAdapterRuntime)
    assert not issubclass(cls, llama3_sft_runtime.EnabledAdapterRuntime)
    assert issubclass(neutral.NeutralM8AdapterRuntime, llama3_sft_runtime.EnabledAdapterRuntime)
    runtime = cls("model", "checkpoint", Qwen3TokenizerDouble(), {"identity": "double"}, lambda: None, arm="dml")
    assert runtime._seed == 0
    assert runtime._runtime_module is qwen3_sft_runtime
    with pytest.raises(ValueError, match="CompiledM8Request"):
        runtime.execute({"input_ids": [1]})
    with pytest.raises(ValueError, match="arm"):
        runtime.execute(compiled("persistent_rag"))
    with pytest.raises(ValueError, match="seed"):
        cls("model", "checkpoint", Qwen3TokenizerDouble(), {"identity": "double"}, lambda: None, arm="dml", seed=-1)
    with pytest.raises(ValueError, match="Unknown neutral runtime module"):
        neutral.neutral_adapter_runtime_class(types.ModuleType("bogus-runtime"))


def test_unknown_consumer_profile_fails_closed_before_filesystem_access(tmp_path):
    with pytest.raises(ModelInputError, match="Unknown M8 neutral consumer profile"):
        neutral.M8NeutralConsumer(tmp_path / "missing", arm="dml", offline=True, consumer_profile="bogus")
    with pytest.raises(ModelInputError, match="Unknown M8 neutral consumer profile"):
        neutral.M8NeutralConsumer(tmp_path / "missing", arm="dml", offline=True, consumer_profile=QWEN3_PROFILE + "-x")


class _AddedToken:
    def __init__(self, content, special):
        self.content = content
        self.special = special

    def __str__(self):
        return self.content


class Qwen3ConsumerTokenizerDouble:
    """Full tokenizer double admitting the qwen3 consumer's layout checks."""

    chat_template = "qwen3-neutral-template-double"

    def __init__(self):
        contents = {
            **qwen3_snapshot.SPECIAL_TOKENS,
            **qwen3_snapshot.CONFIG_ONLY_SPECIAL_TOKENS,
            **qwen3_snapshot.NONSPECIAL_ADDED_TOKENS,
        }
        specials = {**qwen3_snapshot.SPECIAL_TOKENS, **qwen3_snapshot.CONFIG_ONLY_SPECIAL_TOKENS}
        self.added_tokens_decoder = {i: _AddedToken(v, i in specials) for i, v in contents.items()}
        self._by_content = {v: k for k, v in contents.items()}
        self.backend_tokenizer = types.SimpleNamespace(
            no_truncation=lambda: None, no_padding=lambda: None, to_str=lambda: "qwen3-double-backend"
        )
        self.template_kwargs_seen = None

    def __len__(self):
        return qwen3_sft_runtime.TOKENIZER_ROWS

    def get_vocab(self):
        return {f"tok{i}": i for i in range(qwen3_sft_runtime.TOKENIZER_ROWS)}

    def convert_tokens_to_ids(self, value):
        return self._by_content[value]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        self.template_kwargs_seen = kwargs
        return "".join("[" + m["role"] + "]" + m["content"] for m in messages) + "[assistant]"

    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, ids, **kwargs):
        return "".join(chr(i) if i < 256 else "?" for i in ids)


def _qwen3_double_snapshot(tmp_path):
    """Snapshot directory carrying the qwen3 double tokenizer metadata files."""
    model = tmp_path / "model"
    model.mkdir()
    special = [{"id": i, "content": v, "special": True} for i, v in qwen3_snapshot.SPECIAL_TOKENS.items()]
    nonspecial = [{"id": i, "content": v, "special": False} for i, v in qwen3_snapshot.NONSPECIAL_ADDED_TOKENS.items()]
    (model / "tokenizer.json").write_text(json.dumps({"added_tokens": special + nonspecial}))
    configured = {**qwen3_snapshot.SPECIAL_TOKENS, **qwen3_snapshot.CONFIG_ONLY_SPECIAL_TOKENS}
    (model / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "added_tokens_decoder": {str(i): {"content": v, "special": True} for i, v in configured.items()}
                | {str(i): {"content": v, "special": False} for i, v in qwen3_snapshot.NONSPECIAL_ADDED_TOKENS.items()}
            }
        )
    )
    (tmp_path / qwen3_snapshot.MANIFEST_NAME).write_text("{}")
    files = {f"model/{name}": model / name for name in ("tokenizer.json", "tokenizer_config.json")}
    files[qwen3_snapshot.MANIFEST_NAME] = tmp_path / qwen3_snapshot.MANIFEST_NAME
    return files, model


def test_qwen3_profiled_consumer_builds_against_qwen3_doubles(tmp_path, monkeypatch):
    files, model_path = _qwen3_double_snapshot(tmp_path)
    manifest = {"runtime": {"required_environment": {}}}
    monkeypatch.setattr(
        qwen3_snapshot,
        "verify_manifest",
        lambda directory: (tmp_path, manifest, files, model_path, tmp_path / "adapter"),
    )
    monkeypatch.setattr(qwen3_snapshot, "sha", lambda path: "f" * 64)
    monkeypatch.setattr(
        qwen3_snapshot,
        "identity_for_manifest",
        lambda m: types.SimpleNamespace(
            model_digest="1" * 64,
            tokenizer_digest="2" * 64,
            chat_template_digest="3" * 64,
            model_window_tokens=qwen3_sft_runtime.WINDOW,
        ),
    )
    # The double template cannot match the real vendor digest; admit it through
    # the profile entry so every other qwen3 pin is still exercised.
    template_sha256 = hashlib.sha256(Qwen3ConsumerTokenizerDouble.chat_template.encode()).hexdigest()
    monkeypatch.setitem(
        neutral.PROFILES,
        QWEN3_PROFILE,
        dataclasses.replace(neutral.PROFILES[QWEN3_PROFILE], template_sha256=template_sha256),
    )
    tokenizer = Qwen3ConsumerTokenizerDouble()
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    consumer = neutral.M8NeutralConsumer(tmp_path, arm="context_only", offline=True, consumer_profile=QWEN3_PROFILE)
    assert consumer.consumer_profile == QWEN3_PROFILE
    assert consumer._runtime_class is neutral.Qwen3NeutralM8AdapterRuntime
    identity = consumer.identity
    assert identity["policy"]["protocol"]["vendor_template_sha256"] == template_sha256
    assert identity["policy"]["base_identity"]["snapshot_model"] == qwen3_snapshot.MODEL
    assert identity["policy"]["base_identity"]["snapshot_revision"] == qwen3_snapshot.REVISION
    assert identity["policy"]["base_identity"]["adapter"] == qwen3_snapshot.CHECKPOINT
    llama3_identity = neutral.neutral_identity("context_only", base_identity=identity["policy"]["base_identity"])
    assert identity["runtime_identity"] != llama3_identity["runtime_identity"]

    # Compile goes through the qwen3 window and disables thinking at the boundary.
    scope = {"tenant_id": "t", "client_id": None, "session_id": "s", "instance_id": None}
    record = {
        "record_id": 11,
        "text": "Project status is green.",
        "scope": dict(scope),
        "source_id": "bulletin",
        "fact_key": "status",
        "version": 1,
        "timestamp": 1000,
        "corrects": None,
        "metadata": {},
    }
    messages = m8_protocol.initial_messages(
        "context_only", "What is the project status?", records=[record], scope=scope
    )
    request = consumer.compile(messages, scope=scope)
    assert request.model_window_tokens == qwen3_sft_runtime.WINDOW == 262144
    assert tokenizer.template_kwargs_seen == {"enable_thinking": False}
    consumer.close()
    with pytest.raises(ModelInputError, match="closed"):
        consumer.compile(messages, scope=scope)
