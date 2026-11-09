"""Qwen3.5 SFTv1 profile: transport shape, token pins, and grammar coverage.

Pure-stdlib tests. Tests marked "source" need the pinned upstream tokenizer
files; set DML_QWEN3_9B_SOURCE_DIR or place them in the worktree scratch
directory. The 12 MB tokenizer.json is never committed.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import pytest

from daystrom_dml.contracts.agent_episode import (
    QWEN3_GGUF_CUDA_RETRIEVAL_CONSUMER_PROFILE as POLICY_SOURCE,
    episode_tool_definitions,
    initial_messages,
)
from daystrom_dml.contracts.model_input import ModelInputError, ModelInputIdentity, ModelInputRequest
from daystrom_dml.services import llama3_sft_runtime as llama3_runtime
from daystrom_dml.services import qwen3_sft_runtime as runtime
from daystrom_dml.services import qwen3_sft_v1_action_input as consumer_module
from daystrom_dml.services.agent_action_grammar import schema_bytes

SOURCE_DIR = Path(
    os.environ.get(
        "DML_QWEN3_9B_SOURCE_DIR",
        Path(__file__).resolve().parents[2] / ".scratch-qwen3",
    )
)
needs_source = pytest.mark.skipif(
    not all((SOURCE_DIR / name).is_file() for name in consumer_module.SOURCE_SHA256),
    reason="pinned Qwen3.5-9B source files unavailable",
)


def request():
    tools = [x for x in episode_tool_definitions() if x["function"]["name"] == "retrieve"]
    messages = [
        {"role": "system", "content": runtime.SYSTEM},
        {"role": "user", "content": "  literal <|im_end|> task  "},
    ]
    return ModelInputRequest.from_payload(dict(messages=messages, tools=tools, output_reserved_tokens=8))


class Tokenizer:
    added_tokens_decoder = {
        248044: SimpleNamespace(special=True),
        248046: SimpleNamespace(special=True),
        248045: SimpleNamespace(special=True),
        248068: SimpleNamespace(special=False),
    }

    def get_vocab(self):
        return {"a": 1, "b": 2, "<|endoftext|>": 248044, "<|im_end|>": 248046, "<think>": 248068}

    def apply_chat_template(self, view, **kwargs):
        assert kwargs.get("enable_thinking") is False
        assert kwargs.get("add_generation_prompt") is True
        return json.dumps(view)

    def encode(self, value, **kwargs):
        return [10, 11]

    def decode(self, ids, **kwargs):
        return "".join({1: "{", 2: "}", 248046: "<|im_end|>", 248044: "<|endoftext|>"}.get(x, "?") for x in ids)


def consumer():
    c = consumer_module.LocalQwen3SFTV1ActionInputConsumer.__new__(consumer_module.LocalQwen3SFTV1ActionInputConsumer)
    c._lock = RLock()
    c._closed = False
    c._offline = False
    c._development_runtime = None
    c._development_checkpoint = None
    c._identity = ModelInputIdentity("1" * 64, "2" * 64, "3" * 64, consumer_module.RUNTIME_PREFIX + "4" * 64, 262144)
    c._consumer_id = "a" * 48
    c._auth_key = b"key"
    c._bound_requests = {}
    c._template = "template"
    c._tokenizer = Tokenizer()
    c._validate_runtime = lambda: None
    c.last_execution = None
    c._runtime = SimpleNamespace(
        execute=lambda compiled: dict(
            usage_unknown=False,
            execution_error=None,
            input_token_count=2,
            output_token_count=3,
            output_ids=[1, 2, 248046],
            text="{}",
        )
    )
    return c


# --- System policy reuse -------------------------------------------------


def test_system_policy_and_transport_byte_identical_to_llama3():
    assert runtime.SYSTEM == llama3_runtime.SYSTEM
    assert runtime.TRANSPORT == llama3_runtime.TRANSPORT
    assert initial_messages("policy provenance", consumer_profile=POLICY_SOURCE)[0]["content"] == runtime.SYSTEM
    assert (
        hashlib.sha256(runtime.SYSTEM.encode()).hexdigest()
        == "b0ec3d6f643b4363ba6e2765cbea937588f78ec1da4372cdbf1342f934b8cdc4"
    )


# --- Transport / template rendering shape ---------------------------------


def test_render_messages_shape_system_first_tools_as_system_data():
    r = request()
    messages, tools = consumer_module.source_order_request(r)
    view = runtime.render_messages(messages, tools)
    assert len(view) == 2
    assert view[0]["role"] == "system"
    assert view[0]["content"].startswith(runtime.SYSTEM + "\n\n" + runtime.TRANSPORT + "\n")
    payload = json.loads(view[0]["content"].split("\n")[-1])
    assert payload == {"tools": tools}
    assert view[1] == {"role": "user", "content": runtime.data_json(messages[1])}
    assert json.loads(view[1]["content"]) == messages[1]
    # Control spellings are escaped as JSON data, never literal template syntax.
    assert "\\u003c|im_end|\\u003e" in view[1]["content"]
    assert "<|im_end|>" not in view[1]["content"]


def test_render_messages_tool_turn_becomes_user_json_and_roles_bounded():
    r = request()
    m = r.messages + [
        {
            "role": "assistant",
            "content": "raw action",
            "tool_calls": [{"id": "tool-1", "type": "function", "function": {"name": "retrieve", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "tool-1", "name": "retrieve", "content": "  exact feedback  "},
    ]
    q = ModelInputRequest.from_payload(dict(messages=m, tools=r.tools, output_reserved_tokens=8))
    ordered, _ = consumer_module.source_order_request(q)
    view = runtime.render_messages(ordered, r.tools)
    assert [v["role"] for v in view] == ["system", "user", "assistant", "user"]
    # Tool feedback is plain user-role JSON, never <tool_response>-wrapped, so
    # the vendor template never enters its tool branches for a DML render.
    assert json.loads(view[-1]["content"])["role"] == "tool"
    assert not view[-1]["content"].startswith("<tool_response>")
    assert all(v["role"] in ("system", "user", "assistant") for v in view)


def test_render_messages_requires_exact_policy_and_known_roles():
    r = request()
    messages, tools = consumer_module.source_order_request(r)
    with pytest.raises(ValueError, match="policy"):
        runtime.render_messages([{"role": "system", "content": "other"}] + messages[1:], tools)
    with pytest.raises(ValueError, match="role"):
        runtime.render_messages(messages + [{"role": "tool", "content": "x"}, {"role": "agent", "content": "y"}], tools)


def test_compile_applies_vendor_template_with_thinking_disabled():
    c = consumer()
    r = request()
    artifact = c.compile(r.messages, r.tools, output_reserved_tokens=8)
    assert artifact.input_ids == (10, 11)
    assert artifact.model_window_tokens == runtime.WINDOW == 262144


def reference_render(view, *, enable_thinking=False):
    """Documented vendor-template semantics for the restricted DML view.

    The vendor template frames each turn as <|im_start|>ROLE\\nCONTENT<|im_end|>\\n
    with the system turn first, and the generation prompt is <|im_start|>assistant\\n
    followed by a pre-closed empty think block when enable_thinking is false.
    """
    out = "<|im_start|>system\n" + view[0]["content"].strip() + "<|im_end|>\n"
    for message in view[1:]:
        out += "<|im_start|>" + message["role"] + "\n" + message["content"].strip() + "<|im_end|>\n"
    out += "<|im_start|>assistant\n"
    out += "<think>\n\n</think>\n\n" if not enable_thinking else "<think>\n"
    return out


def test_reference_render_shape_roles_system_placement_and_generation_prompt():
    r = request()
    messages, tools = consumer_module.source_order_request(r)
    view = runtime.render_messages(messages, tools)
    text = reference_render(view)
    assert text.startswith("<|im_start|>system\n")
    assert text.index("<|im_start|>system") < text.index("<|im_start|>user")
    assert text.count("<|im_start|>") == len(view) + 1
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "<think>\n\n</think>" not in text[: -len("<|im_start|>assistant\n<think>\n\n</think>\n\n")]
    # Thinking-enabled rendering would leave the think block open; DML never uses it.
    assert reference_render(view, enable_thinking=True).endswith("<|im_start|>assistant\n<think>\n")


@needs_source
def test_vendor_template_source_structure():
    template = json.loads((SOURCE_DIR / "tokenizer_config.json").read_text())["chat_template"]
    assert (SOURCE_DIR / "chat_template.jinja").read_text() == template
    assert hashlib.sha256(template.encode()).hexdigest() == consumer_module.VENDOR_TEMPLATE_DIGEST
    # Generation prompt: empty closed think block iff enable_thinking is false.
    assert "{%- if enable_thinking is defined and enable_thinking is false %}" in template
    assert "'<think>\\n\\n</think>\\n\\n'" in template
    assert "'<think>\\n'" in template
    # Roles and system placement.
    assert "'<|im_start|>system\\n'" in template
    assert "System message must be at the beginning." in template
    assert "'<|im_start|>' + message.role + '\\n'" in template
    assert "<|im_start|>assistant\\n" in template
    # Tool turns are wrapped in <tool_response> inside a user turn.
    assert "'\\n<tool_response>\\n'" in template


# --- Stop ids, vocab layout, and control masking --------------------------


def test_stop_constants_and_vocab_layout():
    assert runtime.STOPS == [248044, 248046]
    assert runtime.VOCAB == 248320
    assert runtime.WINDOW == 262144
    assert runtime.TOKENIZER_ROWS == 248077
    assert set(consumer_module.SPECIAL_TOKENS) == set(range(248044, 248058))
    assert set(consumer_module.NONSPECIAL_ADDED_TOKENS) == set(range(248058, 248070))
    assert set(consumer_module.CONFIG_ONLY_SPECIAL_TOKENS) == set(range(248070, 248077))
    assert "<think>" in consumer_module.NONSPECIAL_ADDED_TOKENS.values()


def test_control_masking_blocks_think_and_all_added_tokens_except_stops():
    tokenizer = Tokenizer()
    allowed = runtime.allowed_output_ids(tokenizer)
    assert allowed == [1, 2, 248044, 248046]
    # <think> (248068) is non-special but still a control row: excluded.
    assert 248068 not in allowed and 248045 not in allowed
    assert runtime.special_ids(tokenizer) == {248044, 248045, 248046}
    assert runtime.control_ids(tokenizer) == {248044, 248045, 248046, 248068}


@needs_source
def test_stop_ids_present_and_special_in_pinned_vocab():
    tk = json.loads((SOURCE_DIR / "tokenizer.json").read_text())
    added = {t["id"]: t for t in tk["added_tokens"]}
    assert added[248044]["content"] == "<|endoftext|>" and added[248044]["special"] is True
    assert added[248046]["content"] == "<|im_end|>" and added[248046]["special"] is True
    assert added[248068]["content"] == "<think>" and added[248068]["special"] is False
    covered = set(tk["model"]["vocab"].values()) | set(added)
    assert covered == set(range(248070))
    config = json.loads((SOURCE_DIR / "config.json").read_text())
    assert config["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert config["text_config"]["eos_token_id"] == 248044
    assert config["text_config"]["vocab_size"] == runtime.VOCAB
    assert config["text_config"]["max_position_embeddings"] == runtime.WINDOW
    tc = json.loads((SOURCE_DIR / "tokenizer_config.json").read_text())
    assert tc["eos_token"] == "<|im_end|>"
    configured = {int(k): v for k, v in tc["added_tokens_decoder"].items()}
    assert set(configured) == set(range(248044, 248077))


# --- Pin mechanics ----------------------------------------------------------


@needs_source
def test_source_pins_match_downloaded_revision_files():
    paths = consumer_module.verify_source_files(SOURCE_DIR)
    assert set(paths) == set(consumer_module.SOURCE_SHA256)
    for name, expected in consumer_module.SOURCE_SHA256.items():
        assert consumer_module.sha(SOURCE_DIR / name) == expected
    assert consumer_module.TOKENIZER_JSON_SHA256 == consumer_module.SOURCE_SHA256["tokenizer.json"]


@needs_source
def test_source_pin_verification_rejects_tampered_and_missing_files(tmp_path):
    for name in consumer_module.SOURCE_SHA256:
        (tmp_path / name).write_bytes((SOURCE_DIR / name).read_bytes())
    consumer_module.verify_source_files(tmp_path)
    data = bytearray((tmp_path / "tokenizer_config.json").read_bytes())
    data[-10] ^= 1
    (tmp_path / "tokenizer_config.json").write_bytes(bytes(data))
    with pytest.raises(ModelInputError, match="digest"):
        consumer_module.verify_source_files(tmp_path)
    (tmp_path / "tokenizer_config.json").write_bytes((SOURCE_DIR / "tokenizer_config.json").read_bytes())
    (tmp_path / "tokenizer.json").unlink()
    with pytest.raises(ModelInputError, match="digest"):
        consumer_module.verify_source_files(tmp_path)


def test_placeholder_pins_are_fail_closed(tmp_path):
    manifest = dict(
        schema_version=consumer_module.MANIFEST_VERSION,
        model_id=consumer_module.MODEL,
        revision=consumer_module.REVISION,
        adapter_enabled=True,
        consumer_profile=consumer_module.QWEN3_SFT_V1_CONSUMER_PROFILE,
        training_provenance=deepcopy(consumer_module.TRAINING_PROVENANCE),
        model_directory="model",
        adapter_directory="adapter",
        files={},
        chat_template_digest=consumer_module.VENDOR_TEMPLATE_DIGEST,
        runtime=dict(
            versions=consumer_module.VERSIONS,
            image=consumer_module.IMAGE,
            required_environment=consumer_module.REQUIRED_ENVIRONMENT,
        ),
    )
    (tmp_path / "model").mkdir()
    (tmp_path / "adapter").mkdir()
    (tmp_path / consumer_module.MANIFEST_NAME).write_text(json.dumps(manifest))
    # Empty inventory can never match the placeholder official-inventory digest.
    with pytest.raises(ModelInputError, match="official revision"):
        consumer_module.verify_manifest(tmp_path)
    assert consumer_module.CHECKPOINT == {name: "0" * 64 for name in consumer_module.CHECKPOINT}
    assert consumer_module.ADAPTER_TENSORS == 0
    # First qualification (2026-11-07) filled the inventory + image pins from the
    # measured snapshot; CHECKPOINT stays zero until a candidate adapter is locked.
    assert consumer_module.OFFICIAL_INVENTORY_DIGEST == "df77b4be88cb5401d49137ad1833107a10fb72436fbdaacf46d764fcf244289f"
    assert consumer_module.IMAGE == "sha256:04563c302537a91aa49ebdfbceda96111c5712275999b7e8804fa598f0b5641d"


def test_revision_and_identity_constants():
    assert consumer_module.MODEL == "Qwen/Qwen3.5-9B"
    assert len(consumer_module.REVISION) == 40
    int(consumer_module.REVISION, 16)
    assert consumer_module.QWEN3_SFT_V1_CONSUMER_PROFILE == "qwen3_5-9b-sft-v1-bf16-action-json-v1"
    assert consumer_module.RUNTIME_PREFIX == "dml-qwen3-sft-action-runtime-v1:"


# --- Grammar literal token coverage on the Qwen token stream ---------------


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _segments(text, vocab_keys):
    """Greedy longest-match segmentation in the byte-mapped BPE alphabet."""
    mapped = "".join(_BYTES_TO_UNICODE[b] for b in text.encode("utf-8"))
    out = []
    while mapped:
        for end in range(len(mapped), 0, -1):
            if mapped[:end] in vocab_keys:
                out.append(mapped[:end])
                mapped = mapped[end:]
                break
        else:
            return None
    return out


_BYTES_TO_UNICODE = _bytes_to_unicode()


@needs_source
def test_action_grammar_literals_expressible_on_qwen_token_stream():
    tk = json.loads((SOURCE_DIR / "tokenizer.json").read_text())
    vocab_keys = set(tk["model"]["vocab"])
    # Byte fallback: every byte value has a single-token row.
    assert all(c in vocab_keys for c in _BYTES_TO_UNICODE.values())
    literals = {schema_bytes(episode_tool_definitions()).decode()}
    literals.add(
        '{"schema_version":"dml-agent-action-v1","kind":"tool","name":"retrieve","arguments":{"query":"subject","top_k":1}}'
    )
    literals.add(
        '{"schema_version":"dml-agent-action-v1","kind":"final","answer":{"claims":[{"key":"k","value":"v","evidence_ids":[1]}]}}'
    )
    literals.update(["dml-agent-action-v1", '"kind":"tool"', '"kind":"final"', "<|im_end|>", "<|endoftext|>"])
    for literal in literals:
        assert _segments(literal, vocab_keys) is not None
    # Stop/control spellings segment only through their added-token rows, not base BPE.
    added_contents = {t["content"] for t in tk["added_tokens"]}
    assert "<|im_end|>" in added_contents and "<|im_end|>" not in vocab_keys


# --- Consumer mechanics (fake tokenizer/runtime, no optional deps) ---------


def test_authenticated_single_use_and_raw_ids():
    c = consumer()
    r = request()
    a = c.compile(r.messages, r.tools, output_reserved_tokens=8)
    result = c.execute(a)
    assert result.output_ids == (1, 2, 248046) and result.text == "{}"
    with pytest.raises(ModelInputError):
        c.execute(a)


def test_tamper_and_cross_consumer_rejected_before_execution():
    c = consumer()
    r = request()
    a = c.compile(r.messages, r.tools, output_reserved_tokens=8)
    with pytest.raises(ModelInputError):
        c.execute(replace(a, input_ids=(99, 11)))
    other = consumer()
    other._auth_key = b"other"
    with pytest.raises(ModelInputError):
        other.execute(a)
    assert a.artifact_digest in c._bound_requests


@pytest.mark.parametrize("stop", [248044, 248046])
def test_terminal_stop_decode_only(stop):
    assert consumer().decode_output([1, 2, stop]) == "{}"
    assert consumer().decode_output([stop, 1]).startswith("<|")


@pytest.mark.parametrize("reservation", [257, 8192])
def test_reservation_limit(reservation):
    c = consumer()
    r = request()
    with pytest.raises(ModelInputError):
        c.compile(r.messages, r.tools, output_reserved_tokens=reservation)


def test_wrong_profile_refused_before_manifest_access(tmp_path):
    with pytest.raises(ModelInputError, match="profile"):
        consumer_module.LocalQwen3SFTV1ActionInputConsumer(
            tmp_path, consumer_profile="llama3-8b-instruct-sft-v3-bf16-action-json-v1"
        )


def test_format_projection_accepts_complete_at_cap_but_not_incomplete_or_control():
    class T(Tokenizer):
        def decode(self, ids, **kwargs):
            return (
                '{"schema_version":"dml-agent-action-v1","kind":"final","answer":{"claims":[]}}' if ids == [5] else "{"
            )

    assert runtime.project_output([5], T(), request().tools)["action_error"] is None
    assert runtime.project_output([1], T(), request().tools)["action_error"] is not None
    embedded = runtime.project_output([5, 248068], T(), request().tools)
    assert "Embedded output control" in embedded["action_error"]


def test_development_runtime_identity_shape_validation():
    valid = {
        "label": "development-x86-20261106",
        "platform_machine": "x86_64",
        "driver": "595.99.02",
        "device_name": "NVIDIA RTX PRO 5000 72GB Blackwell",
        "capability": (12, 0),
    }
    consumer_module._validate_development_runtime(valid)
    with pytest.raises(ModelInputError, match="must declare exactly"):
        consumer_module._validate_development_runtime({**valid, "extra": 1})
    with pytest.raises(ModelInputError, match="development-"):
        consumer_module._validate_development_runtime({**valid, "label": "qualified-x86"})
    with pytest.raises(ModelInputError, match="capability"):
        consumer_module._validate_development_runtime({**valid, "capability": [12, 0]})


def test_development_checkpoint_identity_shape_validation():
    valid = {"label": "development-m9-qwen3-20261106", "manifest_sha256": "a" * 64, "adapter_sha256": "b" * 64}
    consumer_module._validate_development_checkpoint(valid)
    with pytest.raises(ModelInputError, match="must declare exactly"):
        consumer_module._validate_development_checkpoint({**valid, "extra": 1})
    with pytest.raises(ModelInputError, match="development-"):
        consumer_module._validate_development_checkpoint({**valid, "label": "production-m9"})
    with pytest.raises(ModelInputError, match="manifest_sha256"):
        consumer_module._validate_development_checkpoint({**valid, "manifest_sha256": "A" * 64})
    with pytest.raises(ModelInputError, match="adapter_sha256"):
        consumer_module._validate_development_checkpoint({**valid, "adapter_sha256": 123})


def test_development_identity_properties_recorded_and_immutable():
    identity = {
        "label": "development-x86-20261106",
        "platform_machine": "x86_64",
        "driver": "595.99.02",
        "device_name": "NVIDIA RTX PRO 5000 72GB Blackwell",
        "capability": (12, 0),
    }
    c = consumer()
    c._development_runtime = dict(identity)
    assert c.development_runtime_identity == identity
    identity["driver"] = "mutated"
    assert c.development_runtime_identity["driver"] == "595.99.02"
    c2 = consumer()
    assert c2.development_runtime_identity is None
    assert c2.development_checkpoint_identity is None


def test_identity_for_manifest_binds_profile_and_thinking_policy():
    manifest = dict(
        schema_version=consumer_module.MANIFEST_VERSION,
        model_id=consumer_module.MODEL,
        revision=consumer_module.REVISION,
        adapter_enabled=True,
        consumer_profile=consumer_module.QWEN3_SFT_V1_CONSUMER_PROFILE,
        training_provenance=deepcopy(consumer_module.TRAINING_PROVENANCE),
        model_directory="model",
        adapter_directory="adapter",
        files={"model/tokenizer.json": "a" * 64},
        chat_template_digest=consumer_module.VENDOR_TEMPLATE_DIGEST,
        runtime=dict(
            versions=consumer_module.VERSIONS,
            image=consumer_module.IMAGE,
            required_environment=consumer_module.REQUIRED_ENVIRONMENT,
        ),
    )
    identity = consumer_module.identity_for_manifest(manifest)
    assert identity.runtime_identity.startswith(consumer_module.RUNTIME_PREFIX)
    assert identity.tokenizer_digest == "a" * 64
    assert identity.chat_template_digest == consumer_module.VENDOR_TEMPLATE_DIGEST
    assert identity.model_window_tokens == runtime.WINDOW
    changed = deepcopy(manifest)
    changed["training_provenance"]["status"] = "qualified"
    assert consumer_module.identity_for_manifest(changed) != identity


def test_snapshot_paths_cannot_escape(tmp_path):
    with pytest.raises(ModelInputError):
        consumer_module._relative(tmp_path, "../outside")
    with pytest.raises(ModelInputError):
        consumer_module._relative(tmp_path, "/absolute")


def test_candidate_admission_does_not_require_zero_config_hash(tmp_path):
    """Regression: the candidate path spread the zero CHECKPOINT placeholder over
    adapter_config.json, so every candidate failed with 'adapter hash differs'
    even when weights matched. The config is verified via the manifest files
    loop; the candidate check must cover weights only."""
    import hashlib

    model_dir = tmp_path / "model"
    adapter_dir = tmp_path / "adapter"
    model_dir.mkdir()
    adapter_dir.mkdir()
    weights = b"candidate-weights"
    config_bytes = b'{"peft_type": "LORA"}'
    (adapter_dir / "adapter_model.safetensors").write_bytes(weights)
    (adapter_dir / "adapter_config.json").write_bytes(config_bytes)
    (model_dir / "config.json").write_text("{}")

    def sha(payload):
        return hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema_version": consumer_module.MANIFEST_VERSION,
        "model_id": consumer_module.MODEL,
        "revision": consumer_module.REVISION,
        "adapter_enabled": True,
        "consumer_profile": consumer_module.QWEN3_SFT_V1_CONSUMER_PROFILE,
        "training_provenance": consumer_module.TRAINING_PROVENANCE,
        "model_directory": "model",
        "adapter_directory": "adapter",
        "files": {
            "model/config.json": sha(b"{}"),
            "adapter/adapter_model.safetensors": sha(weights),
            "adapter/adapter_config.json": sha(config_bytes),
        },
        "chat_template_digest": consumer_module.VENDOR_TEMPLATE_DIGEST,
        "runtime": {
            "versions": consumer_module.VERSIONS,
            "image": consumer_module.IMAGE,
            "required_environment": consumer_module.REQUIRED_ENVIRONMENT,
        },
    }
    (tmp_path / consumer_module.MANIFEST_NAME).write_text(json.dumps(manifest))
    declaration = {
        "label": "development-test-candidate",
        "manifest_sha256": "f" * 64,  # wrong on purpose: manifest digest check fires first
        "adapter_sha256": sha(weights),
    }
    with pytest.raises(ModelInputError, match="manifest digest differs"):
        consumer_module.verify_manifest(tmp_path, development_checkpoint=declaration)
    declaration["manifest_sha256"] = sha(json.dumps(manifest).encode())
    (tmp_path / consumer_module.MANIFEST_NAME).write_text(json.dumps(manifest))
    declaration["manifest_sha256"] = hashlib.sha256(
        (tmp_path / consumer_module.MANIFEST_NAME).read_bytes()
    ).hexdigest()
    try:
        consumer_module.verify_manifest(tmp_path, development_checkpoint=declaration)
    except ModelInputError as exc:
        assert "adapter hash differs" not in str(exc), (
            "candidate path must not check adapter_config.json against the zero CHECKPOINT placeholder"
        )
