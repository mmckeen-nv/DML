"""Independent negative oracles for the exact local model-input boundary."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import shutil

import pytest

from daystrom_dml.contracts.model_input import (
    ModelInputBudgetError, ModelInputError, ModelInputRequest,
)
from daystrom_dml.services.model_input import (
    LocalTransformersInputConsumer, ModelInputExecutionError,
)
from daystrom_dml.services.model_input_snapshot import (
    SnapshotVerificationError, verify_local_snapshot,
)
from model_input_fixture import create_snapshot


MESSAGES = [{"role": "user", "content": "Read the green notebook."}]
TOOLS = [{"type": "function", "function": {
    "name": "lookup_note", "description": "Read an owner note",
    "parameters": {"type": "object", "properties": {"subject": {"type": "string"}}},
    "strict": True,
}}]


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    return create_snapshot(tmp_path_factory.mktemp("independent-model-input"), context_window=1024)


@pytest.fixture
def consumer(snapshot):
    with LocalTransformersInputConsumer(snapshot.path) as instance:
        yield instance


def _no_dispatch(monkeypatch, consumer):
    calls = []

    def forbidden(**kwargs):
        calls.append(kwargs)
        raise AssertionError("Inference must not run for a refused request")

    monkeypatch.setattr(consumer._model, "generate", forbidden)
    return calls


@pytest.mark.parametrize("payload", [
    {"messages": MESSAGES, "output_reserved_tokens": True},
    {"messages": MESSAGES, "output_reserved_tokens": 0},
    {"messages": MESSAGES, "output_reserved_tokens": 1.0},
    {"messages": MESSAGES, "output_reserved_tokens": 2**31},
    {"messages": MESSAGES, "output_reserved_tokens": 1, "system": "uncounted"},
    {"messages": [{"role": "user", "content": "x", "images": ["x"]}], "output_reserved_tokens": 1},
    {"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}], "output_reserved_tokens": 1},
    {"messages": [{"role": "user", "content": "\ud800"}], "output_reserved_tokens": 1},
    {"messages": [{"role": "tool", "content": "x"}], "output_reserved_tokens": 1},
    {"messages": [{"role": "user", "content": "x", "tool_call_id": "a"}], "output_reserved_tokens": 1},
    {"messages": [{"role": "assistant", "content": "", "tool_calls": [{
        "id": "c", "type": "function", "function": {"name": "x", "arguments": {}}
    }]}], "output_reserved_tokens": 1},
    {"messages": MESSAGES, "tools": [{"type": "function", "function": {
        "name": "x", "parameters": {}, "strict": 1,
    }}], "output_reserved_tokens": 1},
    {"messages": MESSAGES, "tools": [{"type": "function", "function": {
        "name": "x", "parameters": {"default": float("nan")},
    }}], "output_reserved_tokens": 1},
])
def test_independent_malformed_complete_request_is_rejected(payload):
    with pytest.raises(ModelInputError):
        ModelInputRequest.from_payload(payload)


@pytest.mark.parametrize("payload", [
    b'{"messages":[],"messages":[],"output_reserved_tokens":1}',
    b'{"messages":[{"role":"user","content":"x"}],"output_reserved_tokens":NaN}',
    b'{"messages":[{"role":"user","content":"\\ud800"}],"output_reserved_tokens":1}',
    b'{"messages":[{"role":"user","content":"\xff"}],"output_reserved_tokens":1}',
])
def test_independent_noncanonical_wire_payload_is_rejected(payload):
    with pytest.raises(ModelInputError):
        ModelInputRequest.from_json(payload)


def test_consumer_normalizes_snapshot_admission_failure(tmp_path):
    with pytest.raises(ModelInputError, match="snapshot admission failed"):
        LocalTransformersInputConsumer(tmp_path)


def test_malformed_request_fails_before_tokenizer_or_inference(consumer, monkeypatch):
    dispatches = _no_dispatch(monkeypatch, consumer)

    def forbidden(*args, **kwargs):
        pytest.fail("Malformed request reached tokenization")

    monkeypatch.setattr(consumer._tokenizer, "apply_chat_template", forbidden)
    with pytest.raises(ModelInputError):
        consumer.compile([{"role": "user", "content": "x", "hidden_frame": "x"}], output_reserved_tokens=1)
    assert dispatches == []


def test_identity_change_during_tokenization_cannot_issue_artifact(consumer, monkeypatch):
    tokenize = consumer._tokenizer.apply_chat_template
    dispatches = _no_dispatch(monkeypatch, consumer)

    def mutating_tokenizer(*args, **kwargs):
        result = tokenize(*args, **kwargs)
        consumer._model.config.n_positions += 1
        return result

    monkeypatch.setattr(consumer._tokenizer, "apply_chat_template", mutating_tokenizer)
    with pytest.raises(ModelInputError):
        consumer.compile(MESSAGES, output_reserved_tokens=1)
    assert dispatches == []


@pytest.mark.parametrize("field", [
    "input_ids", "attention_mask", "output_reserved_tokens", "request_digest",
    "consumer_id", "nonce", "auth_tag", "identity",
])
def test_artifact_field_forgery_never_dispatches(consumer, monkeypatch, field):
    artifact = consumer.compile(MESSAGES, output_reserved_tokens=2)
    edits = {
        "input_ids": ((artifact.input_ids[0] + 1) % consumer._model.config.vocab_size,) + artifact.input_ids[1:],
        "attention_mask": (0,) + artifact.attention_mask[1:],
        "output_reserved_tokens": 3,
        "request_digest": "a" * 64,
        "consumer_id": "foreign-consumer",
        "nonce": "forged-nonce",
        "auth_tag": "0" * 64,
        "identity": replace(artifact.identity, model_digest="b" * 64),
    }
    forged = replace(artifact, **{field: edits[field]})
    dispatches = _no_dispatch(monkeypatch, consumer)
    with pytest.raises(ModelInputError):
        consumer.execute(forged)
    assert dispatches == []


def test_object_setattr_cannot_bypass_artifact_validation(consumer, monkeypatch):
    artifact = consumer.compile(MESSAGES, output_reserved_tokens=1)
    object.__setattr__(artifact, "input_ids", (True,))
    dispatches = _no_dispatch(monkeypatch, consumer)
    with pytest.raises(ModelInputError):
        consumer.execute(artifact)
    assert dispatches == []


def test_valid_artifact_from_other_consumer_never_dispatches(snapshot, consumer, monkeypatch):
    with LocalTransformersInputConsumer(snapshot.path) as other:
        artifact = other.compile(MESSAGES, output_reserved_tokens=1)
    dispatches = _no_dispatch(monkeypatch, consumer)
    with pytest.raises(ModelInputError):
        consumer.execute(artifact)
    assert dispatches == []


@pytest.mark.parametrize("mutation", ["template", "tokenizer", "padding", "model_config", "generation", "weights", "training", "versions"])
def test_live_identity_drift_refuses_before_inference(consumer, monkeypatch, mutation):
    artifact = consumer.compile(MESSAGES, output_reserved_tokens=1)
    dispatches = _no_dispatch(monkeypatch, consumer)
    if mutation == "template":
        consumer._tokenizer.chat_template = "{{ messages[0].content }}"
    elif mutation == "tokenizer":
        consumer._tokenizer.add_tokens(["new uncounted token"])
    elif mutation == "padding":
        consumer._tokenizer.padding_side = "left"
    elif mutation == "model_config":
        consumer._model.config.n_positions += 1
    elif mutation == "generation":
        consumer._model.generation_config.token_healing = True
    elif mutation == "weights":
        with consumer._torch.no_grad():
            next(consumer._model.parameters()).flatten()[0] += 1
    elif mutation == "training":
        consumer._model.train()
    else:
        monkeypatch.setattr("daystrom_dml.services.model_input._versions", lambda: {"torch": "changed"})
    with pytest.raises(ModelInputError):
        consumer.execute(artifact)
    with pytest.raises(ModelInputError):
        consumer.compile(MESSAGES, output_reserved_tokens=1)
    assert dispatches == []


def test_tools_and_every_message_field_are_in_exact_ids(consumer, snapshot):
    messages = [
        {"role": "system", "content": "Use memory.", "name": "policy"},
        {"role": "user", "content": "Which café notebook? <green> & 'blue'", "name": "owner"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_7", "type": "function", "function": {
                "name": "lookup_note", "arguments": '{"subject":"notebook"}',
            },
        }]},
        {"role": "tool", "content": "Green notebook.", "tool_call_id": "call_7", "name": "lookup_note"},
    ]
    artifact = consumer.compile(messages, TOOLS, output_reserved_tokens=1)
    # Independent direct rendering includes every field, Unicode and HTML-like
    # text, without calling the consumer or Transformers template renderer.
    rendered = (
        "<|messages|>\n" + json.dumps(messages, ensure_ascii=False, sort_keys=True)
        + "\n<|tools|>\n" + json.dumps(TOOLS, ensure_ascii=False, sort_keys=True)
        + "\n<|assistant|>"
    )
    expected = snapshot.tokenizer.encode(rendered, add_special_tokens=False)
    assert artifact.input_ids == tuple(expected)
    without_tools = consumer.compile(messages, output_reserved_tokens=1)
    assert artifact.input_ids != without_tools.input_ids
    for field in ("tool_calls", "tool_call_id", "parameters", "strict", "description", "name"):
        assert field in rendered


def test_exact_window_boundary_is_admitted_and_one_more_token_refused(consumer, monkeypatch):
    counted = consumer.compile(MESSAGES, output_reserved_tokens=1)
    remainder = counted.model_window_tokens - len(counted.input_ids)
    exact = consumer.compile(MESSAGES, output_reserved_tokens=remainder)
    assert len(exact.input_ids) + exact.output_reserved_tokens == exact.model_window_tokens
    dispatches = _no_dispatch(monkeypatch, consumer)
    with pytest.raises(ModelInputBudgetError):
        consumer.compile(MESSAGES, output_reserved_tokens=remainder + 1)
    assert dispatches == []


@pytest.mark.parametrize("bad_output", ["wrong_prefix", "too_long", "wrong_rank", "wrong_dtype", "out_of_vocabulary", "not_tensor"])
def test_invalid_model_output_is_not_acknowledged(consumer, monkeypatch, bad_output):
    artifact = consumer.compile(MESSAGES, output_reserved_tokens=1)
    torch = consumer._torch
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        ids = list(artifact.input_ids)
        if bad_output == "wrong_prefix":
            ids[0] = (ids[0] + 1) % consumer._model.config.vocab_size
        elif bad_output == "too_long":
            ids += [0, 0]
        elif bad_output == "out_of_vocabulary":
            ids += [consumer._model.config.vocab_size]
        if bad_output == "not_tensor":
            return [ids]
        if bad_output == "wrong_rank":
            return torch.tensor(ids, dtype=torch.long)
        return torch.tensor([ids], dtype=torch.float32 if bad_output == "wrong_dtype" else torch.long)

    monkeypatch.setattr(consumer._model, "generate", generate)
    with pytest.raises(ModelInputExecutionError):
        consumer.execute(artifact)
    assert len(calls) == 1


def test_execution_failure_is_sanitized_and_not_retried(consumer, monkeypatch):
    artifact = consumer.compile(MESSAGES, output_reserved_tokens=1)
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("sensitive full prompt and provider details")

    monkeypatch.setattr(consumer._model, "generate", generate)
    with pytest.raises(ModelInputExecutionError) as error:
        consumer.execute(artifact)
    assert "sensitive" not in str(error.value)
    assert len(calls) == 1


def test_closed_consumer_refuses_artifact_and_compile(consumer):
    artifact = consumer.compile(MESSAGES, output_reserved_tokens=1)
    private_path = consumer._snapshot.path
    consumer.close()
    consumer.close()
    assert not private_path.exists()
    with pytest.raises(ModelInputError):
        consumer.execute(artifact)
    with pytest.raises(ModelInputError):
        consumer.compile(MESSAGES, output_reserved_tokens=1)


@pytest.mark.parametrize("filename", ["config.json", "model.safetensors", "tokenizer.json", "chat_template.jinja"])
def test_snapshot_artifact_tampering_fails_hash_admission(snapshot, tmp_path, filename):
    source = tmp_path / "snapshot"
    shutil.copytree(snapshot.path, source)
    with (source / filename).open("ab") as target:
        target.write(b"changed")
    with pytest.raises(SnapshotVerificationError):
        verify_local_snapshot(source)


@pytest.mark.parametrize("filename", ["generation_config.json", "tokenizer_config.json", "modeling_custom.py"])
def test_unlisted_runtime_defaults_and_custom_code_are_rejected(snapshot, tmp_path, filename):
    source = tmp_path / "snapshot"
    shutil.copytree(snapshot.path, source)
    (source / filename).write_text("{}", encoding="utf-8")
    with pytest.raises(SnapshotVerificationError):
        verify_local_snapshot(source)


def test_rehashed_template_that_ignores_tools_is_still_rejected(snapshot, tmp_path):
    source = tmp_path / "snapshot"
    shutil.copytree(snapshot.path, source)
    altered = b"{{ messages[0].content }}"
    (source / "chat_template.jinja").write_bytes(altered)
    manifest = json.loads((source / "snapshot.json").read_text())
    manifest["files"]["chat_template.jinja"] = hashlib.sha256(altered).hexdigest()
    (source / "snapshot.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SnapshotVerificationError):
        verify_local_snapshot(source)


def test_verified_copy_does_not_follow_later_source_mutation(snapshot, tmp_path):
    source = tmp_path / "snapshot"
    shutil.copytree(snapshot.path, source)
    with verify_local_snapshot(source) as verified:
        private_path = verified.path
        original = (private_path / "tokenizer.json").read_bytes()
        (source / "tokenizer.json").write_text("{}", encoding="utf-8")
        assert (private_path / "tokenizer.json").read_bytes() == original
        exposed = verified.config
        exposed["n_positions"] = 1
        assert verified.config["n_positions"] == 1024
    assert not private_path.exists()
