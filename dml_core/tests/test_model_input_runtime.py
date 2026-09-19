"""Real CPU runtime checks for the authenticated final-input consumer."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json

import pytest

from daystrom_dml.contracts.model_input import (
    ModelInputBudgetError, ModelInputError, SUPPORTED_CHAT_TEMPLATE,
)
from daystrom_dml.services.model_input import (
    LocalTransformersInputConsumer, ModelInputExecutionError,
)
from model_input_fixture import create_snapshot


MESSAGES = [{"role": "user", "content": "The owner prefers green notebooks."}]


@pytest.fixture
def snapshot(tmp_path):
    return create_snapshot(tmp_path / "snapshot", context_window=256)


def _rewrite_config(snapshot, changes):
    path = snapshot.path / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update(changes)
    path.write_text(json.dumps(config), encoding="utf-8")
    manifest = json.loads((snapshot.path / "snapshot.json").read_text(encoding="utf-8"))
    manifest["files"]["config.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (snapshot.path / "snapshot.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_real_tokenizer_compiles_once_and_dispatches_exact_ids(snapshot, monkeypatch):
    expected = snapshot.tokenizer.apply_chat_template(
        json.loads(json.dumps(MESSAGES, sort_keys=True)), tools=[],
        chat_template=SUPPORTED_CHAT_TEMPLATE, tokenize=True,
        add_generation_prompt=False, truncation=False, padding=False,
    )
    with LocalTransformersInputConsumer(snapshot.path) as consumer:
        calls = []
        tokenize = consumer._tokenizer.apply_chat_template

        def counted(*args, **kwargs):
            calls.append(kwargs)
            return tokenize(*args, **kwargs)

        monkeypatch.setattr(consumer._tokenizer, "apply_chat_template", counted)
        artifact = consumer.compile(MESSAGES, output_reserved_tokens=3)
        assert len(calls) == 1
        assert artifact.input_ids == tuple(expected)
        assert artifact.attention_mask == (1,) * len(expected)
        actual_generate = consumer._model.generate
        dispatched = []

        def generate(**kwargs):
            dispatched.append(kwargs)
            return actual_generate(**kwargs)

        def tokenize_forbidden(*_args, **_kwargs):
            raise AssertionError("Execution tokenized after the budget was bound")

        monkeypatch.setattr(consumer._model, "generate", generate)
        monkeypatch.setattr(consumer._tokenizer, "apply_chat_template", tokenize_forbidden)
        monkeypatch.setattr(consumer._tokenizer, "encode", tokenize_forbidden)
        result = consumer.execute(artifact)
        assert len(dispatched) == 1
        assert dispatched[0]["input_ids"].tolist() == [expected]
        assert dispatched[0]["attention_mask"].tolist() == [[1] * len(expected)]
        policy = dispatched[0]["generation_config"]
        assert policy.max_new_tokens == 3 and policy.use_cache is False
        assert policy.do_sample is False and policy.token_healing is False
        assert dispatched[0]["use_model_defaults"] is False
        assert result.input_token_count == len(expected)
        assert result.output_token_count == len(result.output_ids) <= 3
        assert result.artifact_digest == artifact.artifact_digest


def test_repeated_real_compile_and_execution_preserve_runtime_identity(snapshot):
    with LocalTransformersInputConsumer(snapshot.path) as consumer:
        first = consumer.compile(MESSAGES, output_reserved_tokens=2)
        original = consumer.execute(first)
        repeated = consumer.execute(first)
        second = consumer.compile(MESSAGES, output_reserved_tokens=2)
        assert second.input_ids == first.input_ids
        assert second.nonce != first.nonce
        assert consumer.execute(second).output_ids == original.output_ids == repeated.output_ids


def test_exact_window_boundary_counts_output_reservation_without_truncation(snapshot):
    with LocalTransformersInputConsumer(snapshot.path) as consumer:
        initial = consumer.compile(MESSAGES, output_reserved_tokens=1)
        available = initial.model_window_tokens - len(initial.input_ids)
        boundary = consumer.compile(MESSAGES, output_reserved_tokens=available)
        assert len(boundary.input_ids) + boundary.output_reserved_tokens == boundary.model_window_tokens
        assert boundary.input_ids == initial.input_ids
        with pytest.raises(ModelInputBudgetError):
            consumer.compile(MESSAGES, output_reserved_tokens=available + 1)


def test_input_mutation_after_compilation_cannot_change_dispatched_artifact(snapshot):
    messages = [{"role": "user", "content": "Original request"}]
    with LocalTransformersInputConsumer(snapshot.path) as consumer:
        artifact = consumer.compile(messages, output_reserved_tokens=2)
        before = artifact.input_ids
        messages[0]["content"] = "Changed caller data" * 100
        assert consumer.execute(artifact).input_token_count == len(before)
        assert artifact.input_ids == before
        with pytest.raises(FrozenInstanceError):
            artifact.output_reserved_tokens = 99


def test_snapshot_generation_policy_cannot_enable_hidden_input_transformations(snapshot, monkeypatch):
    _rewrite_config(snapshot, {"token_healing": True, "forced_bos_token_id": 17,
                              "forced_eos_token_id": 18, "num_return_sequences": 3,
                              "do_sample": True, "use_cache": True})
    with LocalTransformersInputConsumer(snapshot.path) as consumer:
        artifact = consumer.compile(MESSAGES, output_reserved_tokens=2)
        actual_generate = consumer._model.generate
        captured = []

        def generate(**kwargs):
            captured.append(kwargs["generation_config"])
            return actual_generate(**kwargs)

        monkeypatch.setattr(consumer._model, "generate", generate)
        result = consumer.execute(artifact)
        policy = captured[0]
        assert policy.token_healing is False and policy.do_sample is False
        assert policy.forced_bos_token_id is None and policy.forced_eos_token_id is None
        assert policy.num_return_sequences == 1 and policy.use_cache is False
        assert result.output_token_count <= 2


def test_model_failures_are_sanitized_without_retry_or_extra_dispatch(snapshot, monkeypatch):
    with LocalTransformersInputConsumer(snapshot.path) as consumer:
        artifact = consumer.compile(MESSAGES, output_reserved_tokens=2)
        calls = []

        def fail(**_kwargs):
            calls.append(1)
            raise RuntimeError("private-provider-path-secret")

        monkeypatch.setattr(consumer._model, "generate", fail)
        with pytest.raises(ModelInputExecutionError) as error:
            consumer.execute(artifact)
        assert "private" not in str(error.value)
        assert calls == [1]


def test_close_releases_verified_copy_and_refuses_new_work(snapshot):
    consumer = LocalTransformersInputConsumer(snapshot.path)
    copied = consumer._snapshot.path
    artifact = consumer.compile(MESSAGES, output_reserved_tokens=1)
    consumer.close()
    consumer.close()
    assert not copied.exists()
    with pytest.raises(ModelInputError, match="closed"):
        consumer.compile(MESSAGES, output_reserved_tokens=1)
    with pytest.raises(ModelInputError, match="closed"):
        consumer.execute(artifact)


def test_snapshot_refusal_has_consumer_error_before_loader(snapshot, monkeypatch):
    import transformers

    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("Invalid source reached a tokenizer loader")

    monkeypatch.setattr(transformers, "PreTrainedTokenizerFast", forbidden)
    (snapshot.path / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    with pytest.raises(ModelInputError, match="snapshot admission failed"):
        LocalTransformersInputConsumer(snapshot.path)
    assert calls == []
