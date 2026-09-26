"""Real CPU runtime checks for the authenticated final-input consumer."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import gc
import hashlib
import json
import tracemalloc
from types import SimpleNamespace

import pytest

from daystrom_dml.contracts.model_input import (
    ModelInputBudgetError, ModelInputError, SUPPORTED_CHAT_TEMPLATE,
)
from daystrom_dml.services.model_input import (
    LocalTransformersInputConsumer, ModelInputExecutionError,
)
from model_input_fixture import create_snapshot, require_model_input_dependencies


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


def _fingerprint_subject():
    """Exercise the real fingerprint on tensors without running any model."""
    versions = require_model_input_dependencies()
    import torch

    consumer = object.__new__(LocalTransformersInputConsumer)
    consumer._closed = False
    consumer._runtime_versions = versions
    consumer._identity = SimpleNamespace(to_payload=lambda: {"identity": "exact café fixture"})
    consumer._template = "{{ messages }}"
    consumer._tokenizer = SimpleNamespace(
        backend_tokenizer=SimpleNamespace(to_str=lambda: '{"fixture":"café"}'),
        special_tokens_map={"eos_token": "<end>"}, chat_template="{{ messages }}",
        model_max_length=128, padding_side="right", truncation_side="left",
    )
    consumer._model = torch.nn.Module()
    consumer._model.config = SimpleNamespace(to_dict=lambda: {"model_type": "fingerprint-fixture"})
    consumer._model.generation_config = SimpleNamespace(to_dict=lambda: {"use_cache": False})
    consumer._model.eval()
    return consumer


def _old_fingerprint_oracle(consumer):
    """Independent prior algorithm, including its whole-tensor bytes copy."""
    from importlib.metadata import version

    def encoded(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")

    digest = hashlib.sha256()
    tokenizer, model = consumer._tokenizer, consumer._model
    digest.update(encoded({
        "identity": consumer._identity.to_payload(), "template": consumer._template,
        "tokenizer_backend": tokenizer.backend_tokenizer.to_str(),
        "special_tokens": tokenizer.special_tokens_map,
        "tokenizer_template": tokenizer.chat_template,
        "tokenizer_window": tokenizer.model_max_length,
        "tokenizer_padding": tokenizer.padding_side,
        "tokenizer_truncation": tokenizer.truncation_side,
        "model_config": model.config.to_dict(),
        "generation_config": model.generation_config.to_dict(), "training": model.training,
        "versions": {name: version(name)
                     for name in ("torch", "transformers", "tokenizers", "safetensors", "jinja2")},
    }))
    for group, values in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, value in values:
            if value.device.type != "cpu":
                raise ModelInputError("Exact-input model moved outside the CPU runtime")
            digest.update(encoded([group, name, str(value.dtype), list(value.shape), value.requires_grad]))
            digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize("dtype", [
    "float16", "float32", "float64", "complex64", "complex128", "bool",
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
])
def test_fingerprint_matches_prior_bytes_for_tensor_shapes_dtypes_and_order(dtype):
    consumer = _fingerprint_subject()
    import torch

    values = torch.arange(12).to(getattr(torch, dtype))
    if values.is_floating_point():
        values[:4] = torch.tensor([-0.0, float("inf"), -float("inf"), float("nan")])
    # Deliberately nonlexical insertion order exercises the original ordering.
    consumer._model.register_parameter("z_parameter", torch.nn.Parameter(values.clone(), requires_grad=False))
    consumer._model.register_parameter("a_parameter", torch.nn.Parameter(values.reshape(3, 4).T,
                                                                          requires_grad=False))
    consumer._model.register_buffer("z_scalar", values[0].clone().reshape(()))
    consumer._model.register_buffer("a_empty", values[:0].reshape(2, 0, 3))
    consumer._model.register_buffer("sliced", values[::2])
    consumer._model.register_buffer("transposed", values.reshape(3, 4).T)
    assert consumer._fingerprint() == _old_fingerprint_oracle(consumer)


@pytest.mark.parametrize("kind", ["meta", "bfloat16", "sparse"])
def test_fingerprint_preserves_prior_cpu_and_unsupported_tensor_errors(kind):
    consumer = _fingerprint_subject()
    import torch

    if kind == "meta":
        value = torch.empty(2, device="meta")
    elif kind == "bfloat16":
        value = torch.ones(2, dtype=torch.bfloat16)
    else:
        value = torch.sparse_coo_tensor([[0]], [1.0], (2,))
    consumer._model.register_buffer("unsupported", value)
    with pytest.raises((ModelInputError, TypeError, RuntimeError)) as old:
        _old_fingerprint_oracle(consumer)
    with pytest.raises(type(old.value)) as current:
        consumer._fingerprint()
    assert str(current.value) == str(old.value)
    consumer._runtime_digest = "unused"
    expected = "outside the CPU runtime" if kind == "meta" else "identity cannot be verified"
    with pytest.raises(ModelInputError, match=expected):
        consumer._validate_runtime()


@pytest.mark.parametrize("group", ["parameter", "buffer"])
@pytest.mark.parametrize("alias", ["numpy", "data"])
def test_every_fingerprint_scan_detects_alias_writes_without_version_counter_changes(group, alias):
    consumer = _fingerprint_subject()
    import torch

    value = torch.arange(4, dtype=torch.float32)
    if group == "parameter":
        value = torch.nn.Parameter(value, requires_grad=False)
        consumer._model.register_parameter("observed", value)
    else:
        consumer._model.register_buffer("observed", value)
    consumer._runtime_digest = _old_fingerprint_oracle(consumer)
    consumer._validate_runtime()
    version = value._version
    if alias == "numpy":
        value.detach().numpy()[0] = 29.0
    else:
        value.data[0] = 29.0
    assert value._version == version
    assert consumer._fingerprint() == _old_fingerprint_oracle(consumer) != consumer._runtime_digest
    with pytest.raises(ModelInputError, match="identity changed"):
        consumer._validate_runtime()


def test_fingerprint_eliminates_whole_tensor_python_bytes_allocation():
    consumer = _fingerprint_subject()
    import torch

    value = torch.zeros(4 * 1024 * 1024, dtype=torch.float32)
    consumer._model.register_buffer("large_contiguous", value)
    consumer._fingerprint()  # Warm metadata imports before measuring allocations.

    def observed_peak(operation):
        already_tracing = tracemalloc.is_tracing()
        if not already_tracing:
            tracemalloc.start()
        try:
            gc.collect()
            baseline, _ = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            result = operation()
            _, peak = tracemalloc.get_traced_memory()
            return result, peak - baseline
        finally:
            if not already_tracing:
                tracemalloc.stop()

    old_digest, old_peak = observed_peak(lambda: _old_fingerprint_oracle(consumer))
    new_digest, new_peak = observed_peak(consumer._fingerprint)
    tensor_bytes = value.numel() * value.element_size()
    assert old_digest == new_digest
    assert old_peak >= tensor_bytes
    assert new_peak < tensor_bytes // 8
    # tracemalloc measures Python allocation, not a noncontiguous Torch copy,
    # total process memory, or inference speed. No performance claim follows.
    print(json.dumps({"fingerprint_tensor_bytes": tensor_bytes,
                      "old_python_peak_bytes": old_peak, "new_python_peak_bytes": new_peak}, sort_keys=True))
