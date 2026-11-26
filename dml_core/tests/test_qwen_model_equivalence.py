"""Numerical oracle for the Qwen loader's assigned weights and rotary buffers."""
from __future__ import annotations

import gc
import weakref

import pytest

from qwen_model_input_fixture import create_qwen_snapshot


def test_verified_loader_matches_normally_constructed_qwen_logits(tmp_path):
    snapshot = create_qwen_snapshot(tmp_path / "snapshot")

    import torch

    from daystrom_dml.services.qwen_model_input import LocalQwenInputConsumer

    with LocalQwenInputConsumer(snapshot.path) as consumer:
        artifact = consumer.compile([
            {"role": "system", "content": "Use trusted memory to answer the owner's question."},
            {"role": "user", "content": "Which café notebook is current?"},
            {"role": "assistant", "content": "The current preference is green."},
            {"role": "user", "content": "Preserve the earlier context and explain why."},
        ], output_reserved_tokens=4)
        ids = torch.tensor([artifact.input_ids], dtype=torch.long, device="cpu")
        mask = torch.tensor([artifact.attention_mask], dtype=torch.long, device="cpu")
        snapshot.model.eval()
        with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
            expected = snapshot.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            actual = consumer._model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        assert expected.dtype == actual.dtype == torch.float32
        assert expected.device.type == actual.device.type == "cpu"
        assert expected.shape == actual.shape == (1, len(artifact.input_ids), snapshot.model.config.vocab_size)
        assert bool(torch.isfinite(expected).all())
        assert torch.equal(actual, expected)


def test_request_local_cached_greedy_ids_match_independent_uncached_model(tmp_path):
    snapshot = create_qwen_snapshot(tmp_path / "snapshot")

    import torch
    from transformers import GenerationConfig

    from daystrom_dml.services.qwen_model_input import LocalQwenInputConsumer

    with LocalQwenInputConsumer(snapshot.path) as consumer:
        artifact = consumer.compile([
            {"role": "user", "content": "Read the green notebook. Preserve every earlier word."},
        ], output_reserved_tokens=12)
        cached = consumer.execute(artifact)
        # Independent normal construction, explicit uncached greedy policy, and
        # the same exact compiled IDs: no reuse of the consumer's config helper.
        uncached_config = GenerationConfig(
            max_new_tokens=12, do_sample=False, num_beams=1, use_cache=False,
            bos_token_id=snapshot.tokenizer.bos_token_id,
            eos_token_id=snapshot.tokenizer.eos_token_id,
            pad_token_id=snapshot.tokenizer.pad_token_id,
            cache_implementation=None, disable_compile=True,
        )
        with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
            expected = snapshot.model.generate(
                input_ids=torch.tensor([artifact.input_ids], dtype=torch.long),
                attention_mask=torch.tensor([artifact.attention_mask], dtype=torch.long),
                generation_config=uncached_config, use_model_defaults=False,
            )
        assert tuple(expected[0, :len(artifact.input_ids)].tolist()) == artifact.input_ids
        assert cached.output_ids == tuple(expected[0, len(artifact.input_ids):].tolist())
        assert len(cached.output_ids) >= 2  # Exercise reuse within a generation.
        assert getattr(consumer._model, "_cache", None) is None


def test_each_generation_has_a_fresh_empty_cache_and_repeated_requests_match_fresh_consumers(
    tmp_path, monkeypatch,
):
    snapshot = create_qwen_snapshot(tmp_path / "snapshot")

    from transformers import DynamicCache

    from daystrom_dml.services.qwen_model_input import LocalQwenInputConsumer

    requests = [
        [{"role": "user", "content": "The notebook is green. Which notebook?"}],
        [{"role": "user", "content": "A different owner prefers blue ink. What is the current ink?"}],
    ]
    cache_references = []
    observed = []
    with LocalQwenInputConsumer(snapshot.path) as consumer:
        prepare = consumer._model._prepare_cache_for_generation

        def observe_cache(generation_config, model_kwargs, assistant_model, batch_size, max_cache_length):
            assert generation_config.use_cache is True
            assert generation_config.cache_implementation == "dynamic"
            assert model_kwargs.get("past_key_values") is None
            result = prepare(generation_config, model_kwargs, assistant_model, batch_size, max_cache_length)
            cache = model_kwargs["past_key_values"]
            assert type(cache) is DynamicCache
            assert cache.get_seq_length() == 0
            cache_references.append(weakref.ref(cache))
            return result

        monkeypatch.setattr(consumer._model, "_prepare_cache_for_generation", observe_cache)
        for index in (0, 1, 0):
            artifact = consumer.compile(requests[index], output_reserved_tokens=8)
            observed.append(consumer.execute(artifact).output_ids)
            assert getattr(consumer._model, "_cache", None) is None
            gc.collect()
            assert cache_references[-1]() is None
        assert len(cache_references) == 3
    assert observed[0] == observed[2]
    for index, messages in enumerate(requests):
        with LocalQwenInputConsumer(snapshot.path) as fresh:
            artifact = fresh.compile(messages, output_reserved_tokens=8)
            assert fresh.execute(artifact).output_ids == observed[index]
            assert getattr(fresh._model, "_cache", None) is None


@pytest.mark.parametrize("retained_when", ["before", "after"])
def test_retained_cross_call_cache_refuses_dispatch_or_acknowledgment(tmp_path, monkeypatch, retained_when):
    snapshot = create_qwen_snapshot(tmp_path / "snapshot")

    import torch
    from transformers import DynamicCache

    from daystrom_dml.contracts.model_input import ModelInputError
    from daystrom_dml.services.model_input import ModelInputExecutionError
    from daystrom_dml.services.qwen_model_input import LocalQwenInputConsumer

    with LocalQwenInputConsumer(snapshot.path) as consumer:
        artifact = consumer.compile([{"role": "user", "content": "Read memory."}], output_reserved_tokens=1)
        calls = []

        def retained_generate(**kwargs):
            calls.append(kwargs)
            consumer._model._cache = DynamicCache(config=consumer._model.config)
            return torch.tensor([list(artifact.input_ids) + [consumer._tokenizer.eos_token_id]], dtype=torch.long)

        monkeypatch.setattr(consumer._model, "generate", retained_generate)
        if retained_when == "before":
            consumer._model._cache = DynamicCache(config=consumer._model.config)
        error = ModelInputError if retained_when == "before" else ModelInputExecutionError
        with pytest.raises(error, match="cache state"):
            consumer.execute(artifact)
        assert len(calls) == (0 if retained_when == "before" else 1)
