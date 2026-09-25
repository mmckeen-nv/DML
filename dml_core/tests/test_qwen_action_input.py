"""Tiny real-runtime controls for authenticated constrained action execution."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from daystrom_dml.contracts.agent_episode import AgentEpisodeError, episode_tool_definitions, parse_agent_action
from daystrom_dml.contracts.model_input import ModelInputError, ModelInputRequest
from daystrom_dml.services.agent_action_grammar import ActionLogitsProcessor, MAX_BOUND_REQUESTS, compile_action_grammar
from daystrom_dml.services.model_input import ModelInputExecutionError
from daystrom_dml.services.qwen_action_input import LocalQwenActionInputConsumer, constrained_identity
from qwen_model_input_fixture import create_qwen_snapshot
from scripts import agent_campaign_evidence as checker
from test_agent_action_grammar import grammar_runtime as grammar_runtime


MESSAGES = [{"role": "user", "content": "Read the requested memory."}]


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory, grammar_runtime):
    return create_qwen_snapshot(tmp_path_factory.mktemp("tiny-constrained-qwen"), context_window=4096)


@pytest.fixture
def consumer(snapshot):
    with LocalQwenActionInputConsumer(snapshot.path) as instance:
        yield instance


def _tools(name):
    return [tool for tool in episode_tool_definitions() if tool["function"]["name"] == name]


def _action(name):
    arguments = {"query": "arbitrary", "top_k": 4} if name == "retrieve" else {"record_ref": "r77", "reason": "requested"}
    return {"schema_version": "dml-agent-action-v1", "kind": "tool", "name": name, "arguments": arguments}


def _steered_logits(consumer, monkeypatch, mapping):
    """Synthetic logits exercise actual masks; never claim learned quality."""
    calls = []

    def generate(**kwargs):
        prefix = tuple(kwargs["input_ids"][0].tolist())
        calls.append(prefix)
        text = json.dumps(mapping[prefix], ensure_ascii=False, separators=(",", ":"))
        output = consumer._tokenizer.encode(text, add_special_tokens=False) + [consumer._tokenizer.eos_token_id]
        assert len(output) <= kwargs["generation_config"].max_new_tokens
        complete = list(prefix)
        for token in output:
            scores = consumer._torch.zeros((1, consumer._model.config.vocab_size), dtype=consumer._torch.float32)
            scores[0, token] = 20
            scores = kwargs["logits_processor"][0](consumer._torch.tensor([complete], dtype=consumer._torch.long), scores)
            assert consumer._torch.isfinite(scores[0, token]), "Advertised request grammar rejected the steered test token"
            complete.append(int(scores.argmax(-1).item()))
        return consumer._torch.tensor([complete], dtype=consumer._torch.long)

    monkeypatch.setattr(consumer._model, "generate", generate)
    return calls


def test_compile_a_b_then_execute_a_b_a_uses_authenticated_request_owned_grammar(consumer, monkeypatch):
    a = consumer.compile(MESSAGES, _tools("retrieve"), output_reserved_tokens=128)
    b = consumer.compile(MESSAGES, _tools("retire"), output_reserved_tokens=128)
    expected_request = ModelInputRequest.from_payload({
        "messages": MESSAGES, "tools": _tools("retrieve"), "output_reserved_tokens": 128,
    })
    assert a.request_digest == expected_request.request_digest
    calls = _steered_logits(consumer, monkeypatch, {a.input_ids: _action("retrieve"), b.input_ids: _action("retire")})
    results = [consumer.execute(artifact) for artifact in (a, b, a)]
    assert [parse_agent_action(result.text)["name"] for result in results] == ["retrieve", "retire", "retrieve"]
    assert results[0].output_ids == results[2].output_ids
    assert calls == [a.input_ids, b.input_ids, a.input_ids]
    assert len(consumer._bound_requests) == 2
    for result in results:
        assert result.output_token_count == len(result.output_ids)
        assert result.output_ids[-1] == consumer._tokenizer.eos_token_id
    assert getattr(consumer._model, "_cache", None) is None


def test_real_tiny_generation_at_bound_retains_exact_incomplete_ids_and_counts(consumer):
    artifact = consumer.compile(MESSAGES, _tools("retrieve"), output_reserved_tokens=6)
    result = consumer.execute(artifact)
    assert result.output_token_count == len(result.output_ids) == 6
    assert result.text == consumer._tokenizer.decode(result.output_ids, skip_special_tokens=False,
                                                    clean_up_tokenization_spaces=False)
    with pytest.raises(AgentEpisodeError):
        parse_agent_action(result.text)


def test_constrained_ids_match_independently_constructed_model_and_fresh_matcher(snapshot, consumer):
    import torch
    from transformers import GenerationConfig

    tools = _tools("retrieve")
    artifact = consumer.compile(MESSAGES, tools, output_reserved_tokens=64)
    actual = consumer.execute(artifact)
    grammar = compile_action_grammar(snapshot.tokenizer, snapshot.model.config.vocab_size, tools)
    processor = ActionLogitsProcessor(grammar, snapshot.tokenizer, artifact.input_ids, 64)
    generation = GenerationConfig(
        max_new_tokens=64, do_sample=False, num_beams=1, use_cache=True,
        cache_implementation="dynamic", disable_compile=True,
        bos_token_id=snapshot.tokenizer.bos_token_id, eos_token_id=snapshot.tokenizer.eos_token_id,
        pad_token_id=snapshot.tokenizer.pad_token_id,
    )
    with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
        expected = snapshot.model.generate(
            input_ids=torch.tensor([artifact.input_ids], dtype=torch.long),
            attention_mask=torch.tensor([artifact.attention_mask], dtype=torch.long),
            generation_config=generation, use_model_defaults=False, logits_processor=[processor],
        )
    complete = tuple(expected[0].tolist())
    processor.finish(complete)
    assert complete[:len(artifact.input_ids)] == artifact.input_ids
    assert complete[len(artifact.input_ids):] == actual.output_ids
    assert actual.output_token_count == len(actual.output_ids)


def test_capacity_is_explicit_without_eviction_and_close_releases_every_binding(consumer, monkeypatch):
    artifacts = [consumer.compile(MESSAGES, [], output_reserved_tokens=1) for _ in range(MAX_BOUND_REQUESTS)]
    saved = dict(consumer._bound_requests)

    def forbidden(*args, **kwargs):
        pytest.fail("Over-capacity compilation reached tokenization")

    monkeypatch.setattr(consumer._tokenizer, "apply_chat_template", forbidden)
    with pytest.raises(ModelInputError, match="capacity"):
        consumer.compile(MESSAGES, [], output_reserved_tokens=1)
    assert consumer._bound_requests == saved
    assert all(artifact.artifact_digest in saved for artifact in artifacts)
    path = consumer._snapshot.path
    consumer.close()
    consumer.close()
    assert consumer._bound_requests == {} and not path.exists()


@pytest.mark.parametrize("mutation", ["request_bytes", "artifact", "foreign", "runtime"])
def test_forgery_or_runtime_drift_never_reaches_generation(snapshot, consumer, monkeypatch, mutation):
    artifact = consumer.compile(MESSAGES, _tools("retrieve"), output_reserved_tokens=128)
    calls = []

    def forbidden(**kwargs):
        calls.append(kwargs)
        pytest.fail("Invalid constrained binding reached generation")

    monkeypatch.setattr(consumer._model, "generate", forbidden)
    if mutation == "request_bytes":
        changed = ModelInputRequest.from_payload({"messages": MESSAGES, "tools": _tools("retire"), "output_reserved_tokens": 128})
        consumer._bound_requests[artifact.artifact_digest] = json.dumps(changed.to_payload()).encode()
    elif mutation == "artifact":
        artifact = replace(artifact, request_digest="f" * 64)
    elif mutation == "foreign":
        with LocalQwenActionInputConsumer(snapshot.path) as other:
            artifact = other.compile(MESSAGES, _tools("retrieve"), output_reserved_tokens=128)
    else:
        from daystrom_dml.services import agent_action_grammar
        version = agent_action_grammar.metadata.version
        monkeypatch.setattr(agent_action_grammar.metadata, "version", lambda name: "changed" if name == "xgrammar" else version(name))
    with pytest.raises(ModelInputError):
        consumer.execute(artifact)
    assert calls == []


def test_generation_exception_has_no_hidden_retry_or_binding_growth(consumer, monkeypatch):
    artifact = consumer.compile(MESSAGES, _tools("retrieve"), output_reserved_tokens=128)
    calls = []

    def failing(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("private backend details")

    monkeypatch.setattr(consumer._model, "generate", failing)
    with pytest.raises(ModelInputExecutionError) as error:
        consumer.execute(artifact)
    assert "private" not in str(error.value)
    assert len(calls) == 1 and len(consumer._bound_requests) == 1
    consumer.close()
    assert consumer._bound_requests == {}


def test_independent_replay_checks_exact_request_grammar_and_constrained_runtime(consumer, monkeypatch):
    artifact = consumer.compile(MESSAGES, _tools("retrieve"), output_reserved_tokens=128)
    _steered_logits(consumer, monkeypatch, {artifact.input_ids: _action("retrieve")})
    result = consumer.execute(artifact)
    request = {"messages": MESSAGES, "tools": _tools("retrieve"), "output_reserved_tokens": 128}
    events = [
        {"kind": "model_requested", "payload": {"compiled": artifact.signing_payload(), "request": request}},
        {"kind": "model_completed", "payload": {"output_ids": list(result.output_ids), "text": result.text}},
    ]
    assert checker._replay_model(events, consumer._identity.to_payload(), consumer._tokenizer, "qwen2-action-json-v1") == 1
    base = consumer._snapshot.identity
    assert constrained_identity(base) == consumer._identity
    with pytest.raises(ValueError, match="identity"):
        checker._replay_model(events, base.to_payload(), consumer._tokenizer, "qwen2-action-json-v1")
    changed = deepcopy(events)
    wrong = _action("retire")
    changed[1]["payload"]["text"] = json.dumps(wrong, separators=(",", ":"))
    changed[1]["payload"]["output_ids"] = consumer._tokenizer.encode(changed[1]["payload"]["text"], add_special_tokens=False)
    with pytest.raises(ValueError, match="grammar"):
        checker._replay_model(changed, consumer._identity.to_payload(), consumer._tokenizer, "qwen2-action-json-v1")


def test_v2_policy_identity_is_separate_and_v1_bytes_stay_unchanged(monkeypatch):
    from daystrom_dml.contracts import agent_episode as contract
    from daystrom_dml.contracts.model_input import ModelInputIdentity
    from daystrom_dml.services.agent_action_grammar import policy_identity
    from daystrom_dml.services.model_input import _json_bytes
    import hashlib
    base = ModelInputIdentity('1' * 64, '2' * 64, '3' * 64, 'base', 32768)
    old = constrained_identity(base)
    expected = 'dml-qwen-action-runtime-v1:' + hashlib.sha256(_json_bytes(
        {'base_runtime_identity': base.runtime_identity, **policy_identity()})).hexdigest()
    assert old.runtime_identity == expected
    new = constrained_identity(base, consumer_profile=contract.VALIDATION_CONSUMER_PROFILE)
    assert new != old and new.runtime_identity.startswith('dml-qwen-action-runtime-v2:')
    assert contract.execution_policy_identity()['supersede_admission']['missing_binding'].startswith('terminal_')
    monkeypatch.setattr(contract, 'VALIDATION_MODEL_RESULT', contract.VALIDATION_MODEL_RESULT + ' ')
    assert constrained_identity(base) == old
    assert constrained_identity(base, consumer_profile=contract.VALIDATION_CONSUMER_PROFILE) != new


def test_v2_consumer_compiles_bound_identity_and_refuses_v1_artifact(snapshot, consumer, monkeypatch):
    from daystrom_dml.contracts.agent_episode import VALIDATION_CONSUMER_PROFILE
    from daystrom_dml.contracts.model_input import ModelInputError
    with LocalQwenActionInputConsumer(snapshot.path, consumer_profile=VALIDATION_CONSUMER_PROFILE) as newer:
        artifact = consumer.compile([{'role': 'user', 'content': 'test'}], _tools('retrieve'), output_reserved_tokens=16)
        calls = []
        monkeypatch.setattr(newer._model, 'generate', lambda **kw: calls.append(kw))
        with pytest.raises(ModelInputError):
            newer.execute(artifact)
        own = newer.compile([{'role': 'user', 'content': 'test'}], _tools('retrieve'), output_reserved_tokens=16)
        assert own.identity == constrained_identity(newer._base_action_identity, consumer_profile=VALIDATION_CONSUMER_PROFILE)
        assert calls == []


def test_v2_execution_policy_drift_is_refused_before_generation(snapshot, monkeypatch):
    from daystrom_dml.contracts import agent_episode as contract
    from daystrom_dml.contracts.model_input import ModelInputError
    with LocalQwenActionInputConsumer(snapshot.path, consumer_profile=contract.VALIDATION_CONSUMER_PROFILE) as consumer:
        artifact = consumer.compile([{'role': 'user', 'content': 'test'}], _tools('retrieve'), output_reserved_tokens=16)
        calls = []
        monkeypatch.setattr(consumer._model, 'generate', lambda **kw: calls.append(kw))
        monkeypatch.setattr(contract, 'VALIDATION_ERROR_CODE', 'different')
        with pytest.raises(ModelInputError, match='policy'):
            consumer.execute(artifact)
        assert calls == []
