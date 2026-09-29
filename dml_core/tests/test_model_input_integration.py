"""Real CPU model input oracles, including the receipted-memory caller boundary."""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import builtins
import json
import os

import pytest

import model_input_fixture as fixture_module
from model_input_fixture import create_snapshot


SCOPE = {"tenant_id": "owner", "client_id": "client", "session_id": "session", "instance_id": "instance"}


@pytest.fixture
def snapshot(tmp_path):
    return create_snapshot(tmp_path / "model-snapshot", context_window=1024)


@pytest.fixture
def consumer(snapshot):
    # No heavy optional import occurs during collection in ordinary base CI.
    from daystrom_dml.services.model_input import LocalTransformersInputConsumer

    with LocalTransformersInputConsumer(snapshot.path) as instance:
        yield instance


def complete_conversation(context="The owner prefers green notebooks."):
    messages = [
        {"role": "system", "content": "Use verified memory and current instructions.\nMemory:\n" + context},
        {"role": "user", "content": "Earlier I used blue notebooks."},
        {"role": "assistant", "content": "I will check the current notes.", "tool_calls": [
            {"id": "lookup-1", "type": "function", "function": {
                "name": "lookup_note", "arguments": '{"subject":"notebooks"}',
            }},
        ]},
        {"role": "tool", "tool_call_id": "lookup-1", "name": "lookup_note",
         "content": '{"preference":"green notebooks","source":"owner-confirmed"}'},
        {"role": "user", "content": "Which notebook should I buy?"},
    ]
    tools = [{"type": "function", "function": {
        "name": "lookup_note", "description": "Read an owner note", "strict": True,
        "parameters": {"type": "object", "properties": {"subject": {"type": "string"}},
                       "required": ["subject"], "additionalProperties": False},
    }}]
    return messages, tools


def independently_render(messages, tools):
    """Literal frozen wire framing, without calling the consumer or its renderer."""
    def ordered(value):
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False))

    # The pinned Jinja environment strips the template's final newline.
    return ("<|messages|>\n" + json.dumps(ordered(messages), ensure_ascii=False)
            + "\n<|tools|>\n" + json.dumps(ordered(tools), ensure_ascii=False)
            + "\n<|assistant|>")


def observe_real_dispatch(consumer, monkeypatch):
    """Spy on, and still execute, both real generate and actual forward calls."""
    observed = {"generate": [], "forward": []}
    original = consumer._model.generate

    def generate(*args, **kwargs):
        record = {
            "input_ids": tuple(kwargs["input_ids"][0].tolist()),
            "attention_mask": tuple(kwargs["attention_mask"][0].tolist()),
            "reserved": kwargs["generation_config"].max_new_tokens,
            "use_cache": kwargs["generation_config"].use_cache,
            "do_sample": kwargs["generation_config"].do_sample,
        }
        result = original(*args, **kwargs)
        record["result_ids"] = tuple(result[0].tolist())
        observed["generate"].append(record)
        return result

    def forward(_model, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        observed["forward"].append(tuple(ids[0].tolist()))

    monkeypatch.setattr(consumer._model, "generate", generate)
    handle = consumer._model.register_forward_pre_hook(forward, with_kwargs=True)
    return observed, handle


def assert_exact_dispatch(artifact, result, observed, *, expected_ids, reserved):
    assert artifact.input_ids == expected_ids
    assert artifact.attention_mask == (1,) * len(expected_ids)
    assert artifact.output_reserved_tokens == reserved
    assert len(expected_ids) + reserved <= artifact.model_window_tokens
    assert len(observed["generate"]) == 1
    actual = observed["generate"][0]
    assert actual["input_ids"] == expected_ids
    assert actual["attention_mask"] == artifact.attention_mask
    assert actual["reserved"] == reserved
    assert actual["use_cache"] is False
    assert actual["do_sample"] is False
    assert observed["forward"]
    assert observed["forward"][0] == expected_ids
    for step, forward_ids in enumerate(observed["forward"]):
        assert forward_ids == actual["result_ids"][:len(expected_ids) + step]
    assert result.input_token_count == len(expected_ids)
    assert result.output_ids == actual["result_ids"][len(expected_ids):]
    assert result.output_token_count == len(result.output_ids)
    assert 1 <= result.output_token_count <= reserved


def refuse_late_render(*_args, **_kwargs):
    raise AssertionError("Execution rendered or tokenized content after admission")


def test_real_model_receives_complete_frozen_messages_tools_and_reserved_output(consumer, snapshot, monkeypatch):
    messages, tools = complete_conversation()
    rendered = independently_render(messages, tools)
    expected_ids = tuple(snapshot.tokenizer.encode(rendered, add_special_tokens=False))
    assert snapshot.tokenizer.decode(expected_ids) == rendered
    artifact = consumer.compile(messages, tools, output_reserved_tokens=4)
    assert artifact.input_ids == expected_ids

    # Caller mutations cannot change any part of an admitted tool conversation.
    messages[0]["content"] = "caller-mutated system after admission"
    messages[2]["tool_calls"][0]["function"]["arguments"] = '{"changed":true}'
    messages[3]["content"] = "caller-mutated tool result"
    tools[0]["function"]["parameters"]["properties"]["subject"]["type"] = "integer"
    monkeypatch.setattr(type(consumer._tokenizer), "__call__", refuse_late_render)
    monkeypatch.setattr(consumer._tokenizer, "encode", refuse_late_render)
    monkeypatch.setattr(consumer._tokenizer, "apply_chat_template", refuse_late_render)
    observed, handle = observe_real_dispatch(consumer, monkeypatch)
    try:
        result = consumer.execute(artifact)
    finally:
        handle.remove()
    assert_exact_dispatch(artifact, result, observed, expected_ids=expected_ids, reserved=4)


def test_complete_framing_overflow_rejects_before_any_model_invocation(consumer, snapshot, monkeypatch):
    from daystrom_dml.contracts.model_input import ModelInputBudgetError

    messages, tools = complete_conversation()
    complete_ids = snapshot.tokenizer.encode(independently_render(messages, tools), add_special_tokens=False)
    content_ids = snapshot.tokenizer.encode("".join(message["content"] for message in messages), add_special_tokens=False)
    # The content alone fits this reservation. Complete role/tool framing does not.
    reserved = snapshot.manifest["context_window"] - len(complete_ids) + 1
    assert reserved > 0
    assert len(content_ids) + reserved <= snapshot.manifest["context_window"]
    invoked = []

    def forbidden(*_args, **_kwargs):
        invoked.append(True)
        raise AssertionError("Overflow reached the real model")

    monkeypatch.setattr(consumer._model, "generate", forbidden)
    monkeypatch.setattr(consumer._model, "forward", forbidden)
    with pytest.raises(ModelInputBudgetError):
        consumer.compile(messages, tools, output_reserved_tokens=reserved)
    assert invoked == []


@pytest.mark.parametrize("transport", ["python", "http"])
def test_receipted_profile_recall_is_admitted_with_complete_final_model_input(
    consumer, snapshot, tmp_path, monkeypatch, transport,
):
    from fastapi.testclient import TestClient

    from daystrom_dml.dml_adapter import DMLAdapter
    from daystrom_dml.provider_server import create_app
    from test_production_profile_integration import PinnedEmbedder, supported_config

    for name in tuple(os.environ):
        if name.startswith("DML_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    token = "real-model-input-integration-credential"
    monkeypatch.setenv("DML_API_TOKEN", token)
    config_path = tmp_path / "profile.json"
    config_path.write_text(json.dumps(supported_config(tmp_path / "authority")), encoding="utf-8")
    headers = {"Authorization": f"Bearer {token}"}
    text = "The owner prefers green notebooks."
    foreign = "Foreign owner secret must not reach the model."
    with ExitStack() as stack:
        adapter = DMLAdapter(config_path=config_path, embedder=PinnedEmbedder(), start_aging_loop=False)
        stack.callback(adapter.close)
        if transport == "python":
            receipt = adapter.ingest_memory_receipted(text, idempotency_key="one",
                meta={"source_trust": "trusted"}, **SCOPE)
            adapter.ingest_memory_receipted(foreign, idempotency_key="other",
                meta={"source_trust": "trusted"}, **{**SCOPE, "tenant_id": "foreign"})
            report = adapter.retrieve_context("notebook preference", **SCOPE)
        else:
            app = create_app(config_path=str(config_path), adapter_factory=lambda: adapter)
            client = stack.enter_context(TestClient(app))
            response = client.post("/api/remember/receipt", headers=headers, json={
                "text": text, "idempotency_key": "one", "meta": {"source_trust": "trusted"}, **SCOPE,
            })
            assert response.status_code == 200, response.text
            receipt = response.json()
            response = client.post("/api/remember/receipt", headers=headers, json={
                "text": foreign, "idempotency_key": "other", "meta": {"source_trust": "trusted"},
                **{**SCOPE, "tenant_id": "foreign"},
            })
            assert response.status_code == 200, response.text
            response = client.post("/api/recall", headers=headers,
                json={"query": "notebook preference", **SCOPE})
            assert response.status_code == 200, response.text
            report = response.json()
        assert {int(item["id"]) for item in report["items"]} == {receipt["result"]["memory"]["id"]}
        assert text in report["raw_context"]
        assert foreign not in report["raw_context"]
        messages, tools = complete_conversation(report["raw_context"])
        rendered = independently_render(messages, tools)
        expected_ids = tuple(snapshot.tokenizer.encode(rendered, add_special_tokens=False))
        artifact = consumer.compile(messages, tools, output_reserved_tokens=3)
        report["raw_context"] += " UNCHECKED CONTENT AFTER COMPILATION"
        observed, handle = observe_real_dispatch(consumer, monkeypatch)
        try:
            result = consumer.execute(artifact)
        finally:
            handle.remove()
        assert_exact_dispatch(artifact, result, observed, expected_ids=expected_ids, reserved=3)
        assert foreign not in rendered
        assert "UNCHECKED CONTENT" not in snapshot.tokenizer.decode(observed["generate"][0]["input_ids"])


def test_fixture_build_is_deterministic_and_contains_real_model_artifacts(tmp_path):
    first = create_snapshot(tmp_path / "first", context_window=64)
    second = create_snapshot(tmp_path / "second", context_window=64)
    assert first.manifest == second.manifest
    assert set(first.manifest["files"]) == {
        "config.json", "model.safetensors", "tokenizer.json", "chat_template.jinja",
    }
    assert sum(parameter.numel() for parameter in first.model.parameters()) > 1000
    assert first.model.__class__.__name__ == "GPT2LMHeadModel"
    assert first.tokenizer.__class__.__name__ == "PreTrainedTokenizerFast"
    assert first.model.device.type == "cpu"


@pytest.mark.parametrize("mandatory", [False, True])
def test_optional_runtime_admission_checks_versions_before_heavy_imports(monkeypatch, tmp_path, mandatory):
    installed = deepcopy(fixture_module.DEPENDENCY_PINS)
    installed["transformers"] = "unqualified-version"
    monkeypatch.setattr(fixture_module.metadata, "version", installed.__getitem__)
    if mandatory:
        monkeypatch.setenv(fixture_module.REQUIRE_ENV, "1")
    else:
        monkeypatch.delenv(fixture_module.REQUIRE_ENV, raising=False)
    imports = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "tokenizers", "safetensors"}:
            imports.append(name)
            raise AssertionError("Unqualified heavy dependency imported before pin admission")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    exception = pytest.fail.Exception if mandatory else pytest.skip.Exception
    with pytest.raises(exception, match="unqualified-version"):
        fixture_module.create_snapshot(tmp_path / "must-remain-absent")
    assert imports == []
    assert not (tmp_path / "must-remain-absent").exists()
