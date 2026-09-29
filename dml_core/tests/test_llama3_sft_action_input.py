"""Portable contract controls; pinned runtime parity is qualified separately."""

from dataclasses import replace
from copy import deepcopy
from threading import RLock
from types import SimpleNamespace
import hashlib
import json
import pytest

from daystrom_dml.contracts.agent_episode import episode_tool_definitions
from daystrom_dml.contracts.model_input import ModelInputRequest, ModelInputIdentity, ModelInputError
from daystrom_dml.services.llama3_sft_action_input import (
    LocalLlama3SFTActionInputConsumer,
    source_order_request,
    RUNTIME_PREFIX,
)
from daystrom_dml.services import llama3_sft_runtime as runtime


def request():
    tools = [x for x in episode_tool_definitions() if x["function"]["name"] == "retrieve"]
    messages = [
        {"role": "system", "content": runtime.SYSTEM},
        {"role": "user", "content": "  literal <|eot_id|> task  "},
    ]
    return ModelInputRequest.from_payload(dict(messages=messages, tools=tools, output_reserved_tokens=8))


class Tokenizer:
    added_tokens_decoder = {
        128009: SimpleNamespace(special=True),
        128001: SimpleNamespace(special=True),
        128006: SimpleNamespace(special=True),
    }

    def apply_chat_template(self, view, **kwargs):
        return json.dumps(view)

    def encode(self, value, **kwargs):
        return [10, 11]

    def decode(self, ids, **kwargs):
        return "".join({1: "{", 2: "}", 128009: "<|eot_id|>", 128001: "<|end_of_text|>"}.get(x, "?") for x in ids)


def consumer():
    c = LocalLlama3SFTActionInputConsumer.__new__(LocalLlama3SFTActionInputConsumer)
    c._lock = RLock()
    c._closed = False
    c._offline = False
    c._identity = ModelInputIdentity("1" * 64, "2" * 64, "3" * 64, RUNTIME_PREFIX + "4" * 64, 8192)
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
            output_ids=[1, 2, 128009],
            text="{}",
        )
    )
    return c


def test_restores_source_order_and_preserves_literal_data():
    r = request()
    messages, tools = source_order_request(r)
    assert list(messages[0]) == ["role", "content"]
    assert list(tools[0]["function"]) == ["name", "description", "parameters"]
    assert tools == r.tools
    view = runtime.render_messages(messages, tools)
    assert "\\u003c|eot_id|\\u003e" in view[1]["content"]
    assert json.loads(view[1]["content"]) == messages[1]
    assert messages[1]["content"].startswith("  ") and messages[1]["content"].endswith("  ")


def test_feedback_order_preserves_argument_and_result_bytes():
    r = request()
    m = r.messages
    args = '{"query":" a ","top_k":1}'
    m += [
        {
            "role": "assistant",
            "content": "raw action",
            "tool_calls": [{"id": "tool-1", "type": "function", "function": {"name": "retrieve", "arguments": args}}],
        },
        {"role": "tool", "tool_call_id": "tool-1", "name": "retrieve", "content": "  exact feedback  "},
    ]
    q = ModelInputRequest.from_payload(dict(messages=m, tools=r.tools, output_reserved_tokens=8))
    ordered, _ = source_order_request(q)
    assert list(ordered[-2]) == ["role", "content", "tool_calls"]
    assert list(ordered[-2]["tool_calls"][0]) == ["id", "type", "function"]
    assert list(ordered[-1]) == ["role", "tool_call_id", "name", "content"]
    assert ordered[-2]["tool_calls"][0]["function"]["arguments"] == args
    assert ordered[-1]["content"] == "  exact feedback  "
    assert q.to_payload() == ModelInputRequest.from_payload(q.to_payload()).to_payload()


def test_changed_tool_definition_rejected():
    r = request()
    tools = deepcopy(r.tools)
    tools[0]["function"]["description"] = "different"
    q = ModelInputRequest.from_payload(dict(messages=r.messages, tools=tools, output_reserved_tokens=8))
    with pytest.raises(ModelInputError):
        source_order_request(q)


def test_authenticated_single_use_and_raw_ids():
    c = consumer()
    r = request()
    a = c.compile(r.messages, r.tools, output_reserved_tokens=8)
    result = c.execute(a)
    assert result.output_ids == (1, 2, 128009) and result.text == "{}"
    assert result.output_token_count == 3
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


def test_offline_cannot_execute():
    c = consumer()
    r = request()
    a = c.compile(r.messages, r.tools, output_reserved_tokens=8)
    c._offline = True
    with pytest.raises(ModelInputError):
        c.execute(a)


def test_unknown_failure_retains_partial_raw_evidence_and_consumes_artifact():
    from daystrom_dml.services.model_input import ModelInputExecutionError

    c = consumer()
    r = request()
    a = c.compile(r.messages, r.tools, output_reserved_tokens=8)
    c._runtime.execute = lambda _: dict(usage_unknown=True, execution_error="failure", output_ids=[1], raw_text="{")
    with pytest.raises(ModelInputExecutionError):
        c.execute(a)
    assert c.last_execution["output_ids"] == [1]
    with pytest.raises(ModelInputError):
        c.execute(a)


@pytest.mark.parametrize("reservation", [257, 8192])
def test_reservation_limit(reservation):
    c = consumer()
    r = request()
    with pytest.raises(ModelInputError):
        c.compile(r.messages, r.tools, output_reserved_tokens=reservation)


@pytest.mark.parametrize("stop", [128001, 128009])
def test_terminal_stop_decode_only(stop):
    assert consumer().decode_output([1, 2, stop]) == "{}"
    assert consumer().decode_output([stop, 1]).startswith("<|")


def test_format_projection_accepts_complete_at_cap_but_not_incomplete():
    class T(Tokenizer):
        def decode(self, ids, **kwargs):
            return (
                '{"schema_version":"dml-agent-action-v1","kind":"final","answer":{"claims":[]}}' if ids == [5] else "{"
            )

    assert runtime.project_output([5], T(), request().tools)["action_error"] is None
    assert runtime.project_output([1], T(), request().tools)["action_error"] is not None


def test_no_prompt_policy_change():
    from daystrom_dml.contracts.agent_episode import initial_messages, LLAMA3_SFT_CONSUMER_PROFILE

    assert initial_messages("test", consumer_profile=LLAMA3_SFT_CONSUMER_PROFILE)[0]["content"] == runtime.SYSTEM
    assert (
        hashlib.sha256(runtime.SYSTEM.encode()).hexdigest()
        == "b0ec3d6f643b4363ba6e2765cbea937588f78ec1da4372cdbf1342f934b8cdc4"
    )


def test_failure_wrapper_retains_known_response_without_success_credit():
    c = consumer()
    r = request()
    a = c.compile(r.messages, r.tools, output_reserved_tokens=8)
    c.validate_output_tokens = lambda req, ids: c.decode_output(ids)
    result = dict(
        model_identity=a.identity.to_payload(),
        input_ids=list(a.input_ids),
        output_ids=[1, 2, 128009],
        raw_text="{}<|eot_id|>",
        usage_unknown=False,
        input_token_count=2,
        output_token_count=3,
        execution_error=None,
    )
    envelope = dict(
        artifact_digest="b" * 64,
        exception_type="ValueError",
        exception_message="outer binding failure",
        runtime_result=result,
    )
    c.validate_failed_execution(a, r, envelope, recorded_digest="b" * 64)
    result["output_token_count"] = 4
    with pytest.raises(ModelInputError):
        c.validate_failed_execution(a, r, envelope, recorded_digest="b" * 64)


def test_absent_failure_raw_text_requires_decode_error():
    c = consumer()
    r = request()
    a = c.compile(r.messages, r.tools, output_reserved_tokens=8)
    c.validate_output_tokens = lambda req, ids: c.decode_output(ids)
    result = dict(
        model_identity=a.identity.to_payload(),
        input_ids=list(a.input_ids),
        output_ids=[1],
        raw_text=None,
        usage_unknown=True,
        execution_error="generation error",
    )
    envelope = dict(
        artifact_digest=a.artifact_digest,
        exception_type="ModelInputExecutionError",
        exception_message="failed",
        runtime_result=result,
    )
    with pytest.raises(ModelInputError):
        c.validate_failed_execution(a, r, envelope)
    result["decode_error"] = "retained decode failure"
    c.validate_failed_execution(a, r, envelope)


def test_previous_execution_evidence_reset_before_authentication():
    c = consumer()
    r = request()
    a = c.compile(r.messages, r.tools, output_reserved_tokens=8)
    c.execute(a)
    assert c.last_execution is not None
    with pytest.raises(ModelInputError):
        c.execute(a)
    assert c.last_execution is None


def test_offline_release_is_owned_and_cannot_enable_online_reuse():
    c = consumer()
    r = request()
    a = c.compile(r.messages, r.tools, output_reserved_tokens=8)
    with pytest.raises(ModelInputError):
        c.release_compiled(a)
    c._offline = True
    c.release_compiled(a)
    with pytest.raises(ModelInputError):
        c.release_compiled(a)


def test_revision_label_cannot_admit_self_selected_base_inventory(tmp_path):
    from daystrom_dml.services.llama3_sft_action_input import (
        verify_manifest,
        MANIFEST_NAME,
        MANIFEST_VERSION,
        MODEL,
        REVISION,
        IMAGE,
        VERSIONS,
    )

    (tmp_path / "model").mkdir()
    (tmp_path / "adapter").mkdir()
    manifest = dict(
        schema_version=MANIFEST_VERSION,
        model_id=MODEL,
        revision=REVISION,
        adapter_enabled=True,
        model_directory="model",
        adapter_directory="adapter",
        files={},
        runtime={"versions": VERSIONS, "image": IMAGE},
    )
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(ModelInputError, match="official revision"):
        verify_manifest(tmp_path)


@pytest.mark.parametrize("relative", ["../outside", "/absolute"])
def test_snapshot_paths_cannot_escape(tmp_path, relative):
    from daystrom_dml.services.llama3_sft_action_input import _relative

    with pytest.raises(ModelInputError):
        _relative(tmp_path, relative)
