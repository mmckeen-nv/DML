"""Independent value, identity and alias checks for exact-input contracts."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import pytest

from daystrom_dml.contracts.model_input import (
    CompiledModelInput, MAX_INPUT_TOKENS, MAX_REQUEST_BYTES, ModelInputBudgetError,
    ModelInputError, ModelInputIdentity, ModelInputRequest,
    SUPPORTED_CHAT_TEMPLATE, SUPPORTED_CHAT_TEMPLATE_DIGEST,
)


def payload():
    return {
        "messages": [
            {"role": "system", "content": "Use retrieved evidence.", "name": "policy"},
            {"role": "user", "content": "Café 漢字 🌍\nlookup", "name": "caller"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call-1", "type": "function", "function": {
                    "name": "recall", "arguments": '{"query":"Ω"}'}}]},
            {"role": "tool", "content": "Retrieved context", "name": "recall", "tool_call_id": "call-1"},
        ],
        "tools": [{"type": "function", "function": {
            "name": "recall", "description": "Scoped lookup", "strict": True,
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1}},
                "required": ["query"], "additionalProperties": False}}}],
        "output_reserved_tokens": 8,
    }


def identity(**changes):
    values = {"model_digest": "1" * 64, "tokenizer_digest": "2" * 64,
              "chat_template_digest": SUPPORTED_CHAT_TEMPLATE_DIGEST,
              "runtime_identity": "dml-model-input-runtime-v1:pinned", "model_window_tokens": 32}
    values.update(changes)
    return ModelInputIdentity(**values)


def compiled(**changes):
    values = {"identity": identity(), "request_digest": ModelInputRequest.from_payload(payload()).request_digest,
              "input_ids": (1, 8, 3), "attention_mask": (1, 1, 1),
              "output_reserved_tokens": 8, "model_window_tokens": 32,
              "consumer_id": "consumer-1", "nonce": "compile-1", "auth_tag": "0" * 64}
    values.update(changes)
    return CompiledModelInput(**values)


def test_every_message_and_tool_field_survives_canonical_roundtrip():
    original = payload()
    request = ModelInputRequest.from_payload(original)
    assert request.to_payload() == original
    assert request.messages == original["messages"]
    assert request.tools == original["tools"]
    expected = json.dumps(original, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert request.request_digest == hashlib.sha256(expected).hexdigest()
    assert ModelInputRequest.from_json(expected) == request
    assert SUPPORTED_CHAT_TEMPLATE == "<|messages|>\n{{ messages | tojson }}\n<|tools|>\n{{ tools | tojson }}\n<|assistant|>\n"
    assert SUPPORTED_CHAT_TEMPLATE_DIGEST == hashlib.sha256(SUPPORTED_CHAT_TEMPLATE.encode()).hexdigest()


def test_omitted_tools_normalize_to_the_same_empty_definition_list():
    raw = {"messages": [{"role": "user", "content": ""}], "output_reserved_tokens": 1}
    first = ModelInputRequest.from_payload(raw)
    raw["tools"] = []
    second = ModelInputRequest.from_payload(raw)
    assert first == second
    assert first.request_digest == second.request_digest


def test_request_owns_canonical_bytes_and_all_accessors_return_detached_values():
    raw = payload()
    original = deepcopy(raw)
    request = ModelInputRequest.from_payload(raw)
    digest = request.request_digest
    raw["messages"][2]["tool_calls"][0]["function"]["arguments"] = "changed"
    raw["tools"][0]["function"]["parameters"]["required"].append("limit")
    request.messages[0]["content"] = "changed"
    request.tools.clear()
    exported = request.to_payload()
    exported["messages"].clear()
    assert request.to_payload() == original
    assert request.request_digest == digest
    with pytest.raises(FrozenInstanceError):
        request.output_reserved_tokens = 10
    with pytest.raises(FrozenInstanceError):
        request.messages_json = b"[]"
    assert not hasattr(request, "__dict__")


@pytest.mark.parametrize("key", ["messages", "output_reserved_tokens"])
def test_required_request_fields_cannot_be_omitted(key):
    raw = payload()
    del raw[key]
    with pytest.raises(ModelInputError):
        ModelInputRequest.from_payload(raw)


@pytest.mark.parametrize("raw", [None, [], "request", True])
def test_request_root_requires_plain_object(raw):
    with pytest.raises(ModelInputError):
        ModelInputRequest.from_payload(raw)


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, "8", None, 2 ** 31, 10 ** 100])
def test_output_reservation_is_positive_bounded_integer(value):
    raw = payload()
    raw["output_reserved_tokens"] = value
    with pytest.raises(ModelInputError):
        ModelInputRequest.from_payload(raw)


@pytest.mark.parametrize("message", [
    {}, {"role": "user"}, {"role": "developer", "content": "x"},
    {"role": 1, "content": "x"}, {"role": "user", "content": None},
    {"role": "user", "content": [{"type": "image_url", "url": "hidden"}]},
    {"role": "user", "content": "x", "unknown": "private"},
    {"role": "user", "content": "x", "name": None},
    {"role": "user", "content": "x", "name": " "},
    {"role": "tool", "content": "x"},
    {"role": "tool", "content": "x", "tool_call_id": 1},
    {"role": "user", "content": "x", "tool_call_id": "call"},
    {"role": "user", "content": "x", "tool_calls": []},
    {"role": "assistant", "content": "", "tool_calls": None},
    {"role": "assistant", "content": "", "tool_calls": [{}]},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "other", "function": {}}]},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": {}}}]},
])
def test_unsupported_message_shapes_fail_closed(message):
    raw = payload()
    raw["messages"] = [message]
    with pytest.raises(ModelInputError):
        ModelInputRequest.from_payload(raw)


@pytest.mark.parametrize("path", ["request", "message", "call", "call_function", "tool", "tool_function"])
def test_unknown_fields_are_never_silently_dropped(path):
    raw = payload()
    target = {"request": raw, "message": raw["messages"][0],
              "call": raw["messages"][2]["tool_calls"][0],
              "call_function": raw["messages"][2]["tool_calls"][0]["function"],
              "tool": raw["tools"][0], "tool_function": raw["tools"][0]["function"]}[path]
    target["unexpected"] = "private"
    with pytest.raises(ModelInputError):
        ModelInputRequest.from_payload(raw)


@pytest.mark.parametrize("tools", [None, {}, (), [{"type": "other", "function": {}}],
    [{"type": "function", "function": {"name": "f"}}],
    [{"type": "function", "function": {"name": "f", "parameters": []}}],
    [{"type": "function", "function": {"name": "f", "parameters": {}, "strict": 1}}],
    [{"type": "function", "function": {"name": "f", "parameters": {}, "description": None}}],
])
def test_tool_definition_shape_is_strict(tools):
    raw = payload()
    raw["tools"] = tools
    with pytest.raises(ModelInputError):
        ModelInputRequest.from_payload(raw)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"),
    2 ** 63, -(2 ** 63) - 1, ("tuple",), {"set"}, b"bytes", object(), "\ud800"])
def test_nested_tool_schema_requires_finite_strict_utf8_json(value):
    raw = payload()
    raw["tools"][0]["function"]["parameters"]["nested"] = {"value": value}
    with pytest.raises(ModelInputError):
        ModelInputRequest.from_payload(raw)


@pytest.mark.parametrize("location", ["content", "name", "key", "arguments"])
def test_surrogates_are_rejected_in_all_text_positions(location):
    raw = payload()
    if location == "key":
        raw["tools"][0]["function"]["parameters"]["\udfff"] = 1
    elif location == "arguments":
        raw["messages"][2]["tool_calls"][0]["function"]["arguments"] = "\udfff"
    else:
        raw["messages"][0][location] = "\udfff"
    with pytest.raises(ModelInputError):
        ModelInputRequest.from_payload(raw)


@pytest.mark.parametrize("raw", [
    b'{"messages":[],"messages":[],"output_reserved_tokens":1}',
    b'{"messages":[{"role":"user","role":"system","content":"x"}],"output_reserved_tokens":1}',
    b'{"messages":[{"role":"user","content":"x"}],"tools":[],"output_reserved_tokens":NaN}',
    b'{"messages":[{"role":"user","content":"\\ud800"}],"output_reserved_tokens":1}',
    b'\xff', b'[]', b'{', b'{}' + b' ' * MAX_REQUEST_BYTES,
], ids=["duplicate-root-key", "duplicate-message-key", "nonfinite-number", "surrogate",
        "invalid-utf8", "array-root", "incomplete-object", "oversized-json"])
def test_raw_json_rejects_duplicate_invalid_nonfinite_or_oversized_values(raw):
    with pytest.raises(ModelInputError):
        ModelInputRequest.from_json(raw)


def test_size_limits_count_complete_envelope_and_utf8_bytes():
    raw = {"messages": [{"role": "user", "content": "x" * (MAX_REQUEST_BYTES - 85)}],
           "tools": [], "output_reserved_tokens": 1}
    # Compute the independent canonical wire length, including all framing.
    overhead = len(json.dumps({**raw, "messages": [{"role": "user", "content": ""}]},
                             sort_keys=True, separators=(",", ":")).encode())
    raw["messages"][0]["content"] = "x" * (MAX_REQUEST_BYTES - overhead)
    assert len(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()) == MAX_REQUEST_BYTES
    ModelInputRequest.from_payload(raw)
    raw["messages"][0]["content"] += "x"
    with pytest.raises(ModelInputError):
        ModelInputRequest.from_payload(raw)
    raw["messages"][0]["content"] = "é" * (MAX_REQUEST_BYTES // 2)
    with pytest.raises(ModelInputError):
        ModelInputRequest.from_payload(raw)


def test_nested_depth_cycles_and_collection_counts_are_bounded():
    raw = payload()
    nested = {}
    cursor = nested
    for _ in range(65):
        cursor["child"] = {}
        cursor = cursor["child"]
    raw["tools"][0]["function"]["parameters"] = nested
    with pytest.raises(ModelInputError):
        ModelInputRequest.from_payload(raw)
    nested["cycle"] = nested
    with pytest.raises(ModelInputError):
        ModelInputRequest.from_payload(raw)
    for field, value in (("messages", [{"role": "user", "content": ""}] * 1025),
                         ("tools", payload()["tools"] * 129)):
        oversized = payload()
        oversized[field] = value
        with pytest.raises(ModelInputError):
            ModelInputRequest.from_payload(oversized)
    raw = payload()
    raw["messages"][2]["tool_calls"] *= 129
    with pytest.raises(ModelInputError):
        ModelInputRequest.from_payload(raw)


def test_direct_request_constructor_requires_canonical_immutable_bytes():
    request = ModelInputRequest.from_payload(payload())
    with pytest.raises(ModelInputError):
        ModelInputRequest(bytearray(request.messages_json), request.tools_json, 1)
    with pytest.raises(ModelInputError):
        ModelInputRequest(b" " + request.messages_json, request.tools_json, 1)
    with pytest.raises(ModelInputError):
        ModelInputRequest(b"[]", request.tools_json, 1)


@pytest.mark.parametrize("field,value", [
    ("model_digest", "A" * 64), ("tokenizer_digest", "x" * 64),
    ("chat_template_digest", "sha256:" + "a" * 64), ("runtime_identity", " "),
    ("runtime_identity", "\ud800"), ("runtime_identity", "x" * 1025),
    ("model_window_tokens", True), ("model_window_tokens", 0),
    ("model_window_tokens", 32.0), ("model_window_tokens", 2 ** 31),
])
def test_identity_is_strict_and_bounded(field, value):
    with pytest.raises(ModelInputError):
        identity(**{field: value})


def test_identity_digest_binds_every_identity_field_and_is_detached():
    value = identity()
    before = value.identity_digest
    exported = value.to_payload()
    exported["model_digest"] = "0" * 64
    assert value.identity_digest == before
    for field, replacement in {"model_digest": "9" * 64, "tokenizer_digest": "9" * 64,
                              "chat_template_digest": "9" * 64, "runtime_identity": "different",
                              "model_window_tokens": 33}.items():
        assert replace(value, **{field: replacement}).identity_digest != before
    with pytest.raises(FrozenInstanceError):
        value.model_window_tokens = 64
    assert not hasattr(value, "__dict__")


@pytest.mark.parametrize("field,value", [
    ("identity", {}), ("request_digest", "bad"), ("auth_tag", "bad"),
    ("consumer_id", " "), ("nonce", "💡"), ("nonce", "x" * 129),
    ("input_ids", [1, 2, 3]), ("input_ids", ()), ("input_ids", (True, 2, 3)),
    ("input_ids", (-1, 2, 3)), ("input_ids", (1.0, 2, 3)),
    ("input_ids", (2 ** 31, 2, 3)),
    ("attention_mask", [1, 1, 1]), ("attention_mask", (1, 1)),
    ("attention_mask", (True, 1, 1)), ("attention_mask", (1, 2, 1)),
    ("attention_mask", (0, 0, 0)), ("output_reserved_tokens", True),
    ("output_reserved_tokens", 0), ("model_window_tokens", 33),
])
def test_compiled_shape_cannot_coerce_or_drop_inputs(field, value):
    with pytest.raises(ModelInputError):
        compiled(**{field: value})


def test_input_and_reservation_exact_window_boundary_counts_full_sequence():
    accepted = compiled(identity=identity(model_window_tokens=11), model_window_tokens=11)
    assert accepted.input_tokens == 3
    with pytest.raises(ModelInputBudgetError):
        replace(accepted, output_reserved_tokens=9)
    with pytest.raises(ModelInputBudgetError):
        compiled(identity=identity(model_window_tokens=10), model_window_tokens=10,
                 attention_mask=(0, 1, 1))
    with pytest.raises(ModelInputError):
        compiled(input_ids=(1,) * (MAX_INPUT_TOKENS + 1), attention_mask=(1,) * (MAX_INPUT_TOKENS + 1))


def test_artifact_digest_binds_every_consumed_field_except_authentication_tag():
    value = compiled()
    before = value.artifact_digest
    assert replace(value, auth_tag="f" * 64).artifact_digest == before
    replacements = [
        {"identity": identity(model_digest="9" * 64)}, {"request_digest": "9" * 64},
        {"input_ids": (8, 1, 3)}, {"attention_mask": (0, 1, 1)},
        {"output_reserved_tokens": 9},
        {"identity": identity(model_window_tokens=33), "model_window_tokens": 33},
        {"consumer_id": "other"}, {"nonce": "other"},
    ]
    for replacement in replacements:
        assert replace(value, **replacement).artifact_digest != before
    unsigned = value.signing_payload()
    unsigned["input_ids"].clear()
    unsigned["identity"]["model_digest"] = "0" * 64
    assert value.artifact_digest == before
    with pytest.raises(FrozenInstanceError):
        value.input_ids = (9,)
    assert not hasattr(value, "__dict__")


def test_revalidation_rejects_low_level_mutation_instead_of_trusting_cached_digest():
    request = ModelInputRequest.from_payload(payload())
    object.__setattr__(request, "messages_json", b"[]")
    with pytest.raises(ModelInputError):
        request.to_payload()
    value = compiled()
    before = value.artifact_digest
    object.__setattr__(value, "input_ids", (9, 8, 3))
    assert value.artifact_digest != before
    object.__setattr__(value, "attention_mask", (True, 1, 1))
    with pytest.raises(ModelInputError):
        value.validate()


def test_logical_tool_pairing_is_not_claimed_by_the_serialization_contract():
    raw = {"messages": [{"role": "tool", "content": "Standalone record", "tool_call_id": "external-call"}],
           "tools": [], "output_reserved_tokens": 1}
    assert ModelInputRequest.from_payload(raw).to_payload() == raw
