"""Strict ephemeral contracts for one pinned exact-input consumer.

These values describe tokenized input, not a durable storage format or proof of
model quality. Consumer authentication is checked by the owning runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re


SUPPORTED_CHAT_TEMPLATE = "<|messages|>\n{{ messages | tojson }}\n<|tools|>\n{{ tools | tojson }}\n<|assistant|>\n"
SUPPORTED_CHAT_TEMPLATE_DIGEST = hashlib.sha256(SUPPORTED_CHAT_TEMPLATE.encode("utf-8")).hexdigest()
MAX_REQUEST_BYTES = 1024 * 1024
MAX_MESSAGES = 1024
MAX_TOOLS = 128
MAX_TOOL_CALLS = 128
MAX_INPUT_TOKENS = 1024 * 1024
_MAX_INTEGER = 2 ** 31 - 1
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_OPAQUE_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")


class ModelInputError(ValueError):
    """Input or artifact is outside the exact-input contract."""


class ModelInputBudgetError(ModelInputError):
    """The compiled input and reserved output exceed the pinned model window."""


def _text(value, *, nonempty=False, limit=MAX_REQUEST_BYTES):
    if type(value) is not str or (nonempty and not value.strip()):
        raise ModelInputError("Expected valid text")
    try:
        valid = len(value.encode("utf-8")) <= limit
    except UnicodeError:
        valid = False
    if not valid:
        raise ModelInputError("Text is invalid UTF-8 or exceeds its limit")


def _integer(value, *, minimum=0, maximum=_MAX_INTEGER):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ModelInputError("Expected a bounded integer")


def _digest(value):
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ModelInputError("Expected a canonical SHA-256 digest")


def _json_value(value, depth=0):
    if depth > 64:
        raise ModelInputError("JSON nesting exceeds 64 levels")
    if value is None or type(value) is bool:
        return
    if type(value) is str:
        _text(value)
    elif type(value) is int:
        _integer(value, minimum=-(2 ** 63), maximum=2 ** 63 - 1)
    elif type(value) is float:
        if not math.isfinite(value):
            raise ModelInputError("JSON numbers must be finite")
    elif type(value) is list:
        for item in value:
            _json_value(item, depth + 1)
    elif type(value) is dict:
        for key, item in value.items():
            _text(key)
            _json_value(item, depth + 1)
    else:
        raise ModelInputError("Expected strict JSON values")


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ModelInputError("Duplicate JSON object key")
        value[key] = item
    return value


def _constant(_value):
    raise ModelInputError("JSON numbers must be finite")


def _decode(raw):
    if type(raw) is not bytes or len(raw) > MAX_REQUEST_BYTES:
        raise ModelInputError("Expected bounded UTF-8 JSON bytes")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_object, parse_constant=_constant)
        _json_value(value)
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ModelInputError("Invalid exact-input JSON") from exc


def _keys(value, required, optional=()):
    if type(value) is not dict or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise ModelInputError("Missing or unsupported exact-input fields")


def _messages(messages):
    if type(messages) is not list or not 1 <= len(messages) <= MAX_MESSAGES:
        raise ModelInputError("Expected between one and 1024 text messages")
    for message in messages:
        _keys(message, ("role", "content"), ("name", "tool_call_id", "tool_calls"))
        role = message["role"]
        if type(role) is not str or role not in {"system", "user", "assistant", "tool"}:
            raise ModelInputError("Unsupported text message role")
        _text(message["content"])
        if "name" in message:
            _text(message["name"], nonempty=True, limit=256)
        if role == "tool":
            _text(message.get("tool_call_id"), nonempty=True, limit=256)
        elif "tool_call_id" in message:
            raise ModelInputError("tool_call_id is reserved for tool messages")
        if "tool_calls" not in message:
            continue
        calls = message["tool_calls"]
        if role != "assistant" or type(calls) is not list or len(calls) > MAX_TOOL_CALLS:
            raise ModelInputError("Invalid assistant tool calls")
        for call in calls:
            _keys(call, ("id", "type", "function"))
            _text(call["id"], nonempty=True, limit=256)
            if type(call["type"]) is not str or call["type"] != "function":
                raise ModelInputError("Only function tool calls are supported")
            _keys(call["function"], ("name", "arguments"))
            _text(call["function"]["name"], nonempty=True, limit=256)
            _text(call["function"]["arguments"])


def _tools(tools):
    if type(tools) is not list or len(tools) > MAX_TOOLS:
        raise ModelInputError("Expected at most 128 tool definitions")
    for tool in tools:
        _keys(tool, ("type", "function"))
        if type(tool["type"]) is not str or tool["type"] != "function":
            raise ModelInputError("Only function tool definitions are supported")
        function = tool["function"]
        _keys(function, ("name", "parameters"), ("description", "strict"))
        _text(function["name"], nonempty=True, limit=256)
        if type(function["parameters"]) is not dict:
            raise ModelInputError("Tool parameters must be a JSON object")
        if "description" in function:
            _text(function["description"])
        if "strict" in function and type(function["strict"]) is not bool:
            raise ModelInputError("Tool strict must be a boolean")


@dataclass(frozen=True, slots=True)
class ModelInputIdentity:
    model_digest: str
    tokenizer_digest: str
    chat_template_digest: str
    runtime_identity: str
    model_window_tokens: int

    def __post_init__(self):
        self.validate()

    def validate(self):
        _digest(self.model_digest)
        _digest(self.tokenizer_digest)
        _digest(self.chat_template_digest)
        _text(self.runtime_identity, nonempty=True, limit=1024)
        _integer(self.model_window_tokens, minimum=1)

    def to_payload(self):
        self.validate()
        return {"model_digest": self.model_digest, "tokenizer_digest": self.tokenizer_digest,
                "chat_template_digest": self.chat_template_digest, "runtime_identity": self.runtime_identity,
                "model_window_tokens": self.model_window_tokens}

    @property
    def identity_digest(self):
        return hashlib.sha256(_canonical(self.to_payload())).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelInputRequest:
    messages_json: bytes
    tools_json: bytes
    output_reserved_tokens: int

    def __post_init__(self):
        self.validate()

    @classmethod
    def from_payload(cls, payload):
        _keys(payload, ("messages", "output_reserved_tokens"), ("tools",))
        _json_value(payload)
        _messages(payload["messages"])
        tools = payload.get("tools", [])
        _tools(tools)
        return cls(_canonical(payload["messages"]), _canonical(tools), payload["output_reserved_tokens"])

    @classmethod
    def from_json(cls, payload: bytes):
        return cls.from_payload(_decode(payload))

    def validate(self):
        _integer(self.output_reserved_tokens, minimum=1)
        messages, tools = _decode(self.messages_json), _decode(self.tools_json)
        _messages(messages)
        _tools(tools)
        if _canonical(messages) != self.messages_json or _canonical(tools) != self.tools_json:
            raise ModelInputError("Request bytes must be canonical JSON")
        if len(_canonical({"messages": messages, "tools": tools,
                           "output_reserved_tokens": self.output_reserved_tokens})) > MAX_REQUEST_BYTES:
            raise ModelInputError("Exact-input request exceeds 1 MiB")

    def to_payload(self):
        self.validate()
        return {"messages": _decode(self.messages_json), "tools": _decode(self.tools_json),
                "output_reserved_tokens": self.output_reserved_tokens}

    @property
    def messages(self):
        return self.to_payload()["messages"]

    @property
    def tools(self):
        return self.to_payload()["tools"]

    @property
    def request_digest(self):
        return hashlib.sha256(_canonical(self.to_payload())).hexdigest()


@dataclass(frozen=True, slots=True)
class CompiledModelInput:
    identity: ModelInputIdentity
    request_digest: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    output_reserved_tokens: int
    model_window_tokens: int
    consumer_id: str
    nonce: str
    auth_tag: str

    def __post_init__(self):
        self.validate()

    def validate(self):
        if type(self.identity) is not ModelInputIdentity:
            raise ModelInputError("Compiled input requires an exact identity")
        self.identity.validate()
        _digest(self.request_digest)
        _digest(self.auth_tag)
        for value in (self.consumer_id, self.nonce):
            if type(value) is not str or _OPAQUE_ID.fullmatch(value) is None:
                raise ModelInputError("Invalid compiled-input ownership identity")
        if (type(self.input_ids) is not tuple or type(self.attention_mask) is not tuple
                or not 1 <= len(self.input_ids) <= MAX_INPUT_TOKENS
                or len(self.input_ids) != len(self.attention_mask)):
            raise ModelInputError("Compiled input requires bounded immutable token and mask tuples")
        for token in self.input_ids:
            _integer(token)
        for value in self.attention_mask:
            _integer(value, maximum=1)
        if not any(self.attention_mask):
            raise ModelInputError("Compiled input cannot have an entirely masked input")
        _integer(self.output_reserved_tokens, minimum=1)
        _integer(self.model_window_tokens, minimum=1)
        if self.model_window_tokens != self.identity.model_window_tokens:
            raise ModelInputError("Compiled input model window differs from its identity")
        if len(self.input_ids) + self.output_reserved_tokens > self.model_window_tokens:
            raise ModelInputBudgetError("Exact input and output reservation exceed the model window")

    @property
    def input_tokens(self):
        self.validate()
        return len(self.input_ids)

    def signing_payload(self):
        self.validate()
        return {"identity": self.identity.to_payload(), "request_digest": self.request_digest,
                "input_ids": list(self.input_ids), "attention_mask": list(self.attention_mask),
                "output_reserved_tokens": self.output_reserved_tokens,
                "model_window_tokens": self.model_window_tokens,
                "consumer_id": self.consumer_id, "nonce": self.nonce}

    @property
    def artifact_digest(self):
        return hashlib.sha256(_canonical(self.signing_payload())).hexdigest()
