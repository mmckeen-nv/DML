"""Trusted receipt-only tools for the bounded episode companion.

The model controls an action's narrow arguments.  This bridge owns scope, trust,
idempotency and immutable CAS references, which come only from complete receipts.
It is not a new supported-profile API or a provider capability expansion.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Sequence

from ..contracts.profile import PROFILE_ID
from ..contracts.agent_episode import episode_tool_definitions as episode_tool_definitions
from ..persistence import validate_record
from .receipt_ingestion import SCOPE_KEYS


class EpisodeToolError(ValueError):
    """An action cannot be admitted to this episode's tool authority."""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _digest(record: dict) -> str:
    # Receipt CAS uses its canonical JSON's default ASCII escaping.
    raw = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _scope(scope: dict) -> dict:
    if type(scope) is not dict or set(scope) != set(SCOPE_KEYS):
        raise EpisodeToolError("An exact complete trusted scope is required")
    for name, value in scope.items():
        if value is None and name != "tenant_id":
            continue
        if type(value) is not str or not value.strip() or len(value.encode("utf-8")) > 256:
            raise EpisodeToolError("Invalid trusted scope")
    return deepcopy(scope)


_ARGUMENTS: dict[str, dict[str, Any]] = {
    "retrieve": {"query": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 10}},
    "ingest": {"text": {"type": "string"}},
    "update": {"record_ref": {"type": "string"}, "text": {"type": "string"}, "reason": {"type": "string"}},
    "promote": {"record_refs": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 32},
                "text": {"type": "string"}, "reason": {"type": "string"}},
    "supersede": {"record_ref": {"type": "string"}, "replacement_ref": {"type": "string"}, "reason": {"type": "string"}},
    "retire": {"record_ref": {"type": "string"}, "reason": {"type": "string"}},
}


@dataclass(frozen=True, slots=True)
class PreparedEpisodeTool:
    """Owned canonical action and immutable operation identity."""

    name: str
    arguments_json: str
    idempotency_key: str | None

    @property
    def arguments(self) -> dict:
        return json.loads(self.arguments_json)

    def request_payload(self) -> dict:
        return {"name": self.name, "arguments": self.arguments,
                "idempotency_key": self.idempotency_key}


class SelectedProfileEpisodeTools:
    """Join selected-profile reads to trusted receipt records without rebasing CAS.

    References survive later writes unchanged.  A historical reference therefore
    reaches the receipt API with its original digest and can correctly conflict.
    A read that cannot be matched to a complete owned receipt has no CAS reference
    and supplies no complete-record verifier evidence.
    """

    def __init__(self, adapter, *, scope: dict, episode_id: str,
                 seed_receipts: Sequence[dict] = (), source_trust: str = "untrusted",
                 observation_records: Sequence[dict] = (), allowed_tools=None,
                 effective_time=2000000000):
        if adapter.production_profile_id != PROFILE_ID:
            raise EpisodeToolError("Episode tools require the selected receipt profile")
        if type(episode_id) is not str or not episode_id or len(episode_id.encode("utf-8")) > 128:
            raise EpisodeToolError("Invalid episode-owned key prefix")
        if source_trust != "untrusted":
            raise EpisodeToolError("Model-generated memory is always untrusted")
        self._adapter = adapter
        self._scope = _scope(scope)
        self._episode_id = episode_id
        self._source_trust = source_trust
        if type(effective_time) not in (int, float) or not math.isfinite(effective_time):
            raise EpisodeToolError("A finite trusted effective time is required")
        self.effective_time = effective_time
        allowed = tuple(_ARGUMENTS) if allowed_tools is None else tuple(allowed_tools)
        if not allowed or len(set(allowed)) != len(allowed) or set(allowed) - set(_ARGUMENTS):
            raise EpisodeToolError("Invalid frozen tool allowlist")
        self.allowed_tools = allowed
        self._records: dict[str, dict] = {}
        self._references: dict[str, str] = {}
        self._observations: list[dict] = []
        self._keys: dict[str, str] = {}
        self._prepared: dict[int, PreparedEpisodeTool] = {}
        for receipt in seed_receipts:
            self._register_receipt(receipt)
        for record in observation_records:
            validate_record(record)
            if all(record["meta"].get(key) == value for key, value in self._scope.items()):
                self._observations.append(deepcopy(record))

    @property
    def scope(self) -> dict:
        return deepcopy(self._scope)

    def _register_receipt(self, receipt: dict) -> str | None:
        if (type(receipt) is not dict or type(receipt.get("result")) is not dict
                or set(receipt["result"]) != {"memory"}):
            raise EpisodeToolError("A complete committed memory receipt is required")
        record = deepcopy(receipt["result"]["memory"])
        validate_record(record)
        if receipt.get("scope") != {key: record["meta"].get(key) for key in SCOPE_KEYS}:
            raise EpisodeToolError("Receipt record does not match its receipt scope")
        if not all(record["meta"].get(key) == value for key, value in self._scope.items()):
            return None
        digest = _digest(record)
        reference = self._references.setdefault(digest, "r" + str(len(self._references)))
        self._records[reference] = record
        self._observations.append(deepcopy(record))
        return reference

    def _record(self, reference: str) -> dict:
        if type(reference) is not str or reference not in self._records:
            raise EpisodeToolError("Record reference is unavailable in this episode")
        return deepcopy(self._records[reference])

    def prepare(self, name: str, arguments: dict, *, call_id: str) -> PreparedEpisodeTool:
        from ..contracts.agent_episode import parse_agent_action

        action = parse_agent_action(canonical({"schema_version": "dml-agent-action-v1",
            "kind": "tool", "name": name, "arguments": arguments}))
        arguments = action["arguments"]
        if name not in self.allowed_tools:
            raise EpisodeToolError("Tool is outside this task's frozen allowlist")
        for field in ("record_ref", "replacement_ref"):
            if field in arguments:
                self._record(arguments[field])
        for reference in arguments.get("record_refs", []):
            self._record(reference)
        if type(call_id) is not str or not call_id or len(call_id.encode("utf-8")) > 64:
            raise EpisodeToolError("Invalid runner-owned call identity")
        key = None if name == "retrieve" else "episode:" + self._episode_id + ":" + call_id
        frozen = canonical(arguments)
        if key is not None:
            request = canonical([name, arguments])
            if key in self._keys and self._keys[key] != request:
                raise EpisodeToolError("Call identity already owns a different action")
            self._keys[key] = request
        prepared = PreparedEpisodeTool(name, frozen, key)
        self._prepared[id(prepared)] = prepared
        return prepared

    @staticmethod
    def _public(record: dict, reference: str | None) -> dict:
        result = {"id": record["id"], "text": record["text"], "record_ref": reference}
        for name in ("source", "claim_key", "claim_value", "source_trust", "memory_state"):
            if name in record["meta"]:
                result[name] = deepcopy(record["meta"][name])
        return result

    def _match(self, item: dict) -> tuple[str | None, dict] | None:
        # Hydration adds these display-only fields. They are not authority and
        # never enter a CAS digest. No other metadata differences are ignored.
        projection = {"lattice_row", "lattice_col", "lattice_layer", "lattice_neighbors",
                      "lattice_degree", "lattice_policy", "lattice_id", "lattice_index", "lattice_size_hint"}
        metadata = {key: value for key, value in item.get("meta", {}).items() if key not in projection}
        for record in reversed(self._observations):
            if (str(record["id"]) != item.get("id")
                    or {key: value for key, value in record["meta"].items() if key not in projection} != metadata
                    or any(record[field] != item.get(field)
                           for field in ("timestamp", "level", "fidelity", "salience"))):
                continue
            text = record["text"].strip()
            decision = record["meta"].get("content_update_decision")
            if not (type(decision) is dict and decision.get("schema_version") == "dml-content-update-decision-v1"):
                text = str(record["meta"].get("summary") or "").strip() or text
            excerpt = item.get("text")
            if (type(excerpt) is str and (excerpt == text or (
                    excerpt.endswith("...") and text.startswith(excerpt[:-3])))):
                reference = self._references.get(_digest(record))
                return reference, deepcopy(record)
        return None

    def execute(self, prepared: PreparedEpisodeTool) -> tuple[dict, str]:
        if self._prepared.get(id(prepared)) is not prepared:
            raise EpisodeToolError("Only this bridge's prepared actions may execute")
        name, arguments = prepared.name, prepared.arguments
        if name == "retrieve":
            report = self._adapter.retrieve_context(arguments["query"], top_k=arguments["top_k"],
                as_of=self.effective_time, **self._scope)
            records, public = [], []
            for item in report["items"]:
                if not all(item.get("meta", {}).get(key) == value for key, value in self._scope.items()):
                    raise EpisodeToolError("Retrieval returned a record outside the trusted scope")
                match = self._match(item)
                if match is None:
                    public.append({"id": item["id"], "text": item["text"], "record_ref": None})
                else:
                    reference, record = match
                    records.append(record)
                    public.append(self._public(record, reference))
            return {"report": deepcopy(report), "receipt": None, "observed_records": records}, canonical({
                "records": public, "requested_top_k": arguments["top_k"],
                "returned_count": len(public), "limit_reached": len(public) == arguments["top_k"]})

        kwargs = {**self._scope, "idempotency_key": prepared.idempotency_key}
        if name == "ingest":
            receipt = self._adapter.ingest_memory_receipted(arguments["text"],
                meta={"source_trust": self._source_trust,
                      "source": "episode-generated:" + self._episode_id}, **kwargs)
        elif name == "promote":
            records = [self._record(reference) for reference in arguments["record_refs"]]
            receipt = self._adapter.promote_memories_receipted(
                [{"memory_id": record["id"], "expected_memory_digest": _digest(record)} for record in records],
                text=arguments["text"], reason=arguments["reason"], **kwargs)
        else:
            record = self._record(arguments["record_ref"])
            kwargs.update(expected_memory_digest=_digest(record), reason=arguments["reason"])
            if name == "update":
                receipt = self._adapter.update_memory_receipted(record["id"], text=arguments["text"], **kwargs)
            elif name == "retire":
                receipt = self._adapter.retire_memory_receipted(record["id"], **kwargs)
            else:
                replacement = self._record(arguments["replacement_ref"])
                receipt = self._adapter.supersede_memory_receipted(record["id"],
                    replacement_memory_id=replacement["id"], expected_replacement_digest=_digest(replacement), **kwargs)
        reference = self._register_receipt(receipt)
        if reference is None:
            raise EpisodeToolError("Write receipt escaped the trusted scope")
        record = self._record(reference)
        return {"receipt": deepcopy(receipt), "observed_records": [record]}, canonical({
            "records": [self._public(record, reference)]})
