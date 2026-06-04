"""Formal Daystrom Platform contract registry and lightweight validators.

The registry intentionally stays dependency-light: it loads checked-in JSON
Schema artifacts and performs the small subset of validation needed by runtime
contract gates without requiring the optional ``jsonschema`` package.
"""
from __future__ import annotations

import json
from importlib import resources
from typing import Any, Dict, List, Mapping

from daystrom_dml.api_contracts import ContractError

COGNITIVE_PACKET_V1 = "daystrom-cognitive-packet-v1"
SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"

_SCHEMA_FILES = {
    COGNITIVE_PACKET_V1: "cognitive-packet-v1.schema.json",
}


class ContractRegistry:
    """Load versioned Daystrom contract artifacts bundled with the package."""

    @classmethod
    def available_contracts(cls) -> List[str]:
        return sorted(_SCHEMA_FILES)

    @classmethod
    def load_schema(cls, contract_name: str) -> Dict[str, Any]:
        try:
            filename = _SCHEMA_FILES[contract_name]
        except KeyError as exc:
            raise ContractError(f"Unknown Daystrom contract schema: {contract_name}") from exc
        try:
            text = (
                resources.files("daystrom_dml.contracts.schemas")
                .joinpath(filename)
                .read_text(encoding="utf-8")
            )
        except FileNotFoundError as exc:  # pragma: no cover - packaging guard
            raise ContractError(f"Missing Daystrom contract schema artifact: {filename}") from exc
        return json.loads(text)


def validate_cognitive_packet_v1(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a payload against the Cognitive Packet v1 contract.

    Returns a compact success artifact on valid input. Raises ``ContractError``
    with path-qualified messages on invalid input.
    """

    schema = ContractRegistry.load_schema(COGNITIVE_PACKET_V1)
    errors = _validate_schema(payload, schema, path="$", root=schema)
    if errors:
        raise ContractError("Invalid cognitive packet v1: " + "; ".join(errors))
    return {"valid": True, "schema_version": COGNITIVE_PACKET_V1, "errors": []}


def _validate_schema(value: Any, schema: Mapping[str, Any], *, path: str, root: Mapping[str, Any]) -> List[str]:
    if "$ref" in schema:
        schema = _resolve_ref(root, str(schema["$ref"]))

    errors: List[str] = []

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected const {schema['const']!r}, got {value!r}")

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: expected one of {schema['enum']!r}, got {value!r}")

    if "type" in schema and not _matches_type(value, schema["type"]):
        errors.append(f"{path}: expected type {schema['type']!r}, got {type(value).__name__}")
        return errors

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: must be <= {schema['maximum']}")

    if isinstance(value, str) and "minLength" in schema and len(value) < int(schema["minLength"]):
        errors.append(f"{path}: string shorter than minLength {schema['minLength']}")

    if isinstance(value, Mapping):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: required property missing")
        properties = schema.get("properties", {})
        if isinstance(properties, Mapping):
            for key, child_schema in properties.items():
                if key in value and isinstance(child_schema, Mapping):
                    errors.extend(_validate_schema(value[key], child_schema, path=f"{path}.{key}", root=root))

    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        child_schema = schema["items"]
        for index, item in enumerate(value):
            errors.extend(_validate_schema(item, child_schema, path=f"{path}[{index}]", root=root))

    if "not" in schema and isinstance(schema["not"], Mapping):
        negated = schema["not"]
        if "enum" in negated and value in negated["enum"]:
            errors.append(f"{path}: value {value!r} is forbidden")

    return errors


def _resolve_ref(root: Mapping[str, Any], ref: str) -> Mapping[str, Any]:
    if not ref.startswith("#/"):
        raise ContractError(f"Unsupported schema ref: {ref}")
    node: Any = root
    for part in ref[2:].split("/"):
        if not isinstance(node, Mapping) or part not in node:
            raise ContractError(f"Unresolvable schema ref: {ref}")
        node = node[part]
    if not isinstance(node, Mapping):
        raise ContractError(f"Schema ref does not resolve to object: {ref}")
    return node


def _matches_type(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_matches_type(value, item) for item in expected)
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True
