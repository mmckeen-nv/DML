import json

import pytest

from daystrom_dml.api_contracts import ContractError
from daystrom_dml.cognition.schema import CognitivePacket
from daystrom_dml.context import (
    CONTEXT_MANIFEST_V1,
    CONTEXT_PACKET_V1,
    ContextBudget,
    ContextManifest,
    ContextPacket,
    ContextSegment,
    RuntimeCapabilities,
)
from daystrom_dml.context.capabilities import RUNTIME_CAPABILITIES_V1
from daystrom_dml.contracts import (
    COGNITIVE_PACKET_V1,
    ContractRegistry,
    validate_cognitive_packet_v1,
    validate_contract,
)


def test_contract_registry_loads_cognitive_packet_schema():
    schema = ContractRegistry.load_schema(COGNITIVE_PACKET_V1)

    assert schema["$id"].endswith("cognitive-packet-v1.schema.json")
    assert schema["properties"]["packet_version"]["const"] == COGNITIVE_PACKET_V1
    assert "dcn_plan" in schema["required"]


def test_contract_registry_discovers_context_contract_schemas():
    assert {
        COGNITIVE_PACKET_V1,
        CONTEXT_MANIFEST_V1,
        CONTEXT_PACKET_V1,
        RUNTIME_CAPABILITIES_V1,
    }.issubset(set(ContractRegistry.available_contracts()))

    expected_files = {
        CONTEXT_MANIFEST_V1: "context-manifest-v1.schema.json",
        CONTEXT_PACKET_V1: "context-packet-v1.schema.json",
        RUNTIME_CAPABILITIES_V1: "runtime-capabilities-v1.schema.json",
    }
    for contract_name, filename in expected_files.items():
        schema = ContractRegistry.load_schema(contract_name)
        assert schema["$id"].endswith(filename)


def test_default_cognitive_packet_validates_against_registered_v1_schema():
    packet = CognitivePacket()

    result = validate_cognitive_packet_v1(packet.to_dict())

    assert result["valid"] is True
    assert result["schema_version"] == COGNITIVE_PACKET_V1
    assert result["errors"] == []


def test_context_contract_payloads_validate_against_registered_schemas():
    segment = ContextSegment(segment_id="seg-1", kind="message", content="hello", estimated_tokens=3)
    capabilities = RuntimeCapabilities.api_only(model_id="model", backend_id="api")
    manifest = ContextManifest(
        model_id=capabilities.model_id,
        runtime_id=capabilities.backend_id,
        segment_ids=[segment.segment_id],
        estimated_input_tokens=segment.effective_tokens,
    )
    packet = ContextPacket(
        capabilities=capabilities,
        budget=ContextBudget(model_limit_tokens=100, admitted_input_tokens=segment.effective_tokens),
        segments=[segment],
        manifest=manifest,
    )

    for contract_name, payload in {
        CONTEXT_MANIFEST_V1: manifest.to_dict(),
        CONTEXT_PACKET_V1: packet.to_dict(),
        RUNTIME_CAPABILITIES_V1: capabilities.to_dict(),
    }.items():
        result = validate_contract(contract_name, payload)
        assert result["valid"] is True
        assert result["schema_version"] == contract_name
        assert result["errors"] == []


def test_cognitive_packet_v1_validator_rejects_bad_packet_version():
    payload = CognitivePacket().to_dict()
    payload["packet_version"] = "daystrom-cognitive-packet-v2"

    with pytest.raises(ContractError, match="packet_version"):
        validate_cognitive_packet_v1(payload)


def test_context_contract_validation_rejects_invalid_payloads():
    capabilities = RuntimeCapabilities.api_only(model_id="model", backend_id="api")
    manifest = ContextManifest(model_id="model", runtime_id="api")
    packet = ContextPacket(
        capabilities=capabilities,
        budget=ContextBudget(model_limit_tokens=100),
        manifest=manifest,
    )

    bad_manifest = manifest.to_dict()
    bad_manifest["manifest_version"] = "daystrom-context-manifest-v2"
    with pytest.raises(ContractError, match="manifest_version"):
        validate_contract(CONTEXT_MANIFEST_V1, bad_manifest)

    bad_packet = packet.to_dict()
    bad_packet["packet_version"] = "daystrom-context-packet-v2"
    with pytest.raises(ContractError, match="packet_version"):
        validate_contract(CONTEXT_PACKET_V1, bad_packet)

    bad_capabilities = capabilities.to_dict()
    del bad_capabilities["model_id"]
    with pytest.raises(ContractError, match="model_id"):
        validate_contract(RUNTIME_CAPABILITIES_V1, bad_capabilities)


def test_cognitive_packet_schema_artifact_is_json_serializable():
    schema = ContractRegistry.load_schema(COGNITIVE_PACKET_V1)

    encoded = json.dumps(schema, sort_keys=True)

    assert "daystrom-cognitive-packet-v1" in encoded
