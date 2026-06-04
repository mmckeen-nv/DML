import json

import pytest

from daystrom_dml.api_contracts import ContractError
from daystrom_dml.cognition.schema import CognitivePacket
from daystrom_dml.contracts import (
    COGNITIVE_PACKET_V1,
    ContractRegistry,
    validate_cognitive_packet_v1,
)


def test_contract_registry_loads_cognitive_packet_schema():
    schema = ContractRegistry.load_schema(COGNITIVE_PACKET_V1)

    assert schema["$id"].endswith("cognitive-packet-v1.schema.json")
    assert schema["properties"]["packet_version"]["const"] == COGNITIVE_PACKET_V1
    assert "dcn_plan" in schema["required"]


def test_default_cognitive_packet_validates_against_registered_v1_schema():
    packet = CognitivePacket()

    result = validate_cognitive_packet_v1(packet.to_dict())

    assert result["valid"] is True
    assert result["schema_version"] == COGNITIVE_PACKET_V1
    assert result["errors"] == []


def test_cognitive_packet_v1_validator_rejects_bad_packet_version():
    payload = CognitivePacket().to_dict()
    payload["packet_version"] = "daystrom-cognitive-packet-v2"

    with pytest.raises(ContractError, match="packet_version"):
        validate_cognitive_packet_v1(payload)


def test_cognitive_packet_schema_artifact_is_json_serializable():
    schema = ContractRegistry.load_schema(COGNITIVE_PACKET_V1)

    encoded = json.dumps(schema, sort_keys=True)

    assert "daystrom-cognitive-packet-v1" in encoded
