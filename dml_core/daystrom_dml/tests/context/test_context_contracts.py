import json
from pathlib import Path

import pytest

from daystrom_dml.api_contracts import ContractError, DaystromScope
from daystrom_dml.context import (
    CONTEXT_MANIFEST_V1,
    CONTEXT_PACKET_V1,
    ContextAuthority,
    ContextBudget,
    ContextManifest,
    ContextPacket,
    ContextPriority,
    ContextSegment,
    RuntimeCapabilities,
)
from daystrom_dml.contracts import _validate_schema


SCHEMA_DIR = Path(__file__).resolve().parents[2] / "contracts" / "schemas"


def test_context_enums_use_stable_wire_values():
    assert [item.value for item in ContextAuthority] == [
        "immutable",
        "current_instruction",
        "trusted_control",
        "reference",
        "untrusted_data",
    ]
    assert [item.value for item in ContextPriority] == ["critical", "working", "reference", "disposable"]


def test_context_segment_roundtrip_and_effective_tokens():
    segment = ContextSegment(
        segment_id="seg-1",
        kind="memory",
        content={"text": "Keep answers terse."},
        authority="reference",
        priority=ContextPriority.WORKING,
        source={"system": "dml"},
        provenance={"record_id": "mem-1"},
        cache={"cache_key": "abc"},
        retention={"policy": "session"},
        estimated_tokens=7,
        exact_tokens=5,
    )

    assert segment.effective_tokens == 5
    assert segment.to_dict()["authority"] == "reference"
    assert ContextSegment.from_dict(segment.to_dict()) == segment


def test_context_segment_validates_required_identity_and_token_counts():
    with pytest.raises(ContractError, match="segment_id"):
        ContextSegment(segment_id="", kind="memory", content="x")

    with pytest.raises(ContractError, match="estimated_tokens"):
        ContextSegment(segment_id="seg-1", kind="memory", content="x", estimated_tokens=-1)

    with pytest.raises(ContractError, match="exact_tokens"):
        ContextSegment(segment_id="seg-1", kind="memory", content="x", exact_tokens=-1)


def test_context_budget_admission_is_deterministic_and_bounded():
    budget = ContextBudget(
        model_limit_tokens=100,
        output_reserved_tokens=20,
        runtime_reserved_tokens=10,
        admitted_input_tokens=40,
    )

    assert budget.available_input_tokens == 70
    assert budget.remaining_input_tokens == 30
    assert budget.pressure == "normal"
    assert budget.admit(30) is True
    assert budget.remaining_input_tokens == 0
    assert budget.pressure == "exhausted"
    assert budget.admit(1) is False
    assert budget.admitted_input_tokens == 70


def test_context_budget_rejects_negative_and_impossible_values():
    with pytest.raises(ContractError, match="model_limit_tokens"):
        ContextBudget(model_limit_tokens=-1)

    with pytest.raises(ContractError, match="reservations"):
        ContextBudget(model_limit_tokens=10, output_reserved_tokens=8, runtime_reserved_tokens=3)

    with pytest.raises(ContractError, match="admitted_input_tokens"):
        ContextBudget(model_limit_tokens=10, admitted_input_tokens=11)


def test_runtime_capabilities_api_only_defaults_are_conservative():
    capabilities = RuntimeCapabilities.api_only(model_id="gpt-test", backend_id="openai")

    assert capabilities.model_id == "gpt-test"
    assert capabilities.backend_id == "openai"
    assert capabilities.tokenizer_mode == "provider_estimated"
    assert capabilities.prompt_cache_visibility == "opaque"
    assert capabilities.supports_kv_checkpoint_restore is False
    assert capabilities.supports_kv_offload is False
    assert capabilities.supports_nvcache is False
    assert capabilities.supports_context_branching is False
    assert RuntimeCapabilities.from_dict(capabilities.to_dict()) == capabilities


def test_manifest_digest_is_stable_and_excludes_timestamp():
    scope = DaystromScope(tenant_id="tenant", session_id="session")
    manifest_a = ContextManifest(
        scope=scope,
        model_id="model",
        runtime_id="runtime",
        segment_ids=["seg-1", "seg-2"],
        estimated_input_tokens=12,
        exact_input_tokens=10,
        decisions={"selected": ["seg-1", "seg-2"]},
        audit={"trace_id": "trace"},
        created_at=100.0,
    )
    manifest_b = ContextManifest.from_dict({**manifest_a.to_dict(), "created_at": 200.0})

    assert manifest_a.content_digest == manifest_b.content_digest
    changed = ContextManifest.from_dict(
        {**manifest_a.to_dict(), "segment_ids": ["seg-2", "seg-1"], "content_digest": ""}
    )
    assert changed.content_digest != manifest_a.content_digest
    assert ContextManifest.from_dict(manifest_a.to_dict()) == manifest_a


def test_manifest_rejects_unknown_version_and_tampered_digest():
    manifest = ContextManifest(
        model_id="model",
        runtime_id="runtime",
        segment_ids=["seg-1"],
        estimated_input_tokens=5,
    )

    with pytest.raises(ContractError, match="manifest_version"):
        ContextManifest.from_dict({**manifest.to_dict(), "manifest_version": "daystrom-context-manifest-v2"})

    with pytest.raises(ContractError, match="content_digest"):
        ContextManifest.from_dict({**manifest.to_dict(), "content_digest": "0" * 64})


def test_blank_manifest_digest_is_computed_for_fresh_objects():
    manifest = ContextManifest(
        model_id="model",
        runtime_id="runtime",
        segment_ids=["seg-1"],
        estimated_input_tokens=5,
        content_digest="",
    )

    assert len(manifest.content_digest) == 64
    assert manifest.manifest_version == CONTEXT_MANIFEST_V1


def test_context_packet_validates_budget_and_roundtrips_json_payload():
    segment = ContextSegment(segment_id="seg-1", kind="message", content="hello", estimated_tokens=10)
    budget = ContextBudget(model_limit_tokens=100, output_reserved_tokens=30, runtime_reserved_tokens=10)
    capabilities = RuntimeCapabilities.api_only(model_id="model", backend_id="api")
    manifest = ContextManifest(
        scope=DaystromScope(session_id="session"),
        model_id=capabilities.model_id,
        runtime_id=capabilities.backend_id,
        segment_ids=[segment.segment_id],
        estimated_input_tokens=segment.effective_tokens,
    )
    packet = ContextPacket(
        scope=manifest.scope,
        capabilities=capabilities,
        budget=budget,
        segments=[segment],
        manifest=manifest,
        rendered_messages=[{"role": "user", "content": "hello"}],
        decisions={"admitted": ["seg-1"]},
        warnings=[],
    )

    data = json.loads(json.dumps(packet.to_dict(), sort_keys=True))

    assert ContextPacket.from_dict(data) == packet
    assert packet.packet_version == CONTEXT_PACKET_V1

    with pytest.raises(ContractError, match="exceed"):
        ContextPacket(
            scope=manifest.scope,
            capabilities=capabilities,
            budget=ContextBudget(model_limit_tokens=5),
            segments=[segment],
            manifest=manifest,
        )


def test_context_packet_rejects_unknown_version():
    segment = ContextSegment(segment_id="seg-1", kind="message", content="hello", estimated_tokens=1)
    packet = ContextPacket(
        budget=ContextBudget(model_limit_tokens=10),
        segments=[segment],
        manifest=ContextManifest(
            model_id="unknown",
            runtime_id="unknown",
            segment_ids=[segment.segment_id],
            estimated_input_tokens=segment.effective_tokens,
        ),
    )

    with pytest.raises(ContractError, match="packet_version"):
        ContextPacket.from_dict({**packet.to_dict(), "packet_version": "daystrom-context-packet-v2"})


def test_context_schemas_parse_and_reject_bad_required_or_negative_fields():
    segment = ContextSegment(segment_id="seg-1", kind="message", content="hello", estimated_tokens=10)
    budget = ContextBudget(model_limit_tokens=100, output_reserved_tokens=30, runtime_reserved_tokens=10)
    capabilities = RuntimeCapabilities.api_only(model_id="model", backend_id="api")
    manifest = ContextManifest(
        scope=DaystromScope(session_id="session"),
        model_id=capabilities.model_id,
        runtime_id=capabilities.backend_id,
        segment_ids=[segment.segment_id],
        estimated_input_tokens=segment.effective_tokens,
    )
    packet = ContextPacket(
        scope=manifest.scope,
        capabilities=capabilities,
        budget=budget,
        segments=[segment],
        manifest=manifest,
    )
    valid_payloads = {
        "context-packet-v1.schema.json": packet.to_dict(),
        "context-manifest-v1.schema.json": manifest.to_dict(),
        "runtime-capabilities-v1.schema.json": capabilities.to_dict(),
    }

    for filename in [
        "context-packet-v1.schema.json",
        "context-manifest-v1.schema.json",
        "runtime-capabilities-v1.schema.json",
    ]:
        schema = json.loads((SCHEMA_DIR / filename).read_text(encoding="utf-8"))
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        encoded = json.dumps(schema)
        assert '"minimum": 0' in encoded
        assert "required" in schema
        assert _validate_schema(valid_payloads[filename], schema, path="$", root=schema) == []

    packet_schema = json.loads((SCHEMA_DIR / "context-packet-v1.schema.json").read_text(encoding="utf-8"))
    missing_segment_id = packet.to_dict()
    del missing_segment_id["segments"][0]["segment_id"]
    assert any("segment_id" in error for error in _validate_schema(missing_segment_id, packet_schema, path="$", root=packet_schema))

    negative_tokens = packet.to_dict()
    negative_tokens["segments"][0]["estimated_tokens"] = -1
    assert any(
        "estimated_tokens" in error and ">= 0" in error
        for error in _validate_schema(negative_tokens, packet_schema, path="$", root=packet_schema)
    )

    capabilities_schema = json.loads((SCHEMA_DIR / "runtime-capabilities-v1.schema.json").read_text(encoding="utf-8"))
    missing_model_id = capabilities.to_dict()
    del missing_model_id["model_id"]
    assert any("model_id" in error for error in _validate_schema(missing_model_id, capabilities_schema, path="$", root=capabilities_schema))

    manifest_schema = json.loads((SCHEMA_DIR / "context-manifest-v1.schema.json").read_text(encoding="utf-8"))
    negative_total = manifest.to_dict()
    negative_total["estimated_input_tokens"] = -1
    assert any(
        "estimated_input_tokens" in error and ">= 0" in error
        for error in _validate_schema(negative_total, manifest_schema, path="$", root=manifest_schema)
    )
