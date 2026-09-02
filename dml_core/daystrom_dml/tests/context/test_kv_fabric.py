from __future__ import annotations

import hashlib
from typing import Any

import pytest

from daystrom_dml.api_contracts import ContractError, DaystromScope
from daystrom_dml.context.checkpoints import ExecutionCheckpointIdentity
from daystrom_dml.context.kv_fabric import (
    KVObjectDescriptor,
    KVRoutePlanner,
    KVTierEndpointCapability,
    KVTierKind,
    KVTransferDirection,
    KVTransferMechanism,
    KVTransferPlan,
    KVTransferTicket,
    KVTransferTicketIssuer,
    KVTransportCapability,
    runtime_compatibility_digest,
    tier_semantics_for_kind,
)


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _identity(**overrides: Any) -> ExecutionCheckpointIdentity:
    values = {
        "scope": DaystromScope(
            tenant_id="tenant-a",
            client_id="client-a",
            session_id="session-a",
            instance_id="instance-a",
            thread_id="thread-a",
            project_id="project-a",
            relationship_id="relationship-a",
        ),
        "model_id": "model-a",
        "model_digest": _digest("model"),
        "tokenizer_digest": _digest("tokenizer"),
        "positional_config_digest": _digest("positional"),
        "immutable_prefix_digest": _digest("prefix"),
        "packet_digest": _digest("packet"),
        "manifest_digest": _digest("manifest"),
        "runtime_id": "runtime-a",
        "runtime_version": "2026.09.02",
        "adapter_id": "adapter-a",
        "runtime_endpoint_digest": _digest("endpoint"),
    }
    values.update(overrides)
    return ExecutionCheckpointIdentity(**values)


def _descriptor(**overrides: Any) -> KVObjectDescriptor:
    values = {
        "object_id": "kv-object-a",
        "identity": _identity(),
        "token_start": 0,
        "token_end": 128,
        "layout_digest": _digest("layout-a"),
        "dtype_digest": _digest("dtype-fp16"),
        "topology_digest": _digest("topology-tp1"),
        "payload_digest": _digest("payload-a"),
        "byte_size": 4096,
    }
    values.update(overrides)
    identity = values.pop("identity")
    return KVObjectDescriptor.from_checkpoint_identity(identity=identity, **values)


def _transport(
    mechanism: KVTransferMechanism,
    *,
    direction: KVTransferDirection = KVTransferDirection.BIDIRECTIONAL,
    unit_cost: int = 5,
    priority: int = 10,
    max_bytes_per_transfer: int = 32 * 1024,
) -> KVTransportCapability:
    return KVTransportCapability(
        mechanism=mechanism,
        direction=direction,
        unit_cost=unit_cost,
        priority=priority,
        max_bytes_per_transfer=max_bytes_per_transfer,
    )


def _endpoint(
    tier_id: str,
    descriptor: KVObjectDescriptor,
    *,
    tier_kind: KVTierKind,
    transfers: tuple[KVTransportCapability, ...],
    admission_available_bytes: int = 32 * 1024,
    now: float = 100.0,
    expires_at: float = 200.0,
    authority_domain_digest: str | None = None,
    layout_digest: str | None = None,
    topology_digest: str | None = None,
    runtime_id: str | None = None,
    runtime_version: str | None = None,
    adapter_id: str | None = None,
    accepted_runtime_identity_digests: tuple[str, ...] | None = None,
) -> KVTierEndpointCapability:
    semantics = tier_semantics_for_kind(tier_kind)
    return KVTierEndpointCapability(
        tier_id=tier_id,
        tier_kind=tier_kind,
        durability=semantics.durability,
        locality=semantics.locality,
        runtime_id=descriptor.runtime_id if runtime_id is None else runtime_id,
        runtime_version=descriptor.runtime_version if runtime_version is None else runtime_version,
        adapter_id=descriptor.adapter_id if adapter_id is None else adapter_id,
        endpoint_digest=_digest(f"endpoint-{tier_id}"),
        authority_domain_digest=descriptor.authority_domain_digest
        if authority_domain_digest is None
        else authority_domain_digest,
        transfers=transfers,
        accepted_layout_digests=(descriptor.layout_digest if layout_digest is None else layout_digest,),
        accepted_dtype_digests=(descriptor.dtype_digest,),
        accepted_topology_digests=(descriptor.topology_digest if topology_digest is None else topology_digest,),
        accepted_runtime_identity_digests=(
            (descriptor.runtime_compatibility_digest,)
            if accepted_runtime_identity_digests is None
            else accepted_runtime_identity_digests
        ),
        capacity_bytes=64 * 1024,
        admission_limit_bytes=64 * 1024,
        admission_available_bytes=admission_available_bytes,
        declared_at=now,
        expires_at=expires_at,
    )


def test_contracts_roundtrip_and_strict_unknown_field_rejection() -> None:
    descriptor = _descriptor()
    source = _endpoint(
        "gpu-a",
        descriptor,
        tier_kind=KVTierKind.GPU_HBM,
        transfers=(_transport(KVTransferMechanism.PCIE_DMA),),
    )
    destination = _endpoint(
        "cpu-a",
        descriptor,
        tier_kind=KVTierKind.HOST_PINNED,
        transfers=(_transport(KVTransferMechanism.PCIE_DMA),),
    )
    plan = KVRoutePlanner(clock=lambda: 110.0).plan_route(
        descriptor=descriptor,
        source_tier_id=source.tier_id,
        destination_tier_id=destination.tier_id,
        endpoints=[source, destination],
    )

    assert KVObjectDescriptor.from_dict(descriptor.to_dict()) == descriptor
    assert KVTierEndpointCapability.from_dict(source.to_dict()) == source
    assert plan == KVTransferPlan.from_dict(plan.to_dict())

    payload = source.to_dict()
    payload["unknown"] = "x"
    with pytest.raises(ContractError, match="unknown fields"):
        KVTierEndpointCapability.from_dict(payload)

    with pytest.raises(ContractError, match="finite"):
        _endpoint(
            "bad-nan",
            descriptor,
            tier_kind=KVTierKind.HOST_PINNED,
            transfers=(_transport(KVTransferMechanism.PCIE_DMA),),
            now=float("nan"),
        )
    with pytest.raises(ContractError, match="finite"):
        _endpoint(
            "bad-inf",
            descriptor,
            tier_kind=KVTierKind.HOST_PINNED,
            transfers=(_transport(KVTransferMechanism.PCIE_DMA),),
            expires_at=float("inf"),
        )


def test_tamper_detection_rejects_modified_digests() -> None:
    descriptor = _descriptor()
    payload = descriptor.to_dict()
    payload["token_end"] = 64
    with pytest.raises(ContractError, match="integrity"):
        KVObjectDescriptor.from_dict(payload)


def test_route_rejects_cross_authority_domain_mismatch() -> None:
    descriptor = _descriptor()
    source = _endpoint(
        "gpu-a",
        descriptor,
        tier_kind=KVTierKind.GPU_HBM,
        transfers=(_transport(KVTransferMechanism.PCIE_DMA),),
    )
    destination = _endpoint(
        "cpu-a",
        descriptor,
        tier_kind=KVTierKind.HOST_PINNED,
        transfers=(_transport(KVTransferMechanism.PCIE_DMA),),
        authority_domain_digest=_digest("authority-b"),
    )
    with pytest.raises(ContractError, match="authority domain mismatch"):
        KVRoutePlanner(clock=lambda: 110.0).plan_route(
            descriptor=descriptor,
            source_tier_id="gpu-a",
            destination_tier_id="cpu-a",
            endpoints=[source, destination],
        )


def test_descriptor_authority_domain_is_derived_from_checkpoint_scope() -> None:
    first = _descriptor()
    second = _descriptor(
        identity=_identity(
            scope=DaystromScope(
                tenant_id="tenant-b",
                client_id="client-a",
                session_id="session-a",
                instance_id="instance-a",
                thread_id="thread-a",
                project_id="project-a",
                relationship_id="relationship-a",
            )
        )
    )

    assert first.authority_domain_digest != second.authority_domain_digest


def test_route_rejects_layout_topology_and_runtime_compatibility_mismatch() -> None:
    descriptor = _descriptor()
    source = _endpoint(
        "gpu-a",
        descriptor,
        tier_kind=KVTierKind.GPU_HBM,
        transfers=(_transport(KVTransferMechanism.PCIE_DMA),),
    )
    bad_layout = _endpoint(
        "cpu-a",
        descriptor,
        tier_kind=KVTierKind.HOST_PINNED,
        transfers=(_transport(KVTransferMechanism.PCIE_DMA),),
        layout_digest=_digest("layout-b"),
    )
    with pytest.raises(ContractError, match="layout digest mismatch"):
        KVRoutePlanner(clock=lambda: 110.0).plan_route(
            descriptor=descriptor,
            source_tier_id="gpu-a",
            destination_tier_id="cpu-a",
            endpoints=[source, bad_layout],
        )

    bad_topology = _endpoint(
        "cpu-b",
        descriptor,
        tier_kind=KVTierKind.HOST_PINNED,
        transfers=(_transport(KVTransferMechanism.PCIE_DMA),),
        topology_digest=_digest("topology-b"),
    )
    with pytest.raises(ContractError, match="topology digest mismatch"):
        KVRoutePlanner(clock=lambda: 110.0).plan_route(
            descriptor=descriptor,
            source_tier_id="gpu-a",
            destination_tier_id="cpu-b",
            endpoints=[source, bad_topology],
        )

    bad_runtime = _endpoint(
        "cpu-c",
        descriptor,
        tier_kind=KVTierKind.HOST_PINNED,
        transfers=(_transport(KVTransferMechanism.PCIE_DMA),),
        accepted_runtime_identity_digests=(runtime_compatibility_digest(runtime_id="runtime-b", runtime_version="1.0.0", adapter_id="adapter-b"),),
    )
    with pytest.raises(ContractError, match="runtime compatibility mismatch"):
        KVRoutePlanner(clock=lambda: 110.0).plan_route(
            descriptor=descriptor,
            source_tier_id="gpu-a",
            destination_tier_id="cpu-c",
            endpoints=[source, bad_runtime],
        )

    compatible_cross_adapter = _endpoint(
        "cpu-d",
        descriptor,
        tier_kind=KVTierKind.HOST_PINNED,
        transfers=(_transport(KVTransferMechanism.PCIE_DMA),),
        runtime_id="runtime-b",
        runtime_version="2027.01.01",
        adapter_id="adapter-b",
        accepted_runtime_identity_digests=(
            descriptor.runtime_compatibility_digest,
            runtime_compatibility_digest(runtime_id="runtime-b", runtime_version="2027.01.01", adapter_id="adapter-b"),
        ),
    )
    plan = KVRoutePlanner(clock=lambda: 110.0).plan_route(
        descriptor=descriptor,
        source_tier_id="gpu-a",
        destination_tier_id="cpu-d",
        endpoints=[source, compatible_cross_adapter],
    )
    assert plan.destination_tier_id == "cpu-d"


def test_route_rejects_insufficient_capacity_and_stale_capability() -> None:
    descriptor = _descriptor(byte_size=8192)
    source = _endpoint(
        "gpu-a",
        descriptor,
        tier_kind=KVTierKind.GPU_HBM,
        transfers=(_transport(KVTransferMechanism.PCIE_DMA),),
    )
    insufficient = _endpoint(
        "cpu-a",
        descriptor,
        tier_kind=KVTierKind.HOST_PINNED,
        transfers=(_transport(KVTransferMechanism.PCIE_DMA),),
        admission_available_bytes=1024,
    )
    with pytest.raises(ContractError, match="insufficient admission capacity"):
        KVRoutePlanner(clock=lambda: 110.0).plan_route(
            descriptor=descriptor,
            source_tier_id="gpu-a",
            destination_tier_id="cpu-a",
            endpoints=[source, insufficient],
        )

    stale = _endpoint(
        "cpu-b",
        descriptor,
        tier_kind=KVTierKind.HOST_PINNED,
        transfers=(_transport(KVTransferMechanism.PCIE_DMA),),
        expires_at=109.0,
    )
    with pytest.raises(ContractError, match="stale"):
        KVRoutePlanner(clock=lambda: 110.0).plan_route(
            descriptor=descriptor,
            source_tier_id="gpu-a",
            destination_tier_id="cpu-b",
            endpoints=[source, stale],
        )

    future = _endpoint(
        "cpu-c",
        descriptor,
        tier_kind=KVTierKind.HOST_PINNED,
        transfers=(_transport(KVTransferMechanism.PCIE_DMA),),
        now=111.0,
        expires_at=211.0,
    )
    with pytest.raises(ContractError, match="future"):
        KVRoutePlanner(clock=lambda: 110.0).plan_route(
            descriptor=descriptor,
            source_tier_id="gpu-a",
            destination_tier_id="cpu-c",
            endpoints=[source, future],
        )


def test_route_rejects_unsupported_intersection_and_wrong_direction() -> None:
    descriptor = _descriptor()
    src = _endpoint(
        "gpu-a",
        descriptor,
        tier_kind=KVTierKind.GPU_HBM,
        transfers=(_transport(KVTransferMechanism.PCIE_DMA),),
    )
    dst_no_intersection = _endpoint(
        "rdma-a",
        descriptor,
        tier_kind=KVTierKind.REMOTE_RDMA_MEMORY,
        transfers=(_transport(KVTransferMechanism.RDMA),),
    )
    with pytest.raises(ContractError, match="unsupported transport intersection"):
        KVRoutePlanner(clock=lambda: 110.0).plan_route(
            descriptor=descriptor,
            source_tier_id="gpu-a",
            destination_tier_id="rdma-a",
            endpoints=[src, dst_no_intersection],
        )

    dst_wrong_direction = _endpoint(
        "cpu-a",
        descriptor,
        tier_kind=KVTierKind.HOST_PINNED,
        transfers=(
            _transport(KVTransferMechanism.PCIE_DMA, direction=KVTransferDirection.INGRESS),
        ),
    )
    src_wrong_direction = _endpoint(
        "gpu-b",
        descriptor,
        tier_kind=KVTierKind.GPU_HBM,
        transfers=(
            _transport(KVTransferMechanism.PCIE_DMA, direction=KVTransferDirection.INGRESS),
        ),
    )
    with pytest.raises(ContractError, match="direction mismatch"):
        KVRoutePlanner(clock=lambda: 110.0).plan_route(
            descriptor=descriptor,
            source_tier_id="gpu-b",
            destination_tier_id="cpu-a",
            endpoints=[src_wrong_direction, dst_wrong_direction],
        )


def test_route_planning_is_deterministic_by_cost_priority_and_tie_break() -> None:
    descriptor = _descriptor()
    source = _endpoint(
        "gpu-a",
        descriptor,
        tier_kind=KVTierKind.GPU_HBM,
        transfers=(
            _transport(KVTransferMechanism.PCIE_DMA, unit_cost=10, priority=20),
            _transport(KVTransferMechanism.HOST_COPY, unit_cost=10, priority=20),
        ),
    )
    destination = _endpoint(
        "obj-a",
        descriptor,
        tier_kind=KVTierKind.REMOTE_FILE_OBJECT,
        transfers=(
            _transport(KVTransferMechanism.OBJECT_IO, unit_cost=20, priority=10),
        ),
    )
    mid_a = _endpoint(
        "mid-a",
        descriptor,
        tier_kind=KVTierKind.HOST_PINNED,
        transfers=(
            _transport(KVTransferMechanism.PCIE_DMA, unit_cost=5, priority=5),
            _transport(KVTransferMechanism.HOST_COPY, unit_cost=5, priority=5),
            _transport(KVTransferMechanism.OBJECT_IO, unit_cost=5, priority=5),
        ),
    )
    mid_b = _endpoint(
        "mid-b",
        descriptor,
        tier_kind=KVTierKind.HOST_PINNED,
        transfers=(
            _transport(KVTransferMechanism.PCIE_DMA, unit_cost=5, priority=5),
            _transport(KVTransferMechanism.HOST_COPY, unit_cost=5, priority=5),
            _transport(KVTransferMechanism.OBJECT_IO, unit_cost=5, priority=5),
        ),
    )
    planner = KVRoutePlanner(clock=lambda: 110.0)

    plan_a = planner.plan_route(
        descriptor=descriptor,
        source_tier_id="gpu-a",
        destination_tier_id="obj-a",
        endpoints=[source, destination, mid_a, mid_b],
    )
    plan_b = planner.plan_route(
        descriptor=descriptor,
        source_tier_id="gpu-a",
        destination_tier_id="obj-a",
        endpoints=[mid_b, destination, source, mid_a],
    )

    assert plan_a.plan_digest == plan_b.plan_digest
    assert [hop.destination_tier_id for hop in plan_a.hops] == ["mid-a", "obj-a"]

    with pytest.raises(ContractError, match="planner clock"):
        KVRoutePlanner(clock=lambda: float("nan")).plan_route(
            descriptor=descriptor,
            source_tier_id="gpu-a",
            destination_tier_id="obj-a",
            endpoints=[source, destination, mid_a, mid_b],
        )


def test_route_rejects_impossible_path_and_duplicate_tier_declarations() -> None:
    descriptor = _descriptor()
    source = _endpoint(
        "gpu-a",
        descriptor,
        tier_kind=KVTierKind.GPU_HBM,
        transfers=(_transport(KVTransferMechanism.PCIE_DMA),),
    )
    destination = _endpoint(
        "obj-a",
        descriptor,
        tier_kind=KVTierKind.REMOTE_FILE_OBJECT,
        transfers=(_transport(KVTransferMechanism.OBJECT_IO),),
    )
    isolated_mid = _endpoint(
        "mid-a",
        descriptor,
        tier_kind=KVTierKind.LOCAL_SSD,
        transfers=(_transport(KVTransferMechanism.BLOCK_IO),),
    )
    planner = KVRoutePlanner(clock=lambda: 110.0)

    with pytest.raises(ContractError, match="no feasible transfer path"):
        planner.plan_route(
            descriptor=descriptor,
            source_tier_id="gpu-a",
            destination_tier_id="obj-a",
            endpoints=[source, destination, isolated_mid],
        )

    with pytest.raises(ContractError, match="duplicate tier IDs"):
        duplicate_source = _endpoint(
            "gpu-a",
            descriptor,
            tier_kind=KVTierKind.HOST_PINNED,
            transfers=(_transport(KVTransferMechanism.PCIE_DMA),),
        )
        planner.plan_route(
            descriptor=descriptor,
            source_tier_id="gpu-a",
            destination_tier_id="obj-a",
            endpoints=[source, destination, duplicate_source],
        )

    with pytest.raises(ContractError, match="endpoint declaration count exceeds"):
        KVRoutePlanner(clock=lambda: 110.0, max_endpoints=2).plan_route(
            descriptor=descriptor,
            source_tier_id="gpu-a",
            destination_tier_id="obj-a",
            endpoints=[source, destination, isolated_mid],
        )


def test_transfer_tickets_are_single_use_and_expire() -> None:
    descriptor = _descriptor()
    source = _endpoint(
        "gpu-a",
        descriptor,
        tier_kind=KVTierKind.GPU_HBM,
        transfers=(_transport(KVTransferMechanism.PCIE_DMA),),
    )
    destination = _endpoint(
        "cpu-a",
        descriptor,
        tier_kind=KVTierKind.HOST_PINNED,
        transfers=(_transport(KVTransferMechanism.PCIE_DMA),),
    )
    now = [100.0]
    planner = KVRoutePlanner(clock=lambda: now[0])
    plan = planner.plan_route(
        descriptor=descriptor,
        source_tier_id="gpu-a",
        destination_tier_id="cpu-a",
        endpoints=[source, destination],
    )
    signing_key = b"A" * 32
    with pytest.raises(ContractError, match="at least 32 bytes"):
        KVTransferTicketIssuer(signing_key=b"short", clock=lambda: now[0], max_ttl_seconds=30.0)

    issuer = KVTransferTicketIssuer(
        signing_key=signing_key,
        clock=lambda: now[0],
        max_ttl_seconds=30.0,
        nonce_bytes_factory=lambda _: b"N" * 32,
    )
    with pytest.raises(ContractError, match="authority domain does not match plan"):
        issuer.issue(
            plan=plan,
            authority_domain_digest=_digest("authority-b"),
            ttl_seconds=10.0,
        )
    ticket = issuer.issue(plan=plan, authority_domain_digest=descriptor.authority_domain_digest, ttl_seconds=10.0)

    issuer.consume(
        ticket,
        expected_plan_digest=plan.plan_digest,
        expected_authority_domain_digest=descriptor.authority_domain_digest,
    )
    with pytest.raises(ContractError, match="replay"):
        issuer.consume(
            ticket,
            expected_plan_digest=plan.plan_digest,
            expected_authority_domain_digest=descriptor.authority_domain_digest,
        )

    ticket_payload = ticket.to_dict()
    ticket_payload["plan_digest"] = _digest("different")
    with pytest.raises(ContractError, match="integrity"):
        ticket.__class__.from_dict(ticket_payload)

    wrong_key_issuer = KVTransferTicketIssuer(
        signing_key=b"B" * 32,
        clock=lambda: now[0],
        max_ttl_seconds=30.0,
    )
    wrong_key_ticket = issuer.issue(plan=plan, authority_domain_digest=descriptor.authority_domain_digest, ttl_seconds=10.0)
    with pytest.raises(ContractError, match="authentication failed"):
        wrong_key_issuer.consume(
            wrong_key_ticket,
            expected_plan_digest=plan.plan_digest,
            expected_authority_domain_digest=descriptor.authority_domain_digest,
        )

    forged_ticket = KVTransferTicket(
        plan_digest=plan.plan_digest,
        authority_domain_digest=descriptor.authority_domain_digest,
        ticket_nonce_digest=_digest("forged-nonce"),
        issued_at=now[0],
        expires_at=now[0] + 5.0,
        ticket_auth_tag=_digest("forged-auth"),
    )
    with pytest.raises(ContractError, match="authentication failed"):
        issuer.consume(
            forged_ticket,
            expected_plan_digest=plan.plan_digest,
            expected_authority_domain_digest=descriptor.authority_domain_digest,
        )

    tampered = KVTransferTicket(
        plan_digest=ticket.plan_digest,
        authority_domain_digest=ticket.authority_domain_digest,
        ticket_nonce_digest=ticket.ticket_nonce_digest,
        issued_at=ticket.issued_at,
        expires_at=ticket.expires_at + 1.0,
        ticket_auth_tag=ticket.ticket_auth_tag,
    )
    with pytest.raises(ContractError, match="authentication failed"):
        issuer.consume(
            tampered,
            expected_plan_digest=plan.plan_digest,
            expected_authority_domain_digest=descriptor.authority_domain_digest,
        )

    expiring_ticket = issuer.issue(plan=plan, authority_domain_digest=descriptor.authority_domain_digest, ttl_seconds=5.0)
    now[0] = 99.0
    with pytest.raises(ContractError, match="not yet valid"):
        issuer.consume(
            expiring_ticket,
            expected_plan_digest=plan.plan_digest,
            expected_authority_domain_digest=descriptor.authority_domain_digest,
        )
    now[0] = 106.0
    with pytest.raises(ContractError, match="expired"):
        issuer.consume(
            expiring_ticket,
            expected_plan_digest=plan.plan_digest,
            expected_authority_domain_digest=descriptor.authority_domain_digest,
        )
