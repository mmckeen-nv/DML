"""Payload-free control-plane contracts for heterogeneous DCM KV tiers.

This module adds deterministic, adapter-neutral planning contracts for extending
managed KV checkpoints beyond the proven GPU APC + pinned CPU runtime path.
It does not move bytes, mutate tier state, or claim transfer completion.
"""
from __future__ import annotations

import hashlib
import heapq
import hmac
import json
import math
import re
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

from daystrom_dml.api_contracts import ContractError, SerializableDataclass, enum_from_value
from daystrom_dml.context.checkpoints import (
    EXECUTION_CHECKPOINT_IDENTITY_V1,
    EXECUTION_CHECKPOINT_IDENTITY_V2,
    ExecutionCheckpointIdentity,
)

KV_TIER_SEMANTICS_V1 = "daystrom-kv-tier-semantics-v1"
KV_TIER_ENDPOINT_CAPABILITY_V1 = "daystrom-kv-tier-endpoint-capability-v1"
KV_OBJECT_DESCRIPTOR_V1 = "daystrom-kv-object-descriptor-v1"
KV_TRANSFER_PLAN_V1 = "daystrom-kv-transfer-plan-v1"
KV_TRANSFER_TICKET_V1 = "daystrom-kv-transfer-ticket-v1"

_SAFE_ID_RE = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"


class KVTierKind(str, Enum):
    """Storage/memory class for managed KV checkpoint materialization."""

    GPU_HBM = "gpu_hbm"
    HOST_PINNED = "host_pinned"
    REMOTE_RDMA_MEMORY = "remote_rdma_memory"
    LOCAL_SSD = "local_ssd"
    REMOTE_FILE_OBJECT = "remote_file_object"


class KVDurability(str, Enum):
    """Durability semantics for one tier kind."""

    VOLATILE = "volatile"
    EPHEMERAL = "ephemeral"
    PERSISTENT = "persistent"


class KVLocality(str, Enum):
    """Locality semantics for one tier kind."""

    LOCAL_DEVICE = "local_device"
    LOCAL_HOST = "local_host"
    REMOTE_FABRIC = "remote_fabric"
    REMOTE_STORAGE = "remote_storage"


class KVTransferDirection(str, Enum):
    INGRESS = "ingress"
    EGRESS = "egress"
    BIDIRECTIONAL = "bidirectional"


class KVTransferMechanism(str, Enum):
    """Mechanism identity only; adapters own actual movement implementation."""

    PCIE_DMA = "pcie_dma"
    HOST_COPY = "host_copy"
    RDMA = "rdma"
    BLOCK_IO = "block_io"
    OBJECT_IO = "object_io"


@dataclass(frozen=True)
class KVTierSemantics(SerializableDataclass):
    tier_kind: KVTierKind
    durability: KVDurability
    locality: KVLocality
    semantics_version: str = KV_TIER_SEMANTICS_V1

    def __post_init__(self) -> None:
        object.__setattr__(self, "tier_kind", enum_from_value(KVTierKind, self.tier_kind))
        object.__setattr__(self, "durability", enum_from_value(KVDurability, self.durability))
        object.__setattr__(self, "locality", enum_from_value(KVLocality, self.locality))
        if self.semantics_version != KV_TIER_SEMANTICS_V1:
            raise ContractError("unsupported tier semantics version")

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "KVTierSemantics":
        payload = _strict_payload(cls, data)
        return cls(**payload)


_TIER_SEMANTICS_BY_KIND: Mapping[KVTierKind, KVTierSemantics] = {
    KVTierKind.GPU_HBM: KVTierSemantics(
        tier_kind=KVTierKind.GPU_HBM,
        durability=KVDurability.VOLATILE,
        locality=KVLocality.LOCAL_DEVICE,
    ),
    KVTierKind.HOST_PINNED: KVTierSemantics(
        tier_kind=KVTierKind.HOST_PINNED,
        durability=KVDurability.EPHEMERAL,
        locality=KVLocality.LOCAL_HOST,
    ),
    KVTierKind.REMOTE_RDMA_MEMORY: KVTierSemantics(
        tier_kind=KVTierKind.REMOTE_RDMA_MEMORY,
        durability=KVDurability.EPHEMERAL,
        locality=KVLocality.REMOTE_FABRIC,
    ),
    KVTierKind.LOCAL_SSD: KVTierSemantics(
        tier_kind=KVTierKind.LOCAL_SSD,
        durability=KVDurability.PERSISTENT,
        locality=KVLocality.LOCAL_HOST,
    ),
    KVTierKind.REMOTE_FILE_OBJECT: KVTierSemantics(
        tier_kind=KVTierKind.REMOTE_FILE_OBJECT,
        durability=KVDurability.PERSISTENT,
        locality=KVLocality.REMOTE_STORAGE,
    ),
}


def tier_semantics_for_kind(kind: KVTierKind | str) -> KVTierSemantics:
    """Return canonical durability/locality semantics for a tier kind."""

    tier_kind = enum_from_value(KVTierKind, kind)
    return _TIER_SEMANTICS_BY_KIND[tier_kind]


@dataclass(frozen=True)
class KVTransportCapability(SerializableDataclass):
    mechanism: KVTransferMechanism
    direction: KVTransferDirection
    unit_cost: int
    priority: int
    max_bytes_per_transfer: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "mechanism", enum_from_value(KVTransferMechanism, self.mechanism))
        object.__setattr__(self, "direction", enum_from_value(KVTransferDirection, self.direction))
        _require_non_negative_int("unit_cost", self.unit_cost)
        _require_non_negative_int("priority", self.priority)
        _require_positive_int("max_bytes_per_transfer", self.max_bytes_per_transfer)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "KVTransportCapability":
        payload = _strict_payload(cls, data)
        return cls(**payload)


@dataclass(frozen=True)
class KVTierEndpointCapability(SerializableDataclass):
    """Exact endpoint identity and bounded admission declaration for one KV tier."""

    tier_id: str
    tier_kind: KVTierKind
    durability: KVDurability
    locality: KVLocality
    runtime_id: str
    runtime_version: str
    adapter_id: str
    endpoint_digest: str
    authority_domain_digest: str
    transfers: tuple[KVTransportCapability, ...]
    accepted_layout_digests: tuple[str, ...]
    accepted_dtype_digests: tuple[str, ...]
    accepted_topology_digests: tuple[str, ...]
    accepted_runtime_identity_digests: tuple[str, ...]
    capacity_bytes: int
    admission_limit_bytes: int
    admission_available_bytes: int
    declared_at: float
    expires_at: float
    capability_version: str = KV_TIER_ENDPOINT_CAPABILITY_V1
    capability_digest: str = ""

    def __post_init__(self) -> None:
        _safe_id("tier_id", self.tier_id)
        object.__setattr__(self, "tier_kind", enum_from_value(KVTierKind, self.tier_kind))
        object.__setattr__(self, "durability", enum_from_value(KVDurability, self.durability))
        object.__setattr__(self, "locality", enum_from_value(KVLocality, self.locality))
        if self.capability_version != KV_TIER_ENDPOINT_CAPABILITY_V1:
            raise ContractError("unsupported tier endpoint capability version")
        if not isinstance(self.runtime_id, str) or not self.runtime_id:
            raise ContractError("runtime_id must be non-empty")
        if not isinstance(self.runtime_version, str) or not self.runtime_version:
            raise ContractError("runtime_version must be non-empty")
        if self.runtime_version.casefold() == "unknown":
            raise ContractError("runtime_version must be exact")
        if not isinstance(self.adapter_id, str) or not self.adapter_id:
            raise ContractError("adapter_id must be non-empty")
        _strong_digest("endpoint_digest", self.endpoint_digest)
        _strong_digest("authority_domain_digest", self.authority_domain_digest)
        transfers = _coerce_transport_capabilities(self.transfers)
        if not transfers:
            raise ContractError("transfers must be non-empty")
        object.__setattr__(self, "transfers", transfers)
        _require_unique_mechanisms(transfers)
        object.__setattr__(self, "accepted_layout_digests", _coerce_digest_tuple("accepted_layout_digests", self.accepted_layout_digests))
        object.__setattr__(self, "accepted_dtype_digests", _coerce_digest_tuple("accepted_dtype_digests", self.accepted_dtype_digests))
        object.__setattr__(
            self,
            "accepted_topology_digests",
            _coerce_digest_tuple("accepted_topology_digests", self.accepted_topology_digests),
        )
        object.__setattr__(
            self,
            "accepted_runtime_identity_digests",
            _coerce_digest_tuple("accepted_runtime_identity_digests", self.accepted_runtime_identity_digests),
        )
        _require_positive_int("capacity_bytes", self.capacity_bytes)
        _require_positive_int("admission_limit_bytes", self.admission_limit_bytes)
        _require_non_negative_int("admission_available_bytes", self.admission_available_bytes)
        if self.admission_limit_bytes > self.capacity_bytes:
            raise ContractError("admission_limit_bytes cannot exceed capacity_bytes")
        if self.admission_available_bytes > self.admission_limit_bytes:
            raise ContractError("admission_available_bytes cannot exceed admission_limit_bytes")
        _require_numeric("declared_at", self.declared_at)
        _require_numeric("expires_at", self.expires_at)
        if self.expires_at <= self.declared_at:
            raise ContractError("expires_at must be later than declared_at")
        _require_semantics_match(self.tier_kind, self.durability, self.locality)
        computed = self.compute_capability_digest()
        if self.capability_digest and self.capability_digest != computed:
            raise ContractError("tier endpoint capability integrity check failed")
        object.__setattr__(self, "capability_digest", computed)

    def compute_capability_digest(self) -> str:
        return _json_digest(
            {
                "tier_id": self.tier_id,
                "tier_kind": self.tier_kind.value,
                "durability": self.durability.value,
                "locality": self.locality.value,
                "runtime_id": self.runtime_id,
                "runtime_version": self.runtime_version,
                "adapter_id": self.adapter_id,
                "endpoint_digest": self.endpoint_digest,
                "authority_domain_digest": self.authority_domain_digest,
                "transfers": [item.to_dict() for item in self.transfers],
                "accepted_layout_digests": list(self.accepted_layout_digests),
                "accepted_dtype_digests": list(self.accepted_dtype_digests),
                "accepted_topology_digests": list(self.accepted_topology_digests),
                "accepted_runtime_identity_digests": list(self.accepted_runtime_identity_digests),
                "capacity_bytes": self.capacity_bytes,
                "admission_limit_bytes": self.admission_limit_bytes,
                "admission_available_bytes": self.admission_available_bytes,
                "declared_at": self.declared_at,
                "expires_at": self.expires_at,
                "capability_version": self.capability_version,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        if self.capability_digest != self.compute_capability_digest():
            raise ContractError("tier endpoint capability integrity check failed")
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "KVTierEndpointCapability":
        payload = _strict_payload(cls, data)
        if not payload.get("capability_digest"):
            raise ContractError("tier endpoint capability capability_digest is required")
        payload["transfers"] = tuple(KVTransportCapability.from_dict(item) for item in payload.get("transfers", ()))
        return cls(**payload)

    def transport_for(self, mechanism: KVTransferMechanism) -> Optional[KVTransportCapability]:
        for item in self.transfers:
            if item.mechanism is mechanism:
                return item
        return None


@dataclass(frozen=True)
class KVObjectDescriptor(SerializableDataclass):
    """Immutable descriptor for one exact KV object without payload bytes."""

    object_id: str
    checkpoint_binding_digest: str
    checkpoint_identity_version: str
    checkpoint_identity_digest: str
    runtime_id: str
    runtime_version: str
    adapter_id: str
    authority_domain_digest: str
    token_start: int
    token_end: int
    layout_digest: str
    dtype_digest: str
    topology_digest: str
    payload_digest: str
    byte_size: int
    runtime_compatibility_digest: str = ""
    descriptor_version: str = KV_OBJECT_DESCRIPTOR_V1
    descriptor_digest: str = ""

    def __post_init__(self) -> None:
        _safe_id("object_id", self.object_id)
        _strong_digest("checkpoint_binding_digest", self.checkpoint_binding_digest)
        if self.checkpoint_identity_version not in {
            EXECUTION_CHECKPOINT_IDENTITY_V1,
            EXECUTION_CHECKPOINT_IDENTITY_V2,
        }:
            raise ContractError("unsupported checkpoint identity version")
        _strong_digest("checkpoint_identity_digest", self.checkpoint_identity_digest)
        for name in ("runtime_id", "runtime_version", "adapter_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ContractError(f"{name} must be non-empty")
        if self.runtime_version.casefold() == "unknown":
            raise ContractError("runtime_version must be exact")
        computed_runtime_compatibility = runtime_compatibility_digest(
            runtime_id=self.runtime_id,
            runtime_version=self.runtime_version,
            adapter_id=self.adapter_id,
        )
        if self.runtime_compatibility_digest and self.runtime_compatibility_digest != computed_runtime_compatibility:
            raise ContractError("runtime compatibility digest mismatch")
        object.__setattr__(self, "runtime_compatibility_digest", computed_runtime_compatibility)
        _strong_digest("authority_domain_digest", self.authority_domain_digest)
        _require_non_negative_int("token_start", self.token_start)
        _require_non_negative_int("token_end", self.token_end)
        if self.token_end <= self.token_start:
            raise ContractError("token_end must be greater than token_start")
        for name in ("layout_digest", "dtype_digest", "topology_digest", "payload_digest"):
            _strong_digest(name, getattr(self, name))
        _require_positive_int("byte_size", self.byte_size)
        if self.descriptor_version != KV_OBJECT_DESCRIPTOR_V1:
            raise ContractError("unsupported KV object descriptor version")
        computed = self.compute_descriptor_digest()
        if self.descriptor_digest and self.descriptor_digest != computed:
            raise ContractError("KV object descriptor integrity check failed")
        object.__setattr__(self, "descriptor_digest", computed)

    @classmethod
    def from_checkpoint_identity(
        cls,
        *,
        object_id: str,
        identity: ExecutionCheckpointIdentity,
        token_start: int,
        token_end: int,
        layout_digest: str,
        dtype_digest: str,
        topology_digest: str,
        payload_digest: str,
        byte_size: int,
    ) -> "KVObjectDescriptor":
        if not isinstance(identity, ExecutionCheckpointIdentity):
            raise ContractError("identity must be an ExecutionCheckpointIdentity")
        identity_payload = identity.to_dict()
        authority_domain_digest = _json_digest(identity.scope.to_dict())
        return cls(
            object_id=object_id,
            checkpoint_binding_digest=identity.binding_digest,
            checkpoint_identity_version=identity.identity_version,
            checkpoint_identity_digest=_json_digest(identity_payload),
            runtime_id=identity.runtime_id,
            runtime_version=identity.runtime_version,
            adapter_id=identity.adapter_id,
            authority_domain_digest=authority_domain_digest,
            token_start=token_start,
            token_end=token_end,
            layout_digest=layout_digest,
            dtype_digest=dtype_digest,
            topology_digest=topology_digest,
            payload_digest=payload_digest,
            byte_size=byte_size,
        )

    def compute_descriptor_digest(self) -> str:
        return _json_digest(
            {
                "object_id": self.object_id,
                "checkpoint_binding_digest": self.checkpoint_binding_digest,
                "checkpoint_identity_version": self.checkpoint_identity_version,
                "checkpoint_identity_digest": self.checkpoint_identity_digest,
                "runtime_id": self.runtime_id,
                "runtime_version": self.runtime_version,
                "adapter_id": self.adapter_id,
                "runtime_compatibility_digest": self.runtime_compatibility_digest,
                "authority_domain_digest": self.authority_domain_digest,
                "token_start": self.token_start,
                "token_end": self.token_end,
                "layout_digest": self.layout_digest,
                "dtype_digest": self.dtype_digest,
                "topology_digest": self.topology_digest,
                "payload_digest": self.payload_digest,
                "byte_size": self.byte_size,
                "descriptor_version": self.descriptor_version,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        if self.descriptor_digest != self.compute_descriptor_digest():
            raise ContractError("KV object descriptor integrity check failed")
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "KVObjectDescriptor":
        payload = _strict_payload(cls, data)
        if not payload.get("descriptor_digest"):
            raise ContractError("KV object descriptor descriptor_digest is required")
        return cls(**payload)


@dataclass(frozen=True)
class KVTransferHop(SerializableDataclass):
    source_tier_id: str
    destination_tier_id: str
    mechanism: KVTransferMechanism
    hop_cost: int
    hop_priority: int

    def __post_init__(self) -> None:
        _safe_id("source_tier_id", self.source_tier_id)
        _safe_id("destination_tier_id", self.destination_tier_id)
        if self.source_tier_id == self.destination_tier_id:
            raise ContractError("transfer hop requires distinct source and destination")
        object.__setattr__(self, "mechanism", enum_from_value(KVTransferMechanism, self.mechanism))
        _require_non_negative_int("hop_cost", self.hop_cost)
        _require_non_negative_int("hop_priority", self.hop_priority)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "KVTransferHop":
        payload = _strict_payload(cls, data)
        return cls(**payload)


@dataclass(frozen=True)
class KVTransferPlan(SerializableDataclass):
    descriptor: KVObjectDescriptor
    source_tier_id: str
    destination_tier_id: str
    hops: tuple[KVTransferHop, ...]
    total_cost: int
    total_priority: int
    plan_version: str = KV_TRANSFER_PLAN_V1
    plan_digest: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.descriptor, dict):
            object.__setattr__(self, "descriptor", KVObjectDescriptor.from_dict(self.descriptor))
        if not isinstance(self.descriptor, KVObjectDescriptor):
            raise ContractError("descriptor must be a KVObjectDescriptor")
        _safe_id("source_tier_id", self.source_tier_id)
        _safe_id("destination_tier_id", self.destination_tier_id)
        if self.source_tier_id == self.destination_tier_id:
            raise ContractError("transfer plan source and destination must differ")
        hops = _coerce_hops(self.hops)
        if not hops:
            raise ContractError("transfer plan requires at least one hop")
        object.__setattr__(self, "hops", hops)
        _validate_hop_chain(hops, source=self.source_tier_id, destination=self.destination_tier_id)
        _require_non_negative_int("total_cost", self.total_cost)
        _require_non_negative_int("total_priority", self.total_priority)
        if self.total_cost != sum(item.hop_cost for item in hops):
            raise ContractError("transfer plan total_cost mismatch")
        if self.total_priority != sum(item.hop_priority for item in hops):
            raise ContractError("transfer plan total_priority mismatch")
        if self.plan_version != KV_TRANSFER_PLAN_V1:
            raise ContractError("unsupported KV transfer plan version")
        computed = self.compute_plan_digest()
        if self.plan_digest and self.plan_digest != computed:
            raise ContractError("KV transfer plan integrity check failed")
        object.__setattr__(self, "plan_digest", computed)

    def compute_plan_digest(self) -> str:
        return _json_digest(
            {
                "descriptor": self.descriptor.to_dict(),
                "source_tier_id": self.source_tier_id,
                "destination_tier_id": self.destination_tier_id,
                "hops": [item.to_dict() for item in self.hops],
                "total_cost": self.total_cost,
                "total_priority": self.total_priority,
                "plan_version": self.plan_version,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        if self.plan_digest != self.compute_plan_digest():
            raise ContractError("KV transfer plan integrity check failed")
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "KVTransferPlan":
        payload = _strict_payload(cls, data)
        if not payload.get("plan_digest"):
            raise ContractError("KV transfer plan plan_digest is required")
        payload["descriptor"] = KVObjectDescriptor.from_dict(payload["descriptor"])
        payload["hops"] = tuple(KVTransferHop.from_dict(item) for item in payload.get("hops", ()))
        return cls(**payload)


class KVRoutePlanner:
    """Deterministic planner for payload-free endpoint transfer paths."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.time,
        max_hops: int = 6,
        max_endpoints: int = 32,
        max_route_expansions: int = 50000,
    ) -> None:
        _require_positive_int("max_hops", max_hops)
        _require_positive_int("max_endpoints", max_endpoints)
        if max_endpoints < 2:
            raise ContractError("max_endpoints must be at least 2")
        _require_positive_int("max_route_expansions", max_route_expansions)
        self.clock = clock
        self.max_hops = max_hops
        self.max_endpoints = max_endpoints
        self.max_route_expansions = max_route_expansions

    def plan_route(
        self,
        *,
        descriptor: KVObjectDescriptor,
        source_tier_id: str,
        destination_tier_id: str,
        endpoints: Sequence[KVTierEndpointCapability],
        intermediate_tier_ids: Optional[Sequence[str]] = None,
    ) -> KVTransferPlan:
        if not isinstance(descriptor, KVObjectDescriptor):
            raise ContractError("descriptor must be a KVObjectDescriptor")
        _safe_id("source_tier_id", source_tier_id)
        _safe_id("destination_tier_id", destination_tier_id)
        if source_tier_id == destination_tier_id:
            raise ContractError("source and destination must differ")
        if not isinstance(endpoints, Sequence) or not endpoints:
            raise ContractError("endpoints must be a non-empty sequence")
        if len(endpoints) > self.max_endpoints:
            raise ContractError("endpoint declaration count exceeds planner limit")
        endpoint_map = _endpoint_map(endpoints)
        if source_tier_id not in endpoint_map or destination_tier_id not in endpoint_map:
            raise ContractError("source and destination endpoint declarations are required")
        if descriptor.byte_size > endpoint_map[destination_tier_id].admission_available_bytes:
            raise ContractError("insufficient admission capacity at destination tier")

        now = _coerce_finite_number("planner clock", self.clock())
        selected_ids = [source_tier_id, destination_tier_id]
        if intermediate_tier_ids is not None:
            if not isinstance(intermediate_tier_ids, Sequence):
                raise ContractError("intermediate_tier_ids must be a sequence")
            selected_ids.extend(_validated_intermediate_ids(intermediate_tier_ids, endpoint_map))
        else:
            selected_ids.extend(
                tier_id
                for tier_id in sorted(endpoint_map)
                if tier_id not in {source_tier_id, destination_tier_id}
            )
        if len(set(selected_ids)) != len(selected_ids):
            raise ContractError("duplicate tier IDs are not allowed in route declaration")

        selected = {tier_id: endpoint_map[tier_id] for tier_id in selected_ids}
        for endpoint in selected.values():
            self._validate_endpoint_compatibility(endpoint, descriptor, now)

        best = self._search_best_plan(
            descriptor=descriptor,
            source_tier_id=source_tier_id,
            destination_tier_id=destination_tier_id,
            selected=selected,
        )
        if best is None:
            middle = [tier_id for tier_id in selected if tier_id not in {source_tier_id, destination_tier_id}]
            if not middle:
                reason = _edge_rejection_reason(
                    selected[source_tier_id],
                    selected[destination_tier_id],
                    descriptor.byte_size,
                )
                if reason is not None:
                    raise ContractError(reason)
            raise ContractError("no feasible transfer path for declared endpoints")
        return best

    def _validate_endpoint_compatibility(
        self,
        endpoint: KVTierEndpointCapability,
        descriptor: KVObjectDescriptor,
        now: float,
    ) -> None:
        if now >= endpoint.expires_at:
            raise ContractError("stale tier endpoint capability declaration")
        if endpoint.declared_at > now:
            raise ContractError("future tier endpoint capability declaration")
        if descriptor.runtime_compatibility_digest not in endpoint.accepted_runtime_identity_digests:
            raise ContractError("runtime compatibility mismatch")
        if endpoint.authority_domain_digest != descriptor.authority_domain_digest:
            raise ContractError("authority domain mismatch")
        if descriptor.layout_digest not in endpoint.accepted_layout_digests:
            raise ContractError("layout digest mismatch")
        if descriptor.dtype_digest not in endpoint.accepted_dtype_digests:
            raise ContractError("dtype digest mismatch")
        if descriptor.topology_digest not in endpoint.accepted_topology_digests:
            raise ContractError("topology digest mismatch")

    def _search_best_plan(
        self,
        *,
        descriptor: KVObjectDescriptor,
        source_tier_id: str,
        destination_tier_id: str,
        selected: Mapping[str, KVTierEndpointCapability],
    ) -> Optional[KVTransferPlan]:
        adjacency = _adjacency_by_source(selected=selected, transfer_bytes=descriptor.byte_size)
        frontier: list[
            tuple[tuple[int, int, str], int, str, tuple[str, ...], tuple[KVTransferHop, ...], int, int]
        ] = [
            ((0, 0, ""), 0, source_tier_id, (source_tier_id,), (), 0, 0),
        ]
        best_plan: Optional[KVTransferPlan] = None
        expansions = 0

        while frontier:
            _, hop_count, current, path_nodes, hops, total_cost, total_priority = heapq.heappop(frontier)
            expansions += 1
            if expansions > self.max_route_expansions:
                raise ContractError("route search exceeded deterministic work bound")

            if current == destination_tier_id and hops:
                candidate = KVTransferPlan(
                    descriptor=descriptor,
                    source_tier_id=source_tier_id,
                    destination_tier_id=destination_tier_id,
                    hops=hops,
                    total_cost=total_cost,
                    total_priority=total_priority,
                )
                if best_plan is None or _plan_order_key(candidate) < _plan_order_key(best_plan):
                    best_plan = candidate
                continue

            if hop_count >= self.max_hops:
                continue

            for next_tier_id, hop in adjacency.get(current, ()):
                if next_tier_id in path_nodes:
                    continue
                next_hops = hops + (hop,)
                next_cost = total_cost + hop.hop_cost
                next_priority = total_priority + hop.hop_priority
                route_signature = _hop_route_signature(next_hops)
                heapq.heappush(
                    frontier,
                    (
                        (next_cost, -next_priority, route_signature),
                        hop_count + 1,
                        next_tier_id,
                        path_nodes + (next_tier_id,),
                        next_hops,
                        next_cost,
                        next_priority,
                    ),
                )

        return best_plan


@dataclass(frozen=True)
class KVTransferTicket(SerializableDataclass):
    """Optional short-lived single-use authorization for a planned transfer."""

    plan_digest: str
    authority_domain_digest: str
    ticket_nonce_digest: str
    issued_at: float
    expires_at: float
    ticket_auth_tag: str
    ticket_version: str = KV_TRANSFER_TICKET_V1
    ticket_digest: str = ""

    def __post_init__(self) -> None:
        _strong_digest("plan_digest", self.plan_digest)
        _strong_digest("authority_domain_digest", self.authority_domain_digest)
        _strong_digest("ticket_nonce_digest", self.ticket_nonce_digest)
        _require_numeric("issued_at", self.issued_at)
        _require_numeric("expires_at", self.expires_at)
        if self.expires_at <= self.issued_at:
            raise ContractError("expires_at must be later than issued_at")
        _strong_digest("ticket_auth_tag", self.ticket_auth_tag)
        if self.ticket_version != KV_TRANSFER_TICKET_V1:
            raise ContractError("unsupported KV transfer ticket version")
        computed = self.compute_ticket_digest()
        if self.ticket_digest and self.ticket_digest != computed:
            raise ContractError("KV transfer ticket integrity check failed")
        object.__setattr__(self, "ticket_digest", computed)

    def compute_ticket_digest(self) -> str:
        return _json_digest(
            {
                "plan_digest": self.plan_digest,
                "authority_domain_digest": self.authority_domain_digest,
                "ticket_nonce_digest": self.ticket_nonce_digest,
                "issued_at": self.issued_at,
                "expires_at": self.expires_at,
                "ticket_auth_tag": self.ticket_auth_tag,
                "ticket_version": self.ticket_version,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        if self.ticket_digest != self.compute_ticket_digest():
            raise ContractError("KV transfer ticket integrity check failed")
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "KVTransferTicket":
        payload = _strict_payload(cls, data)
        if not payload.get("ticket_digest"):
            raise ContractError("KV transfer ticket ticket_digest is required")
        return cls(**payload)


class KVTransferTicketIssuer:
    """In-memory single-use ticket issuer with fail-closed replay checks."""

    def __init__(
        self,
        *,
        signing_key: bytes,
        clock: Callable[[], float] = time.time,
        max_ttl_seconds: float = 120.0,
        nonce_bytes_factory: Callable[[int], bytes] = secrets.token_bytes,
    ) -> None:
        if not isinstance(signing_key, (bytes, bytearray)):
            raise ContractError("signing_key must be bytes")
        self._signing_key = bytes(signing_key)
        if len(self._signing_key) < 32:
            raise ContractError("signing_key must be at least 32 bytes")
        _require_numeric("max_ttl_seconds", max_ttl_seconds)
        if max_ttl_seconds <= 0:
            raise ContractError("max_ttl_seconds must be positive")
        if not callable(nonce_bytes_factory):
            raise ContractError("nonce_bytes_factory must be callable")
        self.clock = clock
        self.max_ttl_seconds = float(max_ttl_seconds)
        self._nonce_bytes_factory = nonce_bytes_factory
        self._consumed_ticket_digests: set[str] = set()

    def issue(
        self,
        *,
        plan: KVTransferPlan,
        authority_domain_digest: str,
        ttl_seconds: float,
    ) -> KVTransferTicket:
        if not isinstance(plan, KVTransferPlan):
            raise ContractError("plan must be a KVTransferPlan")
        _strong_digest("authority_domain_digest", authority_domain_digest)
        if authority_domain_digest != plan.descriptor.authority_domain_digest:
            raise ContractError("transfer ticket authority domain does not match plan")
        _require_numeric("ttl_seconds", ttl_seconds)
        ttl = float(ttl_seconds)
        if ttl <= 0 or ttl > self.max_ttl_seconds:
            raise ContractError("ttl_seconds is outside allowed bounds")
        now = _coerce_finite_number("ticket clock", self.clock())
        nonce_bytes = self._nonce_bytes_factory(32)
        if not isinstance(nonce_bytes, (bytes, bytearray)) or len(nonce_bytes) < 32:
            raise ContractError("nonce_bytes_factory must return at least 32 bytes")
        nonce = "sha256:" + hashlib.sha256(bytes(nonce_bytes)).hexdigest()
        public_payload = {
            "plan_digest": plan.plan_digest,
            "authority_domain_digest": authority_domain_digest,
            "ticket_nonce_digest": nonce,
            "issued_at": now,
            "expires_at": now + ttl,
            "ticket_version": KV_TRANSFER_TICKET_V1,
        }
        return KVTransferTicket(
            plan_digest=plan.plan_digest,
            authority_domain_digest=authority_domain_digest,
            ticket_nonce_digest=nonce,
            issued_at=now,
            expires_at=now + ttl,
            ticket_version=KV_TRANSFER_TICKET_V1,
            ticket_auth_tag=self._compute_ticket_auth_tag(public_payload),
        )

    def consume(
        self,
        ticket: KVTransferTicket,
        *,
        expected_plan_digest: str,
        expected_authority_domain_digest: str,
    ) -> None:
        if not isinstance(ticket, KVTransferTicket):
            raise ContractError("ticket must be a KVTransferTicket")
        _strong_digest("expected_plan_digest", expected_plan_digest)
        _strong_digest("expected_authority_domain_digest", expected_authority_domain_digest)
        expected_auth_tag = self._compute_ticket_auth_tag(_ticket_auth_payload(ticket))
        if not hmac.compare_digest(ticket.ticket_auth_tag, expected_auth_tag):
            raise ContractError("transfer ticket authentication failed")
        if ticket.plan_digest != expected_plan_digest:
            raise ContractError("transfer ticket plan digest mismatch")
        if ticket.authority_domain_digest != expected_authority_domain_digest:
            raise ContractError("transfer ticket authority domain mismatch")
        now = _coerce_finite_number("ticket clock", self.clock())
        if now < ticket.issued_at:
            raise ContractError("transfer ticket is not yet valid")
        if now >= ticket.expires_at:
            raise ContractError("transfer ticket expired")
        if ticket.ticket_digest in self._consumed_ticket_digests:
            raise ContractError("transfer ticket replay detected")
        self._consumed_ticket_digests.add(ticket.ticket_digest)

    def _compute_ticket_auth_tag(self, payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return "sha256:" + hmac.new(self._signing_key, encoded, hashlib.sha256).hexdigest()


def _validated_intermediate_ids(
    tier_ids: Sequence[str],
    endpoint_map: Mapping[str, KVTierEndpointCapability],
) -> list[str]:
    validated: list[str] = []
    for value in tier_ids:
        _safe_id("intermediate_tier_id", value)
        if value not in endpoint_map:
            raise ContractError("intermediate tier endpoint declaration is required")
        if value in validated:
            raise ContractError("duplicate tier IDs are not allowed in route declaration")
        validated.append(value)
    return validated


def _best_hop(
    source: KVTierEndpointCapability,
    destination: KVTierEndpointCapability,
    transfer_bytes: int,
) -> Optional[KVTransferHop]:
    if transfer_bytes > destination.admission_available_bytes:
        return None
    candidates: list[KVTransferHop] = []
    source_mechanisms = {item.mechanism: item for item in source.transfers}
    destination_mechanisms = {item.mechanism: item for item in destination.transfers}
    for mechanism in sorted(set(source_mechanisms) & set(destination_mechanisms), key=lambda item: item.value):
        source_cap = source_mechanisms[mechanism]
        destination_cap = destination_mechanisms[mechanism]
        if transfer_bytes > source_cap.max_bytes_per_transfer:
            continue
        if transfer_bytes > destination_cap.max_bytes_per_transfer:
            continue
        if not _allows_egress(source_cap.direction):
            continue
        if not _allows_ingress(destination_cap.direction):
            continue
        candidates.append(
            KVTransferHop(
                source_tier_id=source.tier_id,
                destination_tier_id=destination.tier_id,
                mechanism=mechanism,
                hop_cost=source_cap.unit_cost + destination_cap.unit_cost,
                hop_priority=source_cap.priority + destination_cap.priority,
            )
        )
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item.hop_cost, -item.hop_priority, item.mechanism.value))


def _adjacency_by_source(
    *,
    selected: Mapping[str, KVTierEndpointCapability],
    transfer_bytes: int,
) -> Dict[str, tuple[tuple[str, KVTransferHop], ...]]:
    adjacency: Dict[str, tuple[tuple[str, KVTransferHop], ...]] = {}
    for source_id in sorted(selected):
        source = selected[source_id]
        candidates: list[tuple[str, KVTransferHop]] = []
        for destination_id in sorted(selected):
            if source_id == destination_id:
                continue
            destination = selected[destination_id]
            hop = _best_hop(source, destination, transfer_bytes)
            if hop is None:
                continue
            candidates.append((destination_id, hop))
        adjacency[source_id] = tuple(candidates)
    return adjacency


def _edge_rejection_reason(
    source: KVTierEndpointCapability,
    destination: KVTierEndpointCapability,
    transfer_bytes: int,
) -> Optional[str]:
    if transfer_bytes > destination.admission_available_bytes:
        return "insufficient admission capacity at destination tier"
    source_mechanisms = {item.mechanism: item for item in source.transfers}
    destination_mechanisms = {item.mechanism: item for item in destination.transfers}
    common = set(source_mechanisms) & set(destination_mechanisms)
    if not common:
        return "unsupported transport intersection between tiers"
    wrong_direction = False
    for mechanism in common:
        source_cap = source_mechanisms[mechanism]
        destination_cap = destination_mechanisms[mechanism]
        if transfer_bytes > source_cap.max_bytes_per_transfer or transfer_bytes > destination_cap.max_bytes_per_transfer:
            continue
        if not _allows_egress(source_cap.direction) or not _allows_ingress(destination_cap.direction):
            wrong_direction = True
            continue
        return None
    if wrong_direction:
        return "transport direction mismatch between tiers"
    return "insufficient per-transfer transport capacity"


def _allows_ingress(direction: KVTransferDirection) -> bool:
    return direction in {KVTransferDirection.INGRESS, KVTransferDirection.BIDIRECTIONAL}


def _allows_egress(direction: KVTransferDirection) -> bool:
    return direction in {KVTransferDirection.EGRESS, KVTransferDirection.BIDIRECTIONAL}


def _plan_order_key(plan: KVTransferPlan) -> tuple[int, int, str]:
    return (plan.total_cost, -plan.total_priority, _hop_route_signature(plan.hops))


def _hop_route_signature(hops: Sequence[KVTransferHop]) -> str:
    return "|".join(f"{item.source_tier_id}>{item.destination_tier_id}:{item.mechanism.value}" for item in hops)


def _coerce_hops(items: Any) -> tuple[KVTransferHop, ...]:
    if not isinstance(items, (list, tuple)):
        raise ContractError("hops must be a sequence")
    coerced: list[KVTransferHop] = []
    for item in items:
        if isinstance(item, KVTransferHop):
            coerced.append(item)
        elif isinstance(item, dict):
            coerced.append(KVTransferHop.from_dict(item))
        else:
            raise ContractError("hops must contain KVTransferHop entries")
    return tuple(coerced)


def _validate_hop_chain(
    hops: Sequence[KVTransferHop],
    *,
    source: str,
    destination: str,
) -> None:
    if hops[0].source_tier_id != source:
        raise ContractError("transfer plan source hop mismatch")
    if hops[-1].destination_tier_id != destination:
        raise ContractError("transfer plan destination hop mismatch")
    visited = {source}
    for previous, current in zip(hops, hops[1:]):
        if previous.destination_tier_id != current.source_tier_id:
            raise ContractError("transfer plan hop chain is not contiguous")
    for hop in hops:
        if hop.destination_tier_id in visited:
            raise ContractError("transfer plan cannot contain a cycle")
        visited.add(hop.destination_tier_id)


def _endpoint_map(endpoints: Sequence[KVTierEndpointCapability]) -> Dict[str, KVTierEndpointCapability]:
    mapping: Dict[str, KVTierEndpointCapability] = {}
    for item in endpoints:
        if not isinstance(item, KVTierEndpointCapability):
            raise ContractError("endpoints must contain KVTierEndpointCapability entries")
        existing = mapping.get(item.tier_id)
        if existing is not None:
            raise ContractError("duplicate tier IDs are not allowed in endpoint declarations")
        mapping[item.tier_id] = item
    return mapping


def _coerce_transport_capabilities(items: Any) -> tuple[KVTransportCapability, ...]:
    if not isinstance(items, (list, tuple)):
        raise ContractError("transfers must be a sequence")
    normalized: list[KVTransportCapability] = []
    for item in items:
        if isinstance(item, KVTransportCapability):
            normalized.append(item)
        elif isinstance(item, dict):
            normalized.append(KVTransportCapability.from_dict(item))
        else:
            raise ContractError("transfers must contain KVTransportCapability entries")
    return tuple(normalized)


def _coerce_digest_tuple(name: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ContractError(f"{name} must be a sequence")
    out: list[str] = []
    for item in value:
        _strong_digest(name, item)
        if item in out:
            raise ContractError(f"{name} must not contain duplicates")
        out.append(item)
    if not out:
        raise ContractError(f"{name} must be non-empty")
    return tuple(out)


def _require_unique_mechanisms(transfers: Iterable[KVTransportCapability]) -> None:
    seen: set[KVTransferMechanism] = set()
    for item in transfers:
        if item.mechanism in seen:
            raise ContractError("transfers must not declare a mechanism more than once")
        seen.add(item.mechanism)


def _require_semantics_match(kind: KVTierKind, durability: KVDurability, locality: KVLocality) -> None:
    semantics = tier_semantics_for_kind(kind)
    if semantics.durability is not durability or semantics.locality is not locality:
        raise ContractError("tier semantics mismatch for kind")


def _strict_payload(cls: type, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ContractError(f"{cls.__name__}.from_dict expected dict, got {type(data).__name__}")
    known = set(getattr(cls, "__dataclass_fields__", {}))
    unknown = set(data) - known
    if unknown:
        raise ContractError(f"{cls.__name__} contains unknown fields: {sorted(unknown)}")
    return dict(data)


def _strong_digest(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ContractError(f"{name} must be a sha256 digest")
    hex_part = value[7:]
    if len(hex_part) != 64 or any(ch not in "0123456789abcdef" for ch in hex_part):
        raise ContractError(f"{name} must be a sha256 digest")


def _safe_id(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be a safe identifier")
    if re.fullmatch(_SAFE_ID_RE, value) is None:
        raise ContractError(f"{name} must be a safe identifier")


def _require_positive_int(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")


def _require_non_negative_int(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractError(f"{name} must be a non-negative integer")


def _require_numeric(name: str, value: Any) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ContractError(f"{name} must be numeric")
    if not math.isfinite(float(value)):
        raise ContractError(f"{name} must be finite")


def _coerce_finite_number(name: str, value: Any) -> float:
    _require_numeric(name, value)
    return float(value)


def runtime_compatibility_digest(*, runtime_id: str, runtime_version: str, adapter_id: str) -> str:
    for name, value in (
        ("runtime_id", runtime_id),
        ("runtime_version", runtime_version),
        ("adapter_id", adapter_id),
    ):
        if not isinstance(value, str) or not value:
            raise ContractError(f"{name} must be non-empty")
    if runtime_version.casefold() == "unknown":
        raise ContractError("runtime_version must be exact")
    return _json_digest(
        {
            "runtime_id": runtime_id,
            "runtime_version": runtime_version,
            "adapter_id": adapter_id,
        }
    )


def _ticket_auth_payload(ticket: KVTransferTicket) -> Dict[str, Any]:
    return {
        "plan_digest": ticket.plan_digest,
        "authority_domain_digest": ticket.authority_domain_digest,
        "ticket_nonce_digest": ticket.ticket_nonce_digest,
        "issued_at": ticket.issued_at,
        "expires_at": ticket.expires_at,
        "ticket_version": ticket.ticket_version,
    }


def _json_digest(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
