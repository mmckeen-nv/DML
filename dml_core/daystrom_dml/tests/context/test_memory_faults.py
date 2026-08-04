from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from typing import Optional

import pytest

from daystrom_dml.api_contracts import ContractError, DaystromScope
from daystrom_dml.context.adapters.memory import (
    DML2ExactPageFaultAdapter,
    STMHotMemoryFaultAdapter,
)
from daystrom_dml.context.faults import (
    EvidenceAuthority,
    EvidenceHandle,
    MemoryFaultBudget,
    MemoryFaultReason,
    MemoryFaultRequest,
    MemoryFaultResolver,
    MemoryFaultResult,
    MemoryFaultStatus,
    MemorySourceTier,
)
from daystrom_dml.context.paging import MemoryContextPageCache
from daystrom_dml.stm.schema import Commitment, STMState


def scope(**overrides: str) -> DaystromScope:
    values = {
        "tenant_id": "tenant-a",
        "client_id": "client-a",
        "session_id": "session-a",
        "instance_id": "instance-a",
        "thread_id": "thread-a",
        "project_id": "project-a",
        "relationship_id": "relationship-a",
    }
    values.update(overrides)
    return DaystromScope(**values)


def request(**overrides: object) -> MemoryFaultRequest:
    values: dict[str, object] = {
        "request_id": "req-1",
        "scope": scope(),
        "query": "dependency-free",
        "key": "page-1",
        "budget": MemoryFaultBudget(max_items=1, max_payload_bytes=1024, max_payload_tokens=256),
    }
    values.update(overrides)
    return MemoryFaultRequest(**values)


def handle(
    tier: MemorySourceTier,
    *,
    payload: Optional[str] = None,
    digest: Optional[str] = None,
    item_scope: Optional[DaystromScope] = None,
    handle_id: str = "h-1",
) -> EvidenceHandle:
    content_digest = digest
    if content_digest is None and payload is not None:
        content_digest = "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if content_digest is None:
        content_digest = "sha256:" + hashlib.sha256(handle_id.encode("utf-8")).hexdigest()
    return EvidenceHandle(
        handle_id=handle_id,
        tier=tier,
        scope=item_scope or scope(),
        digest=content_digest,
        authority=EvidenceAuthority.REFERENCE,
        size_bytes=len(payload.encode("utf-8")) if payload is not None else 0,
        payload_text=payload,
        provenance={"adapter_id": f"{tier.value}-fake", "source_id": handle_id},
    )


class FakeLookup:
    def __init__(
        self,
        tier: MemorySourceTier,
        result: Optional[EvidenceHandle] = None,
        *,
        status: MemoryFaultStatus = MemoryFaultStatus.MISS,
        raises: bool = False,
        reason: MemoryFaultReason = MemoryFaultReason.NOT_FOUND,
    ) -> None:
        self.tier = tier
        self.result = result
        self.status = status
        self.raises = raises
        self.reason = reason
        self.calls: list[DaystromScope] = []

    def lookup(self, req: MemoryFaultRequest) -> MemoryFaultResult:
        self.calls.append(req.scope)
        if self.raises:
            raise RuntimeError("payload secret should not leak")
        if self.result is not None:
            return MemoryFaultResult.hit(req, self.tier, [self.result], reason=MemoryFaultReason.FOUND)
        return MemoryFaultResult.empty(req, self.status, self.tier, [self.reason])


class FakeDurableLookup(FakeLookup):
    pass


def test_dml1_hit_stops_before_lower_tiers() -> None:
    dml1 = FakeLookup(MemorySourceTier.DML1_HOT, handle(MemorySourceTier.DML1_HOT, payload="hot evidence"))
    dml2 = FakeLookup(MemorySourceTier.DML2_EXACT, handle(MemorySourceTier.DML2_EXACT, payload="page evidence"))
    durable = FakeDurableLookup(MemorySourceTier.DURABLE, handle(MemorySourceTier.DURABLE, payload="durable evidence"))

    result = MemoryFaultResolver(dml1_hot=dml1, dml2_exact=dml2, durable=durable).resolve(request(include_payload=True))

    assert result.status is MemoryFaultStatus.HIT
    assert result.tier is MemorySourceTier.DML1_HOT
    assert result.evidence[0].payload_text == "hot evidence"
    assert len(dml1.calls) == 1
    assert dml2.calls == []
    assert durable.calls == []


def test_dml2_page_hit_after_dml1_miss() -> None:
    cache = MemoryContextPageCache(max_pages=4, max_bytes=4096)
    page = cache.put_text(scope(), "exact page evidence", ttl_seconds=60, page_id="page-1")
    dml2 = DML2ExactPageFaultAdapter(cache)

    result = MemoryFaultResolver(
        dml1_hot=FakeLookup(MemorySourceTier.DML1_HOT),
        dml2_exact=dml2,
    ).resolve(request(include_payload=True, query=None))

    assert result.status is MemoryFaultStatus.HIT
    assert result.tier is MemorySourceTier.DML2_EXACT
    assert result.evidence[0].handle_id == page.page_id
    assert result.evidence[0].digest == page.content_digest
    assert result.evidence[0].payload_text == "exact page evidence"
    assert result.evidence[0].authority is EvidenceAuthority.UNTRUSTED_DATA


def test_dml1_stm_adapter_hit_uses_existing_hot_context_adapter() -> None:
    state = STMState(
        commitments=[
            Commitment(
                id="c-1",
                statement="Keep memory evidence dependency-free",
                confidence=0.99,
                source="design-note",
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        ],
        version=7,
    )
    dml1 = STMHotMemoryFaultAdapter(lambda req_scope: state if req_scope == scope() else None)

    result = MemoryFaultResolver(dml1_hot=dml1).resolve(request(include_payload=True, key=None))

    assert result.status is MemoryFaultStatus.HIT
    assert result.evidence[0].tier is MemorySourceTier.DML1_HOT
    assert "dependency-free" in (result.evidence[0].payload_text or "")
    assert result.evidence[0].digest.startswith("sha256:")


def test_opt_in_durable_hit_and_denied_without_opt_in() -> None:
    durable = FakeDurableLookup(MemorySourceTier.DURABLE, handle(MemorySourceTier.DURABLE, payload="durable"))

    denied = MemoryFaultResolver(durable=durable).resolve(request(allow_durable=False))
    allowed = MemoryFaultResolver(durable=durable).resolve(request(allow_durable=True, include_payload=True))

    assert denied.status is MemoryFaultStatus.DENIED
    assert denied.reason_codes == [MemoryFaultReason.DURABLE_NOT_ALLOWED]
    assert durable.calls == [scope()]
    assert allowed.status is MemoryFaultStatus.HIT
    assert allowed.tier is MemorySourceTier.DURABLE
    assert allowed.evidence[0].payload_text == "durable"


def test_escalation_ordering_is_deterministic() -> None:
    dml1 = FakeLookup(MemorySourceTier.DML1_HOT)
    dml2 = FakeLookup(MemorySourceTier.DML2_EXACT)
    durable = FakeDurableLookup(MemorySourceTier.DURABLE, handle(MemorySourceTier.DURABLE, payload="durable"))

    result = MemoryFaultResolver(dml1_hot=dml1, dml2_exact=dml2, durable=durable).resolve(
        request(allow_durable=True)
    )

    assert result.status is MemoryFaultStatus.HIT
    assert [event["tier"] for event in result.telemetry["attempts"]] == ["dml1_hot", "dml2_exact", "durable"]


@pytest.mark.parametrize(
    ("field", "other_value"),
    [
        ("tenant_id", "tenant-b"),
        ("client_id", "client-b"),
        ("session_id", "session-b"),
        ("instance_id", "instance-b"),
        ("thread_id", "thread-b"),
        ("project_id", "project-b"),
        ("relationship_id", "relationship-b"),
    ],
)
def test_scope_isolation_treats_cross_scope_adapter_results_as_miss(field: str, other_value: str) -> None:
    other_handle = handle(MemorySourceTier.DML1_HOT, payload="cross-scope secret", item_scope=scope(**{field: other_value}))
    result = MemoryFaultResolver(dml1_hot=FakeLookup(MemorySourceTier.DML1_HOT, other_handle)).resolve(request())

    assert result.status is MemoryFaultStatus.CORRUPT
    assert result.evidence == []
    assert result.reason_codes == [MemoryFaultReason.SCOPE_MISMATCH]
    assert "cross-scope secret" not in json.dumps(result.to_dict())


def test_exact_payload_budget_returns_handle_without_payload() -> None:
    dml1 = FakeLookup(MemorySourceTier.DML1_HOT, handle(MemorySourceTier.DML1_HOT, payload="too large"))

    result = MemoryFaultResolver(dml1_hot=dml1).resolve(
        request(include_payload=True, budget=MemoryFaultBudget(max_items=1, max_payload_bytes=3, max_payload_tokens=20))
    )

    assert result.status is MemoryFaultStatus.OVER_BUDGET
    assert result.evidence[0].payload_text is None
    assert result.evidence[0].size_bytes == len("too large".encode("utf-8"))
    assert result.reason_codes == [MemoryFaultReason.PAYLOAD_OVER_BUDGET]


def test_digest_corruption_fails_closed_without_payload_leak() -> None:
    cache = MemoryContextPageCache(max_pages=4, max_bytes=4096)
    page = cache.put_text(scope(), "secret exact page", ttl_seconds=60, page_id="page-1")
    corrupted = replace(page, content_digest="sha256:corrupt")
    cache._entries[((scope().tenant_id, scope().client_id, scope().session_id, scope().instance_id, scope().thread_id, scope().project_id, scope().relationship_id), page.page_id)].page = corrupted  # type: ignore[attr-defined]

    result = MemoryFaultResolver(dml2_exact=DML2ExactPageFaultAdapter(cache)).resolve(request(query=None))

    assert result.status is MemoryFaultStatus.CORRUPT
    assert result.evidence == []
    assert result.reason_codes == [MemoryFaultReason.DIGEST_MISMATCH]
    assert "secret exact page" not in json.dumps(result.to_dict())


def test_adapter_exceptions_are_contained_and_payload_free() -> None:
    result = MemoryFaultResolver(
        dml1_hot=FakeLookup(MemorySourceTier.DML1_HOT, raises=True),
        dml2_exact=FakeLookup(MemorySourceTier.DML2_EXACT),
    ).resolve(request())

    serialized = json.dumps(result.to_dict(), sort_keys=True)
    assert result.status is MemoryFaultStatus.ADAPTER_ERROR
    assert result.reason_codes == [MemoryFaultReason.ADAPTER_EXCEPTION]
    assert "payload secret" not in serialized
    assert "RuntimeError" in serialized


def test_no_payload_telemetry_or_logging() -> None:
    result = MemoryFaultResolver(
        dml1_hot=FakeLookup(MemorySourceTier.DML1_HOT, handle(MemorySourceTier.DML1_HOT, payload="secret evidence"))
    ).resolve(request(include_payload=True))

    assert result.evidence[0].payload_text == "secret evidence"
    telemetry = json.dumps(result.telemetry, sort_keys=True)
    assert "secret evidence" not in telemetry
    assert result.telemetry["attempts"][0]["evidence_count"] == 1


def test_json_roundtrip_and_contract_validation() -> None:
    original = MemoryFaultResolver(
        dml1_hot=FakeLookup(MemorySourceTier.DML1_HOT, handle(MemorySourceTier.DML1_HOT, payload="payload"))
    ).resolve(request(include_payload=True))

    restored = MemoryFaultResult.from_dict(json.loads(json.dumps(original.to_dict())))

    assert restored == original
    with pytest.raises(ContractError, match="schema_version"):
        MemoryFaultRequest.from_dict({**request().to_dict(), "schema_version": "0.9"})
    with pytest.raises(ContractError, match="max_items"):
        MemoryFaultBudget(max_items=-1)
