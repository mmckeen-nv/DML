import json

import pytest

from daystrom_dml.api_contracts import ContractError, DaystromScope
from daystrom_dml.context import (
    ACTIVE_ADMISSION_MODE,
    ContextAuthority,
    ContextPriority,
    ContextSegment,
    WorkingSetManager,
)
from daystrom_dml.context.controller import ContextController


def segment(
    segment_id: str,
    *,
    scope: DaystromScope,
    content: str | None = None,
    tokens: int = 1,
    authority: ContextAuthority = ContextAuthority.REFERENCE,
    priority: ContextPriority = ContextPriority.REFERENCE,
) -> ContextSegment:
    return ContextSegment(
        segment_id=segment_id,
        kind="working-set-test",
        content=content if content is not None else segment_id,
        scope=scope,
        authority=authority,
        priority=priority,
        estimated_tokens=tokens,
    )


def reconcile(manager: WorkingSetManager, scope: DaystromScope, segments, **kwargs):
    return manager.reconcile(
        scope=scope,
        segments=segments,
        model_id="model",
        runtime_id="runtime",
        model_limit_tokens=kwargs.pop("model_limit_tokens", 8),
        **kwargs,
    )


def test_working_set_replacement_tracks_retained_added_replaced_evicted_and_omitted():
    scope = DaystromScope(tenant_id="tenant", session_id="session")
    manager = WorkingSetManager(max_candidates=16, clock=lambda: 100.0)
    first = reconcile(
        manager,
        scope,
        [
            segment("system", scope=scope, authority=ContextAuthority.IMMUTABLE),
            segment("plan", scope=scope, priority=ContextPriority.WORKING, content="plan-v1", tokens=2),
            segment("old-evidence", scope=scope, content="old", tokens=2),
        ],
    )

    second = reconcile(
        manager,
        scope,
        [
            segment("system", scope=scope, authority=ContextAuthority.IMMUTABLE),
            segment("plan", scope=scope, priority=ContextPriority.WORKING, content="plan-v2", tokens=2),
            segment("new-evidence", scope=scope, content="new", tokens=2),
            segment("oversized", scope=scope, content="cold", tokens=20),
        ],
        parent_manifest=first.manifest,
    )

    transition = second.decisions["working_set"]
    assert transition == {
        "version": "daystrom-working-set-transition-v1",
        "parent_manifest_id": first.manifest.content_digest,
        "retained": ["system"],
        "replaced": ["plan"],
        "added": ["new-evidence"],
        "evicted": ["old-evidence"],
        "omitted": ["oversized"],
        "stable_prefix": ["system"],
        "prefill_from_index": 1,
        "prefill_segment_ids": ["plan", "new-evidence"],
    }
    assert second.manifest.parent_manifest_id == first.manifest.content_digest
    assert second.manifest.segment_ids == ["system", "plan", "new-evidence"]
    assert set(second.manifest.segment_digests) == set(second.manifest.segment_ids)
    assert second.manifest.segment_digests["plan"] != first.manifest.segment_digests["plan"]
    assert second.packet_content_digest == second.compute_content_digest()
    assert type(second).from_dict(json.loads(json.dumps(second.to_dict()))) == second


def test_working_set_manifest_and_transition_are_deterministic_across_fresh_managers():
    scope = DaystromScope(tenant_id="tenant", session_id="session")
    segments = [
        segment("system", scope=scope, authority=ContextAuthority.IMMUTABLE),
        segment("task", scope=scope, authority=ContextAuthority.CURRENT_INSTRUCTION),
        segment("evidence", scope=scope),
    ]

    left = reconcile(WorkingSetManager(clock=lambda: 10.0), scope, segments)
    right = reconcile(WorkingSetManager(clock=lambda: 20.0), scope, segments)

    assert left.manifest.created_at != right.manifest.created_at
    assert left.manifest.content_digest == right.manifest.content_digest
    assert left.decisions == right.decisions


def test_working_set_reorder_invalidates_prefill_at_first_divergence():
    scope = DaystromScope(tenant_id="tenant", session_id="session")
    manager = WorkingSetManager()
    first = reconcile(
        manager,
        scope,
        [
            segment("system", scope=scope, authority=ContextAuthority.IMMUTABLE),
            segment("a", scope=scope),
            segment("b", scope=scope),
        ],
    )
    second = reconcile(
        manager,
        scope,
        [
            segment("system", scope=scope, authority=ContextAuthority.IMMUTABLE),
            segment("b", scope=scope),
            segment("a", scope=scope),
        ],
        parent_manifest=first.manifest,
    )

    transition = second.decisions["working_set"]
    assert transition["stable_prefix"] == ["system"]
    assert transition["prefill_from_index"] == 1
    assert transition["prefill_segment_ids"] == ["b", "a"]
    assert transition["retained"] == ["system", "b", "a"]


def test_working_set_preflight_rejects_parent_drift_before_page_out_side_effects():
    scope = DaystromScope(tenant_id="tenant", session_id="session")
    manager = WorkingSetManager()
    first = reconcile(manager, scope, [segment("old", scope=scope)])
    paged: list[str] = []

    for changed in (
        {"scope": DaystromScope(tenant_id="other", session_id="session")},
        {"model_id": "other-model"},
        {"runtime_id": "other-runtime"},
    ):
        args = {
            "scope": scope,
            "segments": [segment("too-large", scope=scope, tokens=20)],
            "model_id": "model",
            "runtime_id": "runtime",
            "model_limit_tokens": 8,
            "parent_manifest": first.manifest,
            "page_out": lambda _, item: paged.append(item.segment_id),
        }
        args.update(changed)
        with pytest.raises(ContractError, match="parent manifest"):
            manager.reconcile(**args)

    assert paged == []


def test_working_set_candidate_limit_fails_before_admission_or_page_out():
    scope = DaystromScope(tenant_id="tenant", session_id="session")
    manager = WorkingSetManager(max_candidates=2)
    paged: list[str] = []

    with pytest.raises(ContractError, match="max_candidates"):
        reconcile(
            manager,
            scope,
            (segment(f"seg-{index}", scope=scope, tokens=20) for index in range(3)),
            page_out=lambda _, item: paged.append(item.segment_id),
        )

    assert paged == []


def test_working_set_copies_parent_and_candidates_and_detects_parent_digest_tampering():
    scope = DaystromScope(tenant_id="tenant", session_id="session")
    manager = WorkingSetManager()
    candidate = segment("task", scope=scope, authority=ContextAuthority.CURRENT_INSTRUCTION, content="original")
    first = reconcile(manager, scope, [candidate])
    parent_wire = first.manifest.to_dict()
    second = reconcile(manager, scope, [candidate], parent_manifest=first.manifest)

    candidate.content = "mutated"
    first.manifest.segment_ids.append("tampered")
    assert second.segments[0].content == "original"
    assert second.manifest.parent_manifest_id == parent_wire["content_digest"]

    parent_wire["segment_digests"]["task"] = "0" * 64
    with pytest.raises(ContractError, match="content_digest"):
        manager.reconcile(
            scope=scope,
            segments=[segment("task", scope=scope)],
            model_id="model",
            runtime_id="runtime",
            model_limit_tokens=8,
            parent_manifest=parent_wire,
        )


def test_controller_exposes_working_set_reconciliation_only_in_active_mode():
    scope = DaystromScope(tenant_id="tenant", session_id="session")
    args = {
        "scope": scope,
        "segments": [segment("task", scope=scope, authority=ContextAuthority.CURRENT_INSTRUCTION)],
        "model_id": "model",
        "runtime_id": "runtime",
        "model_limit_tokens": 8,
    }

    with pytest.raises(ContractError, match="active_admission"):
        ContextController().reconcile_working_set(**args)

    packet = ContextController(mode=ACTIVE_ADMISSION_MODE, clock=lambda: 7.0).reconcile_working_set(**args)
    assert packet.manifest.created_at == 7.0
    assert packet.decisions["working_set"]["prefill_segment_ids"] == ["task"]


def test_working_set_rejects_duplicate_parent_segments_and_invalid_limits():
    scope = DaystromScope(tenant_id="tenant", session_id="session")
    manager = WorkingSetManager()
    first = reconcile(manager, scope, [segment("task", scope=scope)])
    wire = first.manifest.to_dict()
    wire["segment_ids"] = ["task", "task"]
    wire["content_digest"] = ""

    with pytest.raises(ContractError, match="content_digest"):
        manager.reconcile(
            scope=scope,
            segments=[segment("task", scope=scope)],
            model_id="model",
            runtime_id="runtime",
            model_limit_tokens=8,
            parent_manifest=wire,
        )

    with pytest.raises(ContractError, match="max_candidates"):
        WorkingSetManager(max_candidates=0)
