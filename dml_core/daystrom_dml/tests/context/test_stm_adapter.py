from __future__ import annotations

from datetime import datetime, timezone

from daystrom_dml.api_contracts import DaystromScope
from daystrom_dml.context.adapters.stm import STMHotContextAdapter
from daystrom_dml.stm.schema import Commitment, EntityRecord, Note, PlanState, STMState


FIXED_A = datetime(2026, 1, 1, tzinfo=timezone.utc)
FIXED_B = datetime(2026, 1, 2, tzinfo=timezone.utc)


def scope() -> DaystromScope:
    return DaystromScope(
        tenant_id="tenant-a",
        client_id="client-a",
        session_id="session-a",
        thread_id="thread-a",
        project_id="project-a",
    )


def make_state(*, last_updated: datetime = FIXED_A) -> STMState:
    return STMState(
        commitments=[
            Commitment(
                id="c-low",
                statement="Lower priority commitment",
                confidence=0.25,
                source="model",
                created_at=FIXED_A,
                updated_at=FIXED_A,
                tags=["minor"],
            ),
            Commitment(
                id="c-high",
                statement="Keep the API dependency-free",
                confidence=0.95,
                source="design-note-7",
                created_at=FIXED_A,
                updated_at=FIXED_A,
                tags=["architecture"],
            ),
        ],
        goals=["Implement context paging", "Preserve exact payloads"],
        constraints=["Do not write DML2 pages to disk"],
        entities={
            "STMController": EntityRecord(
                name="STMController",
                type="class",
                attributes={"module": "daystrom_dml.stm.controller"},
                relations=[{"source_segment_id": "seg-controller", "kind": "defined-in"}],
            )
        },
        intermediate=[
            Note(text="User: do not include this raw transcript\nAssistant: agreed", source="system", created_at=FIXED_A)
        ],
        plan=PlanState(steps=["Write tests", "Implement adapter", "Run validation"], current_step=1, status="active"),
        last_updated=last_updated,
        version=3,
    )


def test_snapshot_is_compact_json_friendly_and_deterministic() -> None:
    adapter = STMHotContextAdapter(top_commitments=2)
    first = adapter.snapshot(scope(), make_state(last_updated=FIXED_A))
    second = adapter.snapshot(scope(), make_state(last_updated=FIXED_B))

    assert first == second
    assert first["scope"]["tenant_id"] == "tenant-a"
    assert first["goals"] == ["Implement context paging", "Preserve exact payloads"]
    assert first["current_plan_step"] == {
        "index": 1,
        "total": 3,
        "status": "active",
        "text": "Implement adapter",
    }
    assert [item["id"] for item in first["commitments"]] == ["c-high", "c-low"]
    assert first["entities"][0]["name"] == "STMController"
    assert first["evidence_refs"] == ["design-note-7", "model", "seg-controller"]
    assert "intermediate" not in first
    assert "last_updated" not in first


def test_checkpoint_digest_changes_with_semantic_state_not_timestamps() -> None:
    adapter = STMHotContextAdapter()
    digest_a = adapter.checkpoint_digest(scope(), make_state(last_updated=FIXED_A))
    digest_b = adapter.checkpoint_digest(scope(), make_state(last_updated=FIXED_B))
    changed = make_state(last_updated=FIXED_A)
    changed.goals.append("New semantic goal")

    assert digest_a == digest_b
    assert digest_a != adapter.checkpoint_digest(scope(), changed)
    assert digest_a.startswith("sha256:")


def test_render_outputs_deterministic_segments_respecting_hard_budget() -> None:
    adapter = STMHotContextAdapter(top_commitments=2)

    segments = adapter.render(scope(), make_state(), budget_tokens=18, estimate_tokens=lambda text: len(text.split()))

    assert segments
    assert sum(segment["tokens"] for segment in segments) <= 18
    assert [segment["rank"] for segment in segments] == list(range(len(segments)))
    assert all(segment["metadata"]["scope"]["tenant_id"] == "tenant-a" for segment in segments)
    assert all("User:" not in segment["text"] for segment in segments)
    assert all("Assistant:" not in segment["text"] for segment in segments)


def test_render_zero_budget_returns_no_segments() -> None:
    adapter = STMHotContextAdapter()

    assert adapter.render(scope(), make_state(), budget_tokens=0) == []


def test_adapter_wraps_existing_stm_controller_summary_without_raw_notes() -> None:
    adapter = STMHotContextAdapter()

    summary = adapter.controller.build_stm_summary(make_state())

    assert "Keep the API dependency-free" in summary
    assert "User:" not in "\n".join(segment["text"] for segment in adapter.render(scope(), make_state(), 512))
