from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from daystrom_dml.api_contracts import ContractError, DaystromScope
from daystrom_dml.context import (
    ACTIVE_ADMISSION_MODE,
    ContextAuthority,
    ContextPriority,
    ContextSegment,
    ExecutionCheckpointIdentity,
    ExecutionCheckpointRecord,
    NativeContextCheckpointBinding,
    WorkingSetManager,
)
from daystrom_dml.context.controller import ContextController
from daystrom_dml.context.native_transition import NativeContextTransitionCompiler


def strong(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


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
        kind="native-transition-test",
        content=content if content is not None else segment_id,
        scope=scope,
        authority=authority,
        priority=priority,
        estimated_tokens=tokens,
    )


def generations():
    scope = DaystromScope(tenant_id="tenant", session_id="session")
    manager = WorkingSetManager(max_candidates=16, clock=lambda: 100.0)
    parent = manager.reconcile(
        scope=scope,
        segments=[
            segment("system", scope=scope, tokens=4, authority=ContextAuthority.IMMUTABLE),
            segment("plan", scope=scope, content="plan-v1", tokens=8, priority=ContextPriority.WORKING),
            segment("old-evidence", scope=scope, tokens=6),
        ],
        model_id="model",
        runtime_id="runtime",
        model_limit_tokens=64,
    )
    current = manager.reconcile(
        scope=scope,
        segments=[
            segment("system", scope=scope, tokens=4, authority=ContextAuthority.IMMUTABLE),
            segment("plan", scope=scope, content="plan-v2", tokens=5, priority=ContextPriority.WORKING),
            segment("new-evidence", scope=scope, tokens=7),
        ],
        model_id="model",
        runtime_id="runtime",
        model_limit_tokens=64,
        parent_manifest=parent.manifest,
    )
    return parent, current


def checkpoint_for(packet, *, expires_at: float = 500.0, tokens_saved: int | None = None):
    identity = ExecutionCheckpointIdentity(
        scope=packet.scope,
        model_id=packet.capabilities.model_id,
        model_digest=strong("model"),
        tokenizer_digest=strong("tokenizer"),
        positional_config_digest=strong("position"),
        immutable_prefix_digest=strong("immutable"),
        packet_digest="sha256:" + packet.packet_content_digest,
        manifest_digest="sha256:" + packet.manifest.content_digest,
        runtime_id=packet.capabilities.backend_id,
        runtime_version="0.20.0",
        adapter_id="test-adapter",
        runtime_endpoint_digest=strong("endpoint"),
    )
    return NativeContextCheckpointBinding(
        record=ExecutionCheckpointRecord(
            checkpoint_id="parent-checkpoint",
            checkpoint_name="parent-checkpoint.bin",
            identity=identity,
            tokens_saved=tokens_saved or sum(item.effective_tokens for item in packet.segments),
            bytes_saved=4096,
            created_at=100.0,
            expires_at=expires_at,
        ),
        runtime_checkpoint_digest=strong("runtime-parent-checkpoint"),
    )


def test_compile_binds_semantic_transition_to_native_checkpoint_work() -> None:
    parent, current = generations()
    plan = NativeContextTransitionCompiler(clock=lambda: 200.0).compile(
        parent_packet=parent,
        current_packet=current,
        parent_checkpoint=checkpoint_for(parent),
        model_native_limit=262144,
        served_limit=65536,
    )

    assert plan.stable_prefix_span_ids == ["system"]
    assert plan.stable_prefix_tokens == 4
    assert plan.suffix_span_ids == ["plan", "new-evidence"]
    assert plan.suffix_tokens == 12
    assert [item.span_id for item in plan.page_out] == ["plan", "old-evidence"]
    assert [item.span_id for item in plan.page_in] == ["plan", "new-evidence"]
    assert plan.page_out[0].span_digest == parent.manifest.segment_digests["plan"]
    assert plan.page_in[0].span_digest == current.manifest.segment_digests["plan"]
    assert [step.operation for step in plan.steps] == [
        "restore_parent_prefix",
        "prefill_suffix",
        "checkpoint_current_generation",
    ]
    assert plan.steps[0].checkpoint_id == "parent-checkpoint"
    assert plan.steps[0].checkpoint_digest == strong("runtime-parent-checkpoint")
    assert plan.steps[0].checkpoint_digest != checkpoint_for(parent).record.record_digest
    assert plan.steps[0].token_count == 4
    assert plan.steps[1].span_ids == ["plan", "new-evidence"]
    assert plan.steps[2].checkpoint_digest == plan.current_checkpoint_digest
    assert plan.model_native_limit == 262144
    assert plan.served_limit == 65536
    assert plan.served_limit_shortfall == 196608
    assert plan.feasible is True


def test_compile_without_parent_checkpoint_falls_back_to_full_prefill() -> None:
    parent, current = generations()
    plan = NativeContextTransitionCompiler(clock=lambda: 200.0).compile(
        parent_packet=parent,
        current_packet=current,
        parent_checkpoint=None,
        model_native_limit=64,
        served_limit=64,
    )

    assert plan.stable_prefix_tokens == 0
    assert plan.suffix_span_ids == ["system", "plan", "new-evidence"]
    assert [step.operation for step in plan.steps] == [
        "prefill_full",
        "checkpoint_current_generation",
    ]
    assert "parent_checkpoint_unavailable" in plan.reason_codes


def test_compile_is_deterministic_and_payload_free() -> None:
    parent, current = generations()
    compiler = NativeContextTransitionCompiler(clock=lambda: 200.0)
    left = compiler.compile(
        parent_packet=parent,
        current_packet=current,
        parent_checkpoint=checkpoint_for(parent),
        model_native_limit=64,
        served_limit=64,
    )
    right = compiler.compile(
        parent_packet=parent,
        current_packet=current,
        parent_checkpoint=checkpoint_for(parent),
        model_native_limit=64,
        served_limit=64,
    )

    assert left.plan_digest == right.plan_digest
    wire = left.to_dict()
    assert type(left).from_dict(json.loads(json.dumps(wire))) == left
    serialized = json.dumps(wire, sort_keys=True)
    assert "plan-v1" not in serialized
    assert "plan-v2" not in serialized
    assert "old-evidence" in serialized  # IDs are permitted; payload text is not.
    assert "rendered_messages" not in serialized
    assert "content" not in serialized

    left.suffix_span_ids.append("tampered")
    with pytest.raises(ContractError, match="plan_digest"):
        left.to_dict()


def test_checkpoint_binding_requires_distinct_prefixed_runtime_identity() -> None:
    parent, _ = generations()
    record = checkpoint_for(parent).record

    with pytest.raises(ContractError, match="runtime_checkpoint_digest"):
        NativeContextCheckpointBinding(
            record=record,
            runtime_checkpoint_digest="0" * 64,
        )
    with pytest.raises(ContractError, match="binding version"):
        NativeContextCheckpointBinding(
            record=record,
            runtime_checkpoint_digest=strong("runtime-parent-checkpoint"),
            binding_version="future-version",
        )


@pytest.mark.parametrize("field", ["packet_digest", "manifest_digest", "model_id", "runtime_id"])
def test_compile_rejects_checkpoint_identity_drift(field: str) -> None:
    parent, current = generations()
    binding = checkpoint_for(parent)
    wire = binding.record.to_dict()
    if field in {"packet_digest", "manifest_digest"}:
        wire["identity"][field] = strong("drift")
    else:
        wire["identity"][field] = "drift"
    wire["identity"]["binding_digest"] = ""
    wire["record_digest"] = ""

    with pytest.raises(ContractError, match="checkpoint"):
        # Rebuilding an internally coherent but parent-incompatible record must
        # still fail before the compiler emits executable work.
        identity_wire = wire["identity"]
        identity_wire.pop("binding_digest")
        identity = ExecutionCheckpointIdentity(**identity_wire)
        record = ExecutionCheckpointRecord(
            checkpoint_id=wire["checkpoint_id"],
            checkpoint_name=wire["checkpoint_name"],
            identity=identity,
            tokens_saved=wire["tokens_saved"],
            bytes_saved=wire["bytes_saved"],
            created_at=wire["created_at"],
            expires_at=wire["expires_at"],
        )
        NativeContextTransitionCompiler(clock=lambda: 200.0).compile(
            parent_packet=parent,
            current_packet=current,
            parent_checkpoint=NativeContextCheckpointBinding(
                record=record,
                runtime_checkpoint_digest=binding.runtime_checkpoint_digest,
            ),
            model_native_limit=64,
            served_limit=64,
        )


def test_compile_rejects_expired_or_undersized_checkpoint() -> None:
    parent, current = generations()
    compiler = NativeContextTransitionCompiler(clock=lambda: 200.0)

    with pytest.raises(ContractError, match="expired"):
        compiler.compile(
            parent_packet=parent,
            current_packet=current,
            parent_checkpoint=checkpoint_for(parent, expires_at=199.0),
            model_native_limit=64,
            served_limit=64,
        )
    with pytest.raises(ContractError, match="stable prefix"):
        compiler.compile(
            parent_packet=parent,
            current_packet=current,
            parent_checkpoint=checkpoint_for(parent, tokens_saved=2),
            model_native_limit=64,
            served_limit=64,
        )


def test_compile_rejects_tampered_lineage_and_served_overflow() -> None:
    parent, current = generations()
    current.manifest.parent_manifest_id = strong("wrong").removeprefix("sha256:")
    current.manifest.content_digest = current.manifest.compute_content_digest()
    current.packet_content_digest = current.compute_content_digest()

    with pytest.raises(ContractError, match="parent manifest"):
        NativeContextTransitionCompiler().compile(
            parent_packet=parent,
            current_packet=current,
            parent_checkpoint=None,
            model_native_limit=64,
            served_limit=64,
        )

    parent, current = generations()
    plan = NativeContextTransitionCompiler().compile(
        parent_packet=parent,
        current_packet=current,
        parent_checkpoint=None,
        model_native_limit=64,
        served_limit=8,
    )
    assert plan.feasible is False
    assert plan.served_overflow_tokens == 8
    assert "current_generation_exceeds_served_limit" in plan.reason_codes


def test_controller_exposes_native_transition_only_in_active_mode() -> None:
    parent, current = generations()
    args = {
        "parent_packet": parent,
        "current_packet": current,
        "parent_checkpoint": checkpoint_for(parent),
        "model_native_limit": 64,
        "served_limit": 64,
    }

    with pytest.raises(ContractError, match="active_admission"):
        ContextController().compile_native_transition(**args)

    plan = ContextController(
        mode=ACTIVE_ADMISSION_MODE,
        clock=lambda: 200.0,
    ).compile_native_transition(**args)
    assert plan.stable_prefix_span_ids == ["system"]


def test_transition_cli_executes_directly_and_emits_no_payload(tmp_path: Path) -> None:
    parent, current = generations()
    source = tmp_path / "transition-input.json"
    artifact = tmp_path / "transition-plan.json"
    source.write_text(
        json.dumps(
            {
                "parent_packet": parent.to_dict(),
                "current_packet": current.to_dict(),
                "parent_checkpoint": checkpoint_for(parent).to_dict(),
                "model_native_limit": 262144,
                "served_limit": 65536,
                "observed_at": 200.0,
            }
        )
    )
    script = Path(__file__).parents[3] / "scripts" / "dcm_native_context_transition.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(source),
            "--artifact",
            str(artifact),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(artifact.read_text())
    assert report["pass"] is True
    assert report["stable_prefix_tokens"] == 4
    assert report["served_limit_shortfall"] == 196608
    serialized = json.dumps(report, sort_keys=True)
    assert "plan-v1" not in serialized
    assert "plan-v2" not in serialized
    assert "rendered_messages" not in serialized


def test_transition_cli_rejects_unknown_fields_without_leaking_input(tmp_path: Path) -> None:
    source = tmp_path / "transition-input.json"
    artifact = tmp_path / "transition-plan.json"
    source.write_text(json.dumps({"raw_prompt": "PRIVATE CONTEXT"}))
    script = Path(__file__).parents[3] / "scripts" / "dcm_native_context_transition.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--input", str(source), "--artifact", str(artifact)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    report = json.loads(artifact.read_text())
    assert report["pass"] is False
    assert report["error_type"] == "ContractError"
    assert "PRIVATE" not in json.dumps(report)
