# Read-lane promotion manifest

Date: 2026-04-15
Lane: project only (`projects/dpm`)
Status: rehearsal-ready manifest refreshed against current source snapshot
Promotion target: durable/main read-only seam only (not executed by this note)
Source state: local project-lane working tree snapshot; repository has no commits yet on `master`; working tree remains entirely uncommitted and review should treat this manifest as the deterministic source checklist

## Promotion intent
Promote only the current read-only DPM contract seam needed for a staging rehearsal.

This manifest is intentionally narrow:
- read-only / `active-read` contract surface only
- no durable plugin writes
- no `active-write` behavior
- no durable/main edits performed here
- no broad cleanup or redesign beyond the current packet

## Canonical seam being proposed
Frozen behavioral seam for rehearsal:
- rollout target remains `active-read`
- retrieval plus bounded overlay emission are allowed
- durable plugin writes remain forbidden
- explicit current-turn user instructions remain highest precedence
- retrieval precedence remains:
  1. explicit current-turn instruction
  2. thread-local continuity
  3. project-scoped continuity
  4. relationship memory
  5. weighted preference graph

## Promotion packet contents
### Required notes
- `notes/sprint-freeze-readonly-plugin-state.md`
- `notes/read-lane-promotion-runbook-note.md`
- `notes/read-lane-promotion-manifest.md`
- `notes/read-lane-rollback-note.md`
- `notes/next-step.md`
- `notes/packet-2-boundary-contract.md`

### Contracts and schemas
- `specs/dpm-plugin-contract.md`
- `specs/config/dpm-lifecycle-config-contract.md`
- `specs/config/dpm-config.schema.json`
- `specs/runtime-source-overlay-coherence-spec.md`
- `specs/project-continuity-source-contract.md`
- `specs/relationship-continuity-source-contract.md`
- `specs/replay-overlay-schema.md`
- `specs/preference-graph-schema.md`
- `specs/continuity-metadata-foundation.md`
- `specs/continuity-checkpoint-contract.md`
- `specs/dpm-plugin-blueprint.md`

### Runtime/script seam needed for rehearsal validation
- `scripts/continuity_recall.py`
- `scripts/run_continuity_validation.sh`
- `scripts/thread_safe_key.py`
- `scripts/dml_checkpoint_thread.sh`
- `scripts/find_thread_checkpoint.sh`
- `scripts/recall_thread_checkpoint.sh`

### Tests and fixtures needed to verify the seam after staging promotion
- `tests/smoke/test_layout.py`
- `tests/test_continuity_checkpoint_contract.py`
- `tests/unit/test_continuity_recall_no_leak.py`
- `tests/unit/test_dpm_config_contract.py`
- `tests/unit/test_dpm_plugin_validation_scaffold.py`
- `tests/unit/test_preference_graph_schema.py`
- `tests/unit/test_project_continuity_source_contract.py`
- `tests/unit/test_relationship_continuity_source_contract.py`
- `tests/unit/test_replay_overlay_schema.py`
- `tests/unit/test_runtime_coherence_regressions.py`
- `tests/fixtures/open_loops.fixture.json`
- `tests/fixtures/preference_graph.relationship.fixture.json`
- `tests/fixtures/preference_graph.thread_override.fixture.json`
- `tests/fixtures/project_source.boundedness_violation.fixture.json`
- `tests/fixtures/project_source.thread_compatible.fixture.json`
- `tests/fixtures/project_source.thread_conflict.fixture.json`
- `tests/fixtures/project_source.valid.fixture.json`
- `tests/fixtures/relationship_source.boundedness_violation.fixture.json`
- `tests/fixtures/relationship_source.thread_compatible.fixture.json`
- `tests/fixtures/relationship_source.thread_conflict.fixture.json`
- `tests/fixtures/relationship_source.valid.fixture.json`
- `tests/fixtures/replay_overlay.relationship.fixture.json`
- `tests/fixtures/replay_overlay.thread_override.fixture.json`
- `tests/fixtures/self_state.fixture.json`
- `tests/fixtures/thread_registry.fixture.json`

### Example configs for operator reference
- `examples/dpm/config.active-read.json`
- `examples/dpm/config.observe-only.json`
- `examples/dpm/config.disabled.json`

Excluded from rehearsal promotion packet:
- `examples/dpm/config.active-write.json` as an enabled target
- `runtime/`
- `out/`
- `scratch/`
- Python cache artifacts

## Validation evidence for this exact source snapshot
Executed in `projects/dpm` on 2026-04-15:
- `bash scripts/run_continuity_validation.sh`

Observed result:
- bundled contract/regression + layout smoke suite currently returns `46 passed, 1 failed in 0.31s`
- the bundled suite now includes the ingress preflight guard: `tests/unit/test_ingress_preflight_packet_guard.py`
- the current failure is `tests/test_continuity_checkpoint_contract.py::test_writer_finder_registry_and_recall_align_on_canonical_checkpoint_contract[...]`, where the `self-architecture-portrait` recall path returned `retrieval_status == "registry_only"` instead of the asserted `"hit"`
- standalone layout smoke was not rerun separately in this refresh because the bundled suite already exercised `tests/smoke/test_layout.py`

Snapshot hygiene note:
- ignored cache directories are present locally (`.pytest_cache/`, `scripts/__pycache__/`, `tests/**/__pycache__/`) but remain excluded by `.gitignore` and are not part of the promotion packet

## Rehearsal promotion gate
A staging rehearsal may proceed only if all of the following remain true at copy/promote time:
- `notes/sprint-freeze-readonly-plugin-state.md` still matches the source lane state
- the packet remains read-only and additive
- no active-write semantics are introduced during promotion
- the target lane receives the packet without interpretation or omission
- the same validation commands are rerun in the target lane and pass

## Operator handoff summary
Use this manifest as the exact file checklist for a narrow staging rehearsal of the read-only seam.
If any file in the packet is missing, changed in meaning, or supplemented with active-write behavior, stop and refresh the manifest before promotion.