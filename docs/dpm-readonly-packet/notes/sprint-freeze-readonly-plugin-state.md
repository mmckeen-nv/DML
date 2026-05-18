# Sprint Freeze: read-only plugin contract state

Date: 2026-04-15
Lane: project only (`projects/dpm`)
Status: handoff-ready snapshot refreshed against current uncommitted source tree

## Frozen state
- DPM remains a post-foundation additive plugin layer; the continuity metadata foundation stays authoritative for identity and retrieval-safe scoping.
- The canonical plugin contract is `specs/dpm-plugin-contract.md`.
- The canonical lifecycle/config contract is `specs/config/dpm-lifecycle-config-contract.md` with schema `specs/config/dpm-config.schema.json`.
- The current safe rollout target is `active-read`: the runtime seam may read thread continuity plus compatible project continuity and emit a bounded replay overlay, while durable plugin writes remain forbidden.
- Explicit current-turn user instructions remain the top-precedence constraint and must suppress conflicting plugin guidance.
- Retrieval precedence is frozen as a contract target: explicit current-turn instruction -> thread-local continuity -> project-scoped continuity -> relationship memory -> weighted preference graph.
- In the current runtime seam implementation for this lane, the exercised read path is narrower than the full contract target: thread reads are primary, project reads are included only after a compatibility check passes, and the emitted overlay stays bounded by configured audit caps.
- Relationship-memory reads, preference-graph reads, explicit-instruction seam wiring inside `override_state`, and any write-capable seam behavior are not yet implemented in the active runtime seam for this packet.
- Runtime coherence work is documented but still contract-first; no durable/main promotion or product behavior change is part of this lane snapshot.

## Touched artifacts in this lane
Contracts/specs already defining the current read-only state:
- `specs/dpm-plugin-contract.md`
- `specs/config/dpm-lifecycle-config-contract.md`
- `specs/config/dpm-config.schema.json`
- `specs/runtime-source-overlay-coherence-spec.md`
- `specs/project-continuity-source-contract.md`
- `specs/relationship-continuity-source-contract.md`
- `specs/replay-overlay-schema.md`

Implementation/tests exercising the current contract surface:
- `scripts/continuity_recall.py`
- `scripts/run_continuity_validation.sh`
- `tests/unit/test_ingress_preflight_packet_guard.py`
- `tests/unit/test_dpm_plugin_validation_scaffold.py`
- `tests/unit/test_dpm_config_contract.py`
- `tests/unit/test_project_continuity_source_contract.py`
- `tests/unit/test_continuity_recall_no_leak.py`
- `tests/unit/test_runtime_coherence_regressions.py`
- `tests/unit/test_replay_overlay_schema.py`
- `tests/unit/test_preference_graph_schema.py`
- `tests/smoke/test_layout.py`

## Validation snapshot
- `scripts/run_continuity_validation.sh` is the nearest bundled regression path for the frozen read-only contract surface and now includes `tests/unit/test_ingress_preflight_packet_guard.py`.
- Current bundled evidence for this source snapshot is mixed rather than all-green: `46 passed, 1 failed in 0.31s`.
- The observed failure is in `tests/test_continuity_checkpoint_contract.py`, where the `self-architecture-portrait` case returned `retrieval_status == "registry_only"` instead of the asserted `"hit"`.
- `tests/smoke/test_layout.py` remains the nearest lane-layout smoke path and still ran inside the bundled suite.

## Notes
- This note is a factual freeze marker for the project lane only.
- No write-safe plugin contract is accepted here beyond the documented future `active-write` mode semantics.
- No durable/main files were touched for this packet.
