# Promotion-readiness stopline

Date: 2026-04-15
Lane: project only (`projects/dpm`)
Status: go/no-go boundary for future narrow read-lane promotion work

## Green now
- The project lane is green for a narrow read-lane promotion review packet only.
- The validated safe target remains `active-read`.
- The currently exercised seam is still narrow and truthful: thread continuity reads are primary, compatible project continuity reads are optional refinements, and replay overlay output is bounded by configured audit caps.
- Durable plugin writes remain forbidden in the current lane state.
- Explicit current-turn user instruction precedence remains the top contract constraint.
- Retrieval/order contract remains frozen as: explicit current-turn instruction -> thread-local continuity -> project-scoped continuity -> relationship memory -> weighted preference graph.
- Current validation evidence in this lane is green:
  - `bash scripts/run_continuity_validation.sh` -> `45 passed`
  - `pytest tests/smoke/test_layout.py -q` -> `1 passed` (standalone confirmation recheck)

## Frozen for promotion review
- The authoritative freeze marker remains `notes/sprint-freeze-readonly-plugin-state.md`.
- Promotion review must stay additive and packetized around the existing read-only seam only.
- The promotion packet is the existing manifest-defined set in `notes/read-lane-promotion-manifest.md`.
- Promotion language must stay exact: `active-read` only, read-only seam only, no reinterpretation, no omitted packet files.
- No durable/main edits are authorized by this note itself.
- If any packet file changes meaning, or if runtime behavior expands beyond the frozen seam, stop and refresh the freeze marker and manifest before any promotion step.

## Definitely not happening yet
- No promotion execution is approved by this note alone.
- No `active-write` implementation, enablement, or write-safe runtime claim.
- No durable plugin writes.
- No claim that relationship-memory reads are implemented in the active runtime seam.
- No claim that preference-graph reads are implemented in the active runtime seam.
- No claim that explicit-instruction seam wiring inside `override_state` is implemented in the active runtime seam.
- No broad durable/main redesign, cleanup wave, or documentation rewrite.
- No retrieval precedence changes beyond the frozen contract target.

## Stopline rule
Proceed only if the lane still matches the freeze marker, the promotion packet remains narrow/read-only, and the documented validation commands are rerun and pass in the target lane.

If any of those conditions fail, the boundary is red: do not promote; refresh docs/contracts/tests first.
