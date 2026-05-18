# Read-lane freeze criteria and promotion runbook note

Date: 2026-04-15
Lane: project only (`projects/dpm`)
Status: note-only handoff refreshed against current validation snapshot

## Read-lane freeze criteria
The project lane is considered frozen for read-lane promotion review when all of the following stay true:
- The rollout target remains `active-read` only.
- The currently exercised runtime seam stays narrow: thread continuity reads are primary, compatible project continuity reads are optional refinements, and emitted replay overlay text stays bounded by the configured audit cap.
- Durable plugin writes remain forbidden and no runtime path in this lane claims write-safe behavior.
- Explicit current-turn user instruction precedence remains intact over plugin-derived guidance.
- Retrieval ordering stays aligned with the freeze marker and contract set: explicit current-turn instruction -> thread-local continuity -> project-scoped continuity -> relationship memory -> weighted preference graph.
- No touched artifact requires `active-write` runtime support or durable/main behavior changes.
- The local regression bundle and layout smoke check pass in `projects/dpm`.

## Promotion prerequisites
Before any separate promotion step is considered, confirm:
- `notes/sprint-freeze-readonly-plugin-state.md` still matches the lane state.
- The read-only contract/spec set remains the authoritative source for current behavior.
- Example config and tests still describe `active-read` as the safe rollout target.
- Promotion language stays narrow and truthful: active-read seam only, thread + compatible project reads only, bounded overlay only, no writes.
- Any durable `adapter.get_context(...)` drift in durable/main is explicitly quarantined from the rehearsal evidence surface and is not cited as proof of this project-lane packet.
- No durable/main files are modified as part of this note refresh.
- Promotion intent is narrow and additive, not a rewrite or active-write expansion.

## Evidence quarantine for rehearsal honesty
For this read-only promotion rehearsal, the evidence surface is restricted to the project-lane contract/spec set, the local validation bundle, and the note artifacts named in this runbook.

Exclude from rehearsal evidence:
- durable/main behavior claims derived from `adapter.get_context(...)`
- screenshots, logs, or notes that imply durable/main parity for the project-lane read seam
- any proof artifact that depends on undocumented durable/main retrieval behavior rather than the validated `projects/dpm` read-only contract surface

Interpretation rule:
- if `adapter.get_context(...)` in durable/main has drifted relative to this lane, treat that drift as an external quarantine item and a separate follow-up, not as confirming or disconfirming read-only promotion readiness for this packet

## Post-promotion validation commands
Run the same checks after promotion in the target lane:
- `bash scripts/run_continuity_validation.sh`
- optional confirmation recheck: `pytest tests/smoke/test_layout.py -q`

Current source-lane evidence before any promotion:
- the bundled validator now includes the ingress preflight guard (`tests/unit/test_ingress_preflight_packet_guard.py`)
- the latest bundled run in `projects/dpm` is not green: `46 passed, 1 failed in 0.31s`
- the current failing assertion is in `tests/test_continuity_checkpoint_contract.py` for the `self-architecture-portrait` recall path (`retrieval_status` returned `registry_only` instead of `hit`)

Expected result for a promotion-ready target lane:
- the bundled contract/regression + layout smoke suite passes
- the optional standalone layout smoke recheck passes if run
- note artifacts referenced by the smoke test remain present

## Not yet in scope
This note does not approve or define:
- promotion execution itself
- relationship-memory or preference-graph reads as active runtime seam behavior for this packet
- explicit-instruction seam wiring beyond the documented precedence contract
- `active-write` implementation or any write-safe runtime behavior
- durable/main design changes beyond a future narrow promotion
- broad documentation rewrites
- new retrieval precedence changes
