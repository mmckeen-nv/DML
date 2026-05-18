# Read-lane rollback note

Date: 2026-04-14
Lane: project source `projects/dpm`
Status: rollback-oriented staging rehearsal note
Scope: rollback guidance for a narrow read-only seam promotion only

## Rollback intent
If the read-only DPM seam is promoted to a staging or durable target and the target no longer matches the frozen packet, rollback means restoring the target lane to its immediately previous state and re-establishing the pre-promotion read path.

This note does not authorize destructive history rewrite.
Prefer revert/restore or replacement from a known-good pre-promotion snapshot.

## Rollback triggers
Rollback the staged promotion if any of the following is observed:
- validation bundle fails in the target lane
- `tests/smoke/test_layout.py` fails because required note artifacts are absent or misplaced
- the target exposes or implies `active-write` behavior
- retrieval precedence differs from the frozen order
- explicit current-turn user instructions are no longer top precedence
- the promoted packet omits a manifest-listed contract, script, test, or note
- the target lane contains unintended runtime/output/cache artifacts mixed into the seam

## What must be restored
Restore the target lane to the exact pre-promotion state for the read-only seam, specifically:
- remove or revert the promoted DPM packet files introduced by the rehearsal if they are the source of divergence
- restore the previous target versions of any replaced files
- restore note presence/state so the lane matches the previous known-good baseline
- restore target behavior to pre-promotion retrieval/order semantics

## Preferred rollback method
Choose the first method compatible with the target lane's governance:
1. revert the narrow promotion commit or change-set as a single unit
2. restore the target lane from a pre-promotion backup/snapshot
3. replace the promoted files with the exact pre-promotion copies from the target lane record

Do not widen rollback into unrelated cleanup.
Do not introduce new content while rolling back.

## Minimum rollback checklist
Before rollback:
- identify the exact promotion change-set or copied file set
- identify the last known-good target state
- confirm rollback owner and target path
- capture failing evidence from the target validation run

Perform rollback:
- undo the promoted read-only packet as one narrow unit
- ensure no `active-write` or runtime-output artifacts remain from the rehearsal
- preserve operator notes and failure evidence outside the target code path if needed

After rollback, rerun:
- `bash scripts/run_continuity_validation.sh`
- `pytest tests/smoke/test_layout.py -q`

Rollback is considered successful when:
- the target lane again matches its pre-promotion state
- the validation bundle passes
- the layout smoke test passes
- no read-only seam note or contract remains in a partially promoted state

## Failure taxonomy for rollback rehearsal
- `GATE_FAILED`: target validation or smoke tests fail after promotion attempt; contain by reverting the narrow promotion unit
- `ARTIFACT_MISSING`: a required manifest file is absent or mismatched; contain by restoring the prior target snapshot and refreshing the manifest
- `ROLLBACK_UNVERIFIED`: rollback performed but post-rollback validation not rerun or not passing; contain by blocking further promotion until verification is complete
- `ENVIRONMENT`: target Python/tooling/path issue prevents validation; contain by fixing the environment before judging the packet itself
- `PERMISSION_BOUNDARY`: target lane cannot be reverted/restored under current permissions; contain by pausing and handing rollback execution to the authorized operator

## Operator note
Because the source repository currently has no commits yet on `master`, this note assumes rollback in the target lane is anchored to that lane's own pre-promotion snapshot or revertable change-set, not to source-lane git history.

Use this note together with:
- `notes/read-lane-promotion-manifest.md`
- `notes/read-lane-promotion-runbook-note.md`
- `notes/sprint-freeze-readonly-plugin-state.md`

If the packet changes, refresh this rollback note before any staging rehearsal.