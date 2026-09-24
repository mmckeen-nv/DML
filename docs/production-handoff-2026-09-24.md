# M7 continuation — 2026-09-24

## Actual backing filesystem repair (r2)

[CI 333](artifacts/agent-episode-ci-333-summary-2026-09-24.json) finished **17/20 successful**
on `c18f022fe39ad5f9585cbed007501a3a28aed3f0` and established that both
candidate directories resolve to the same root ext4 block device with barriers
disabled. Selection correctly failed closed; both Linux recovery jobs rejected
the same mount condition. No live campaign was frozen or launched on that source.
Its strict model admission completed with zero generations and remains a
historical receipt.

The r2 source adds a shared helper restricted to disposable GitHub-hosted Linux
runners. Only an actual root ext4 block filesystem rejected solely for disabled
barriers may be remounted with `barrier=1`. Device/source/target identity, other
mount options and the unchanged admission predicate are checked before and
after; failed attempts retain evidence. Existing recovery/concurrency recorders,
scales, timing bounds and qualification guards remain unchanged. No loopback
filesystem or admission waiver is used.

Independent source grading accepted this repair at **9.6/10**, with all **34
focused fault controls passing and zero skips/errors/failures**. The original
434-source manifest expands to 436, including both new source files. Fresh
validation passed the exact **784-case/21-module model lane**, all **34 separate
filesystem controls**, both strict lint scopes and maintained mypy. All **436
source hashes remained unchanged**, with zero test skips/errors/failures. Actual runner repair/readback and successful new-source
CI remain required. The live campaign remains on hold pending fresh admission
and an independently reviewed freeze.

Raw CI logs stay local. Public evidence uses compact outcomes and hashes, as
requested; summaries are not substitutes for raw bytes in a replay.

## Earlier September 24 checkpoint

M7 remains **IN PROGRESS**. PR #118 remains open and unmerged. Milestones 1–6
remain closed; M8–11 remain open; M12–13 remain deferred. No new milestone or
serial source gate is introduced. DML remains alpha, `production_ready=false`.

Public source `3ba05702e256d3fc10837fd8e42b84e739b71a69` contains the independently
reviewed retrieval metadata change and passed 784 mandatory tests locally.
[CI 332](https://github.com/mmckeen-nv/DML/actions/runs/35761991252) completed
19/20 jobs successfully. Its mandatory CPU lane passed 784 cases with zero
skips; the Linux concurrency lane passed all 359 pytest cases but correctly
rejected its measured ext4 filesystem because it was mounted with `nobarrier`.
The guard and acceptance thresholds remain unchanged.

The [compact CI receipt](artifacts/agent-episode-ci-332-summary-2026-09-24.json)
retains source identity, counts, failure cause, artifact identities and hashes,
and verification limits. Per user instruction, raw CI logs and their archive
are not uploaded. The compact receipt does not replace raw bytes for replay.

The execution environment lost its model assets again. The existing Python
packages survived; restoring the interpreter launcher recovered the same 66
package versions and Python/SQLite runtime. All five model payloads were reacquired from the immutable published pins, and
offline preparation completed. Independent all-tensor verification is next. Prior receipts remain historical evidence;
new preparation, independent tensor review and zero-generation admission are
required for the new execution.

Attempt 7 has not been frozen or launched. The CI repair, independently graded **9.6/10** with 12 focused controls passing, selects an
existing filesystem through the unchanged admission predicate and fails closed
if none qualifies. The reviewed repair is integrated. Fresh validation passed **784 mandatory tests
with zero skips/errors/failures**, both strict lint scopes and maintained mypy,
with all 434 source hashes unchanged. Exact-source CI and independently verified live campaign evidence
are both required before M7 can close. The source, model and runtime remain
fixed from a reviewed freeze through the complete campaign and two replays.

[Previous handoff](production-handoff-2026-09-22.md) preserves the distinction
between attempt 5's interruption and attempt 6's observed completion/rejection
with currently unavailable raw evidence. Neither becomes qualifying evidence
through this continuation.
