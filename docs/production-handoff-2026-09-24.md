# M7 continuation — 2026-09-24

## Current attempt 9 after execution connection loss

M7 remains **IN PROGRESS**. Attempt 8 lost its execution connection after five
retained task reports: four successes and one `tool_error`. The model requested
self-supersession (`r1` as both source and replacement), which the existing
service rejects. The partial reports contain **9,659 known tokens**; final costs,
effects and outcomes of the full nine-task invocation remain unknown. No final
campaign or execution receipt appeared after the next task's declared deadline.
No missing campaign, replay, completion review or termination proof is fabricated.

A separately identified attempt 9 uses an isolated checkout. The original
attempt-8 checkout stays at `35b5cf2d5400705ee57b9b0685dc55b97590a960`, with its
source, runtime inputs and work paths preserved for any late evidence. The new
candidate only clarifies the public `supersede` description: source and replacement
must be distinct records with observed references; obtain the missing record
before calling. The service already enforces this precondition. No parser,
schema, model weights, task truth, policy, limits or live gates change.

The independent source review accepted that clarification at **9.6/10**. Fresh
803-case/21-module validation plus 34 filesystem controls, current strict model
admission, published-source freeze and pre-generation review are required for
the new full nine-task run. Two sequential replays, independent completion review
and passing exact-source CI remain mandatory for closure. The next run cannot
repair historical retention gaps. Raw logs stay local; published records remain
compact. Earlier checkpoints below are historical.

## Current recovery and attempt 8

M7 remains **IN PROGRESS**. Attempt 7 completed nine tasks on published source
`a6f54953241ea644da1770dd86dc86ec6b7fda8f`, with eight verifier successes. Both
sequential replays matched and rejected the required model-owned supersession:
the model retrieved the current preference and answered without executing the
requested lifecycle operation. The independent grader released that source freeze.

After another workspace rollback, the attempt-7 raw campaign, replays and local
reviews are unavailable. Their previously observed results and hashes are
historical evidence only; no missing raw events will be reconstructed or treated
as freshly verified. Attempt 8 uses a distinct identity and fresh evidence.

The pending semantic candidate changes the public descriptions of `retrieve` and
`supersede`. It clarifies retrieval cardinality, observed write references and
the difference between reading a record and committing a lifecycle change.
Task truth, parser, policy, learned model weights, limits, verifier and acceptance
gates remain unchanged. The revised public descriptions produce a newly declared
action-runtime identity. Fresh source validation, model admission, independent pre-generation
review, the full nine-task campaign, two sequential replays and passing exact-source
CI remain required before closure. Earlier checkpoints below are historical.

The restored execution environment enforces an 8 GiB memory limit. The original
model preparation was killed by the memory limit before publication; that failed
preparation is retained and is not a model admission or a live attempt. The
canonical preparer now converts and serializes tensors in bounded
chunks. This resource repair retains the original immutable source pins,
exact finite BF16-to-F32 conversion and round-trip checks, complete tensor coverage,
serialized readback verification and exclusive publication. Fresh validation and
independent all-tensor comparison are required for its new source hash.
Fresh validation passed **803 mandatory tests across 21 modules** (the original
784 plus 19 serialization, corruption and memory-bound controls), **34 separate
filesystem controls**, both strict lint scopes and maintained mypy. All 436 source
hashes remained unchanged. These results establish source validation, not live
qualification; actual bounded preparation and model admission are next.

CI 334 passed its three filesystem qualification lanes. Its aggregate result was
19 original successful jobs plus one successful, independently reviewed diagnostic
retry of the production stress job. The original lock-acquisition timeout remains
a recorded failure and repeatability limitation; the retry does not establish a
fairness fix. Candidate 8 requires new-source CI.

Only compact outcomes, source changes and necessary review records are published.
Raw logs and campaign events are not uploaded. PR #118 remains open and unmerged.

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
