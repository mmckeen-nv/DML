# M7 continuation — updated 2026-09-25

## Current work: attempt 11 guidance candidate

M7 remains **IN PROGRESS**. Attempt 10 completed all nine tasks on published
source `443f444da6d1daa529b78c467cbe7652d40d4f1f`, with eight verifier successes,
nine measurable finals across eight intents, **19 model calls and 18,918 known
tokens**, and zero unknown usage or effects. The trusted rejection worked, but
the model then finalized without the required supersession or a supporting
citation. Both completed sequential replays matched and rejected only
`verified_model_owned_supersession`. The
[completion review](artifacts/agent-episode-live-completion-grader-review-attempt-10-isolated-2026-09-25.json)
retains the earlier failed reviewer-launcher path setup separately from the two
completed replays, rejects M7 closure, and releases only attempt 10's freeze.

[CI 337](https://github.com/mmckeen-nv/DML/actions/runs/36080001253) passed all
**20 jobs on the first attempt** for that exact source. Its
[independent review](artifacts/agent-episode-ci-337-grader-review-2026-09-25.json)
accepted **9.6/10**, including the exact 891 model case IDs with zero skips,
filesystem qualification and production stress. This does not override the
failed live gate. The original attempt-8 source and inputs remain frozen;
attempts 5–7 retain their disclosed evidence gaps.

Attempt 11 is implemented in a separate checkout. It tests one fixed,
generic recovery-guidance paragraph under explicit profile
`qwen2-action-json-recovery-v3`. It explains that a rejected operation does not
complete requested work, earlier successful observations retain their meaning,
and the model must choose its next valid action within the existing limits.
There is no demonstrated wiring defect: the previous model received the complete
history and error. This is an uncertain policy experiment, with no predicted
improvement and no further wording iterations if it fails.

Old v1 and attempt-10 v2 policy/runtime identities remain exact. The new profile
inherits the same validated pre-dispatch mechanics and artifact schemas; grammar,
public tool descriptions, task truth, verifier, learned weights, limits and live
gates remain unchanged. All **925 existing cases** remain, with **50 new guidance controls** separately
enumerated. [Full validation](artifacts/agent-episode-attempt-11-isolated-validation-2026-09-25.json)
passed **975 cases with zero skips, failures or errors**, both strict lint scopes
and maintained mypy. All **436 source hashes** remained unchanged. The independent
[integrated review](artifacts/agent-episode-attempt-11-integrated-grader-review-2026-09-25.json) accepted the candidate at **9.6/10**. Exact-source CI, zero-generation
admission, a reviewed freeze, one complete
nine-task campaign and two sequential replays remain required. PR #118 stays
open and unmerged; M1–6 are closed, M7–11 open and M12–13 deferred.
`production_ready=false`. Raw logs and events stay local; published records are
compact. Earlier checkpoints below are historical.

## Historical validated attempt 10 source

M7 remains **IN PROGRESS**. Attempt 9 completed all nine tasks on published source
`2a2fb0891d14213c5a593157aab64a1d5ea41834`: eight successes and one `tool_error`,
with **17,525 known tokens** and one task with unknown effects. The model received
the distinct-record contract but still requested self-supersession. Both actual,
sequential evidence replays agree: intent coverage, verified supersession and
complete effects accounting fail. The
[independent completion review](artifacts/agent-episode-live-completion-grader-review-attempt-9-isolated-2026-09-24.json)
rejects qualification and releases only attempt 9's completed freeze.
[CI 336](https://github.com/mmckeen-nv/DML/actions/runs/36073124885) passed all
**20 jobs on the first attempt** for that exact source; CI does not override the
failed live gates. All nine reports and both replay records remain retained locally.

The new candidate is implemented in a separate checkout. It adds an
explicitly versioned capability to return a trusted, provably pre-dispatch
validation rejection for self-supersession to the model. The original proposal and fixed error remain
in the evidence; any correction must be a new model decision charged to the same
six-step, token and 300-second budgets. Existing execution failures remain
terminal and conservative. The old protocol retains its behavior. A separate
resource change removes unnecessary whole-tensor byte copies from fingerprint
hashing while preserving every fresh integrity scan and exact digest. Its
[independent review](artifacts/agent-episode-fingerprint-allocation-grader-review-2026-09-25.json)
accepted the allocation scope at **9.6/10**, with **132 focused passes** including
22 new controls. A 16 MiB contiguous fixture used 569,267 bytes of peak Python
allocation versus 16,845,541 for the old algorithm, with identical digests.
This does not establish total process memory or inference-speed improvement.

The explicit profile `qwen2-action-json-validation-v2` binds execution protocol
`dml-agent-predispatch-validation-v2` into the actual runtime identity. Both
supersession references must have appeared as immutable full records in prior
model-visible results. Only a same-record rejection is recoverable; private or
missing references remain terminal. Replay derives that boundary independently
and binds successful mutation receipts to the model's exact request and call key.

[Fresh integrated validation](artifacts/agent-episode-attempt-10-isolated-validation-2026-09-25-r2.json)
passed **925 cases with zero skips, failures or errors**: all 803 original model
cases, 34 filesystem controls, 66 new protocol controls and 22 fingerprint controls.
Both strict lint scopes and maintained mypy passed; all **436 source hashes**
remained unchanged. The initial validation's missing ledger type annotation is
retained as a rejected source-validation record; the one-line correction changes
no runtime behavior. The independent source review accepted the corrected scope
at **9.6/10**; the
[integrated review](artifacts/agent-episode-attempt-10-integrated-grader-review-2026-09-25.json)
binds the final source, scope and actual validation evidence.

Fresh published-source CI, strict zero-generation admission, pre-generation
review, a full nine-task run and two sequential replays are still required.
No attempt-10 model outcome or inference-speed improvement is claimed. Task truth,
action grammar, public tool descriptions, agent policy, learned weights and live
acceptance gates remain unchanged.

The original attempt-8 source and inputs remain frozen; its full outcome is still
unknown. Attempts 5–7 retain their disclosed historical evidence gaps. PR #118
remains open and unmerged. Milestones 1–6 are closed, M7–11 remain open and M12–13
are deferred; the count remains 20 serial source gates plus foundations.
`production_ready=false`. Only compact records and source changes are published;
raw logs and campaign events stay local. Earlier checkpoints below are historical.

## Historical attempt 9 preparation after execution connection loss

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

## Historical recovery and attempt 8

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
