# Milestone 7 live qualification record

Current continuation: [September 24 handoff](production-handoff-2026-09-24.md). M7 remains open after attempt 7 failed required model-owned supersession. Attempt 8 is being prepared with fresh evidence after workspace rollback. Earlier dated status below is historical.

## Recovery after the execution-environment disconnect — 2026-09-22

M7 remains **IN PROGRESS**. Public Coder-recovery source `1c4ccd5` passed
[CI 331](https://github.com/mmckeen-nv/DML/actions/runs/35697661942), all 20 jobs.
The current workspace can access the public repository and the separate reviewed
retrieval-metadata checkout, but the previous main workspace is absent.

Before the disconnect, attempt 6 was observed to finish all nine tasks with 8/9
verifier successes, nine measurable finals across eight intents and 15,534 known
tokens. Both sequential replays matched and the independent completion review
rejected M7 solely because the model never performed the required supersession.
Those are retained observations in the [PR recovery handoff](https://github.com/mmckeen-nv/DML/pull/118),
not a replacement for raw evidence. The campaign, both replay artifacts, trained
provenance/admission artifacts, integrated validation receipt and unpublished
27-file commit are currently unavailable. Do not reconstruct them from prose,
claim their bytes remain verified locally, or reclassify attempt 6 as qualifying.
Attempt 5 remains separately interrupted/unreplayable with unknown full costs.

The unavailable local commit was `97df9a569baf0c4c1a81370f91094a13b777946f`, tree
`97d7aed92e0cea92e467404bcf173069a58039ba`. This recovery does not recreate or claim
that exact tree. The surviving three-file metadata change and its original
[independent 9.6/10 review](artifacts/agent-episode-retrieval-metadata-review-2026-09-22.json)
remain byte-matched. Its 234-case independent JUnit also survives with zero skips,
failures or errors. A fresh clone differs from public source only in those three
reviewed source files before recovery documentation is added.

The change exposes only requested retrieval cap, returned count and equality to
the cap. It asserts no hidden-record total or completeness guarantee and does not
force model actions, add reads, change task truth or relax acceptance gates.
Fresh [integrated validation](artifacts/agent-episode-metadata-recovery-validation-2026-09-22.json)
passed **784 mandatory cases with zero skips/errors/failures**, all 21 modules,
both strict lint scopes and maintained mypy, with 434 unchanged source hashes.
[Independent supervisor review](artifacts/agent-episode-metadata-recovery-supervisor-review-2026-09-22.json)
accepts this recovered source for publication at **9.6/10**, with no source blockers.
Published-source CI, recreated pinned model assets, strict admission and
independent pre-generation review remain required before a new campaign can qualify. Attempt 7 has not been frozen or launched. No attempt ID or
successful result will be substituted for missing historical evidence.

Milestones 1–6 remain closed; M7 is active; M8–11 remain open; M12–13 remain
deferred. This continues serial source gate 20. PR #118 stays unmerged and DML
remains alpha with `production_ready=false`.

## Earlier evidence chronology and status

Earlier recovery status, retained historically: **2026-09-22: IN PROGRESS**. Four predeclared
trained campaigns completed and failed their unchanged live gates. Attempt 5
started on September 20, but its temporary model and raw output paths did not
survive the frozen session. Its final result, total cost, effects and completed-task
count are unavailable. It is **interrupted/unreplayable and unqualified**, not a
completed fifth campaign. Contemporaneous PR observations remain attributed to
their original update; they do not substitute for raw evidence.

Reviewed source `0d4364d116d75ed24b60e441845518b47f39ad57` passed
[CI 330](https://github.com/mmckeen-nv/DML/actions/runs/35518344343) **20/20 jobs**,
with **774 mandatory CPU passes / zero skips**. The
[session recovery review](artifacts/agent-episode-session-recovery-review-2026-09-22.json)
records the missing evidence and limited process visibility. It releases the
operational hold only for new, separately identified work; it does not assert that
attempt 5 finished or passed replay. Historical declarations and reviews remain
unchanged. The separately pinned Coder preparer has been reconstructed with fresh
[independent source acceptance at 9.5/10](artifacts/agent-episode-coder-reconstruction-review-2026-09-22.json)
and [780 mandatory CPU passes / zero skips](artifacts/agent-episode-coder-recovery-validation-2026-09-22.json).
All 434 recorded source hashes stayed unchanged; strict lint and mypy passed.
Actual model preparation/admission and a fresh pre-generation declaration remain
required. No new live qualification is claimed.

[PR #118](https://github.com/mmckeen-nv/DML/pull/118) remains open and unmerged.

This continues **serial source gate 20** within release milestone 7. It does not
create a twenty-first source gate. The [finite release plan](production-remaining-work-2026-09-18.md)
retains **six of eleven first-release milestones closed, five open and two broader
milestones deferred**. Milestone 5 remains closed by reconciliation. DML remains
alpha, the supported profile remains candidate and `production_ready` is false.

## Evidence chronology

| Evidence | Scope and result |
| --- | --- |
| [Initial source review](artifacts/agent-episode-review-2026-09-20.json) | 9.2/10 rejected; repairs accepted at **9.6/10**. The [initial validation manifest](artifacts/agent-episode-local-validation-2026-09-20.json) records 529 focused passes / zero skips, 4,667 full-suite passes / 24 declared skips and 417 unchanged source hashes. The selections overlap. |
| [CI 326](https://github.com/mmckeen-nv/DML/actions/runs/35500751139) on `f29befb` | 18 jobs passed, one Windows path-length failure and one cancellation. Failed/cancelled jobs are retained and do not qualify source. |
| [Windows repair review](artifacts/agent-episode-windows-review-2026-09-20.json) and [CI 327](https://github.com/mmckeen-nv/DML/actions/runs/35502295087) | Test-ID repair accepted at **9.8/10**. Source `6db5a837eb7377e08356a76a2d4f226c3e9b26fd` passed **20/20 jobs**, with **529 mandatory CPU passes / zero skips**. All 354 manifest source hashes independently matched published tree `caca07eac5cf867e65ce145dbb62c511dd5e1262`; tested merge `ea3361a638266ff6a1587f57149b4bb8de5b5b7a` has that tree. |
| [Completion validation](artifacts/agent-episode-completion-validation-2026-09-20.json) | Completion source accepted at **9.6/10**; **698 mandatory CPU cases passed with no failures or skips**, with **428 unchanged source hashes**. Maintained Ruff and mypy over 76 files passed. Earlier 4,829-pass / 24-skip regression changed source during execution and is retained as **interim**, not final-source qualification. |
| [Initial live declaration](artifacts/agent-episode-live-spec-2026-09-20.json) | Rejected before any corpus generation because duplicate installed-distribution metadata produced an inconsistent dependency inventory. No live result is attributed to this declaration. |
| [Corrected live declaration](artifacts/agent-episode-live-spec-r2-2026-09-20.json) | Frozen before corpus generation, with source, model, runtime, ordered tasks, bounds and acceptance criteria. Installed versions resolve to selected distributions and critical actual imports are checked. |
| [First actual corpus campaign](artifacts/agent-episode-live-campaign-attempt-1-2026-09-20.json) | Nine attempted tasks, nine `invalid_action` terminals, zero completed tasks or measurable intents. Thirteen actual model calls and four successful retrieval calls consumed 3,934 input + 477 output = 4,411 tokens. Both independent replays rejected coverage, supersession and feedback gates. All raw results and failures are retained. |
| [General protocol clarification](artifacts/agent-episode-protocol-validation-2026-09-20.json) | Accepted at **9.7/10** with **710 mandatory CPU cases passing / zero failures or skips**, all 428 source hashes unchanged. Only the model-facing `AGENT_POLICY` and twelve strict regression cases changed; the parsed source outside the policy is identical. The second campaign uses its separately frozen declaration, pinned to source `a8a50a46945592a39a5b56b82957a15537e8483b`. |
| [Second actual corpus campaign](artifacts/agent-episode-live-campaign-attempt-2-2026-09-20.json) | Nine attempts, 18 model calls, nine retrieves; seven measurable finals across six intents, four verifier successes, two invalid actions and three typed-value semantic failures. Total 10,688 tokens, including every failure. [Independent replay](artifacts/agent-episode-live-evidence-attempt-2-2026-09-20.json) rejected missing near-duplicate coverage and model-owned supersession. |
| [Native-content renderer v2 review](artifacts/agent-episode-native-review-2026-09-20.json) and [validation](artifacts/agent-episode-native-validation-2026-09-20.json) | Generic native ChatML content plus indexed full-field metadata, reversible control-marker escaping and exact template identity; independent **9.6/10**, **715 mandatory CPU passes / zero skips**, **429 unchanged source hashes**. Fresh bundle prepared and admitted with zero new generations before its separate third campaign freeze. Prior campaigns retain their original JSON framing. |
| [Third live declaration](artifacts/agent-episode-live-spec-attempt-3-2026-09-20.json) | Independently reviewed before generation; all nine tasks completed on source `2915b65`: two measurable finals/two intents, one task success, six invalid actions and one timeout. Coverage remains incomplete; independent final completion review **7.0/10: rejected**. Corpus, trained weights, bounds and acceptance gates unchanged. |
| [Final published completion-source CI 328](https://github.com/mmckeen-nv/DML/actions/runs/35514202959) | **20/20 latest job results successful** after one same-source production-lane retry. Original **19 successful / one failed** retained: 256-client 30-second ownership timeout, cause undetermined; retry test passed in 22.25 seconds. CPU evidence unchanged: **715/0**, 18 modules and **366 matching source hashes**. |

The corrected declaration's SHA-256 is
`923ea74d37b9604825677f78ce9bb71e7caa541e4bb87dc43087397a451b08e5`.
Its rejected predecessor has SHA-256
`f2582f0ba270cc97d83a40e5cf99500a593bc5773cc7535a1608f0752fe7387f`.
These are separate immutable artifacts. No threshold or source change after
generation may be silently folded into the original declaration.

## Rejected live attempt 1

The first trained execution completed all **nine attempts** under the corrected
frozen declaration. Generated fences, missing final envelopes and string record
handles where the final grammar requires integer citation IDs caused strict
parser rejection. Retrieval alone did not satisfy the frozen measurable-answer
gate. There were **13 model calls, four successful retrievals, nine invalid-action
terminals, zero completed tasks and zero measurably covered intents**. Both
independent evidence replays rejected the campaign; required supersession and
actual-answer feedback were not established.

The campaign consumed **3,934 input tokens and 477 output tokens: 4,411 total**.
No usage or effects were unknown. Cost per completed task is undefined because
there were no successes; contradiction, false-memory and repeated-error rates
are unavailable because there were no measurable final answers. These are
retained failed outcomes, not zero semantic error rates.

| Retained artifact | SHA-256 |
| --- | --- |
| [Raw campaign](artifacts/agent-episode-live-campaign-attempt-1-2026-09-20.json) | `17d8df590d3c457ff697f29912921d362ac3d807854b1125dcd5129ee53ed626` |
| [Evidence replay](artifacts/agent-episode-live-evidence-attempt-1-2026-09-20.json) | `e05d51185bc438d2e3d40b1051381b0c5105d72639457c3ca5de79b2c7d60a37` |
| [Execution record](artifacts/agent-episode-live-execution-attempt-1-2026-09-20.json) | `c1d2fcfeb0ca3fe7a2279f82d2b1bb3950aa9d410350941871fcc89dff61ccc6` |

The correction is confined to general protocol instructions and twelve strict
regression cases. Independent review accepted it at **9.7/10**; the mandatory lane
passed **710 cases with zero failures or skips**, with **428 unchanged source
hashes**. The [policy review](artifacts/agent-episode-protocol-review-2026-09-20.json)
retains that separate acceptance. The
[second pre-generation declaration](artifacts/agent-episode-live-spec-attempt-2-2026-09-20.json)
pins source `a8a50a46945592a39a5b56b82957a15537e8483b`, with SHA-256
`5c7f01d7d94fad5c04228f2508006f491404baf4d31377d803301a64c8f648e2`.
Its acceptance gates, trained model and bounds match attempt 1. This declaration
was frozen before attempt 2 generation; its complete rejected result is recorded
below. The action parser, task truth, model, limits and acceptance gates were not
relaxed. Attempt 1 and its failed
cost remain part of the record; a later attempt cannot overwrite or silently
retry it. Milestone 7 remains open.

## Rejected live attempt 2

The second frozen campaign retained all **nine attempts**, with **18 actual model
calls and nine successful retrieves**. It produced **seven schema-valid,
measurable finals across six of eight intents**, of which **four passed their
independent verifiers**. Two attempts returned invalid actions, and three
schema-valid answers failed exact typed-value verification. These three answers
are semantic failures even though their terminal protocol status is `completed`.
The required supersession did not mutate authority, and near-duplicate coverage
had no valid final. The independent replay therefore rejected qualification.

Measured cost was **9,601 input + 1,087 output = 10,688 tokens**, including failed
attempts, or **2,672 tokens per independently successful task**. No usage or
effects were unknown. The feedback task measured an actual repeated error in its
one observed opportunity (**1/1**); the aggregate repeated-error rate remains
null because eight tasks have unavailable coverage. Aggregate contradiction and
false-memory rates likewise remain null, with measured partial numerator and
denominator retained. They are not silently reduced to zero.

| Retained artifact | SHA-256 |
| --- | --- |
| [Raw campaign](artifacts/agent-episode-live-campaign-attempt-2-2026-09-20.json) | `7d28b0ef74173d1593730cf55a17fb41e056d061befd32e680100cea951b9d50` |
| [Independent replay](artifacts/agent-episode-live-evidence-attempt-2-2026-09-20.json) | `fbd363fd1fb960cda60eeb1f194cae288b1e504510d82680776f2867c6fbe485` |
| [Execution record](artifacts/agent-episode-live-execution-attempt-2-2026-09-20.json) | `a21fd8937141abd3904496cc3f59903b766f6317ee7a4366a63a0812724699cd` |
| [Independent attempt review](artifacts/agent-episode-live-attempt-2-review-2026-09-20.json) | `81a7f2d703622b8a33ea2b02fb8038f59f5319cd8b557d667994e3f13ef6f4b0` |

The generic native ChatML renderer correction preserves complete indexed
metadata, reversible control-marker escaping and strict template identity. It
received independent **9.6/10** source acceptance after **715 mandatory CPU passes
with zero skips**, with **429 unchanged source hashes**. The
[Qwen contract](qwen-model-input-v1.md#complete-framing-and-output) records its v2
format and template digest. The matching fresh bundle and third declaration are
recorded below; attempts 1 and 2 retain their historical full-message JSON framing. The two failed campaign artifacts and their acceptance
criteria remain unchanged. Four task successes do not close milestone 7.

## Completed live attempt 3: qualification rejected

Reviewed native-renderer source is published as
[`2915b65290315e4eead5ee05bbb5179e31cfc189`](https://github.com/mmckeen-nv/DML/commit/2915b65290315e4eead5ee05bbb5179e31cfc189), tree
`4846330e9ea97ba156467d973fda90c488857b1b`. The
[third declaration](artifacts/agent-episode-live-spec-attempt-3-2026-09-20.json)
was frozen and independently reviewed before generation, with SHA-256
`aa8aa676f8b2c5e42a8432f16e9b55bc26fbb3716a3940bd197870752266439e`.
It retains the same trained learned weights, configuration, tokenizer, corpus,
ordered tasks, limits and acceptance gates. The separately identified native v2
template and exact source revision are the declared changes.

The actual campaign retained all **nine attempts**. It recorded **16 completed
model calls and eight retrievals**, with **two measurable final answers across
two intents and one independent task success**. Six terminals are invalid actions,
two are protocol-completed (one semantic failure) and one is a timeout. Coverage,
required model-owned supersession and actual-answer feedback remain unmet; independent completion review is **7.0/10: rejected**.

The timeout leaves **one task with unknown usage and effects** and one incomplete
input/output call. Exact input, output and total token cost therefore remain
**null**, as does exact cost per successful task. Recorded lower bounds are
**9,600 input + 592 output = 10,192 known tokens**. Those are lower bounds, not a
claimed total. Aggregate semantic rates and TTFT remain unavailable.
The [raw campaign](artifacts/agent-episode-live-campaign-attempt-3-2026-09-20.json)
has SHA-256
`f9229c20615526225d7ca7dac5438120f225eb251b970f730a621b3c1e899fe5`.
The [execution record](artifacts/agent-episode-live-execution-attempt-3-2026-09-20.json)
and [evidence replay](artifacts/agent-episode-live-evidence-attempt-3-2026-09-20.json)
retain the incomplete operation and failed gates.

The [independent attempt-3 review](artifacts/agent-episode-live-attempt-3-review-2026-09-20.json)
rejects milestone completion at **7.0/10**, below the required 9.5/10. Its SHA-256
is `e42738000ff3011a56ed5d119a7434694106996dfc710ec37c832009898ecbbc`.
This grade applies to observed live completion, not the separately accepted
9.6/10 source-correctness review. The next candidate is the explicit
[agent-action JSON profile](qwen-agent-action-v1.md), with syntax-only constrained
decoding and a separately declared shared-policy clarification. Source review,
validation and a new frozen campaign are required; no improvement is attributed
to grammar alone or inferred before execution.

## Rejected live attempt 4 and source follow-up

The [action profile](qwen-agent-action-v1.md) applies public syntax constraints
and clarifies the shared harness policy. Its
[source review](artifacts/agent-episode-action-review-2026-09-20.json) accepts
**9.6/10** with [760 mandatory passes / zero skips](artifacts/agent-episode-action-validation-2026-09-20.json),
**433 unchanged source hashes**, strict Ruff and maintained mypy over **78 files**.
The [admission record](artifacts/agent-episode-qwen-action-admission-2026-09-20.json)
contains zero generations.

The [fourth pre-generation declaration](artifacts/agent-episode-live-spec-attempt-4-2026-09-20.json)
has SHA-256
`5b4e70f02bf3d45e80dd2d714f7fdb63e6d569748769c666b11f7ded47ac016d`.
It records base `2915b65` plus exact reviewed source hashes; that source was
subsequently published as
[`da526eb58d7556c455b2b534f9261cc68f74becf`](https://github.com/mmckeen-nv/DML/commit/da526eb58d7556c455b2b534f9261cc68f74becf), tree
`57e7fda56cb2dbb2aef5b0a265367dd20da39a7b`. The
[preflight](artifacts/agent-episode-live-preflight-attempt-4-2026-09-20.json)
retains pre-generation checks. This campaign changes both the explicit grammar
profile and shared policy; any observed difference cannot be attributed to grammar
alone. The complete campaign retained **nine attempts and eight measurable finals**;
only **six intents** had both retrieval and a measurable final, and **two tasks**
passed their independent verifiers. One task timed out. There was no model-owned
supersession; an instruction-like-memory answer invented a citation without
retrieval, and the feedback task lacked the required new retrieval. Both
independent replays rejected the frozen gates. The
[independent attempt-4 review](artifacts/agent-episode-live-attempt-4-review-2026-09-20.json)
scored milestone completion **8.0/10: rejected**, with SHA-256
`f2e457c390ba51d4a06ef4b1482aeb76ab7518b672714a657c6df628053a3c16`.

Recorded lower bounds are **12,652 input + 710 output = 13,362 known tokens**.
The timeout leaves one unknown-usage/effect task and an incomplete model call;
exact input/output/total cost and exact per-success cost remain null. Two successes
do not convert unknown cost or incomplete quality coverage into zeroes.

| Retained attempt-4 artifact | SHA-256 |
| --- | --- |
| [Raw campaign](artifacts/agent-episode-live-campaign-attempt-4-2026-09-20.json) | `1815413cb0708d530657b6b64918b25ba8cc10a42293aeaa9313744348f29ad6` |
| [Evidence replay](artifacts/agent-episode-live-evidence-attempt-4-2026-09-20.json) | `2b7472fa19057ff442beb48e63081b81132856be71352028114e69ebf4ddb923` |
| [Execution](artifacts/agent-episode-live-execution-attempt-4-2026-09-20.json) | `5963c3383f32e94a874cab71499bfa16a35d034b44d3103e111da6bd5cd024b4` |
| [Supervised resource cleanup](artifacts/agent-episode-live-resource-cleanup-attempt-4-2026-09-20.json) | `aa545a3db3795dfb5c1ab0e9007dfa68bfdc1cc2794a2f3da6f1dd694961dc64` |

[CI 329](https://github.com/mmckeen-nv/DML/actions/runs/35516379118) passed
**20/20 jobs** on public `da526eb`. The [compact receipt](artifacts/agent-episode-ci-329-2026-09-20.json),
SHA-256 `dfb8b7ab1a3522f89dc6585b864955989ea6d954c0f4102e92ddae574afbcb92`,
retains **760 mandatory CPU passes / zero skips**, **20 modules** and **370
matching source hashes**. Both full-suite jobs passed **4,672 tests with 250
skips each**; quiet logs do not enumerate every skip reason, and these overlapping
counts are not added to CPU coverage.

## Integrated cleanup and grounding candidate

The [cleanup blocker review](artifacts/agent-episode-worker-cleanup-blocker-review-2026-09-20.json)
scored the published source's completion state **9.2/10: rejected**, SHA-256
`f9208ec744d2a7f6f1272ff083972ec4051740a4efdebc16bec2709bcbeb9e59`.
The supervisor survived the killed child, so leaving its private snapshot copy
behind is within milestone 7's runtime scope. Attempt 4 required separately
recorded supervised cleanup; that action is not credited to the frozen producer.

The [reviewed repair](artifacts/agent-episode-worker-cleanup-review-2026-09-20.json)
received scoped **9.6/10** acceptance. The parent creates a worker-specific
scratch directory, pins its device/inode ownership and redirects child temporary
files into it. After child termination, the surviving parent removes only that
owned tree, handles read-only private copies without following links or escaping
the resolved root, and exposes cleanup failure instead of reporting success.
Real killed-child and injected-permission tests retain their scoped evidence.
This does not claim cleanup after supervisor death.

The [grounding revision](artifacts/agent-episode-grounding-review-2026-09-20.json)
received scoped **9.7/10** acceptance. It removes fictitious query/value/ID examples
and clarifies public tool descriptions and final-answer versus mutation semantics.
The complete public tool schemas and policy still bind the constrained runtime
identity. Model choice, trained weights, task truth, grammar/parser boundaries,
bounds and live acceptance gates remain unchanged. Improvement cannot be inferred
before actual execution or attributed to a single simultaneous change.

Both repairs were integrated only after the frozen campaign and independent
replays completed. [Combined final mandatory validation](artifacts/agent-episode-grounding-cleanup-validation-2026-09-20.json)
passed **774 tests / zero failures, errors or skips**, 20 modules, **433 unchanged
hashes**, both strict Ruff scopes and maintained mypy. A fresh strict admission
performed **zero generations**, with all five bundle files unchanged and the new
public-tools-bound runtime identity recorded separately. Combined review is
accepted at **9.6/10 with no source blockers** in the [combined final review](artifacts/agent-episode-grounding-cleanup-review-2026-09-20.json); publication, new-source CI and a fifth pre-generation freeze remain required. Historical reviews, all four campaigns and their failed costs remain
immutable; milestone 7 stays open.

## Interrupted attempt 5 — evidence unavailable after session recovery

The [fifth pre-generation declaration](artifacts/agent-episode-live-spec-attempt-5-2026-09-20.json),
SHA-256 `46177ba2958f1a1c18927d6c6761d2c5250a0abc4eb9e7bb57aded831e9d2ad0`,
passed independent review before launch. It binds 433 source hashes and 278
dependency versions, retaining the same trained model, nine tasks, limits and
acceptance gates. The first task timed out after **300.14 seconds**, with four
completed retrieves; that observed failure prevents qualification under the fixed
gates. That statement described the September 20 run. On September 22 the raw
work directory and final campaign are unavailable, so completion and replay cannot
be established. The original declaration remains immutable; a new run must receive
a new attempt identifier and freeze. Completed episode reports for future runs
will be stored outside `/tmp`. In-progress parent-death replay remains outside
this operational recovery; milestone 10 is not closed by this change.

Repaired source is published as
[`0d4364d116d75ed24b60e441845518b47f39ad57`](https://github.com/mmckeen-nv/DML/commit/0d4364d116d75ed24b60e441845518b47f39ad57), exact tree
`d8b2c5d4c699fdde85fe6c70d190c2375b9fb0de`. [CI 330](https://github.com/mmckeen-nv/DML/actions/runs/35518344343)
passed **20/20 jobs without retry**; PR #118 remains unmerged. Its
[immutable receipt](artifacts/agent-episode-ci-330-2026-09-20.json), SHA-256
`7ae8722a64d4e14a0795cc2a9c78cb2bf86e199df4968e119867a4dfbfc4ce0a`,
retains **774 mandatory CPU passes / zero skips**, exactly **20 modules**, and
**370 independently matched source hashes** with verified merge/tree identity.
Both full suites passed **4,675 tests with 261 skips each**, plus hygiene;
these overlap other selections and do not replace zero-skip CPU evidence.
Source remains accepted at 9.6/10; trained qualification stays open.
A separately assessed compatible stock
coder-model alternative is only a prospective path; no new model selection,
admission or semantic qualification is asserted here. All four prior failed campaigns remain retained and
milestone 7 remains open.

## Published-source CI 328

The [overall CI receipt](artifacts/agent-episode-ci-328-2026-09-20.json) has SHA-256
`8ffb2d30360a124ee73fcbe83ab233a8f76f9667ff40de81cb1d15f2620e4636`.
It preserves both production-job attempts, artifact identities and exact source
attribution, alongside the separately retained CPU receipt. Both full-suite jobs
passed **4,672 tests with 205 declared skips each**; those counts are not added to
the mandatory CPU selection, which independently passed with zero skips.

The [independently verified CPU receipt](artifacts/agent-episode-ci328-cpu-receipt-2026-09-20.json)
records **715 tests / zero failures, errors or skips**, across **18 modules**, with
**366 source-file hashes** matching the published source. Tested merge
`e33d70c3eb9724d65a0f0b1bfd6621e100826641` has tree
`4846330e9ea97ba156467d973fda90c488857b1b`. CPU artifact `10606421338` has ZIP
SHA-256 `31221c2ccf8c06b90ee2418a66c0ba48fc8a7dd892e6eb9b8ed11ac2aa8290d5`.
The compact receipt's SHA-256 is
`93a689b24b3d3c03e22040783ac528f5ffffdf90e4f4ace17e050328a87a02e0`;
its pending-overall-CI state is historical at receipt creation and is not rewritten.

The initial workflow had **19 successful jobs and one failure**, the production
256-client test's 30-second ownership acquisition timeout. Cause remains
undetermined. A single same-source diagnostic retry passed the test in **22.25
seconds**, leaving **all 20 latest job results successful**. The original failure
is retained and no source or bound changed for the retry. CPU archive/provenance
are unchanged. This accepts the published source CI gate; it does not replace
failed trained-model coverage or qualify a later source revision.

## Trained snapshot and strict execution boundary

The selected model is `Qwen/Qwen2.5-1.5B-Instruct`, pinned to revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. The
[normalization provenance](artifacts/agent-episode-qwen-provenance-2026-09-20.json)
retains upstream file identities, license, learned-tensor identities and exact
BF16-to-float32 round trips. No training or corpus conditioning is performed.
The normalized `model.safetensors` is **6,174,895,536 bytes**, SHA-256
`03c6c9523873b288883dfe9e4e37ec2d85a9bedccf761c20eddaa55d93363f95`.
The weights are local execution inputs, not repository source artifacts.

The separate [Qwen companion](qwen-model-input-v1.md) admits exactly five local
files and the built-in architecture; execution uses CPU/float32, greedy decoding,
eager attention and complete fixed framing. Explicit `qwen2-instruct-v1`
selection prevents architecture fallback. Input/output and tokenizer identities
remain bound to the compiled request. Each call owns a transient dynamic cache
that is discarded afterwards; no cache is received from, returned to or retained
for a caller. This does not qualify native-KV checkpoint reuse.

The [strict admission record](artifacts/agent-episode-qwen-admission-2026-09-20.json)
and [generic timing record](artifacts/agent-episode-qwen-timing-2026-09-20.json)
confirm an actual non-corpus greeting generation. Its exact input and greedy
output IDs match the retained no-cache measurement. These are admission and
feasibility evidence only. The original preparation record retains its original
runtime identity; the cache-policy change has its own admitted runtime identity,
which the campaign pins explicitly. The earlier GPT-2 tokenizer preflight,
[GPT-2 normalization](artifacts/agent-episode-gpt2-provenance-2026-09-20.json) and
random tiny fixtures do not supply this campaign's semantic outcomes.

The v2 [preparation provenance](artifacts/agent-episode-qwen-native-provenance-2026-09-20.json),
[preparation execution](artifacts/agent-episode-qwen-native-preparation-execution-2026-09-20.json)
and [strict admission](artifacts/agent-episode-qwen-native-admission-2026-09-20.json)
now retain the fresh native-content bundle. Its learned-weight, configuration and
tokenizer bytes match the preceding bundle; the template identity is the v2 hash
recorded in the contract. Strict load and close performed **zero generations**.
This establishes preparation/admission only, not a new semantic campaign result.

The reviewed completion source has a retained Git checkpoint
[`902fda192117ebcbccc269a650f5ce127884f0b7`](https://github.com/mmckeen-nv/DML/commit/902fda192117ebcbccc269a650f5ce127884f0b7), tree
`7872c61b9708b9abc571c33b8941bbe6fb282f38`. At this checkpoint the PR branch still
points to `6db5a837eb7377e08356a76a2d4f226c3e9b26fd`; an uploaded commit is not a
passing published-source CI result.

## Predeclared campaign

The campaign freezes **one attempt per task, zero automatic retries**, across
nine ordered tasks covering all eight adversarial intents. Its controlled fixtures
remain `synthetic_fixture`; scheduled peer writes do not qualify live two-agent
concurrency. Every attempt uses a fresh receipt authority. The follow-up task
receives the preceding actual model answer and terminal reference as explicitly
untrusted context, with the same answer supplied to the independent verifier.

| Bound or identity | Frozen value |
| --- | --- |
| Consumer and corpus | `qwen2-instruct-v1`; `dml-agent-corpus-v1` |
| Model context | 32,768 tokens |
| Model calls per task | At most 6 |
| Reserved output per call | 256 tokens |
| Cumulative input / output per task | 32,768 / 1,536 tokens |
| Transcript / frame / producer-event limits | 256 KiB / 4 MiB / 16 MiB |
| Child wall time per task | 300 seconds, including bootstrap |
| CPU threads | 4 |
| Decoding | Greedy, one beam, no sampling; request-local dynamic cache |
| Fixture effective time | `2000000000` |

Cleanup, bounded final authority reads and artifact publication follow child work;
the wall bound is not a promised end-to-end campaign latency ceiling. The exact
source hashes, learned weights, tokenizer, template, imported runtime versions,
complete installed dependency inventory and command are in the frozen declaration.

## Acceptance and interpretation

Before generation, the declaration requires:

- Exactly nine attempted tasks and complete failure-inclusive raw events and
  terminal results, with no omitted failure or silent retry.
- Actual model generation for every task and successful model-requested retrieval
  plus a schema-valid, measurable final answer for every intent.
- Independent verification of the superseded-preference task, including actual
  model-owned supersession, its receipt and authoritative final state.
- Actual retrieval and measurable answers in both feedback tasks, with the exact
  preceding terminal answer carried into the follow-up as untrusted context.
- Independent source/provenance/execution review and passing required published
  completion-source CI before milestone closure.

The harness qualification does **not** predeclare that every model answer must
be correct. Independently measured task failures remain failures; they do not
become a producer-authored success. A malformed or absent final answer cannot
establish semantic coverage. An unsuccessful required supersession cannot pass
the qualification gate. This distinction was frozen before model execution.

All unsuccessful-attempt cost contributes to the campaign. Per-completed-task
cost is undefined if no task succeeds. Missing measurements stay unavailable:
TTFT is null because this consumer does not measure the first-token boundary.
A repeated-error rate is undefined when there is no observed actual wrong-prior /
new-correction opportunity; an incorrect prior answer is never manufactured to
supply the denominator.

The evidence checker verifies declared coverage, causal consistency and accounting.
It cannot authenticate a producer on its own. Its replay therefore requires
independent model provenance, execution and source inspection. The CLI's local
`live_qualified=false` and `source_ci_qualified=false` flags are retained; an
independent qualification record, rather than editing producer output, establishes
acceptance after all gates pass.

## Closure and next milestone

Until genuine corpus evidence, independent review and final-source CI pass,
milestone 7 remains open and no milestone 8 work is credited. Once those gates
pass, the running PR tab can record the exact source, CI run and artifact digests
without changing the immutable pre-generation or source-review records. That
closure changes the count to **seven first-release milestones closed, four open
and two deferred**; the source-gate count remains 20 plus foundations.

Milestone 8 is the fair held-out baseline comparison. Milestone 9 owns recurring
1k/10k measurements and the 100k live campaign; milestone 10 owns durable decision
replay/export/retention; milestone 11 owns final release qualification. No bounded
harness result establishes a baseline advantage, production retrieval quality,
long-horizon quality, physical power-loss safety or overall production readiness.
