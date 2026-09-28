# Current status pointer — September 28, 2026

M7 remains open. The remote native-tools-v4 candidate completed nine tasks once (eight passed); `supersede_then_answer` failed and `read_both_commits` passed. Both sequential evidence replays completed and matched byte for byte. See the [current milestone ledger](production-remaining-work-2026-09-18.md) and [compact verified outcome](artifacts/native-vllm-m7-outcome-2026-09-28.json). Nothing is merged; `production_ready=false`.

The latest native-tools-v6 budget-guidance development qualification passed 3/6 cases; both recovery cases still failed (0/2). Each recovered a verified conflict, committed supersession and read back, then repeated retrieval on the last permitted call despite visible budget metadata. The baseline supersession case produced incomplete native syntax at its output cap. No new campaign was frozen or launched; M7 remains open. Exact-source CI passed 21/21 jobs on `0351ba9`; primary and independent replay passed all 21 calls, preserving every failure. Independent campaign approval is withheld. See [v6 qualification outcome](artifacts/native-vllm-v6-budget-qualification-outcome-2026-09-28.json).

The newer native-tools-v5 recovery development qualification failed both declared cases (0/2), despite both offline evidence replays passing. The source case truncated before exercising recovery; the replacement case recovered a verified conflict, retrieved fresh state, committed a model-owned supersession and read back, but reached the step limit without a final answer. No new nine-task campaign was frozen or launched. Exact-source [CI 36468583308](https://github.com/mmckeen-nv/DML/actions/runs/36468583308) passed all 21 jobs on `68fdc74`; it does not override failed live qualification. See [v5 recovery qualification](artifacts/native-vllm-v5-recovery-qualification-outcome-2026-09-28.json).

## Historical candidate records

The original records below are preserved and do not supersede the current status linked above.

# M7 model upgrade — September 28, 2026 UTC

M7 remains **IN PROGRESS**. The completed relation candidate on `8c792ae39514624ca6ec28392daa6a67b4dba857` passed exact-source CI345 (20/20 first-attempt jobs) but failed live qualification: nine completed terminals, six successes, three failures, 20 model calls and 21,986 tokens. Both independent sequential data replays matched byte for byte. Supersession persisted correctly but the model answered with the retired value; both related commits were retrieved but the answer added unsupported citation IDs; the conflicting-runbooks answer omitted A. Both requested tasks remain unresolved. The [compact completed outcome](artifacts/agent-episode-attempt-14-relation-outcome-2026-09-27.json) preserves the actual results and hashes. Earlier interrupted evidence remains separate and nonresumable.

The user authorized a stronger-model upgrade. The selected successor is the official **Qwen3-8B GGUF Q4_K_M**, revision `7c41481f57cb95916b40956ab2f0b139b296d974`, exact 5,027,783,488-byte artifact SHA256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`. This is an explicit quantized-model/runtime comparison, not original-BF16 tensor equivalence or proven model improvement. Its metadata identifies the vendor artifact as `Qwen3 8B Awq Compatible Instruct`. The 8 GiB CPU-only workspace cannot safely load a larger BF16 model; actual quantized memory admission remains required.

The new `qwen3-8b-gguf-action-json-sampled-v1` profile uses pinned llama-cpp-python 0.3.35, exact authenticated HF input token IDs, a fresh bounded CPU context for each request, F16 KV, unchanged action grammar and fixed Torch sampling. It retains the exact existing Qwen3 ChatML transport/escaping and non-thinking prefix. The vendor default tool template is not substituted because its XML tool-call instructions conflict with the unchanged DML action grammar. Logical system/task messages, tools, retrieval, corpus, sampling values, budgets and milestone gates remain unchanged. Context is allocated for actual input plus reserved output; no silent truncation or reduction of the episode input cap is permitted.

Local impact checks passed 139 tests with zero skips, plus lint and eight-file type checks. The [actual native prequalification](artifacts/agent-episode-qwen3-8b-prequalification-2026-09-28.json) loaded the exact vendor artifact, verified full token-row parity, and ran one fixed synthetic forward with zero generated tokens (peak RSS 5,433,688,064 bytes, no OOM, clean close). The initial download TLS and smoke-harness accessor failures remain recorded. Source CI, published-source admission, a new frozen declaration, one complete nine-task evaluation and two sequential evidence replays must establish qualification. No new task generation or milestone-completion claim is made here. Raw logs, model files and campaign traces remain local. PR118 stays unmerged; six milestones are closed, five open and two deferred; `production_ready=false`.

## Historical status before the model upgrade

# M7 continuation — updated 2026-09-27

M7 remains **IN PROGRESS**. [CI344](https://github.com/mmckeen-nv/DML/actions/runs/36334302503) passed all 20 jobs on its first attempt for `c51b51440401037873a19e3863888b3ae31f5abf` (tree `ecd4211d6ba6ab41586575f250dbebdde597889a`). The [independent source qualification](artifacts/agent-episode-ci344-qualification-2026-09-27.json) records the actual six evidence archives and both core lanes, including the prior MCP stall case and four telemetry regressions. This qualifies that exact source, not a later correction.

The full nine-task retry launched once at 19:44:20 UTC after fresh pinned-model acquisition, independent tensor proof and zero-generation admission. Its execution session subsequently became unavailable. Two actual completed reports remain, but the complete campaign and producer execution receipt are absent. The [compact partial observation](artifacts/agent-episode-attempt-14-retry-partial-2026-09-27.json) preserves the results and their limits: `preserve_conflict` omitted runbook A; `supersede_then_answer` correctly persisted the model-requested supersession but then returned the old verbose value. Both overall tasks failed. `read_both_commits` has no retained completion in this retry. No complete replay, global process-termination claim or reconstructed completion is made.

The next correction exposes the receipt-backed `superseded_by` relationship already present on the changed record, with matching integrity validation. It does not inject a replacement answer or perform an additional read. Prompts, model weights, sampling, retrieval, corpus, budgets and acceptance gates remain unchanged. Fresh exact-source CI, admission and a complete independently replayed campaign must establish any improvement. The interrupted frozen checkout and its failures remain intact. The revised v2 projection requires the relationship when it exists in the acknowledged record. Earlier c51 displays remain attributable to their original frozen source; their raw reports must not be rewritten or replayed with the revised source as though the interface were identical.

The [independent partial audit](artifacts/agent-episode-attempt-14-retry-grader-2026-09-27.json) verified the two retained reports against the frozen source, tokenizer, grammar, receipt authority and task verifier. It records a lower bound of six model calls and 7,393 known tokens; activity without a retained report remains unknown. The original freeze is incomplete and nonresumable. This review accepts an isolated source correction, not a completed campaign or another generation run.

The [local correction checks](artifacts/agent-episode-supersession-projection-validation-2026-09-27.json) passed 281 focused cases. The [final follow-up](artifacts/agent-episode-supersession-projection-followup-2026-09-27.json) passed all 13 projection and 71 campaign-evidence cases after live/replay alignment; the baseline fails the intended missing-field regression. Ruff and actual Python 3.12 type checks passed. Test-only isolated bytecode caches avoided a corrupt retained pytest cache; shared runtime files were not modified. Fresh supported-version CI remains required.

Both requested tasks remain unresolved. Six milestones are closed, five open and two deferred; PR118 remains unmerged and `production_ready=false`.

## Historical telemetry correction and CI343 — September 27

M7 remains **IN PROGRESS**. [CI343](https://github.com/mmckeen-nv/DML/actions/runs/36331022633) on `cb37774b` ended with **19 successful jobs and one cancelled Python 3.11 job** at the 45-minute limit. The added stack dump identifies the blocked path: Prometheus histogram child creation triggered `DMLAdapter.__del__`; adapter cleanup refreshed state and recorded another operation while the same thread held the metric-family lock. Seven MCP initializer waiters and the main test thread then waited behind it. This establishes the CI343 deadlock; CI342 lacks a stack trace, so its cause remains unproven. The [compact terminal record](artifacts/agent-episode-ci343-deadlock-2026-09-27.json) retains the failed run and log hash. All six evidence archives passed independent checks, but the overall source is CI-unqualified.

The correction guards DML telemetry against same-thread reentry before touching Prometheus. Nested optional metric recording is omitted while adapter cleanup and business operations continue. Ordinary metrics and recording from other threads remain enabled. The complete CI selection, workloads, assertions and milestone gates remain unchanged; fresh CI must qualify the corrected source.

A bounded regression using real Prometheus locks and a real adapter finalizer deadlocked on the unchanged baseline and reached its five-second limit; the corrected code completed in 0.306 seconds with cleanup and subsequent metrics intact. The [focused validation](artifacts/agent-episode-metrics-finalizer-correction-2026-09-27.json) passed 14 cases, with six existing optional MCP skips and no failures. It also checks exception cleanup, independent threads, cross-metric reentry and scrape reentry. Maintained lint and the 87-file type-check scope passed locally. These Python 3.12 results support the correction; fresh supported-version CI remains required.

Fresh recovery reacquired and verified all seven pinned Qwen2.5-3B files, prepared the original BF16 weights and independently checked all 434 tensors. Actual zero-generation admission on `cb37774b` passed in 35.615 seconds, with peak process RSS 6,707,212,288 bytes, no OOM events and a closed consumer. The nine-task declaration was frozen and independently reviewed, but **no model generation occurred** because CI343 failed. Its admission and freeze retain their original source attribution. Reviewed byte/source equivalence may retain these model assets; corrected-source admission, CI, a new declaration, the complete campaign and both sequential replays remain required.

Neither `supersede_then_answer` nor `read_both_commits` is resolved. Six milestones remain closed, five open and two deferred. PR118 stays unmerged and `production_ready=false`.

## Historical CI observability and recovery — September 27

M7 remains **IN PROGRESS**. [CI342](https://github.com/mmckeen-nv/DML/actions/runs/36216312985) on `45cca794` finished with **19 successful jobs and one cancelled Python 3.11 job**. GitHub's job annotation confirms its six-hour execution limit. The last rendered test progress was 29%; no active test name or traceback was recorded. The paired journal-read correction passed production stress, but the cancelled lane leaves this source CI unqualified. No rerun is counted as a success.

A single bounded diagnostic covered the corresponding 72-case window in the local collection order: **66 passed, six optional MCP skips**, in 5.02 pytest seconds. This used Python 3.12 and different retained dependencies, so it does not reproduce or explain the CI 3.11 stall. No production synchronization or SQLite correction is justified by that result. The [compact terminal and diagnostic record](artifacts/agent-episode-ci342-stall-2026-09-27.json) retains the failure and exact hashes.

The new CI-only correction keeps the full test selection, assertions, dependencies and workloads. It adds named test progress (`-vv`), a Python stack dump after a test exceeds 120 seconds, and a 45-minute core-job limit. Stack dumps do not terminate a test or turn failures into passes. The operational limit prevents another six-hour opaque wait; it is not evidence that the underlying stall is fixed. Fresh full CI must qualify the new source.

After a workspace rollback, all 48 checkpoint files and all 800 published source hashes were recovered and verified. The r2 declaration remains historical and unchanged. Raw attempt13 campaign data and model assets are unavailable locally; retained summaries are not a fresh replay or a complete historical archive. A separately declared recovery must reacquire exact model bytes and perform fresh preparation, independent proof, admission, freeze and the complete nine-task campaign with both sequential replays. Neither failed model task is resolved. Six milestones remain closed, five open and two deferred; PR118 stays unmerged and `production_ready=false`.

## Historical r2 prequalification — September 26

M7 remains **IN PROGRESS**. The original attempt 14 source `dff77b11` passed source review and strict zero-generation Qwen2.5-3B admission, but [CI341](https://github.com/mmckeen-nv/DML/actions/runs/36215232626) ended **19 successes / 1 failure**, with no rerun. The unchanged 256-client mixed-transaction test hit its 30-second ownership-acquisition timeout. No attempt 14 declaration or trained-model campaign was started. The [compact prequalification record](artifacts/agent-episode-attempt-14-prequalification-2026-09-26.json) preserves this failed CI and successful admission with exact hashes.

Prequalification revision 2 removes one demonstrated redundant full journal validation from legacy journal refresh. It routes those adapters through the existing paired payload/revision read while preserving file-format stamp callbacks, auxiliary RAG refresh and full integrity checks even at unchanged revisions. The [red/green regression](artifacts/agent-episode-paired-read-correction-2026-09-26.json) observed two full reads before the correction and one afterward. This removes avoidable work under the store lock; the original CI scheduling cause remains unproven, and no fairness guarantee is claimed. Workers, workload, lock timeout and acceptance gates are unchanged.

[Current-source impact validation](artifacts/agent-episode-attempt-14-r2-source-validation-2026-09-26.json) passed **136 tests**, with one existing optional FAISS skip, across six persistence/transaction modules. All maintained lint/type checks and adapter lint passed; all **531 nondocumentation hashes** remained unchanged. The unchanged **256-client / 32-worker** test passed once locally (JUnit case 18.657 seconds; whole command 41.283 seconds), retaining the **30-second acquisition budget and 120-second campaign deadline**. The original 1,457 model/source cases retain their `dff77b11` attribution and were not repeated locally; fresh CI must qualify the corrected source.

The model, fixed sampling, prompts, tools, grammar, corpus, verifiers and live budgets remain unchanged from the reviewed larger-model candidate. Exact prepared model bytes may be reused only with a reviewed source-equivalence record; the corrected source still requires fresh zero-generation admission, exact-source CI, a frozen declaration, one complete nine-task campaign and both sequential evidence replays. This is prequalification correction before any attempt 14 model generation. Attempt 13's two failed tasks remain unresolved. Six milestones are closed, five open and two deferred; PR118 stays unmerged and `production_ready=false`.

## Historical attempt 14 original prequalification

M7 remains **IN PROGRESS**. Attempt 13 completed on `7838cce671d76a1e18c71b53b29518055dbc2d73` with nine terminals, seven verifier successes, 18 model calls and **19,103 known tokens** (17,966 input + 1,137 output), with no unknown usage or effects. Fixed-seed sampling did not resolve either requested task. Supersession retrieved only the current preference and finalized without mutation. The concurrent-commits task retrieved only `validation.status` and omitted `deployment.owner` from its final answer.

Both actual sequential evidence replays are byte-identical (`cebfe90df89e02e76d466d7f4887e0a540d8b0c3aba64a2a01e7de38348fd917`) and reject `verified_model_owned_supersession`. Independent completion grade: **7.0/10, rejected**. Only attempt 13's completed freeze is released; older incomplete freezes and evidence gaps remain. The [compact actual outcome](artifacts/agent-episode-attempt-13-outcome-2026-09-26.json) retains every task result, failure, cost and evidence hash. Both named tasks must succeed before their correction is claimed; the original milestone gates remain unchanged.

[CI340](https://github.com/mmckeen-nv/DML/actions/runs/36212861681) qualified exact source with **20/20 first-attempt successes**, including 1,192 mandatory CPU cases with zero skips. Independent CI grade: **9.6/10**. Source qualification does not override the live rejection. Prior CI339 diagnostic results, original failures and dependency drift remain in the retained history.

Attempt 14 prepares a separate `qwen2-action-json-sampled-bf16-v1` candidate using `Qwen/Qwen2.5-3B-Instruct`, immutable revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`. This changes model family, learned weights and tokenizer transport as well as parameter count; it is not a pure capacity comparison or a proven improvement. It retains original BF16 weights, the fixed sampling settings from attempt 13, task prompts, public tools, grammar, verifiers, budgets and gates. The model's research license limits this candidate to evaluation. Short-input fit under 8 GiB requires measured admission; full 32K-context fit is not claimed.

[Attempt 14 local source validation](artifacts/agent-episode-attempt-14-source-validation-2026-09-26.json) passed **1,457 tests** (1,226 retained + 231 new), zero skips/failures/errors, all three strict lint scopes and the maintained 82-file mypy scope with the actual Python 3.12 target. All **531 nondocumentation file hashes** remained unchanged. Synthetic runtime tests establish implementation behavior; live task correction is still unproven.

Reviewed source, fresh exact-source CI, independent tensor proof, zero-generation admission, a declaration frozen before generation, one complete nine-task campaign and two sequential evidence replays are required. No seed search, selective retry, forced action or automatic answer repair is introduced. M7 stays open, PR118 stays unmerged, and `production_ready=false`. See the [new profile contract](qwen2-bf16-action-v1.md).

## Historical attempt 13 preparation

M7 remains **IN PROGRESS**. Attempt 12 recovery3 completed on `88b76025` with nine terminals, seven verifier successes, 18 model calls and **19,002 known tokens**, with no unknown usage or effects. The model retrieved one record and finalized too early in both failed tasks: it never performed the requested supersession, and it omitted `validation.status` after the peer writes. Both sequential evidence replays are byte-identical and reject the required model-owned supersession gate. Independent completion grade: **7.0/10, rejected**. Only recovery3's completed freeze was released; older incomplete freezes and disclosed evidence gaps remain.

[Compact actual outcome](artifacts/agent-episode-attempt-12-recovery3-outcome-2026-09-25.json) retains every task result, failure, cost and evidence hash. CI339 qualified exact source with 18 original successes and two diagnostic successes. The original production lock timeout and Python 3.11 cancellation remain recorded, including unpinned dependency drift; diagnostics do not establish their causes.

Attempt 13 introduces the separate `qwen3-action-json-nonthinking-sampled-bf16-v1` candidate. It uses Qwen's recommended non-thinking sampling settings (temperature 0.7, top_p 0.8, top_k 20, min_p 0), with fixed CPU seed 0 per call and restored RNG state. The legacy greedy profile, learned weights, full messages, tools, syntax grammar, tasks, verifiers, limits and acceptance gates remain unchanged. The two failures establish a model planning problem, not a demonstrated runtime defect; improved behavior is unproven until a fresh campaign completes. There is one candidate and one full nine-task campaign, with no seed or parameter sweep, selective retry or automatic action repair. See the [sampled profile contract](qwen3-sampled-action-v1.md).

[Local source validation](artifacts/agent-episode-attempt-13-source-validation-2026-09-26.json) passed all **1,226 cases** (1,183 retained plus 43 new), zero skips/failures/errors, three lint scopes and the maintained 82-file mypy scope. All 521 nondocumentation file hashes stayed unchanged. The initial local mypy invocation used the repository's Python 3.10 default against installed Python 3.12-only NumPy stubs; its failure is retained. Correcting the local target to 3.12 passed without a source change or repeating tests. Fresh CI still checks actual supported Python environments. Exact-source CI, admission, frozen declaration and two evidence replays must establish the new candidate's result. M7 stays open, PR118 stays unmerged, and `production_ready=false`.

## Historical attempt 12 preparation


## Historical attempt 12 model and runtime comparison

M7 remains **IN PROGRESS**. Attempt 11 completed all nine task terminals on
published source `e08c38e3fc2fd73d94acd9f89461ed6f49db117f`: eight verifier
successes, nineteen model calls, **20,576 known tokens** (19,362 input and 1,214
output), and no unknown usage or effects. Both completed sequential replays
match byte for byte and reject only `verified_model_owned_supersession`.
The model again proposed self-supersession, received the trusted no-operation
rejection, then finalized without the operation or a supporting citation. The
[completion review](artifacts/agent-episode-live-completion-grader-review-attempt-11-isolated-2026-09-25.json)
rejects M7 closure, releases only attempt 11's completed freeze, and closes the
fixed guidance hypothesis as failed. No wording or placement iteration follows.

[CI 338](https://github.com/mmckeen-nv/DML/actions/runs/36083908802) passed all
**20 jobs on the first attempt** for that source. The
[independent CI review](artifacts/agent-episode-ci-338-grader-review-2026-09-25.json)
accepted **9.6/10**, including 941 mandatory model cases with zero skips, verified
filesystem/concurrency evidence and production stress. Both broad suites passed
4,796 cases with 341 declared skips. CI does not override the failed live gate.

Attempt 12 is implemented in a distinct checkout after that full rejection
and scoped release. It is a separately declared model/runtime capability comparison:
`Qwen/Qwen3-1.7B`, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, under explicit
profile `qwen3-action-json-nonthinking-bf16-v1`. Original BF16 shards and index remain
the prepared weights, with ordinary isolated copies and complete tensor/alias
verification. A new transport records and charges the explicit non-thinking prefix.
The first system-message bytes, grammar, public tools, validation mechanics, tasks,
verifier, budgets and live gates remain unchanged; old profiles and all 975 existing
cases remain supported. No better behavior, memory fit or latency is assumed.

The [full validation](artifacts/agent-episode-attempt-12-isolated-validation-2026-09-25-r2.json)
passed **1,183 cases with zero skips, failures or errors**: all 975 retained cases
plus 118 snapshot and 90 consumer controls. All **445 source hashes** remained
unchanged; all three strict lint scopes and the maintained 82-file mypy scope passed.
The first combined run stopped on two preparer typing errors after 859 passing cases.
That failed receipt and its logs remain retained; a narrowly reviewed typing correction
preceded this fresh full run. No trained-model attempt occurred during either validation.
The [independent source review](artifacts/agent-episode-attempt-12-integrated-grader-review-2026-09-25.json)
covers the corrected source and validation. Fresh exact-source CI remains required.
Model acquisition additionally requires a reviewed helper and a fresh two-copy disk
check with the existing 1 GiB reserve. Exact trained tensor proof, strict zero-generation
admission, frozen specification review, one complete nine-task campaign and two
sequential independent replays are still required. No successor model has been run.

Original attempt 8 remains incomplete with its source and inputs frozen; attempts
5–7 retain disclosed evidence gaps. PR #118 stays open and unmerged. Milestones
1–6 are closed, M7–11 open and M12–13 deferred; 20 serial source gates plus foundations
remain unchanged. DML remains alpha and `production_ready=false`. Raw events and logs
stay local; published records remain compact. Earlier checkpoints below are historical.

## Historical validated attempt 11 guidance candidate

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
