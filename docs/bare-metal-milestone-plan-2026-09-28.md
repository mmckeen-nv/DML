# DML milestone status and M7 execution plan
As of September 29, 2026. Latest full campaign: remote Nemotron native-tools-v4; latest qualification: Qwen3-8B CUDA retrieval-policy v2, planning 0/2 and readiness unrun. Prior untuned diagnostic: original Llama 3 8B Instruct, 0/3. Llama SFT v2 subsequently passed 8/8 held-out synthetic cases versus 2/8 with the adapter disabled; both evidence replays passed. M7 remains open.

## Current position

The [Llama SFT v2 diagnostic](llama3-8b-sft-v2-2026-09-29.md) passed **8/8** predeclared held-out synthetic cases after correcting training/deployment serialization. The same frozen runtime with the adapter disabled passed **2/8**. Both replays verified all 16 episodes and 36 responses, including supersession/read-back with a valid cited final. Preserve the zero-episode failed deployment and its reviewed correction. This qualifies the limited synthetic diagnostic only: station-only SFT tooling still requires source integration and exact-source CI, followed by the original planning/readiness gates and a new complete M7 campaign. [Compact outcome](artifacts/llama3-8b-sft-v2-outcome-2026-09-29.json).

The [Llama 3 SFT v1 experiment](llama3-8b-sft-v1-2026-09-29.md) completed 90 GPU training updates, but **0/432 training/development targets** were admitted by the unchanged inference grammar: the serializer used the wrong JSON field order. Independent review denied evaluation; no held-out model calls or new M7 campaign occurred. Preserve the adapter and failed evidence. A separately versioned serialization correction requires full token-wise grammar checks and a newly approved run from the untouched base. [Compact outcome](artifacts/llama3-8b-sft-v1-outcome-2026-09-29.json).

The latest [original Llama 3 8B Instruct diagnostic](llama3-8b-instruct-trial-2026-09-29.md) passed **0/3** fresh cases, once each. Both primary and independent replays passed all five responses. No candidate was promoted and no new M7 campaign ran; the original gates and historical 8/9 result remain unchanged.

Latest diagnostics: the [paired protocol comparison](qwen3-protocol-comparison-2026-09-28.md) passed **1/3 baseline versus 2/3 native**, and the subsequent [fresh native-envelope diagnostic](qwen3-native-envelope-2026-09-28.md) passed **1/3**. Both primary and independent replays passed for each run. The envelope addressed syntax admission, but unsupported factual claims still failed the original verifier. Neither result promotes a candidate or changes the original gates. **M7 remains open**; the historical full campaign remains 8/9, and no new M7 campaign ran.

The Qwen3-8B CUDA retrieval-policy v2 candidate failed both fresh read-only planning controls (**0/2**). Each finalized without covering both requested stored facts. The staged runner correctly left all four readiness cases unrun. Exact-source CI passed 21/21 on `37471b6`, with 1,589 model-input tests and zero skips. This guidance did not establish readiness; no new M7 campaign was frozen or launched. The original model/runtime, retrieval, validation, budgets and acceptance gates remain unchanged. [Compact v2 outcome](artifacts/qwen3-gguf-retrieval-v2-outcome-2026-09-28.json).

The separately versioned Qwen3-8B CUDA completion candidate also passed **3/4** fresh predeclared noncorpus controls. GPU admission and live process evidence confirmed inference on the GB300. Supersession failed: the model attempted to supersede the current record with itself, received an authentic predispatch rejection with no effects, raised the retrieval cap without changing the query, and finalized without finding or superseding the old record. All four cited finals were valid, but the required mutation was absent. Exact-source CI passed 21/21 on `a3fb6f7`; the run used 12 calls with no execution errors or timeouts. The 4/4 readiness requirement was not met. No new campaign was frozen or launched, and M7 remains open. [Compact GPU outcome](artifacts/qwen3-gguf-cuda-qualification-outcome-2026-09-28.json).

The separate Qwen3-8B ARM64 CPU candidate passed **3/4** predeclared noncorpus controls. Retrieval, dependent feedback and injection resistance passed. Supersession failed: two identical `top_k=1` lookups exposed only the current record, then the model finalized without the requested mutation or read-back. The tool was exposed and grammar-admitted; no returned action was discarded. All four cited finals were valid, but task completion also requires the state change. Both replays verified all 10 completions; exact-source CI passed 21/21 on `5c41f1e`. No errors, timeouts or unknown usage occurred. The 4/4 readiness requirement was not met, so no new campaign was frozen or launched and M7 remains open. This readiness check does not replace the original M7 gates. [Compact Qwen outcome](artifacts/qwen3-gguf-arm64-qualification-outcome-2026-09-28.json).

The latest native-tools-v6 budget-guidance development qualification passed 3/6 cases; both recovery cases still failed (0/2). Each recovered a verified conflict, committed supersession and read back, then repeated retrieval on the last permitted call despite visible budget metadata. The baseline supersession case produced incomplete native syntax at its output cap. No new campaign was frozen or launched; M7 remains open. Exact-source CI passed 21/21 jobs on `0351ba9`; primary and independent replay passed all 21 calls, preserving every failure. Independent campaign approval is withheld. See [v6 qualification outcome](artifacts/native-vllm-v6-budget-qualification-outcome-2026-09-28.json).

The newer native-tools-v5 recovery development qualification failed both declared cases (0/2), despite both offline evidence replays passing. The source case truncated before exercising recovery; the replacement case recovered a verified conflict, retrieved fresh state, committed a model-owned supersession and read back, but reached the step limit without a final answer. No new nine-task campaign was frozen or launched. Exact-source [CI 36468583308](https://github.com/mmckeen-nv/DML/actions/runs/36468583308) passed all 21 jobs on `68fdc74`; it does not override failed live qualification. See [v5 recovery qualification](artifacts/native-vllm-v5-recovery-qualification-outcome-2026-09-28.json).

**13 milestones: 6 closed, 5 open for the first release, 2 deferred.** M7 remains active. The frozen `nemotron-remote-vllm-native-tools-v4` candidate completed nine tasks once: eight passed; `supersede_then_answer` failed with `tool_error`; `read_both_commits` passed. Both sequential evidence replays completed and matched byte for byte. The named-task failure and two original failed gates prevent closure. Nothing is merged; `production_ready=false`.

Execution source: `e970252d1b2afff582fd741f137cb4334e95a06c`. Recorded usage: 30,713 input + 1,363 output tokens; no unknown usage or effects; serving epoch remained stable. Earlier campaigns and development diagnostics remain separate, retained and nonresumable. The remote profile is a new model/runtime candidate with explicit native protocol extensions; it does not inherit the old local profile's exact-token guarantees.

The [compact verified outcome](artifacts/native-vllm-m7-outcome-2026-09-28.json) records the failed gates `all_intents_reached_retrieval_and_final` and `verified_model_owned_supersession`. The model repeated a stale supersession after the first mutation committed and readback was provided; the gateway rejected it without additional effects, and no final answer followed. Both replays passed integrity checks; their exit 1 reflects failed acceptance.

## Milestone ledger
| # | Milestone | Status and remaining acceptance |
|---|---|---|
| 1 | Freeze supported production profile | Closed at reviewed-source gate 16. Supported boundaries and limits recorded; does not imply release readiness. |
| 2 | Extract persistence and transaction coordination | Closed at gate 15. Cross-file crash atomicity remains excluded. |
| 3 | Bind exact model input and tokenizer budgets | Closed at gate 17. Exact identities, framed input accounting and immutable dispatch enforced. New model profiles still need their own qualification. |
| 4 | Qualify crash recovery and supported filesystems | Closed at gate 18; CI320 17/17, six measured environments, 15 real ENOSPC cases. No physical power-loss claim. |
| 5 | Complete persisted-format and migration coverage | Closed by reconciliation September 19. Credit existing foundations and gates 3, 8 and 18; do not reopen or recount. |
| 6 | Qualify mixed-operation concurrency | Closed at gate 19; CI324 20/20, 96 independently verified platform histories. |
| 7 | Wire live-agent semantic and outcome harness | ACTIVE. Native-tools-v4 completed nine tasks once (8 pass); `supersede_then_answer` failed, `read_both_commits` passed. Both evidence replays matched; two original gates failed. Both named tasks and all original gates must pass before closure. |
| 8 | Demonstrate fair baseline value | OPEN. Independent durable baseline exists. Freeze fairness/thresholds; run paired held-out no-memory/baseline/DML comparison with equal resources, confidence intervals and quality/latency gates. |
| 9 | Run continuous 1k/10k lanes and 100k campaign | OPEN. Offline runner exists. Provision recurring live lanes and complete 100k-turn campaign with growing-store, recovery and quality/latency evidence. |
| 10 | Deliver durable decision replay and audit export/retention | OPEN. Mutation decisions/response traces exist. Demonstrate complete product decision/context reconstruction across restart, export/access controls and bounded retention. |
| 11 | Complete release qualification and support documentation | OPEN. Contracts/runbooks exist. Assemble exact-source M1–10 evidence, supported matrix and installation/upgrade/backup/recovery/support guidance; qualify release. |
| 12 | Extract remaining legacy retrieval/lifecycle orchestration | DEFERRED beyond first release. Preserve behavior and ownership while extracting remaining legacy boundaries. |
| 13 | Audit native-KV restore identity end to end | DEFERRED beyond first release. Validate all restore/import/continuation identity dimensions and supported hardware/runtime combinations. Native KV restore remains experimental. |

## Historical M7 execution plan
The following Qwen CPU plan predates the completed remote Nemotron candidate. Its acceptance gates and ownership rules remain applicable; its prospective host/model instructions are historical, not authorization to rerun. These are work packages under M7, not additional milestones.

| Step | Owner | Work and evidence required before advancing |
|---|---|---|
| A. Reconcile and bind a durable host | Supervisor + execution agent | Confirm no existing campaign is running on its actual execution host. Recover immutable source, preparation/review records and model bytes from authentic sources; reverify hashes. Select an always-on Linux host or managed persistent runner with durable storage outside the chat workspace. Record actual access, CPU/RAM/disk and retention. No eligible host is yet verified. |
| B. Implement and qualify the execution lifecycle | Execution agent + evidence agent; independent grader reviews | Proposed design: a host-managed service with automatic campaign restart disabled, unique attempt ID, exclusive start lock, persistent manifest, progress heartbeat, atomic per-task evidence and terminal exit receipt. Chat only starts/observes the job. Test controller disconnect/reconnect, duplicate start, worker termination and retained partial evidence using a synthetic job before loading the model. Host reboot/worker death records interruption; it must not silently resume the model campaign. |
| C. Qualify and freeze the successor | Model/runtime agent + supervisor + grader | Keep the pinned Qwen3-8B artifact, task policy, prompts, retrieval, grammar, sampling and budgets. Record the actual new host/runtime identity; do not assume native builds transfer identically. Perform fresh required admission, source checks and independent review. If lifecycle source changes, qualify that exact source in CI. Create a new uniquely identified full-campaign declaration referencing interrupted attempt15. |
| D. Execute once | Supervisor controls one producer | Run all nine tasks/eight intents; continue through ordinary task failures. No selective retries, seed search, answer repair or altered gates. Preserve raw events and authentic predecessor feedback on the durable host. Do not upload weights, raw logs or raw campaigns to GitHub or the summary deliverable. |
| E. Replay, grade and update the ledger | Primary reviewer then independent grader | Perform two sequential data-only replays and reconcile their evidence. Close M7 only when both named tasks pass and every original gate passes: generation per task, retrieval/measurable final for each intent, complete failure-inclusive evidence, actual predecessor feedback and verified model-owned supersession. All-nine success is not a newly added requirement. Otherwise record exact failed gates and stop this candidate from being called qualified. |

Proposed starting host target: CPU-only Linux, four inference threads, at least 16 GiB RAM and 40 GiB free persistent disk, isolated from competing inference jobs. These are planning allowances, not measured qualification or a promise that every task meets its unchanged 300-second limit. Actual admission and storage budget must validate the chosen host. A GPU backend would require a separately declared runtime comparison.

A detached shell on the same disposable workspace does not establish durability. Prefer a host-managed service (for example systemd) whose lifetime is independent of the chat controller, with no automatic generation restart. Verify disconnect survival on the actual chosen host. Durable local evidence is required for both replays; no remote raw-evidence upload is assumed or authorized by this plan.

## Ownership, reporting and scope controls
- Supervisor: source of truth, dependencies, single launch ownership and milestone status.
- Execution agent: host/service lifecycle and bounded failure diagnostics.
- Evidence agent: durable receipts, progress and integrity accounting.
- Independent grader: lifecycle tests, exact freeze approval and final replay/gate decision.
- Only light observation during actual model execution; no competing model loads.
- Compact progress record: attempt/source/model IDs, completed tasks out of nine, named-task outcomes, current state, last heartbeat, elapsed time and next blocking gate.
- Target public status summary under 10 KB; retain raw evidence privately on the execution host. No model/raw-log upload or merge.
- Maintain one current top-of-file milestone table and append-only historical attempt records. Replace stale present-tense M7 rows during implementation; do not rewrite immutable reviews.
- M7 durable campaign receipts do not substitute for M10 product replay/export/retention qualification.

After M7 acceptance, proceed through M8 fair comparison, M9 live scale/soak, M10 product replay/audit, then M11 release assembly. Preparation can overlap where useful, but closure evidence remains separately attributed. M12–13 remain deferred.

Independent planning review: **9.5/10**, accepted for planning, conditional on binding and verifying the actual execution host. No launch approval is implied.

## Evidence and freshness
- [Current PR118](https://github.com/mmckeen-nv/DML/pull/118), freshly read September 28: exact head, CI and interrupted-attempt status.
- [Published milestone ledger](https://github.com/mmckeen-nv/DML/blob/8677d1cce8e003ab8f9ea9e66ac18918a448ad87/docs/production-remaining-work-2026-09-18.md): authoritative milestone definitions; M7 pre-CI prose and historical rows are stale relative to the PR and saved outcome.
- [CI346](https://github.com/mmckeen-nv/DML/actions/runs/36373086151).
- Saved dense status version 5, modified September 28 03:45 UTC, freshly read for this plan. Older local status copies were not treated as current.
- [systemd service reference](https://github.com/systemd/systemd/blob/main/man/systemd.service.xml), for the proposed host-managed service mechanism. Actual host configuration and behavior require qualification.
