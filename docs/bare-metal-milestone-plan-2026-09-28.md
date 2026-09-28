# DML milestone status and M7 execution plan
As of September 28, 2026, 08:02 MDT. Planning baseline; no campaign launched by this plan.

## Current position
**13 milestones: 6 closed, 5 open for the first release, 2 deferred.** M7 is active. These are acceptance gates, not five untouched implementations. PR118 remains open and unmerged; production_ready=false.

Published code: `8677d1cce8e003ab8f9ea9e66ac18918a448ad87`; tree `6f0084e5b769b0f8e4a075e8832b87d44cba5a56`.
Qwen3-8B official Q4_K_M integration, native admission and independent pre-generation review passed. CI346: 20/20 first-attempt jobs; model-input lane 1,469 passes, zero skips. Local impact: 139 passes.

Attempt15 is interrupted, not completed: one retained successful task (`preserve_conflict`), two known calls and 2,043 known tokens. Both `supersede_then_answer` and `read_both_commits` have no retained terminal results. Unretained work is unknown; interruption cause is unproven. The upgrade has not established overall model improvement or M7 closure. Preserve attempt15 unchanged and nonresumable.

## Milestone ledger
| # | Milestone | Status and remaining acceptance |
|---|---|---|
| 1 | Freeze supported production profile | Closed at reviewed-source gate 16. Supported boundaries and limits recorded; does not imply release readiness. |
| 2 | Extract persistence and transaction coordination | Closed at gate 15. Cross-file crash atomicity remains excluded. |
| 3 | Bind exact model input and tokenizer budgets | Closed at gate 17. Exact identities, framed input accounting and immutable dispatch enforced. New model profiles still need their own qualification. |
| 4 | Qualify crash recovery and supported filesystems | Closed at gate 18; CI320 17/17, six measured environments, 15 real ENOSPC cases. No physical power-loss claim. |
| 5 | Complete persisted-format and migration coverage | Closed by reconciliation September 19. Credit existing foundations and gates 3, 8 and 18; do not reopen or recount. |
| 6 | Qualify mixed-operation concurrency | Closed at gate 19; CI324 20/20, 96 independently verified platform histories. |
| 7 | Wire live-agent semantic and outcome harness | ACTIVE. Source integration and CI passed; complete, durable trained-model evaluation and independent evidence acceptance still required. Both named tasks must pass. |
| 8 | Demonstrate fair baseline value | OPEN. Independent durable baseline exists. Freeze fairness/thresholds; run paired held-out no-memory/baseline/DML comparison with equal resources, confidence intervals and quality/latency gates. |
| 9 | Run continuous 1k/10k lanes and 100k campaign | OPEN. Offline runner exists. Provision recurring live lanes and complete 100k-turn campaign with growing-store, recovery and quality/latency evidence. |
| 10 | Deliver durable decision replay and audit export/retention | OPEN. Mutation decisions/response traces exist. Demonstrate complete product decision/context reconstruction across restart, export/access controls and bounded retention. |
| 11 | Complete release qualification and support documentation | OPEN. Contracts/runbooks exist. Assemble exact-source M1–10 evidence, supported matrix and installation/upgrade/backup/recovery/support guidance; qualify release. |
| 12 | Extract remaining legacy retrieval/lifecycle orchestration | DEFERRED beyond first release. Preserve behavior and ownership while extracting remaining legacy boundaries. |
| 13 | Audit native-KV restore identity end to end | DEFERRED beyond first release. Validate all restore/import/continuation identity dimensions and supported hardware/runtime combinations. Native KV restore remains experimental. |

## M7 work sequence
These are work packages under M7, not additional milestones.

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

