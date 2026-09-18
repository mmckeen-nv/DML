# Finite production release plan

Date: 2026-09-18. This is the current remaining-work ledger for the
[production contract](contracts/production-v1.md), after fifteen reviewed serial
hardening gates, including the transaction-coordinator extraction. The
[original ten areas](productionization-plan-2026-09-12.md) remain the organizing
workstreams; a serial gate or pull request is not another release milestone.

The audited scope contains 13 milestones: 11 for the first production release
and 2 deferred beyond it. Milestone 2 is now closed at the reviewed-source gate,
leaving **12 remaining: 10 for the first release and 2 deferred**. Publication and
exact-commit CI are pending at this source snapshot; outcomes will be recorded in
PR #118. This is not a merged release.
These are closure decisions, each of which may require
several implementations and evidence runs. Completing the coordinator
gate does not complete all persistence qualification or promote the repository
from alpha. The table records remaining obligations, not a claim that each area
starts from zero.

## First production release: 10 remaining of 11 milestones

| # | Milestone | Status | Closure criteria | Original areas |
| --- | --- | --- | --- | --- |
| 1 | Freeze the supported production profile | Open | Publish the exact supported APIs, authoritative journal/receipt format, local-host/filesystem/platform matrix, trusted-caller scope model, dependencies, limits and error/retry outcomes. Experimental features must be outside the profile's correctness dependencies. | 1, 7, 9 |
| 2 | Extract persistence and transaction coordination | Closed at reviewed-source gate: serial gate 15 | Narrow services integrated with characterized legacy and receipt-schema-2/3/4 behavior; independent review **9.6/10**, 164 focused passes, 256-client stress pass, and 2,948 full-suite passes with 9 skips. [Evidence and limits](transaction-coordinator-hardening-2026-09-18.md) explicitly exclude cross-file crash atomicity. | 2, 3, 9 |
| 3 | Bind exact model input and tokenizer budgets | Open | Pin model, tokenizer and chat-template identities; count the complete final input, including tools/framing and reserved output, and reject overflow before inference. Verify all supported entry points and prevent unchecked additions after counting. | 1, 2, 7 |
| 4 | Qualify crash recovery and supported filesystems | Open | Finish the supported-profile mutation/component inventory and deterministic plus seeded fault matrix; account for acknowledged, rejected and uncertain operations on restart; qualify the advertised filesystem/platform failure guarantees and publish a tested recovery runbook. Keep process-kill and power-loss evidence distinct. | 3, 9 |
| 5 | Complete persisted-format and migration coverage | Open | Inventory every persisted family admitted by the supported profile; enforce supported versions and compatibility; pass source-version, future-version, interrupted migration, export and restore cases using accurately labeled release or commit-pinned fixtures. Document excluded families and rollback limitations. | 3, 8 |
| 6 | Qualify mixed-operation concurrency | Open | Exercise supported reads, writes, lifecycle operations, checkpoints and recovery through threads, processes and HTTP/provider callers at 1/16/64/256-client levels with an independent history checker. Show zero lost updates, dirty reads, deadlocks or scope leakage and satisfy predeclared cancellation, timeout, starvation and lock-latency requirements. | 3, 9 |
| 7 | Wire the live-agent semantic and outcome harness | Open | Run pinned tool-driven agent episodes with task verifiers, raw events and failure-inclusive cost/quality metrics; cover the adversarial semantic cases and incorrect-retrieval feedback loops. Distinguish live outcomes from deterministic state checks and offline retrieval smoke. | 4, 6 |
| 8 | Demonstrate fair baseline value | Open | Freeze a fairness manifest and acceptance thresholds before held-out evaluation; compare no memory, the independent durable baseline and the supported DML profile with equal models, embeddings, budgets, tools and compaction. Meet the predeclared value, quality and latency gates with paired episodes and confidence intervals. | 5, 6 |
| 9 | Run continuous 1k/10k lanes and the 100k campaign | Open | Provision recurring 1k/10k live workload lanes and a completed 100k-turn release campaign with growing-store measurements, recovery checks, raw events, seeds, configuration/runtime identities and quality/latency distributions. Report turns and record counts separately; skips and offline simulations cannot close this milestone. | 3, 5, 6, 9 |
| 10 | Deliver durable decision replay and audit export/retention | Open | Persist the supported profile's complete decision and context-replay inputs; reconstruct a deliberately bad answer across restart using retained source versions and pinned policies/renderers. Verify export behavior, access controls and bounded retention; state exactly when retention prevents reconstruction. | 10 |
| 11 | Complete release qualification and support documentation | Open | Assemble exact-source passing evidence for milestones 1–10, the supported matrix, installation/upgrade/backup/recovery procedures, known limitations and support policy. Promote only the qualified profile through the maturity inventory and release process; preserve experimental labels elsewhere. | All |

## Broader completion: 2 deferred milestones

| # | Milestone | Status | Closure criteria | Original areas |
| --- | --- | --- | --- | --- |
| 12 | Extract remaining legacy retrieval and lifecycle orchestration | Deferred beyond the first release | Capture legacy public behavior, move the remaining hybrid retrieval and automatic lifecycle orchestration behind narrow boundaries, and verify compatibility and ownership. First-release support must not depend on unqualified legacy or experimental paths. | 2, 4, 7 |
| 13 | Audit native-KV restore identity end to end | Deferred beyond the first release | Verify every restore/continuation/import path against complete model, tokenizer/template, runtime/endpoint, topology, dtype/layout, position/prefix, scope/authority and payload identities. Every missing or mismatched dimension must reject before native restore; qualify each explicitly supported hardware/runtime combination. Native KV remains experimental until then. | 7 |

## Scope and accounting

The first release builds on existing receipted ingestion, retirement, supersession,
content correction and first-level promotion/merge, verified retention inspection,
projection delivery/migration and scoped retrieval/context services. Those are
implemented candidate boundaries with their own evidence, not new remaining
milestones. Their broader release qualification is captured above.

Physical erasure is excluded from this finite plan. Retirement suppresses normal
retrieval while preserving history; retention inspection does not erase or certify
erasure. Recursive promotion, cascading invalidation, additional experimental
features and extra deployment profiles are also excluded unless separately agreed.
They must not silently grow the first-release milestone count.

A defect discovered while closing a milestone is repaired and recorded under that
milestone. A new pull request, test file or serial hardening gate does not add a
milestone. A material change to the supported product scope requires an explicit
revision of this ledger. Historical results remain in the
[foundations record](production-foundations-2026-09-12.md) and linked hardening
documents; current release readiness is determined by this ledger and the
production contract.
