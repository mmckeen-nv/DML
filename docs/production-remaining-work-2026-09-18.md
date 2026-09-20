# Finite production release plan

Originally recorded 2026-09-18; current status reconciled **2026-09-20** against
review artifacts and the published PR/CI record. The [original ten areas](productionization-plan-2026-09-12.md)
remain workstreams. Twenty independently accepted serial source gates, plus the
initial foundations tranche, supply implementation and evidence toward the release gates
below; they are not nineteen additional release milestones.

## Current completion tab — 2026-09-20

The finite scope remains **13 release milestones: 11 first-release and 2 deferred**.
Milestones **1–6 are closed** and **milestone 7 is IN PROGRESS**, leaving
**5 unclosed first-release gates (including active milestone 7) and 2 deferred**.
These are remaining acceptance obligations, **not five untouched implementation
projects**. Existing implementations and evidence must be credited before assigning
new work. DML remains alpha and the supported profile remains a candidate.

Milestone 4 closed after independent acceptance at **9.6/10** and published source
[`3763303`](https://github.com/mmckeen-nv/DML/commit/3763303) passing all **17 jobs** in
[CI run 320](https://github.com/mmckeen-nv/DML/actions/runs/35359517124).
Six measured recovery environments and all **15 real ext4 ENOSPC cases** were
verified. This qualifies the documented process/storage-failure boundary; physical
power loss remains unclaimed. [PR #118](https://github.com/mmckeen-nv/DML/pull/118)
remains open and unmerged. Prior stage-16 source `dc85123` passed CI 315 and
stage-17 source `49368a9` passed [CI 317](https://github.com/mmckeen-nv/DML/actions/runs/35348088493).

**Milestone 5 closed by reconciliation of previously completed work on 2026-09-19.**
The initial tranche and serial gates 3, 8 and 18 already supplied its implementation
and accepted evidence. The earlier “unstarted” description and open status failed
to credit that work. This closure recognizes existing results; it adds no serial
hardening gate and claims no new implementation or test run. Commit-pinned fixtures
are valid when accurately labeled; multiple semantic releases are not a prerequisite.

### Milestone 5 closure mapping

| Existing criterion | Previously completed implementation and evidence |
| --- | --- |
| Inventory admitted persisted families | The [recovery component inventory](profile-recovery-v1.md#admitted-mutation-and-component-inventory) covers the admitted authority, receipts, decisions/outbox, identity/coordination sidecars and recovery artifacts, with explicit exclusions. The [supported profile](supported-production-profile-v1.md) fixes that scope. |
| Enforce supported versions and compatibility | [Journal validation](../dml_core/daystrom_dml/journal.py), [profile admission](../dml_core/daystrom_dml/services/profile_runtime.py) and [backup verification](../dml_core/daystrom_dml/services/authority_backup.py) enforce admitted schemas and compatible authority. [Profile admission tests](../dml_core/tests/test_production_profile.py), [receipt tests](../dml_core/tests/test_receipt_ingestion.py), [outbox migration tests](../dml_core/tests/test_outbox_migration.py) and [backup tests](../dml_core/tests/test_authority_backup.py) cover explicit paths and schema-2/3/4 operation; there is no automatic upgrade. |
| Reject unsupported future versions | [Foundations tests](../dml_core/tests/test_production_foundations.py), [journal hardening tests](../dml_core/tests/test_journal_hardening.py), [outbox tests](../dml_core/tests/test_journal_outbox.py) and [backup tests](../dml_core/tests/test_authority_backup.py) exercise unsupported journal/decision/outbox/backup versions and fail-closed validation. |
| Source-version and accurately labeled historical fixtures | Explicit schema-0→1, schema-1→2 and schema-2→4 paths are exercised in [foundations](../dml_core/tests/test_production_foundations.py), [receipt](../dml_core/tests/test_receipt_ingestion.py) and [migration](../dml_core/tests/test_outbox_migration.py) tests. Fixtures include [schema 0](../dml_core/tests/fixtures/journal_schema0.sql), [schema 1 at b8626b7](../dml_core/tests/fixtures/journal_schema1_b8626b7.sql), [schema 2 at 8a08efa](../dml_core/tests/fixtures/journal_schema2_8a08efa.sql) and its [pinned historical reader](../dml_core/tests/fixtures/journal_schema2_reader_8a08efa.py). The schema-0 fixture is a synthetic legacy compatibility case; the named schema-1/schema-2 fixtures and reader establish commit-pinned compatibility, not semantic-release compatibility. |
| Interrupted migration preserves source and blocks incomplete output | [Receipt migration tests](../dml_core/tests/test_receipt_ingestion.py), [outbox migration recovery tests](../dml_core/tests/test_outbox_migration_recovery.py) and [independent adversarial tests](../dml_core/tests/test_migration_adversarial.py) cover interruption and invalid destinations. The [accepted migration record](outbox-migration-hardening-2026-09-14.md) records seven real process-kill boundaries, preserved source/history and safe offline cutover. |
| Export and restore outcomes preserve the authority contract | [Receipt tests](../dml_core/tests/test_receipt_ingestion.py) and [adapter tests](../dml_core/tests/test_receipt_adapter.py) reject lossy snapshot/lattice-only checkpoint export of receipt authority. The [backup service](../dml_core/daystrom_dml/services/authority_backup.py) and [60-case backup evidence](../dml_core/tests/test_authority_backup.py) cover full schema-2/3/4 round trips, historical receipts, committed WAL, corruption, interruption, existing destinations and stale-backup detection using retained receipts. The [recovery procedure](profile-recovery-v1.md#offline-operator-recovery-procedure) distinguishes full authority backups from incomplete exports. |
| Document excluded families and rollback limitations | The [profile](supported-production-profile-v1.md), [migration runbook](outbox-migration-hardening-2026-09-14.md#offline-runbook) and [recovery contract](profile-recovery-v1.md) exclude unsupported families, require separate destinations/offline cutover, forbid in-place downgrade and explain post-upgrade writes, captured backup revision and receipt-ledger limitations. Native KV, legacy file stores and excluded runtime APIs do not expand this gate. |

The implementation evidence retains its original attribution: migration gate 8 was
accepted at **9.5/10** with **139 focused passes** and **1,601 full-suite passes**;
recovery gate 18 was accepted at **9.6/10** and subsequently passed CI 320 as recorded
above. These results overlap other selections and are not new tests run for this
reconciliation. Milestone 6 closes separately below; milestones 7–11 retain their
distinct, unmet release criteria.

### Completed milestone 6 / serial gate 19

**Milestone 6 is closed**, with independent source acceptance at **9.6/10** and
all **20 jobs** passing in [CI 324](https://github.com/mmckeen-nv/DML/actions/runs/35470900431)
on source `6647a0d`. The
[qualification manifest](artifacts/profile-concurrency-qualification-2026-09-19.json)
binds exact source/tree, tested merge and archive/JUnit digests. All **96 retained
histories** were independently verified twice: **48 Linux, 24 macOS and 24 Windows**.
Linux passed 359 cases with zero skips; macOS passed 335 with zero skips; Windows
passed 334 with one declared fork skip. Both full-suite jobs passed **4,343 tests
with 75 skips each**. Its
[predeclared qualification contract](production-concurrency-2026-09-19.md)
credits existing CAS, lifecycle and process tests, then fixes the remaining
mixed-operation matrix: schemas 2/3/4, actual 1/16/64/256 ready/in-flight callers,
threads, spawned-process callers and real HTTP/provider requests. It requires
independent history checking, selected-profile lifetime fencing, uncertain
acknowledgement recovery and bounded progress/ownership-wait evidence.

The original reviewed local selection passed **318 tests with zero failures or skips**, with
all **48 matrix cells** and unchanged before/after hashes for 23 frozen files.
Static checks passed. The grader independently replayed all 48 histories and
accepted the finite source scope at **9.6/10**, after the earlier **9.2/10**
review was rejected and repaired. CI 322 then failed Windows negative controls;
a separate causal-order checker gap led to a rejected 9.3/10 follow-up and a
repaired 9.6/10 review. Those immutable reviews retain their own source evidence.

Corrected source `2cc3a3a` finished
[CI 323](https://github.com/mmckeen-nv/DML/actions/runs/35468625699) with **19/20
jobs passed**. macOS qualified 24 bounded cells with **300 passes**; Windows
qualified 24 with **299 passes and one declared fork skip**. Both full-suite jobs
passed **4,308 tests with 75 skips each**. Linux reported **322 passes and two
failures**: the separate-adapter schema-3/256 campaign expired its 90-second
deadline, and HTTP schema-4/256 exceeded the 10-second first-progress bound
(exact latter timing unavailable). The subsequent repair removed duplicate validated
scans/serialization without weakening checks and added rejected-campaign diagnostics. The [source-pinned chronology](production-concurrency-2026-09-19.md#rejected-ci-attempts-and-current-repair)
retains those rejected results. They were not waived or counted as closure;
the successful CI 324 qualification above met the original unchanged bounds.

The [focused correction record](production-concurrency-2026-09-19.md#correction-evidence-before-final-review)
now credits reduced duplicate scans/serialization without reduced validation,
a reproducible single-client diagnostic profile, and three passing contention
cells with **336 unchanged source-file hashes**. A later cleanup-only shared-budget
adjustment has been reviewed. The new
[contention correction review](artifacts/profile-concurrency-contention-review-2026-09-19.json)
accepts final source at **9.6/10**, with three independently replayed histories
(1,320 events / one typed rejection). These are not final-tree qualification
results. Final qualification comes from CI 324 and its verified artifacts above.
The plan now has **six first-release gates closed, five first-release gates open
and two deferred gates**; DML remains alpha and `production_ready` remains false.

The subsequent documentation closure at source
`a5fdf71677ccefa198f8a84bdb8722a66779c87a` passed **20/20 jobs** in
[CI 325](https://github.com/mmckeen-nv/DML/actions/runs/35472033511).
That documentation-head result supplements milestone 6's record; it does not
replace the immutable source `6647a0d` / CI 324 qualification evidence or qualify
new milestone 7 implementation.

### Active: milestone 7 — live-agent semantic and outcome harness

**IN PROGRESS as of 2026-09-20.** Bounded source implementation is independently
accepted at **9.6/10**: a model-driven episode/event producer, strict tool protocol
around the admitted receipt APIs, eight independent task verifiers, and versioned
raw events and failure-inclusive outcomes. Published-source CI and trained-model
live qualification remain pending at the reviewed-source freeze. Subsequent
publication and exact-source CI results are tracked in
[PR #118](https://github.com/mmckeen-nv/DML/pull/118); the pending-at-freeze wording
below is not a statement about a later CI run. The existing
`dml-task-outcome-v1` consumer contract
is preserved. The [milestone 7 working record](live-agent-outcomes-2026-09-20.md)
tracks the scope, acceptance gates, reused evidence and remaining qualification.
This is accepted **serial implementation gate 20**, within milestone 7; the
completed source index now has 20 serial gates plus foundations. It does not
close milestone 7. The first independent source iteration scored **9.2/10 and
was rejected**; repaired source received final **9.6/10 acceptance with no blocking
findings**. It passed **529 focused CPU tests with zero failures or skips** and
**4,667 full-suite tests with 24 declared skips and no failures or errors**,
independently confirmed. These selections overlap. The
[review](artifacts/agent-episode-review-2026-09-20.json) and
[local validation manifest](artifacts/agent-episode-local-validation-2026-09-20.json)
retain source identities and the qualification limits.

| Tracking tab | Current evidence or obligation |
| --- | --- |
| Completed | Milestones 1–6; milestone 6 source `6647a0d`, independent 9.6/10, CI 324 all 20 jobs passing and 96 retained histories independently verified twice. Historical reviews and qualification records remain unchanged. |
| Reused for milestone 7 | Existing eight-scenario adversarial intent, lifecycle and scope regressions, admitted receipt APIs, outcome reducer and exact-input/tokenizer companion. These establish deterministic behavior and plumbing, not live-agent outcomes. |
| Accepted source, serial gate 20 | Independent **9.6/10**: eight bounded semantic scenarios, six gateway tools with task allowlists, model-owned supersession with independent receipt/state verification, supervised execution and failure-inclusive artifacts. Scripted execution is `test_injected`; lexical fixture ranking is `synthetic_fixture`. Publication and source-pinned CI remain pending at this snapshot. |
| Remaining milestone 7 gate | Publish the accepted source and verify exact-source CI, then separately review a provenance-bound compatible trained snapshot, freeze campaign identities/bounds/thresholds, run genuine model-driven episodes covering all eight intents, and retain independently verified failure-inclusive artifacts. Source acceptance and live qualification remain separate. |
| Later first-release gates | Milestone 8 fair held-out baseline comparison; milestone 9 recurring 1k/10k lanes and the 100k live campaign; milestone 10 durable replay/export/retention; milestone 11 final release qualification. Two broader milestones remain deferred. |

Live qualification requires a pinned **trained execution model**. The current
exact-input companion admits local GPT-2/CPU/float32 with fixed JSON framing;
no suitable trained snapshot is yet present in the workspace. A pinned
[tokenizer-only preflight](artifacts/agent-episode-model-preflight-2026-09-20.json)
retains historical first requests, a separately attributed planned prior-context
follow-up, a default-bound supersession-path overflow and a tighter-bound planned
path. It executes no trained model; actual generated history still requires
runtime budget admission.
Random tiny fixtures establish plumbing only. Missing model evidence keeps milestone 7 open even if the bounded
source implementation passes review. The frozen source scope retains that
consumer. Synthetic lexical fixture ranking and scheduled receipted peer writes do not qualify production retrieval
quality or live multi-agent concurrency.

### Numbering legend

| Numbering system | What “5” means |
| --- | --- |
| Original ten workstreams | Independent boring baseline |
| Original delivery order P05 | Crash runner and recovery runbook |
| Serial hardening gate 5 | Projection delta delivery |
| Finite release milestone 5 | Complete persisted-format and migration coverage |

Historical hardening documents and review artifacts retain the source snapshots,
local results and pending-CI statements recorded at their creation. This current tab
supersedes those statements only for present completion accounting; it does not
rewrite historical evidence or review hashes. The [foundations running record](production-foundations-2026-09-12.md)
contains the completed serial-gate index.

## First production release: 5 unclosed gates of 11 milestones

| # | Milestone | Status | Closure criteria | Original areas |
| --- | --- | --- | --- | --- |
| 1 | Freeze the supported production profile | Closed at reviewed-source gate: serial gate 16 | Explicit candidate APIs, journal/receipt authority, platform target versus demonstrated qualification, trusted-caller scope, dependencies, limits and retry outcomes enforced with experimental paths excluded. Independent review **9.6/10**, 441 focused passes, and **3,342 local full-suite passes with 9 skips**. [Evidence and limits](production-profile-hardening-2026-09-18.md); subsequent published stage-16 source passed all nine CI jobs in run 315. | 1, 7, 9 |
| 2 | Extract persistence and transaction coordination | Closed at reviewed-source gate: serial gate 15 | Narrow services integrated with characterized legacy and receipt-schema-2/3/4 behavior; independent review **9.6/10**, 164 focused passes, 256-client stress pass, and 2,948 full-suite passes with 9 skips. [Evidence and limits](transaction-coordinator-hardening-2026-09-18.md) explicitly exclude cross-file crash atomicity. | 2, 3, 9 |
| 3 | Bind exact model input and tokenizer budgets | Closed at reviewed-source gate: serial gate 17 | The Python-only [local Transformers companion](model-input-contract-v1.md) pins model/tokenizer/template identities, counts complete messages/tools/framing plus reserved output, rejects overflow and dispatches the immutable token IDs. The memory profile retains its nine-route/no-generation boundary. Independent review **9.6/10**, **256 focused passes with 0 skips**, and **3,598 full-suite passes with 9 skips**. [Evidence and limits](model-input-hardening-2026-09-18.md); subsequent corrected source `49368a9` passed all ten CI jobs in run 317. | 1, 2, 7 |
| 4 | Qualify crash recovery and supported filesystems | Closed: serial gate 18, 9.6/10; source `3763303`, CI 320 all 17 jobs passed; six measured environments and 15 real ENOSPC cases verified | Finish the supported-profile mutation/component inventory and deterministic plus seeded fault matrix; account for acknowledged, rejected and uncertain operations on restart; qualify the advertised filesystem/platform failure guarantees and publish a tested recovery runbook. Keep process-kill and power-loss evidence distinct. | 3, 9 |
| 5 | Complete persisted-format and migration coverage | Closed by reconciliation on 2026-09-19: previously accepted foundations and serial gates 3, 8 and 18; mapping above | Inventory every persisted family admitted by the supported profile; enforce supported versions and compatibility; pass source-version, future-version, interrupted migration, export and restore cases using accurately labeled release or commit-pinned fixtures. Document excluded families and rollback limitations. | 3, 8 |
| 6 | Qualify mixed-operation concurrency | Closed: serial gate 19, **9.6/10**, source `6647a0d`, **CI 324 all 20 jobs passed**, 48/24/24 platform histories independently verified twice. [Qualification evidence](artifacts/profile-concurrency-qualification-2026-09-19.json) | Exercise supported reads, writes, lifecycle operations and journal checkpoint/recovery behavior through threads, processes and HTTP/provider callers at 1/16/64/256-client levels with an independent history checker. Show zero lost updates, dirty reads, deadlocks or scope leakage in the qualified histories and satisfy the predeclared cancellation, timeout, finite progress and OS ownership-wait requirements. | 3, 9 |
| 7 | Wire the live-agent semantic and outcome harness | **IN PROGRESS**: serial source gate 20 accepted **9.6/10**; 529 focused passes with zero skips and 4,667 full-suite passes / 24 declared skips (overlapping selections); publication/CI were pending at reviewed-source freeze (subsequent results in PR #118); trained-model live qualification remains open | Run pinned tool-driven agent episodes with task verifiers, raw events and failure-inclusive cost/quality metrics; cover the adversarial semantic cases and incorrect-retrieval feedback loops. Distinguish live outcomes from deterministic state checks and offline retrieval smoke. | 4, 6 |
| 8 | Demonstrate fair baseline value | Independent durable baseline implemented; fair held-out live comparison pending | Freeze a fairness manifest and acceptance thresholds before held-out evaluation; compare no memory, the independent durable baseline and the supported DML profile with equal models, embeddings, budgets, tools and compaction. Meet the predeclared value, quality and latency gates with paired episodes and confidence intervals. | 5, 6 |
| 9 | Run continuous 1k/10k lanes and the 100k campaign | Offline runner implemented; recurring live lanes and completed 100k campaign pending | Provision recurring 1k/10k live workload lanes and a completed 100k-turn release campaign with growing-store measurements, recovery checks, raw events, seeds, configuration/runtime identities and quality/latency distributions. Report turns and record counts separately; skips and offline simulations cannot close this milestone. | 3, 5, 6, 9 |
| 10 | Deliver durable decision replay and audit export/retention | Durable mutation decisions and response traces implemented; complete durable replay/export/retention pending | Persist the supported profile's complete decision and context-replay inputs; reconstruct a deliberately bad answer across restart using retained source versions and pinned policies/renderers. Verify export behavior, access controls and bounded retention; state exactly when retention prevents reconstruction. | 10 |
| 11 | Complete release qualification and support documentation | Contracts and recovery procedures implemented; final evidence assembly and release qualification pending | Assemble exact-source passing evidence for milestones 1–10, the supported matrix, installation/upgrade/backup/recovery procedures, known limitations and support policy. Promote only the qualified profile through the maturity inventory and release process; preserve experimental labels elsewhere. | All |

## Broader completion: 2 deferred milestones

| # | Milestone | Status | Closure criteria | Original areas |
| --- | --- | --- | --- | --- |
| 12 | Extract remaining legacy retrieval and lifecycle orchestration | Deferred beyond the first release | Capture legacy public behavior, move the remaining hybrid retrieval and automatic lifecycle orchestration behind narrow boundaries, and verify compatibility and ownership. First-release support must not depend on unqualified legacy or experimental paths. | 2, 4, 7 |
| 13 | Audit native-KV restore identity end to end | Deferred beyond the first release | Verify every restore/continuation/import path against complete model, tokenizer/template, runtime/endpoint, topology, dtype/layout, position/prefix, scope/authority and payload identities. Every missing or mismatched dimension must reject before native restore; qualify each explicitly supported hardware/runtime combination. Native KV remains experimental until then. | 7 |

## Scope and accounting

The first release builds on existing receipted ingestion, retirement, supersession,
content correction and first-level promotion/merge, verified retention inspection,
projection delivery/migration and scoped retrieval/context services. The selected
profile admits journal outbox formats but excludes runtime projection/consumer
backend APIs; their prior feature evidence does not enlarge its runtime surface. Those are
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
