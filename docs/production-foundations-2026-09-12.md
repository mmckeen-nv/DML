# Production foundations implementation

This is the first implementation tranche of the [ten-area plan](productionization-plan-2026-09-12.md).
The repository remains alpha. The [production contract](contracts/production-v1.md)
defines implemented behavior, migration and remaining release gates.
The current [finite remaining-work ledger](production-remaining-work-2026-09-18.md)
contains eleven first-release milestones and two deferred broader milestones;
**six milestones are closed**, leaving **five unclosed first-release gates and
two deferred gates**. **Milestone 7 is IN PROGRESS as of 2026-09-20.**
These are acceptance obligations, not five unstarted
implementation projects. Milestone 5 is closed by reconciliation of
previously accepted migration/versioning evidence, not by a new implementation.

The table summarizes the foundations and subsequent serial gates. Historical
sections below retain the evidence and scope recorded when each gate was built.

| Area | Implemented through the serial gates | Still required |
| --- | --- | --- |
| 1. Production contract | Explicit target, maturity/health, durable receipt APIs and revision-pinned scoped retrieval/context integration; reviewed supported-profile freeze and exact-input companion | Complete broader release qualification |
| 2. Adapter decomposition | Narrow persistence, cache, scoped selection, context/report, lifecycle, receipt, projection, delivery and retention-inspection services; reviewed transaction coordination extraction | Remaining legacy retrieval/lifecycle extraction is deferred beyond the first release |
| 3. Crash consistency | Verified journals/checkpoints, atomic receipts/outbox, explicit migrations, real process-kill, quota, corruption and serialization cases | Completed supported-profile process/storage-failure qualification and runbook are recorded below; physical power loss remains unclaimed |
| 4. Adversarial memory | Eight-case corpus plus lifecycle-specific conflict, scope, authority, provenance and historical-replay regressions | Real-agent semantic outcomes and long-running incorrect-retrieval feedback cases |
| 5. Baseline | Independent durable SQLite + embeddings + recency + top-k + compaction implementation and runner | Held-out live episodes, equalized compaction and statistically supported value gate |
| 6. Agent outcomes | Outcome reducer, failure-inclusive costs, quality/latency/recovery distributions, CI offline baseline and journal-history cost evidence | Wire real agent harnesses; continuous 1k/10k and 100k-turn release campaigns |
| 7. Stability boundaries | Provider contract inventory, reviewed supported-profile boundary and exact model/tokenizer input companion; native KV remains experimental | Release qualification; native-KV restore identity audit and hardware canaries are deferred |
| 8. Version migration | Explicit side-by-side schema 0/1/2/4 paths, schema-3 fresh authorities, preserved historical receipts and commit-pinned compatibility fixtures | Milestone 5 closed by reconciliation of the admitted-profile coverage; preserve compatibility as versions evolve, with accurately labeled commit-pinned fixtures |
| 9. Concurrency | Journal CAS/pinned reads, targeted lifecycle/process tests, mixed selected-profile thread/process/HTTP campaigns, lifetime fencing and snapshot/revision race repair | Milestone 6 closed at 9.6/10 after CI 324 passed all 20 jobs and the 48/24/24 platform histories were verified; growing-store/live-agent qualification remains separate |
| 10. Observability | Durable mutation decisions/source proofs, response retrieval traces, degradation status and scoped retention inspection | Durable full request/context replay, complete decision coverage and bounded audit retention/export |

## Current completion reconciliation — 2026-09-20

Published source [`3763303`](https://github.com/mmckeen-nv/DML/commit/3763303)
passed all **17 jobs** in [CI 320](https://github.com/mmckeen-nv/DML/actions/runs/35359517124),
including six measured recovery environments and **15 real ext4 ENOSPC cases**.
With independent review at **9.6/10**, this closes release milestone 4.
[PR #118](https://github.com/mmckeen-nv/DML/pull/118) remains open and unmerged.
The candidate/alpha classification and absence of physical power-loss claims remain.

The historical sections below retain their original source-snapshot accounting and
pending-CI language. This dated tab and the [finite release ledger](production-remaining-work-2026-09-18.md)
provide current status; accepted historical review artifacts are not rewritten.
Original area 5 is the baseline, original P05 is crash recovery, serial gate 5 is
projection deltas, and release milestone 5 is persisted-format coverage. These
numbering systems are not interchangeable.

### Completed implementation index

Grades apply to the bounded feature reviewed, not whole-platform production readiness.

| Gate | Completed implementation and evidence | Accepted grade |
| --- | --- | --- |
| Foundations | Contract, version validation, schema-0→1 upgrade, adversarial corpus, service seams, independent baseline/outcome scaffold and CAS stress; [initial evidence](#recorded-local-evidence-2026-09-12) | Foundational evidence; no separate grade asserted here |
| 1 | [Journal initialization, integrity and recovery](journal-hardening-2026-09-12.md) | 9.5 |
| 2 | [Semantic checkpoint publication, retention and shutdown](checkpoint-hardening-2026-09-12.md) | 9.5 |
| 3 | [Append-only ingestion receipts and schema-1→2 migration](receipt-hardening-2026-09-12.md) | 9.5 |
| 4 | [Disposable SQLite snapshot projection](projection-hardening-2026-09-14.md) | 9.5 |
| 5 | [Coalesced projection deltas](projection-delta-hardening-2026-09-14.md) | 9.5 |
| 6 | [Bounded projection worker, retries and shutdown](projection-worker-hardening-2026-09-14.md) | 9.5 |
| 7 | [Transactional outbox and ordered delivery](outbox-hardening-2026-09-14.md) | 9.5 |
| 8 | [Explicit schema-2→4 migration and historical compatibility](outbox-migration-hardening-2026-09-14.md) | 9.5 |
| 9 | [Receipted retirement](retirement-hardening-2026-09-14.md) | 9.6 |
| 10 | [Receipted supersession](supersession-hardening-2026-09-14.md) | 9.6 |
| 11 | [Receipted content correction](content-update-hardening-2026-09-16.md) | 9.6 |
| 12 | [Receipted first-level promotion/merge](promotion-hardening-2026-09-17.md) | 9.6 |
| 13 | [Verified retention inspection](retention-inspection-hardening-2026-09-17.md) | 9.6 |
| 14 | [Scoped retrieval/context services](retrieval-context-hardening-2026-09-17.md) | 9.8 |
| 15 | [Persistence/transaction coordinator](transaction-coordinator-hardening-2026-09-18.md) | 9.6 |
| 16 | [Supported-profile freeze](production-profile-hardening-2026-09-18.md) | 9.6 |
| 17 | [Exact model-input/tokenizer binding](model-input-hardening-2026-09-18.md) | 9.6 |
| 18 | [Supported-profile crash qualification and verified backup/restore](profile-recovery-hardening-2026-09-18.md) | 9.6 |
| 19 | [Supported-profile mixed-operation concurrency](production-concurrency-2026-09-19.md) and [exact-source qualification](artifacts/profile-concurrency-qualification-2026-09-19.json) | 9.6 |
| 20 | [Bounded live-agent episode harness](live-agent-outcomes-2026-09-20.md): initial source CI 327 passed 20/20; trained-model completion source accepted at 9.6/10; general protocol clarification at 9.7/10 with 710 mandatory CPU passes; [live campaign and final-source qualification](agent-episode-live-qualification-2026-09-20.md) remain open at this documentation freeze | 9.6; Windows test-ID repair 9.8; protocol clarification 9.7 |

Milestone 5 **closed by reconciliation of previously completed work on 2026-09-19**.
The initial version guards and schema-0→1 path, gate 3's schema-1→2 migration,
gate 8's schema-2→4 path with preserved history and seven process-kill boundaries,
and gate 18's verified schema-2/3/4 backup/restore satisfy the existing criteria.
The [seven-criterion closure mapping](production-remaining-work-2026-09-18.md#milestone-5-closure-mapping)
credits the existing component inventory, commit-pinned fixtures, version rejection,
interruption, export/restore and rollback evidence. This accounting correction
claims no new implementation or test run and does not add a nineteenth serial gate.
Milestones 7–11 and the two deferred milestones remain open within their stated scope.

### Completed implementation: serial gate 19 / milestone 6

[Supported-profile mixed-operation concurrency](production-concurrency-2026-09-19.md)
**is closed at 9.6/10** after source `6647a0d` passed all **20 jobs** in
[CI 324](https://github.com/mmckeen-nv/DML/actions/runs/35470900431). The
[qualification manifest](artifacts/profile-concurrency-qualification-2026-09-19.json)
binds the reviewed tree, tested merge, archive/JUnit digests and all **96 raw
histories**, independently verified twice. Linux passed **359 cases / 48 cells**;
macOS passed **335 / 24**; Windows passed **334 / 24**, with one declared fork
skip. Both full-suite jobs passed **4,343 tests with 75 skips each**.
The frozen matrix covers schemas 2/3/4 and
1/16/64/256 actual ready/in-flight clients through threads, spawned-process
callers and real HTTP/provider routes. Independent history checks, uncertain
HTTP acknowledgements, bounded progress and selected-profile close/fork behavior
are required. The contract distinguishes OS ownership acquisition from complete
request latency and records finite qualification limits.

The original reviewed local selection passed **318 tests with zero failures or skips** in
**572.87 seconds**, including all **48 matrix cells**. All 23 frozen file hashes
matched before and after the run. Maintained/static checks passed; the
independent grader replayed all 48 histories and accepted the bounded source at
**9.6/10**, following rejection and repair of the earlier 9.2/10 result.
Local environment qualification
correctly remains unaccepted because the tree is unpublished and dirty and the
filesystem is an overlay with `fsync=volatile`. The separately recorded earlier
4,058-pass / 24-skip integration run predates the final paired-revision and
typed-timeout fixes. Published corrected source `2cc3a3a` subsequently passed
both full-suite jobs with **4,308 passes and 75 skips each** in
[CI 323](https://github.com/mmckeen-nv/DML/actions/runs/35468625699).

CI 323 finished **19/20 jobs passed**. macOS qualified 24 bounded cells with
300 passes; Windows qualified 24 with 299 passes and one declared fork skip.
Linux had **322 passes and two failures**: the separate-adapter schema-3/256
campaign exceeded its 90-second deadline, and HTTP schema-4/256 exceeded the
10-second first-progress bound; its exact timing is unavailable. The
[concurrency record](production-concurrency-2026-09-19.md#rejected-ci-attempts-and-current-repair)
preserves CI 322's earlier rejection, follow-up review and exact source/tree
identities. The subsequent correction removed duplicate scans/serialization while
preserving validation and added better rejected-campaign diagnostics. The
original bounds remained unchanged.

Focused correction evidence now records one removed duplicate refresh scan,
unchanged semantic validation counts, and three passing contention cells on a
snapshot with **336 unchanged source-file hashes**. The
[diagnostic profile and cell measurements](production-concurrency-2026-09-19.md#correction-evidence-before-final-review)
remain limited evidence. The later cleanup-budget adjustment and final correction
source received **9.6/10** in the new
[contention review](artifacts/profile-concurrency-contention-review-2026-09-19.json),
which independently replayed all three histories. These earlier source-review
measurements retain their attribution; final-tree qualification and milestone 6
closure come from CI 324 and its verified artifacts above.

This is the nineteenth serial implementation gate, not a new release milestone.
Gate 19 is now in the completed index: **six first-release milestones closed,
five first-release milestones open and two deferred**. Milestone 7 is
**IN PROGRESS**. Its initial harness source passed CI 327, and the completion
work now admits a pinned instruction-trained Qwen snapshot through a separate
strict exact-input consumer. The [finite plan](production-remaining-work-2026-09-18.md#active-milestone-7--live-agent-semantic-and-outcome-harness)
and [live qualification record](agent-episode-live-qualification-2026-09-20.md)
track current source, campaign and CI gates. The historical tokenizer-only
preflight remains feasibility evidence; the actual trained corpus campaign and
final completion-source CI require their own acceptance. PR #118 remains
unmerged and `production_ready` remains false.

The subsequent documentation closure at source
`a5fdf71677ccefa198f8a84bdb8722a66779c87a` passed **20/20 jobs** in
[CI 325](https://github.com/mmckeen-nv/DML/actions/runs/35472033511).
This is separate documentation-head evidence. Milestone 6 retains source
`6647a0d` / CI 324 as its immutable qualification; CI 325 does not qualify the
new milestone 7 source work.

### Serial gate 20 / milestone 7 remains open

The [live-agent harness](live-agent-outcomes-2026-09-20.md) remains one serial
source gate, with **20 gates plus foundations**. It adds eight bounded semantic
scenarios, strict model-driven tool actions, independent task verifiers and
failure-inclusive raw outcomes. The [Qwen companion](qwen-model-input-v1.md)
and [action profile](qwen-agent-action-v1.md) use pinned trained weights and explicit
execution identities. Fixture ranking remains synthetic, and scheduled peer
writes do not qualify live two-agent concurrency.

Initial source scored 9.6/10 after a rejected 9.2 iteration; its 529 focused tests
and 4,667 full-suite passes / 24 skips retain their original attribution. The
Windows test-ID repair scored 9.8 and passed CI 327. Later strict Qwen, policy,
renderer and action-profile source reviews retained 9.6/9.7 grades and their own
698/710/715/760 mandatory selections. Counts overlap and are not summed.
The [qualification record](agent-episode-live-qualification-2026-09-20.md) links
all source reviews, validation and complete campaign evidence.

Public `da526eb` passed [CI 329](artifacts/agent-episode-ci-329-2026-09-20.json)
**20/20 jobs**, with **760 mandatory CPU passes / zero skips**, 20 modules and
370 independently matched source hashes. A subsequent surviving-supervisor
snapshot-cleanup finding rejected its completion state at **9.2/10**. All **four
trained campaigns** completed and failed their unchanged live gates; the fourth
received **8.0/10: rejected**. Passing source CI never substitutes for those results.

The scoped cleanup repair (**9.6/10**) and generic grounding revision (**9.7/10**)
are now integrated. [Combined final validation](artifacts/agent-episode-grounding-cleanup-validation-2026-09-20.json)
passed **774 mandatory cases / zero failures, errors or skips**, across **20
modules**, with **433 unchanged source hashes**. Both strict Ruff scopes and
maintained mypy passed. The [combined final review](artifacts/agent-episode-grounding-cleanup-review-2026-09-20.json) accepted **9.6/10 with no source blockers**; publication,
new-source CI and freshly frozen **attempt 5** remain required. Earlier source
and campaign evidence stays immutable. Milestone 7 remains open, the finite
count remains **five first-release gates plus two deferred**, and PR #118 remains
unmerged.

## Validation commands

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DML_SKIP_VENV_REEXEC=1 \
  python -m pytest dml_core/daystrom_dml/tests dml_core/tests integrations/hermes/tests openclaw-wrapper/tests
DML_STRESS_CLIENTS=256 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest dml_core/tests/test_production_foundations.py::test_concurrent_clients_use_compare_and_swap_without_lost_updates
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m scripts.production_baseline --turns 1000 --output production-artifacts/baseline.json
python -m scripts.production_baseline --task-events /path/terminal-events.jsonl --output production-artifacts/agent-outcomes.json
```

The baseline runner supports 10,000 and 100,000 turns, but does not invoke a model.
Only metrics from actual terminal task events may be labeled agent outcomes.
Each event uses `schema_version: dml-task-outcome-v1`, unique `episode_id` and
`task_id`, verified boolean `success`, input/output/maintenance token totals,
`latency_ms` and `retrieval_ms`. Optional measurements include `ttft_ms`,
`repeat_errors`/`repeat_opportunities`, `contradictions`/`factual_outputs`,
`false_memory_claims`/`recalled_claims`, `store_bytes`, `memory_records`, and
`recovery_ms`. Missing quality denominators yield null, never an invented zero.

## Recorded local evidence (2026-09-12)

On Python 3.12.14, the complete core, Hermes and wrapper suite passed: **931 passed,
9 skipped** (optional integrations), with one upstream Starlette deprecation warning.
The separate **256-client** journal CAS stress test passed without lost updates.
Ruff and the maintained mypy selection, expanded to include services (42 files),
passed; local mypy used `--python-version 3.12` to match installed NumPy stubs.
CI retains Python 3.10/3.11 type checks and Linux/macOS/Windows portability coverage.
The Hermes hygiene smoke also passed. Remote CI results remain separate evidence.

The [recorded 1,000-turn offline comparison](artifacts/production-baseline-2026-09-12.json)
found the expected evidence in 1,000/1,000 turns for both systems. Baseline retrieval
p95 was **1.46 ms**, versus **3.21 ms** for DML in this local run. DML has not earned
a performance advantage in this smoke. These single-run measurements over 100
fixed records are not statistical estimates of production performance or agent
quality. No 10k/100k real-agent campaign has been run.

No user memory stores were migrated and no runtime/GPU services were contacted.

## Serial receipt follow-up

The [append-only receipt contract](receipt-hardening-2026-09-12.md) adds an opt-in
schema-2 journal and a narrow ingestion service. It advances areas 1, 2, 3, 7, 8,
9 and 10: scoped historical receipts, atomic receipt/decision persistence, explicit
side-by-side migration, embedding compatibility, process-death/concurrency evidence
and retryable API outcomes. Stable APIs remain empty in the maturity inventory.
Later serial gates added retirement, supersession, content correction and
first-level promotion/merge receipts, plus reference projection delivery. External
backend qualification remains outside those bounded results. The original ten
areas remain workstreams; the finite ledger defines the current release scope.


## Serial projection follow-up

The [disposable SQLite projection](projection-hardening-2026-09-14.md) advances
P03 with atomic full-snapshot publication and pinned-source queries. It reuses
the accepted journal format, keeps source writes independent of backend outages,
and adds crash/concurrency/query regression evidence. Later gates added incremental
delivery and durable outboxes. Existing FAISS qualification and automatic provider
routing were not established by this gate.


## Serial delta delivery follow-up

[Coalesced projection deltas](projection-delta-hardening-2026-09-14.md) add checked
base/target publication, a narrow backend protocol and explicit adapter/CLI integration.
Receipt availability remains independent of backend I/O. Full-state verification
and complete ID manifests remain necessary. Later gates added workers and durable
operation outboxes; existing FAISS backend qualification remains unproven.


## Serial worker follow-up

[Bounded projection retry scheduling](projection-worker-hardening-2026-09-14.md)
adds explicit adapter-owned delivery with coalesced notifications, capped failure
backoff, terminal shutdown and historical pinned status. Process-death recovery
derives pending work from durable source/target state. Receipt acknowledgements
remain independent of backend availability. The following gate added durable
operation outboxes. Growing-store costs and qualification of the existing FAISS
backend were not established by this worker gate.


## Serial transactional outbox follow-up

[Transactional operation events](outbox-hardening-2026-09-14.md) add opt-in journal
schema 3 with full-state events committed beside memory/receipts/decisions, verified
historical prefixes, and ordered idempotent delivery to a dedicated SQLite consumer.
Existing schema-1/2 authorities are preserved. Later gates added explicit migration
and the supported lifecycle receipts. Bounded audit retention and ordered
background scheduling were not established by this gate.

## Serial outbox migration follow-up

[Explicit schema-2 to schema-4 migration](outbox-migration-hardening-2026-09-14.md)
preserves receipt and decision bytes, records an honest baseline for previously
unavailable full-state history, and adds versioned consumer adoption. Seven actual
process-kill boundaries, commit-pinned legacy reader/state fixtures and simultaneous
consumer delivery qualify the migration path. Offline manual cutover remains required.
The subsequent retirement through promotion gates added the supported lifecycle
receipt coverage. Audit retention and real-agent value remain in the finite
release ledger; this migration did not qualify ordered background delivery.

## Serial retirement follow-up

[Receipted memory retirement](retirement-hardening-2026-09-14.md) adds the first
qualified lifecycle mutation to the receipt profile: scoped tombstones guarded by
the exact current record digest, atomic historical receipts and outbox decisions,
and normal retrieval suppression. It retains historical content and capacity.
Subsequent gates added supersession, content updates and first-level promotion/merge.
Physical erasure remains unsupported and is excluded from the finite release plan.

## Serial supersession follow-up

[Receipted supersession](supersession-hardening-2026-09-14.md) extends lifecycle
receipt coverage to explicit same-scope replacement links. Both exact record
digests guard the decision, only the old memory is changed, and normal retrieval
suppresses it without modifying replacement trust. This is the tenth serial gate,
not completion of the ten original areas. Later gates added content updates,
first-level promotion/merge and retention inspection. Full mutation qualification,
decision replay and real-agent value at growing-store scale remain in the finite
ledger; physical erasure is excluded and native-KV qualification is deferred.

## Serial content-update follow-up

[Receipted content correction](content-update-hardening-2026-09-16.md) is the
eleventh serial hardening gate. Exact record and embedding-space checks guard
atomic text/vector updates while preserving scope, trust, lifecycle and creation
time. Preparation runs outside ownership and historical retries bypass the model.
Later gates added first-level promotion/merge and retention inspection. Broader
qualification and replay remain in the finite ledger; physical erasure is excluded
and native compatibility is deferred.

## Serial first-level promotion/merge follow-up

[Receipted promotion/merge](promotion-hardening-2026-09-17.md) is the twelfth
serial hardening gate, extending P03's durable mutations and P07's explicit
lifecycle boundaries. The caller selects base memories and supplies derived text;
one atomic append preserves every source and records complete source provenance.
Multi-source merges respect `no_merge`; output scope/trust remain unchanged and
ranking attributes cannot increase. First-level derivation is an independent
snapshot with its own lifecycle. The next gate added retention inspection.
Recursive promotion, cascading invalidation and physical erasure are excluded
from the finite first-release scope; broader production qualification remains open.

## Serial retention-inspection follow-up

[Verified retention inspection](retention-inspection-hardening-2026-09-17.md)
is the thirteenth serial gate. One pinned verified transaction counts known
same-scope memory and first-level source-proof occurrences across current records,
lineage, the archived journal snapshot, receipts and outbox states. Historical-only
records remain inspectable without exposing payloads or receipt keys. The static
contract and report explicitly state that retirement does not erase content and
that physical erasure is unsupported. No purge, history compaction or capacity
reclamation is implemented; those require a separately versioned history/retry
contract and external-copy policy. This is a bounded operator-evidence gate within
the original workstreams, not completion of their production qualification.

## Serial scoped retrieval/context follow-up

[Scoped retrieval and context services](retrieval-context-hardening-2026-09-17.md)
are the fourteenth serial gate, advancing P06 and areas 1, 2 and 9. Frozen resolved
inputs drive scoped selection and response evidence; narrow services handle recent
fallback, suppression, ledger lookup and detached context reports. The existing
store retains ranking, and the adapter retains model preparation, ownership,
routing, DPM and metrics. Baseline characterization preserves exact rendered output
and evidence across legacy, schema-1 and receipt-schema-2/3/4 paths. Receipt reads
retain ownership through report construction, with explicit embedding-identity
checks after preparation. Exact model-input budget binding and durable replay
remain first-release gates; remaining legacy retrieval/lifecycle extraction is
deferred beyond that release. Maturity
remains candidate; this does not qualify the whole repository for production.

## Serial transaction-coordinator follow-up

The reviewed [transaction and persistence coordinator gate](transaction-coordinator-hardening-2026-09-18.md)
is the fifteenth serial gate and milestone 2 of the finite release ledger. It
separates ownership/nesting/rollback orchestration from component
snapshot/restore/refresh/commit behavior, retaining adapter compatibility shims.
It also addresses caught publish-then-error outcomes in legacy persistence.
Receipt-schema-2/3/4 authority and retry semantics remain unchanged. Compensating
legacy file writes do not provide crash atomicity across components. Final
independent review accepted 9.6/10; 164 focused tests, 256-client stress and
2,948 full-suite tests passed, with 9 full-suite skips. Evidence and qualification
limits are recorded in the gate's hardening record;
this work alone does not close the remaining release qualification milestones.

## Serial supported-profile follow-up

The [supported-profile freeze](production-profile-hardening-2026-09-18.md) is the
sixteenth serial gate and closes milestone 1 at the reviewed-source gate. The explicit
[`dml-receipted-local-v1` candidate](supported-production-profile-v1.md) fixes
admitted runtime APIs, receipt-journal authority, trusted caller scope, strict
configuration, dependency declarations, limits and retry behavior. Its HTTP
surface admits nine routes with explicit tenant input and startup-bound service
credentials. Experimental features, legacy RAG files, generation, semantic
checkpoints and backend projection operations are outside that selected runtime.
Profile admission on Linux/macOS/Windows and CPython 3.10–3.13 is separate from
demonstrated portability and filesystem/power-loss qualification. Independent
final review accepted **9.6/10**, with **441 focused tests passing**. The final root
full suite passed **3,342 tests**, with **9 skips** and **3 warnings**, in 163.26
seconds; maintained/new-surface Ruff, the 63-file mypy selection, Hermes hygiene
and the diff check passed. The [review record](artifacts/production-profile-review-2026-09-18.json)
and hardening document retain the evidence and limits. Publication and exact-commit
CI remain pending at this source snapshot; outcomes will be recorded in PR #118.
Eleven milestones remain: nine for the first release and two deferred. Exact
model-input and tokenizer budget binding is next; no maturity promotion occurred.

## Serial exact model-input follow-up

Before this gate began, the published stage-16 source `dc85123` passed all nine
jobs in CI run 315 (`35341742222`). Both Linux full-suite jobs passed 3,343 tests
with 8 skips, the production lane passed 394 profile tests, and all six portability
jobs passed. The tested PR merge tree matched the published source tree. These
remote results supplement the earlier local source snapshot; they do not qualify
the later stage-17 implementation or imply that PR #118 has merged.

The [exact-input companion](model-input-hardening-2026-09-18.md) is the seventeenth
serial gate and addresses milestone 3. It verifies a local GPT-2 snapshot, pins the
real tokenizer and fixed messages/tools template, and rejects an oversized complete
input plus output reservation before inference. Its owning consumer dispatches the
immutable compiled token IDs directly. A dedicated pinned CPU lane exercises a real
tiny model and tokenizer with zero permitted skips. Independent review accepted
**9.6/10**, with **256 focused tests passing and 0 skips** in 5.59 seconds.
The final root full suite passed **3,598 tests**, with **9 skips** and **3 warnings**,
in **146.54 seconds**; all 256 new model-input cases passed without skips.
Maintained/strict new-surface Ruff, the 66-file mypy selection, Hermes hygiene and
the diff check passed. The [review record](artifacts/model-input-review-2026-09-18.json)
retains source hashes and separately attributed evidence. Milestone 3 is closed
at the reviewed-source gate. Publication and exact-commit CI remain pending at this
source snapshot; outcomes will be recorded in PR #118. The ledger has ten remaining
milestones: eight first-release and two deferred. Crash/filesystem qualification
is next. The memory profile retains its nine HTTP routes and no
generation, and retrieval token counts remain labeled estimates.


## Serial supported-profile recovery follow-up

Before this gate, corrected stage-17 source `49368a9` passed all ten jobs in
[CI run 317](https://github.com/mmckeen-nv/DML/actions/runs/35348088493), including
256 mandatory model-input cases with zero skips. Those published-source results
supplement the earlier stage-17 snapshot; its review artifact remains historical.
PR #118 remains unmerged.

The [profile recovery gate](profile-recovery-hardening-2026-09-18.md) is the
eighteenth serial gate and addresses milestone 4. It exercises the existing
five-mutation, schema-2/3/4 candidate boundary using actual selected-profile
process termination, seeded changing histories, fault handling and offline
backup/restore. The [recovery contract](profile-recovery-v1.md) records the full
component inventory, operator commands and acknowledged/rejected/uncertain
outcomes. Internal consistency cannot prove absence of a consistent rollback;
independently retained receipts are required. Backup recovery stops at its
captured revision. The selected profile additionally requires a linked SQLite
runtime with the admitted WAL-reset fix.

The independent five-module selection passed **416 tests**, with **zero failures,
errors or skips**, in **114.35 seconds**. The actual six-module recorder run passed
**450 tests with zero skips** in **114.92 seconds**, accurately recording the local
overlay with `fsync=volatile` as unqualified. The root full suite passed **4,048
tests with 24 skips and 3 warnings** in **257.17 seconds**; all 450 regular recovery
cases passed, while 15 dedicated ext4 capability cases and 9 existing cases
skipped. Maintained/strict Ruff, mypy over 67 files, Hermes hygiene and the diff
check passed. Independent software review accepted **9.6/10**, with no blockers;
the [review artifact](artifacts/profile-recovery-review-2026-09-18.json) hashes the
24 reviewed files and distinguishes each evidence source.

Publication and exact-source CI remain pending at this snapshot. All six new
measured-platform lanes and all 15 real ext4 ENOSPC cases must pass before the
environment qualification closes. Local overlay process-failure evidence does
not replace those results, and physical power loss is unclaimed. Milestone 4
remains open, leaving **10 milestones: 8 first-release and 2 deferred**. PR #118
will record subsequent CI results and closure; the next milestone's source
ledger update will carry that history forward.
