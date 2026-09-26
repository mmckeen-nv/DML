# Supported-profile mixed-operation concurrency qualification

This is **serial hardening gate 19 and release milestone 6** of the
[finite production plan](production-remaining-work-2026-09-18.md).
**Milestone 6 is closed**, with source review at **9.6/10**, all **20 jobs**
in [CI 324](https://github.com/mmckeen-nv/DML/actions/runs/35470900431) passing,
and all three platform artifacts independently verified twice. The fixed
requirements and earlier rejected attempts retain their separate evidence below.
Milestones **1–6 are closed**: six of eleven first-release gates, leaving **five
first-release gates and two deferred milestones**.

The work qualifies the existing `dml-receipted-local-v1` candidate profile on one
host, with cooperating trusted callers and local SQLite authority. It adds no
runtime endpoint, persisted schema, native checkpoint API or maturity promotion.
The profile remains **candidate**, DML remains **alpha**, and
`production_ready` remains **false**. The [supported-profile contract](supported-production-profile-v1.md)
and [recovery contract](profile-recovery-v1.md) define the admitted product and
storage boundaries.

## Existing evidence and the remaining gap

Existing receipt, lifecycle, transaction and recovery tests establish scoped
idempotency, compare-and-swap rejection, revision-pinned reads, deterministic
races and competing processes. The earlier coordinator's 256-client campaign
used at most 32 active worker threads and legacy mutation operations. Its
[accepted evidence](transaction-coordinator-hardening-2026-09-18.md#evidence-and-acceptance-status)
remains valid within that scope; it does not qualify this milestone's actual
overlapping-client matrix or real HTTP behavior.

This gate requires mixed selected-profile operations, independently checked
histories, actual overlapping callers, bounded failure handling, and transport
acknowledgement-loss evidence. Lifecycle admission and shutdown defects found
while doing that work belong to milestone 6; they do not add release milestones.

## Frozen campaign matrix

Every campaign uses one of the admitted journal schemas **2, 3 or 4** and a
requested client count of **1, 16, 64 or 256**. The complete Linux matrix has
**48 schema/campaign/client-count cells**: three schemas, four campaign modes
and four scales. Shared-adapter and separate-adapter threads are distinct
campaign modes, so neither substitutes for the other.

| Campaign mode | Required execution and recorded evidence |
| --- | --- |
| Shared-adapter threads | Actual worker threads reach a synchronized start gate and operate through one shared adapter. Record requested clients, ready callers and observed overlap. |
| Separate-adapter threads | Actual worker threads reach a synchronized start gate and use separate adapter instances against one authority. Record requested clients, ready callers, observed overlap and adapter count. |
| Spawned processes | At most four actual spawned processes host the requested total number of caller threads. Record process count and caller count separately. A 256-client case must not be described as 256 processes. |
| HTTP/provider | A child server runs the selected profile through real loopback sockets. Authenticated clients exercise provider recall and memory routes. Record arrived/in-flight HTTP requests separately from simultaneously executing server workers. |

For a single client, run the complete mixed recipe sequentially. Larger cases
distribute that recipe across the requested callers and prove the declared
ready/in-flight overlap. Launching 256 tasks without observing overlap is not
sufficient evidence. A bounded server thread pool may queue arrived requests;
the report must distinguish transport concurrency from active handler execution.

The recipe includes all five receipted mutations—ingestion, retirement,
supersession, content update and first-level promotion—plus scoped retrieval,
retention inspection and exact same-key replay. Fixed disjoint lifecycle targets
allow independent mutations, while additional forced related-state races test
compare-and-swap conflicts. Wrong-scope calls and authentication failures must
remain isolated under contention.

Completed mixed campaigns use the complete-history checker below. Their callers
may retry an explicitly identified OS ownership timeout using the identical
request, scope and key, within the original campaign deadline. Each logical
intent must ultimately receive its acknowledgement; a timeout or rejection is
not counted as a completed intent. Allow at most **three total attempts** per
logical operation: its first call and at most two retries. All attempts and their outcomes remain in
the recorded history. Targeted rejected-conflict and uncertain-acknowledgement
cases have separate independent
before/after SQL and receipt assertions. They are additional evidence, not extra
matrix cells or a claim that every error case uses the mixed-history checker.

Each campaign burst admits **at most 40 unique mutation commits**, excluding
the seven seed commits and any schema-4 migration baseline. The captured initial
authority revision records setup separately; `committed_revisions` reports
unique burst commits. Additional callers exercise reads and duplicate requests.
This bounds qualification cost while preserving contention and mixed behavior.
Client count, operation count, unique
commits and record count are distinct report fields. Growing-store and
1k/10k/100k-turn measurements remain milestone 9.

## Independent history checking

Record each caller's operation intent, complete scope, invocation and response
interval, terminal response or error, receipt identity, and observed authority
revision. For a retried logical operation, also retain every attempt's invocation
and response times and error code; preserve the original logical invocation and
deadline. Retain enough independently captured state to check reads at their
reported revision and reconstruct the committed history. The checker must derive
expected state from test intents and acknowledged historical receipts, rather
than use the runtime's own mutation or verification helpers as its oracle.

Acceptance requires the checker to verify:

- Every committed mutation appears exactly once, with matching record state,
  scope, provenance and lineage; duplicate requests resolve the original
  historical receipt without creating another commit.
- Rejected conflicts do not introduce state changes. Uncertain responses remain
  unresolved until a same-key retry or independently observed history establishes
  their outcome.
- A read describes one committed revision consistent with its invocation and
  response interval. It cannot combine records from different revisions, expose
  an uncommitted or future revision, or cross the requested scope.
- Non-overlapping completed operations respect real-time precedence. Final
  state alone cannot substitute for checking the intermediate history.

Deliberately corrupted histories must be rejected by negative tests, including
lost or duplicated mutations, invalid receipt identity, incorrect read revisions
and scope leakage. These tests establish that the checker can detect the failure
classes for which the campaign claims evidence. Passing finite histories is not
a proof of linearizability under every possible scheduler.

## Predeclared timing and failure bounds

These bounds are qualification thresholds for the declared fixture and measured
environment. They are not production service-level objectives or performance
promises for arbitrary stores, embeddings or hardware.

| Measurement | Acceptance requirement and domain |
| --- | --- |
| Released mixed burst | All logical intents receive acknowledgements within **90 seconds** of the synchronized release, with **at most three total attempts per logical operation**. Explicit OS ownership-timeout rejections may be retried with the identical request/scope/key inside that same deadline; they do not complete an intent or reset its clock. Record all attempts and outcomes. A test-runner timeout or unresolved intent is a failure. |
| Progress after a deliberate blocker | Observe waiting-client progress within **10 seconds** of releasing the controlled blocker; finish the bounded scenario without a deadlock or stranded caller. |
| Cooperating OS ownership acquisition | Exercise an actual held OS lock using the existing **30-second** ownership budget, allowing at most **2 seconds** for scheduling in that qualification case. Shorter test budgets must be labeled and cannot replace the default-budget case. |

The OS ownership budget starts in the file-lock acquisition routine. It does
not bound earlier adapter-local lock waits, request/embedding preparation,
backend execution, SQL preflight, or the complete request. Admission tracking
does not serialize otherwise concurrent operations, impose a universal
request deadline, or cancel a blocked backend. Fixtures must bound startup,
barrier waits and teardown so a defect produces an actionable failure.

Only the typed `daystrom_dml.store_lock.StoreLockTimeout` may trigger a Python
mixed campaign's automatic retry. On HTTP a mutation's exact 503
`receipt_ownership_unavailable` response is eligible. Retrieval is eligible only
when its 503 detail is exactly
`{"code":"retrieval_outcome_unavailable","reason":"store_ownership_timeout"}`.
The additional reason identifies actual `StoreLockTimeout`; the existing
retrieval response code is unchanged. A generic retrieval error without that
reason and all retention errors remain ineligible.
Generic `TimeoutError`, backend/callback timeouts, unexpected HTTP
errors and transport `ReadError` remain campaign failures. Deliberately injected
transport-loss cases use their separate uncertainty protocol; they do not make
arbitrary failures acceptable in the mixed matrix. Retry attempts, their timing,
error codes and counts must remain visible in both the checked history and the
qualification report.

Forced delay cases must demonstrate progress after lock release, a rejected
related-state race, and a blocked embedding that leaves an independent caller
and public HTTP health able to make progress. These finite schedules do not
establish strict FIFO fairness or absence of starvation under unbounded load.

## Selected-profile lifetime target

The five receipted mutation APIs, `retrieve_context`,
`inspect_memory_retention` and `durability_status` share explicit
operation-lifetime admission: **eight protected public operations**. The
adapter's creator process is checked before taking a lifetime mutex. An adapter
inherited after a fork must reject these operations; the child opens a fresh
adapter. This does not make arbitrary inherited Python or provider state
fork-safe.

Guarding `durability_status` also prevents inherited persistence-lock access and
allows the existing HTTP health handler to report degraded status while the
selected adapter is closing or closed. `production_profile_status` remains an
informational startup-status call outside operation admission; it does not
certify that a closed adapter can accept memory work.

`close(projection_timeout=5.0)` first fences new selected-profile admissions,
then waits for already admitted operations using the existing timeout argument.
Dependencies must remain available to those operations while they drain. A drain
timeout raises `TimeoutError`, keeps admission fenced and leaves dependencies
live. The caller may retry `close` after in-flight work finishes. Backend work is
not forcibly cancelled, and timing out during drain does not mean that an
admitted mutation was rolled back.

The same timeout bounds waiting for another close call's dependency cleanup.
It does not bound dependency cleanup once this caller begins executing it.
Consequently, the timeout is an admission-drain/cleanup-wait budget, not a
guarantee that the complete `close` call returns within that many seconds.

Concurrent close calls serialize dependency cleanup; successful close is
idempotent, and failed dependency cleanup remains retryable. Calling close from
inside an admitted operation's callback or that same thread's dependency-cleanup
callback must fail immediately instead of waiting on itself. A successful close
may not return while an admitted operation can still commit. An exact historical
receipt remains resolvable through a fresh live adapter. These changes apply to the selected
profile; they preserve existing unselected legacy behavior.

## Ownership error and snapshot revision fixes

`StoreLockTimeout` subclasses `TimeoutError` and is raised only when actual OS
ownership acquisition exhausts its configured wait. Existing Python callers
catching `TimeoutError` remain compatible. A timeout raised by code inside an
already acquired ownership context is not relabeled as an acquisition failure.
The selected HTTP receipt handler reserves `receipt_ownership_unavailable` for
the typed exception. A generic callback `TimeoutError` instead produces
`receipt_outcome_unavailable` with `retry_same_key: true`: that callback may have
run after commit. Embedding-provider failures retain the existing
`embedding_unavailable` classification. Neither is an automatic ownership retry
in the mixed campaign. Recall retains `retrieval_outcome_unavailable`, adding
`reason: store_ownership_timeout` only for actual `StoreLockTimeout`. That exact
detail permits bounded same-request retrieval retry; the generic code alone
does not. Retention remains ineligible. Mutation and eligible retrieval retries
both retain the original 90-second campaign deadline and three-total-attempt
limit; transport `ReadError` remains a campaign failure.

A deterministic shared-read regression exposed a separate race. A delayed old
`JournalStateStore.read_snapshot` could overwrite mutable `journal.revision`
after refresh had loaded a newer payload. Reading that mutable field later could
therefore label the newer payload with the older reader's revision.
`LatticePersistence.load_with_revision` carries the validated payload and the
exact revision returned by its own snapshot read together. Refresh uses that
pair when recording the loaded authority; startup retains the existing
payload-only `load()` interface. The regression spans schemas 2, 3 and 4; the independent
review checks this fix as part of milestone 6. It introduces no schema or API
maturity change.

## HTTP uncertainty and authority maintenance

Real socket disconnect and client read-timeout cases must cover controlled
boundaries before and after SQLite commit. A lost response does not establish
that a mutation failed. Restart or reconnect and repeat the complete original
request, scope and idempotency key; the outcome must resolve without an invented
receipt or duplicate mutation. Client cancellation is not server-side rollback.
The test controls these boundaries through child-process test fixtures, without
adding privileged production HTTP routes.

The HTTP qualification retains one shared `AsyncClient` connection pool, with
an explicitly configured client idle keep-alive expiry of **1 second**, below
the server's unchanged explicit **5-second** idle keep-alive timeout. The
evidence records that policy and the HTTP dependency versions. A controlled
real-socket probe reproduced a request selected just before the server's idle
deadline racing connection closure, yielding
`ConnectionResetError` → `ReadError`. With the 1-second client control, the probe
opened a fresh socket and received HTTP 200. A deterministic regression retains
this mechanism check.

That probe demonstrates the controlled idle-close mechanism; it does **not**
establish the exact cause of the earlier 640-operation campaign's transport
failure. The policy is fixed before the final rerun and does not add `ReadError`
retries, reset any deadline, change the three-attempt limit, or promise immunity
to arbitrary scheduling and network loss. A transport error in the mixed
qualification remains a failure.

Exercise cooperating SQLite checkpoint maintenance and verified full-authority
backup with mixed callers, followed by verification or reopening at the captured
revision. A backup is a captured revision, not a copy of all future writes.
Operator migration and backup/cutover requirements in the recovery contract
remain applicable. SQLite WAL maintenance does not enable the profile's excluded
semantic or native-KV checkpoint APIs.

## Development finding and retry clarification

During development, a real 256-client schema-3 HTTP campaign exhausted the
configured 30-second ownership wait. The reproduced case retained a valid
authority history; profiling identified full retained-outbox validation as the
dominant work. It reproduced after the other 256-client campaign stopped, while
the existing full suite was still running, so it cannot be dismissed as a test
scheduling artifact. A separate transport `ReadError` remains an unaccepted
campaign failure. The subsequent controlled keep-alive probe and declared client
policy above do not retrospectively prove that earlier failure's cause; a fresh
complete qualification run is still required.

Before rerunning qualification, the ownership-timeout retry rule above makes
the original completion/documented-rejection requirement explicit: retry only
that identified rejection, retain every attempt, and require eventual
acknowledgement within the original 90-second logical campaign. The 30-second
ownership timeout, 90-second campaign deadline and mixed workload are unchanged;
the 37-unique-commit high-scale recipe is not reduced to obtain a pass. This
clarification is recorded before accepting results. Finite passing campaigns
will not establish that DML has a supported 256-client service capacity.

## Implementation and retained evidence

| Source | Responsibility |
| --- | --- |
| [Profile runtime](../dml_core/daystrom_dml/services/profile_runtime.py) and [adapter](../dml_core/daystrom_dml/dml_adapter.py) | Selected-profile operation admission, creator-process checks and shutdown coordination. |
| [Persistence service](../dml_core/daystrom_dml/services/persistence.py), [store lock](../dml_core/daystrom_dml/store_lock.py) and [profile HTTP handler](../dml_core/daystrom_dml/services/profile_http.py) | Snapshot-bound payload/revision pairs and precise ownership-timeout classification without relabeling callback or backend failures. |
| [Shared concurrency fixture](../dml_core/tests/profile_concurrency_fixture.py) and [thread/process tests](../dml_core/tests/test_profile_concurrency.py) | Bounded mixed intents, independent revision/history checking, shared/separate adapters, spawned callers, corrupted-history negatives and authority maintenance. |
| [HTTP child fixture](../dml_core/tests/profile_concurrency_http_fixture.py) and [HTTP tests](../dml_core/tests/test_profile_concurrency_http.py) | Real loopback provider, measured arrival/worker overlap, disconnect/timeout/cancellation recovery and independent progress while embedding is blocked. |
| [Runtime tests](../dml_core/tests/test_profile_concurrency_runtime.py) and [independent adversarial tests](../dml_core/tests/test_profile_concurrency_adversarial.py) | Lifetime and fork regressions, read/mutation draining, real OS ownership waits, related-state races and independently authored failure probes. |
| [Evidence recorder](../dml_core/scripts/profile_concurrency_evidence.py) | Exact matrix and required-case validation, measured bounds, source/runtime identity, raw-history verification and fail-closed qualification output. |

The three CI jobs use CPython **3.12** and publish artifacts named
`profile-concurrency-ubuntu-latest-full`,
`profile-concurrency-macos-latest-bounded` and
`profile-concurrency-windows-latest-bounded`. Each artifact retains
`evidence.json`, `concurrency.xml` and `histories/*.json.gz`. Histories contain
the synthetic fixture's initial authority, request/response events and final
authority, with checksums recorded in the evidence. They permit independent
replay of the checker; counts and timing summaries alone are insufficient.

The recorder must reject a missing matrix cell, failed required case, unexpected
skip, invalid timing/overlap claim, missing or mismatched history, or source
changes during the run. The POSIX fork-inheritance case may be explicitly skipped
on Windows, where that process-start method is unavailable; the spawned-process
campaigns still must run. Required high-scale Linux cells may not be silently
deselected. Local default-scale runs are development checks, not the complete
Linux qualification.

## Acceptance record

| Gate | Current status |
| --- | --- |
| Source and regression implementation | Complete: lifetime fencing, paired snapshot revisions, precise ownership errors, duplicate-work correction and rejected-campaign diagnostics qualified on source `6647a0d`. |
| Complete campaign matrix and failure cases | Linux **359 passed, 0 skipped / 48 histories**; macOS **335 passed, 0 skipped / 24 histories**; Windows **334 passed, 1 declared POSIX-fork skip / 24 histories**. All 96 raw histories and platform artifacts were independently verified twice. |
| Independent review | Initial **9.2/10** and follow-up **9.3/10** results were rejected. Their repaired snapshots retain their immutable [original](artifacts/profile-concurrency-review-2026-09-19.json) and [follow-up](artifacts/profile-concurrency-followup-review-2026-09-19.json) reviews. The [contention correction review](artifacts/profile-concurrency-contention-review-2026-09-19.json) accepts qualified source at **9.6/10**. |
| Full integration and maintained static checks | Both full-suite jobs in CI 324 passed **4,343 tests with 75 skips each**. Maintained Ruff, changed-test/benchmark Ruff, mypy over **67 source files**, Hermes smoke and all other required CI jobs passed. |
| Exact-source CI and retained artifacts | [CI 324](https://github.com/mmckeen-nv/DML/actions/runs/35470900431) passed **20/20 jobs**. The [qualification manifest](artifacts/profile-concurrency-qualification-2026-09-19.json) binds source/tree, tested merge, jobs, archive/JUnit digests and retained histories. [PR #118](https://github.com/mmckeen-nv/DML/pull/118) remains open and unmerged. |
| Ledger closure | **Serial gate 19 / milestone 6 closed.** The finite plan now has **six first-release milestones closed, five first-release milestones open and two deferred**. `production_ready` remains false. |

### Qualified source and CI 324

Qualified source is
[`6647a0d37b93858da2e14817a7d329b6de347140`](https://github.com/mmckeen-nv/DML/commit/6647a0d37b93858da2e14817a7d329b6de347140).
CI tested merge `30f15f41f1d86c70b4789c8d41cb61ac87f86a55`, whose tree
`2009c1e999e81e0a53db2b54222808bc7468b240` is the same reviewed source tree.
[Run 35470900431](https://github.com/mmckeen-nv/DML/actions/runs/35470900431)
completed successfully with all 20 jobs passing. The
[qualification manifest](artifacts/profile-concurrency-qualification-2026-09-19.json)
retains exact artifact and JUnit digests and verification of the **48 Linux,
24 macOS and 24 Windows raw histories**. The evidence reviewer and independent
grader each verified all three artifact lanes.

The Linux full matrix's maximum burst was **59.321369302 seconds**, and its
maximum first progress was **8.365432448 seconds**, within the original 90-second
and 10-second bounds. Its **230 typed ownership rejections** remain visible;
every logical intent ultimately completed under the fixed retry policy.
Qualification covers the declared finite corpus and platform/scale combinations,
not a 256-client capacity guarantee, arbitrary-scheduler proof or production SLO.
The bounded macOS/Windows lanes do not establish 64/256-client qualification on
those platforms. Source and historical review artifacts remain immutable.

### Rejected CI attempts and current repair

The initial published source `cbb8fb9` finished
[CI 322](https://github.com/mmckeen-nv/DML/actions/runs/35467434181) with **19/20
jobs passed**. Windows negative controls failed. Follow-up review also identified
a same-client causal-order gap when timestamps tied and rejected that snapshot
at **9.3/10**. The corrected checker and negative controls subsequently received
**9.6/10**, with **167 focused passes and 144 retained-history replays** attributed
to that follow-up review.

Corrected source
[`2cc3a3a47c10d99aae12e186e9913c52621cda4b`](https://github.com/mmckeen-nv/DML/commit/2cc3a3a47c10d99aae12e186e9913c52621cda4b),
tree `e4951b20ef071d003058513488b9545bbdf4813a`, then finished
[CI 323](https://github.com/mmckeen-nv/DML/actions/runs/35468625699) with **19/20
jobs passed**. macOS qualified all **24 bounded cells** with **300 passes**;
Windows qualified all **24 bounded cells** with **299 passes and one declared
POSIX-fork skip**. Their retained artifacts were verified. These results do not
qualify the larger Linux matrix.

Linux reported **322 passes and two failures**: the 256-client, schema-3,
separate-adapter thread campaign expired its original **90-second** deadline,
and the 256-client, schema-4 HTTP campaign exceeded the **10-second** first-progress
bound. The exact latter timing was not retained in the available failure
diagnostic and is not invented here. Both full-suite jobs passing does not waive
either failure. The subsequent repair removed duplicate validated scans/serialization
without weakening validation and added rejected-campaign diagnostics. The original
workload, deadlines and retry limits remained in force. The correction review below
accepted that source; CI 324 above subsequently completed qualification.

### Correction evidence before final review

The runtime correction removes one duplicate owned refresh scan: the measured
changed-authority path goes from **five validated scans to four**, while the
unchanged-authority path remains at four. Private SQL outbox validation avoids a
redundant clone while retaining the same **3,850 checked objects and 950 semantic
validations**. The reproducible
[diagnostic provenance](artifacts/m6-validation-profile-provenance.json) and
[profiling script](artifacts/m6-validation-profile.py) record a local single-client
observation of **1.113 → 0.877 seconds**. This is diagnostic evidence, not a
capacity claim or qualification result.

A focused three-cell run preserved all **336 source-file hashes** before/after:

| Campaign | Burst seconds | First progress seconds | Logical events / unique commits | Typed retries |
| --- | --- | --- | --- | --- |
| Native callers, schema 3, 256 clients | 31.410857234 | 2.232992435 | 640 / 37 | 0 |
| HTTP, schema 4, 256 clients | 40.206062539 | 5.22891469 | 640 / 37 | 1 |
| Four spawned processes, schema 3, 16 clients | 0.597604161 | 0.04367761 | 40 / 7 | 0 |

These cells met the existing bounds on that frozen snapshot. The grader
independently replayed all three histories: **1,320 logical events and one typed
HTTP rejection**. A later cleanup-only shared-budget correction has also been
reviewed; the earlier cell measurements do not qualify that final source tree. The
[contention correction review](artifacts/profile-concurrency-contention-review-2026-09-19.json)
accepts final correction source at **9.6/10** for publication and renewed CI.
At that source-review point milestone 6 remained open. CI 324 above subsequently
qualified the final source with the original workload, validation requirements,
deadlines and retry limits intact.

The correction adds **35 focused controls**: 10 native, 2 HTTP, 8 runtime,
13 independent and 2 evidence cases. Reported passing selections are **44 native**,
**28 HTTP plus one real-network smoke**, **146 existing runtime/persistence plus
8 new runtime**, **39 independent**, and **109 evidence** cases. These selections
overlap and are not summed into one full-suite result. Root maintained Ruff,
changed-test/benchmark Ruff, mypy over 67 source files and Hermes smoke passed.
The predeclared renewed-CI collection was **359 cases for the full mode and 335
for the bounded mode**, permitting only Windows' declared POSIX-fork skip.
CI 324 delivered those collections, with completed results recorded above.

### Original local source-review evidence

The original reviewed local selection passed **318 tests with zero failures or
skips in 572.87 seconds**, using Linux, CPython **3.12.14** and linked SQLite **3.53.1**.
Its 48 checked histories contained **10,320 logical events** and **107 explicitly
recorded ownership rejections**; every logical intent completed, with at most
**two attempts** used. The slowest campaign took **50.558 seconds**, the slowest
first progress was **6.658 seconds**, and HTTP measurement reached **256
in-flight requests / 40 active workers**. Six actual default-budget OS lock
timeouts measured **30.043–30.065 seconds**; release-to-progress measured
**0.009815 seconds**. These measurements satisfy the declared local test bounds
and do not establish production capacity or an SLO.

That local evidence recorder correctly retained `accepted: false`. Its only reported
qualification errors were the unpublished dirty working tree and the local
overlay filesystem with `fsync=volatile`. The separate 23-file before/after hash
comparison established that the tested files did not change during the run;
it does not turn an unpublished tree or unqualified filesystem into accepted
exact-source platform evidence. The grader independently replayed all 48
retained histories and verified the exact 318 passing cases before accepting
that snapshot at 9.6/10. Subsequent CI rejections above retain their separate
attribution; they do not replace the later exact-source qualification in CI 324.

The [independent review](artifacts/profile-concurrency-review-2026-09-19.json)
retains reviewed-source hashes and separately attributed evidence. Preliminary development selections
may overlap; they will not be summed into an inflated final test count.

CI 324 ran the complete **48-cell Linux matrix** and a bounded **24-cell
selection on each of macOS and Windows**, using all four campaign modes and
three schemas at 1 and 16 clients. Reports must name the exact
OS/schema/transport/scale combinations actually run; a smaller portability
selection cannot be described as 64/256-client qualification on those platforms.
Retain raw histories or their verifiable artifacts, measured timing and overlap,
configuration/runtime identities, source identity and test outcomes. A configured
CI lane does not demonstrate a passing run.

This gate does not qualify multi-host or network-filesystem coordination,
noncooperating writers, arbitrary backend cancellation, physical power loss,
unbounded-load fairness, or full production readiness. Existing recovery
qualification remains credited, and milestones 7–11 retain their own remaining
acceptance criteria. Native-KV compatibility and broader legacy orchestration
remain the two deferred milestones.
