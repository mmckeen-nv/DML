# Transaction and persistence coordinator boundary

This fifteenth serial gate advances persistence decomposition and ownership in
P03 and original areas 2, 3 and 9. It is milestone 2 of the
[finite remaining-work ledger](production-remaining-work-2026-09-18.md), not a new
release milestone. The bounded candidate passed independent review at 9.6/10 and
the local integration gates below. It introduces no storage schema, endpoint or maturity
promotion.

## Responsibilities and compatibility

`services.transactions.TransactionCoordinator` owns thread-local nesting,
outer-operation labels, ownership/refresh sequencing and the shared
rollback-capable mutation frame. `services.persistence.PersistenceCoordinator`
owns component snapshot/restore, changed-state refresh and lattice/RAG commits.
Its `PersistenceState` owns persistence/refresh locks, observed component stamps
and process-local durability failures. `LatticePersistence` retains the selected
JSON, JSONL or journal encoding and authority interface.

The adapter composes those services through explicit capabilities and keeps thin
compatibility methods for existing call sites and interception points. Existing
imports of `PersistenceCommitError` and `PersistenceRollbackError` remain valid.
Legacy serialized mutations and `atomic_batch` use one rollback implementation. The
ownership-only `mutation_transaction` remains distinct: it serializes and
refreshes but does not independently snapshot, commit or roll back a caller's body.

The extraction preserves the legacy mutation model, including existing provider
work inside legacy ownership. It does not move every legacy model call outside
the lock or complete legacy lifecycle/retrieval extraction. Receipt preparation
and schema-2/3/4 transaction behavior retain their existing boundaries.

## Ownership, nesting and rollback

An outer scope acquires the existing cooperating-writer lock, sets its thread's
depth and operation label, and refreshes external state before yielding. Nested
scopes on that thread reuse ownership and the outer operation label. They neither
reacquire the process lock nor refresh again. Other threads and processes must
acquire their own ownership. Exit, including failed refresh, restores nesting and
the prior operation label so a stale label cannot affect a later write.

An inherited `TransactionCoordinator` detects a changed process ID after a fork
and rejects transaction/mutation entry with `RuntimeError` before ownership
callbacks or the caller's body run. The child must create a fresh adapter.
Inherited runtime, locks and thread-local state are not reset or adopted. This
boundary prevents inherited nesting from bypassing ownership; it does not qualify
the adapter, providers or runtime as generally fork-safe.

A rollback-capable mutation captures its baseline only after the refresh and
tracks durable components touched by its own frame. A batch performs its final
persistence inside that frame. An exception from the body or final write restores
runtime state and attempts compensation while outer ownership remains held.
This includes `KeyboardInterrupt` and `SystemExit` through `BaseException` handling;
it cannot run after abrupt process death.

Every inner frame passes its touched components to the enclosing rollback frame,
including when inner compensation succeeds. The inner baseline can contain
uncommitted outer changes: restoring that baseline may publish those changes.
An eventual outer failure must therefore compensate from the outer baseline.
Failed inner compensation also retains component tracking so outer rollback can
retry it. Nesting does not create independent database savepoints or a crash-safe
transaction spanning the component files.

Successful compensation re-raises the original exception object. Failed
compensation records the root rollback failure and raises
`PersistenceRollbackError`, retaining `original_error` and `rollback_error` and
chaining from the original error. A wrapped persistence error exposes its cause
as the rollback error. This is a degraded outcome, not confirmation that the
pre-mutation durable state was restored; it does not install a permanent poisoned
adapter state or guarantee that subsequent operations repair every component.

## Component persistence and uncertain publication

The component coordinator snapshots the public lattice, legacy RAG and optional
persistent-RAG capabilities. Rollback restores runtime snapshots and invalidates
the query cache before attempting durable compensation in lattice,
persistent-RAG, legacy-RAG order. A failure can interrupt that compensation
sequence; the error and health state remain visible. This is best-effort recovery
under ownership, not an all-component atomic commit.

The independent adversarial review reproduced a pre-existing gap: a component
writer could publish and then raise `OSError` before the adapter marked it as
committed. Runtime rollback then left rejected content durable. This gate adds
failure reconciliation before deciding which components require compensation:

- JSON/JSONL and RAG file writes fingerprint the prior authoritative file or
  persistent-RAG manifest. On failure, a changed or unreadable/unknown result is
  tracked for compensation. Equal timestamp/size stamps alone cannot prove that
  publication did not occur. This adds a full prior-file read to ordinary file
  writes; it is not a performance optimization.
- A failed journal write verifies the current revision and snapshot. The
  expected next revision must contain the attempted payload before compensation
  may use that revision. An explicit compare-and-swap rejection does not authorize
  rebasing onto a peer's commit. Diverged or unverifiable authority produces a
  rollback failure instead of overwriting it. Successful journal writes add no
  reconciliation read.

These checks apply to legacy compensation. They do not change schema-2/3/4 receipt
atomicity, historical idempotency or outbox commits. A committed receipt whose
acknowledgement is lost remains committed and resolves through the existing retry
key; it is not undone by the legacy compensation path. Receipt profiles continue
to reject legacy mutation batches and lattice-only persistence/checkpoints.

Changed-state refresh and existing process-local health reporting move behind the
component service. The adapter retains reload logging/metrics. Missing initialized
lattice authority still fails explicitly; failed hydration must not advance the
observed stamp or clear the receipt-runtime degradation marker. The extraction
does not introduce a durable recovery log, change ordinary file refresh to a
content-addressed protocol, or qualify arbitrary noncooperating file writers.

## Evidence and acceptance status

Baseline characterization covers public success, provider/persistence failure,
nested ownership, batches, reopened state and receipt compatibility across JSON,
JSONL, journal schema 1 and receipt schemas 2, 3 and 4. Independent adversarial
cases target publication-before-error, misleading stat stamps, nested compensation,
control-flow exceptions, authority divergence and refresh/hydration failure.

The recovery selection checks ownership across threads and spawned processes,
waiting peers during compensation, legacy process death between lattice and RAG
publication, and receipt process-death/acknowledgement boundaries. A killed legacy
batch can leave a published lattice with an unpublished RAG component: that is an
explicitly tested limitation, not a passing claim of cross-file crash atomicity.
Receipt pre/post-commit recovery uses the separate authoritative SQLite contract.

The mixed-client harness uses separate initially stale adapters for ingestion,
atomic batches, owned memory ingestion and aborted-batch retries, checked against
an independent committed-history oracle. The CI stress setting requests 256
adapter clients with at most 32 worker threads active at once; it is not evidence
of 256 simultaneously executing threads or a complete HTTP/provider concurrency
qualification.

| Evidence | Final acceptance record |
| --- | --- |
| Independent review | Accepted at **9.6/10**, no unresolved blocking findings; [review and source hashes](artifacts/transaction-coordinator-review-2026-09-18.json). Acceptance requires at least 9.5/10. |
| Focused service/integration/recovery/adversarial tests | **164 passed** independently: 130 new cases plus 10 existing fail-closed persistence and 24 receipt-adapter cases. |
| Mixed-client stress | **256 clients passed** in 23.64 seconds, maximum 32 active workers; one expanded run against frozen source. |
| Full local integration, lint, types and Hermes hygiene | **2,948 passed, 9 skipped** in 145.63 seconds; maintained Ruff, strict Ruff on the two services and five new test files, mypy on 59 maintained source files, and Hermes hygiene passed. |
| Exact-commit remote CI | Pending publication and exact-commit CI at this source snapshot; outcomes will be recorded in [PR #118](https://github.com/mmckeen-nv/DML/pull/118). Remote success is not inferred from local checks. PR remains unmerged. |

CI adds all five new test files to Linux/macOS/Windows portability on Python
3.10/3.13 and a separate production-evidence JUnit record. The existing full
regression, recovery, retrieval/context and prior stress selections remain in
place. Configuration of a CI lane does not establish that the lane passed.

## Qualification limits

The supported first-release profile and its crash/filesystem, mixed-operation
concurrency, migration, exact model-input budget, durable replay and live-agent
value gates remain open in the finite ledger. Legacy compensation is **not
crash-atomic across files** and does not cover runtime KV, DPM/DCN or all external
projections as one transaction. This gate provides no power-loss, network
filesystem, multi-host, full cancellation/deadline, starvation or lock-latency
guarantee. File fingerprint cost and journal history-validation cost require
growing-store evidence. Physical erasure and additional experimental features
are outside the finite release scope; native-KV identity qualification and
remaining legacy retrieval/lifecycle extraction are deferred beyond it.
