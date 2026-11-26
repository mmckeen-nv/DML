# Receipted first-level promotion and merge

This twelfth serial gate extends P03's durable mutation boundary and P07's
explicit lifecycle semantics. The feature remains a candidate. Completing this
gate does not graduate DML or complete the original ten production areas.

## What is committed

`DMLAdapter.promote_memories_receipted(sources, *, text, reason, idempotency_key,
tenant_id, client_id=None, session_id=None, instance_id=None)` derives one new
level-1 memory from explicitly selected level-0 sources. One source is a promotion;
multiple sources are a merge into one promoted result. The caller supplies the
result text and reason. No summarizer, similarity threshold, automatic conflict
adjudication or factual-verification model decides what to merge.

Each source is an exact object containing `memory_id` and
`expected_memory_digest`, calculated with `services.receipt_lifecycle.memory_digest`.
There must be 1–32 distinct source IDs. Canonical requests sort them by ID, so caller
ordering does not change intent. Requests use version `dml-promotion-request-v1`
and are frozen as finite JSON with the existing 1 MiB request bound, strict scope
and 1,024-byte reason bounds. Full records, not only their text, are preconditions.

The narrow `services.receipt_promotion` service appends a fresh record and commits
the receipt, journal decision (`promote-receipt-v1`) and enabled outbox event in one
transaction. Journal schemas 2, 3 and 4 and existing receipt/event envelopes remain
unchanged. The receipt acknowledges the new record. Every source, unrelated record,
existing lineage entry and historical receipt remains unchanged.

The new ID is greater than every live/lineage ID and respects the persisted
allocator. `summary_of` and `children` contain the sorted immediate source IDs.
The new record's timestamp, salience and fidelity are the respective minima across
its sources. Derivation therefore adds no freshness, salience or fidelity boost.
Capacity increases by one; full stores reject instead of evicting sources.

## Authority and provenance

All sources must be live level-0 records with no nontrivial prior ancestry. Empty
or self-only lineage is accepted; previously promoted or abstracted records are
refused. This gate does not qualify recursive promotion.

All four receipt scope fields must match exactly, with absence distinguished from
an explicit null. The service checks accessibility of every source before checking
digests or accessing a model. Missing, lineage-only and wrong-scope sources return
the same not-found outcome. A changed source conflicts, including typed metadata
changes that Python equality might otherwise equate.

Sources must explicitly declare `source_trust` as `trusted` or `verified` and be
eligible for normal retrieval. Both lifecycle-state aliases are checked separately;
an active primary field cannot hide a suppressed or terminal legacy state.
Retirement/supersession decisions, quarantined namespaces, expiry and malformed
authority values cannot be erased by deriving a new record. Eligibility and every
source digest are checked again on each commit attempt.

Multi-source merges refuse `no_merge` protections, malformed merge flags and
`merge_policy=never`; there is no override. Singleton promotion preserves the
source's merge prohibition. Normal receipt ingestion sets `no_merge=true`, so those
records support singleton promotion but cannot be merged. Multi-source merge is
available for eligible explicitly imported/migrated records. This gate introduces
no policy-change endpoint or implicit permission grant.

Source metadata must match as canonical typed JSON after excluding only these
source-specific or historical fields: `summary`, `source`, `source_id`, `source_ids`,
`provenance` and `content_update_decision`. Every remaining field, including unknown
fields, claim values, expiry, project/thread boundaries and authority markers,
must agree. Unknown prior content-update decision formats are refused.

The output copies the compatible metadata and sets `no_merge=true`. Historical
source summaries are omitted so context uses the supplied result text. It also
stores `promotion_decision` with version `dml-promotion-decision-v1`, the reason,
and a sorted `sources` array. Each entry contains `memory_digest` and the complete
original `memory` record. This proof preserves all source provenance, text,
vectors and prior decisions without overwriting shared lineage entries. The proof
and final output each have a 1 MiB canonical JSON bound; nothing is silently cut.

These are operator-supplied trust labels and reproducible provenance, not evidence
that the supplied text is true or follows logically from its sources. A derived
memory is an independent snapshot with its own lifecycle. Later source edits,
retirement or supersession do not automatically retract it. Operators must retire
or supersede derived records explicitly when needed. Historical proofs and already
built context packets retain the earlier content.

## Model preparation and retries

The existing declared embedding space is required; no identity adoption, dimension
conversion or model migration occurs. Model preparation runs outside store ownership.
Identity descriptors and returned vectors are detached from callback-owned objects,
with finite float32 values, exact dimensions and strict declared identity checked
before/after preparation and inside each commit attempt.

The service uses at most three CAS attempts. Unrelated writes can be rebased; any
changed source or embedding contract invalidates the prepared derivation. Guard
failures and model failures reconcile a competing receipt, including a commit
between an initial lookup and a subsequent snapshot. Lost acknowledgements never
trigger compensating deletion or reversal of a durable append.

All receipted operations share the same scoped key namespace. An exact retry
returns the original receipt before accessing current sources, clocks, capacity or
models. A different request under the same key conflicts. Receipt validation checks
the embedded source digests and the requested result/lineage/authority invariants.
Source-proof vectors must share one nonempty historical dimension, and the result
must have that same dimension, without consulting the current backend contract;
a generic receipt cannot acknowledge a different derivation. Separate keys describe
separate intentional derivations and may produce separate records from the same
sources; there is no automatic semantic deduplication.

Hydration or observability failure cannot revoke a durable receipt. Later owned
retrieval refreshes authoritative state. Mutation adds no projection/backend I/O;
projection and outbox delivery carry the preserved sources and new result through
their existing revision checks.

## API and qualification

`POST /api/memory/promote/receipt` exposes the same arguments under existing bearer
authentication. Strict nested models reject coercion and extra fields. Missing or
wrong-scope sources return 404, stale/lifecycle/key/capacity conflicts 409, invalid
requests 400/422, and unavailable embedding/ownership/durable outcomes 503. Fixed
error codes omit backend/storage exception details. Token holders remain trusted
to select scopes; per-token tenant authorization is unchanged.

The adapter requires opt-in receipt mode and refuses nested compensating
transactions and external RAG mirroring. No legacy automatic promotion path is
enabled by this API. The recovery harness covers process death during preparation
and receipt/outbox mutation hooks, duplicate clients, competing processes and
independent overlapping derivations across schemas 2, 3 and 4. CI retains the
recovery and 256-client concurrency results and runs Linux/macOS/Windows coverage.

The [independent review](artifacts/promotion-review-2026-09-17.json) rejected the
initial implementation at 9.1/10 for insufficient vector-dimension validation on
historical receipts. The corrected implementation binds the result dimension to
its immutable source proof and was accepted at **9.6/10**. All **285 focused cases**
passed: 131 service, 83 adapter/API, 38 recovery/concurrency and 33 independent
adversarial cases. The separate 256-client campaign passed for every supported
schema. Parameter identifiers are bounded to 211 UTF-16 units including pytest's
lifecycle suffix, preserving Windows coverage for oversized inputs.

The final full local suite passed **2,501 tests**, with 9 optional-dependency skips
and 2 known warnings, in 266.38 seconds. Ruff, all 55 maintained mypy source files
and Hermes hygiene passed. This was the single full-suite run after source freeze.

Three builders separately owned the service, adapter/API and recovery harness;
an independent agent graded the completed feature. Nine source/test/CI hashes pin
the accepted code. Exact-commit CI is recorded on PR #118. No user memory stores
were modified. Process death does not qualify physical power
loss. Full snapshots/proofs/events retain growth costs; recursive promotion,
cascading invalidation, retention and erasure, remaining adapter extraction,
native KV compatibility, real-agent outcomes and complete decision replay remain
separate production gates.
