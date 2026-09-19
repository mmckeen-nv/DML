# Append-only ingestion receipts: candidate contract

This serial hardening step implements the receipt portion of P03 in the
[productionization plan](productionization-plan-2026-09-12.md). It is an opt-in
candidate for one host, local storage and cooperating trusted service callers.
External projection transactions are the next gate. This is not a declaration
that DML or its other mutation APIs are production ready.

## What a successful call means

`DMLAdapter.ingest_memory_receipted(text, tenant_id=..., idempotency_key=...)`
and `POST /api/remember/receipt` atomically commit one append-only lattice memory,
its historical result receipt and its journal decision. Scope is the exact
`tenant_id/client_id/session_id/instance_id` tuple, including null members.
A key is unique within that full tuple. Repeating the same canonical request and
key returns the original receipt, even after restart or later low-level deletion.
A receipt acknowledges a historical commit; it does not promise continued liveness.

The canonical request includes text, full scope, kind and effective metadata.
It uses strict finite JSON, rejects conflicting metadata and forces `no_merge=true`.
It excludes generated timestamp and embedding. Keys and scope strings are limited
to 256 UTF-8 bytes; the canonical request is limited to 1 MiB. Unknown HTTP fields
are rejected. Two calls with one key and different canonical requests conflict;
a different full scope uses a separate namespace. The HTTP default kind is `note`;
the adapter default is `memory`, so cross-interface retries must specify the same
kind and scope explicitly.

Embedding preparation runs before store ownership; a second receipt check inside
the transaction resolves races. Multiple competing clients may compute embeddings,
but only one memory and receipt commit for their shared key. Capacity exhaustion
rejects the append without eviction. The API does not merge, promote, mirror to
RAG or generate a survival ledger. It cannot participate in a legacy compensating
batch transaction.

A newly committed receipt remains success if runtime hydration fails. Health records
`receipt_runtime` degradation using only an exception type; a later successful
refresh repairs the cache. An exception around commit is reconciled against the
ledger without compensating away a possibly committed receipt. If the ledger
cannot establish an outcome, retry the identical request and key after recovery.
Never infer that a timeout or disconnected client means no commit occurred.

## Opt-in and embedding compatibility

Configure `persistence.enable=true`, `persistence.journal=true` and
`persistence.receipts=true` on a fresh store to select journal schema 2. Existing
schema-1 journals do not silently upgrade. The supported adapter surface is receipted append, historical receipt lookup and
scoped `retrieve_context`. Legacy retrieval/generation front doors are refused
for schema 2 until their ownership and freshness boundaries receive separate review.
The receipt adapter profile also refuses legacy lattice writes and disables automatic lattice persistence/aging, because
those paths have not been qualified to preserve the receipt embedding contract.
Lattice-only semantic checkpoints and JSON snapshot export are refused for schema 2:
they would omit the receipt ledger. Preserve a consistent full SQLite backup and
its identity sidecar using SQLite's backup facility while writers are stopped;
do not copy only the main database file while a WAL is active.

The first append binds an embedding identity and vector dimension in authoritative
metadata. New writes and embedding-based retrieval fail closed when that contract
does not match. Matching dimensions alone are insufficient. Schema-2 queries bypass the legacy
query cache, pin identity across embedding preparation and validate the result
against the current contract before ranking. Custom or model-backed
embedders require an immutable declared identity through
`persistence.receipt_embedding_identity` or `embedder.receipt_embedding_identity`.
The declaration must pin the model revision and preprocessing configuration.
This is an operator assertion, not cryptographic attestation of model weights.
The built-in deterministic random embedder has a versioned descriptor.
Committed retries return before embedding compatibility checks or embedding calls;
they can resolve with an injected unavailable embedder. This does not eliminate
package/model requirements of constructing a default embedder at startup.

## Explicit migration and downgrade behavior

Stop writers before cutover, retain the source, and run:

```sh
dml-journal enable-receipts /absolute/source.sqlite3 /absolute/new-destination.sqlite3
```

The separate destination preserves store identity, revision, records, metadata and
all decisions, translating schema-1 decisions to schema-2 decisions with no receipt.
A durable migration marker is published before destination creation. Every normal
opener rejects an incomplete target. After verification, identity publication and
marker removal publish the target. A killed migration leaves the source intact;
preserve the failed destination and retry into a new path. Do not remove a marker
to force an incomplete target to open.

A populated schema-1 source has no proven embedding identity. Migration preserves
its data but does not invent that provenance: new receipt appends and embedding
retrieval remain blocked until a separately reviewed conversion can establish it.
This migration is format preservation, not an automatic operational cutover for
legacy vectors. There is no in-place downgrade. Older schema-1-only readers reject
schema 2; reverting software uses the retained original source and excludes any
new destination writes. Reconcile those writes explicitly before rollback.

## HTTP failure contract

| Status | Code | Meaning |
| --- | --- | --- |
| 409 | `idempotency_conflict` | The scoped key belongs to another canonical request. |
| 409 | `revision_conflict` | Retry the identical request and key. |
| 503 | `receipt_ownership_unavailable` | Ownership timed out; retry the identical request and key. |
| 503 | `receipt_outcome_unavailable` | Storage cannot establish a trustworthy outcome. Recover, then retry the same request/key. |
| 503 | `receipt_not_committed` | Reconciliation found no receipt at that check; retry the same request/key. |
| 503 | `embedding_unavailable` | Embedding preparation failed; retry the same request/key. |
| 400 | `invalid_or_unsupported_receipt_request` | Request or configured profile is unsupported. |
| 422 | FastAPI validation detail | Invalid HTTP request shape or forbidden extra field. |

Existing bearer authentication applies. Service token holders remain trusted to
select scopes; this endpoint does not introduce per-token tenant authorization.
Error responses omit underlying exception messages and memory payloads.

## Independent acceptance

An independent grading agent rejected the first reviewed revision at **9.2/10**
after reproducing a stale query vector following failed hydration. The builder
removed schema-2 dependence on the unversioned query cache and pinned query identity
through retrieval ownership. A second deterministic interleaving now verifies
that a model change and first commit between preparation and ranking fail closed.
Unqualified legacy retrieval/generation front doors are explicitly refused.

The final bounded feature received **9.5/10**, with **89 targeted tests passing**
including the 256-client setting. The [review record](artifacts/receipt-review-2026-09-12.json)
pins the accepted source hashes and review history. The final local full suite
passed **1,096 tests**, with nine optional-dependency skips. Ruff, the 45-file
maintained mypy selection and Hermes hygiene passed. This is engineering review
of this feature; final-commit CI is a separate integration gate.

## Evidence and limits

The maintained harness kills real subprocesses at all eight receipt transaction
hooks and six migration publication hooks. A parent-held oracle checks acknowledged
receipts, exact replay and explicit incomplete-target failure. Other cases cover
256 simultaneous same-key clients, eight competing processes, quota-full SQLite,
serialization failures, receipt deletion/tampering and forged historical results,
restart replay, scope conflicts, runtime failure and HTTP behavior. CI retains
JUnit evidence and runs the receipt selection in the portability matrix.

The journal validates the full decision and receipt history on normal operations.
That cost grows with history; this feature has not qualified 1k/10k/100k growing
receipt stores, hardware power loss, hostile filesystem writers, multi-host storage,
external vector projections, automatic lifecycle transitions or durable full response
reconstruction. Checksums detect accidental inconsistency, not authorized tampering
with every linked checksum. Each remaining feature requires its own serial gate.
