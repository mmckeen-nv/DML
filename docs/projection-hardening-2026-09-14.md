# Disposable transactional vector projection

This is the next bounded P03 step after the accepted append-only receipt feature.
A schema-2 receipt journal remains authoritative. A separate SQLite projection
publishes its complete live-memory snapshot and source cursor in one transaction.
The source format is unchanged. This is full-snapshot reconciliation, not an
incremental transactional outbox or a distributed commit across two stores.
Existing FAISS/RAG mirroring and native KV are not qualified by this feature.

## Contract

`JournalStateStore.verified_snapshot()` reads verified store identity, revision and
detached state in one SQLite read transaction. `prepare_source()` validates its
embedding contract and computes a canonical SHA-256 digest of the full source
state. The projection stores live records, the embedding contract, and the bound
source identity/revision/digest. It deliberately excludes receipt and lineage
ledgers and cannot serve as an authoritative backup or ingestion store.

`SQLiteProjection.publish()` freezes its input before backend I/O. Target records
and cursor commit together under compare-and-swap. Repeated publication of the
same snapshot has no additional commit. Delayed publishers cannot replace a
newer source revision; equal revisions with different contents or a different
authority identity fail closed. Whole-snapshot replacement propagates deletion
and leaves no stale target row representing a removed source memory.

No source ownership lock crosses target I/O. Source ingestion remains available
when projection writes stall or fail. If a publication acknowledgement is lost,
the publisher reads the target to reconcile the committed outcome. It never
compensates or changes source memory. Other errors remain explicit; an exception
after publication does not establish that publication was absent.

`reconcile()` rechecks authority after target publication. Its
`matches_pinned_source` field refers to that final verified source read, never a
promise of perpetual currency. `published` reports the snapshot this operation
published; a concurrent newer publisher may already have advanced the target.
`projection_status()` performs the same comparison without changing target data.
Status and query errors leave authority intact.

## Query boundary

`query_projection()` freezes the caller's vector, embedding identity and exact
four-member tenant/client/session/instance scope before source or target I/O.
It pins one verified authority snapshot and requires the target's cursor and full
projected contents to match it. Missing, uninitialized, stale, corrupt or mismatched
projections do not serve results. A query can return the pinned older revision if
source changes after its authority read; returned source revision/digest identify
that snapshot. It does not claim freshness at response completion.

The reference backend uses exact cosine ranking, descending score with memory ID
as the deterministic tie breaker. Explicit untrusted, quarantined, superseded and
expired records are suppressed with reasons under a required finite nonnegative
`as_of` timestamp. Queries require a finite, nonzero representable vector norm and
a matching embedding identity/dimension. Stored zero vectors score zero. `top_k`
is an integer from 1 to 1,000. Scope includes null members; null is not a wildcard.
An empty authority without an embedding contract cannot serve vector queries.
Embedding declarations remain operator/backend assertions, not weight attestation.

## Operator commands

The source must already exist and use journal schema 2. The target must have a
separate directory. Existing unrelated journals, even empty journals, are refused.
A projection binds permanently to its first source identity.

```sh
dml-projection /absolute/authority/source.sqlite3 /absolute/projection/index.sqlite3 sync
dml-projection /absolute/authority/source.sqlite3 /absolute/projection/index.sqlite3 status
dml-projection /absolute/authority/source.sqlite3 /absolute/projection/index.sqlite3 query /absolute/query.json
```

The query file contains exactly `vector`, `embedding_identity`, `scope`, `as_of`
and optional `top_k`. Supply the embedding identity recorded by the source,
including `backend`, `revision`, `model` and `mode`. This is an explicit vector API;
it does not silently choose or call a model. Unknown fields fail. Commands return
JSON; errors return exit code 2 and an exception type without raw backend messages,
paths or memory text. Status and query refuse missing targets without creating them.

There is no background worker or automatic adapter routing in this candidate.
Operators explicitly synchronize and inspect status. Receipt success acknowledges
authority only; projection availability is independent. Large-store scheduling,
incremental delivery, backend adapters and provider routing need their own gates.

## Recovery and versions

The projection envelope is `dml-sqlite-projection-v1` inside journal schema 1.
The authority remains journal schema 2. Unknown/mismatched target formats fail
before publication. Do not configure a DML adapter to ingest into the projection.
Reverting this feature leaves authority readable by the accepted receipt version;
there is no source migration or implicit downgrade.

Real process death during target construction either leaves a complete recognized
projection or an explicitly rejected incomplete target. Preserve incomplete/corrupt
targets and rebuild from verified authority into a new dedicated directory. Missing
initialized target databases also fail closed while their identity marker remains.
Never remove identity or coordination files to make an incomplete target appear
healthy. Receipt ledgers remain solely in the source and survive target rebuilds.

Checksums and linked digests detect accidental inconsistency, not a filesystem
writer who can coherently rewrite every record and checksum. Callers of prepared
publication are trusted: query-time comparison against verified authority rejects
a fabricated prepared snapshot. This is not signed remote replication.

## Independent acceptance

The independent grader rejected the initial revision at **9.1/10**. Corrections
closed target-version and mutable-query boundaries and strengthened initialization,
concurrency and staleness evidence. Final contract review also reproduced an
identity-change window before snapshot `BEGIN`; identity and state are now read
and checked inside the same transaction.

Final grade: **9.5/10** for this bounded feature. The final focused harness passed
**65 tests** with 256-publisher stress enabled. The [review record](artifacts/projection-review-2026-09-14.json)
pins source hashes, findings and category scores. Final local integration passed
**1,161 tests**, with nine optional-dependency skips. Ruff, 46 maintained mypy
files and Hermes hygiene passed. Final-commit CI remains a separate
integration gate; this does not qualify all DML features for production.

## Evidence and remaining limits

The independent harness covers all seven target transaction hooks and fourteen
construction/initial-format hooks using actual subprocess death, 256 simultaneous
same-snapshot publishers, eight publisher processes, delayed old snapshots,
quota-full rollback, lost acknowledgements, corruption, deletion, source progress
during backend stalls, foreign targets, frozen requests and pinned query evidence.
CI runs the projection tests across Linux/macOS/Windows and retains JUnit evidence.

Full-source copying, canonical hashing, linear exact ranking and whole-history
journal validation have growing time and memory costs. No 1k/10k/100k growing-store
SLO, real-agent performance gain, hardware power-loss, multi-host storage or existing
FAISS compatibility is claimed. All-component atomicity, lifecycle mutation receipts,
incremental outbox delivery and durable full model-response reconstruction remain
separate production release gates.
