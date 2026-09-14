# Production contract v1: scope and implementation status

The smallest target is **durable, scoped, attributed memory in; bounded,
attributable context out, with explicit recovery and version compatibility**.
This is the production target, not a declaration that the current alpha meets it.
`GET /api/contracts` and `/health` expose the current maturity inventory.

## Scope

One host, local filesystem, cooperating trusted callers. Memory scope is the exact
tenant/client/session/instance tuple. A service bearer token does not authorize
individual tenants: holders are trusted service callers. Native KV reuse,
automatic promotion/abstraction, DPM/DCN, ANN and fabric routing remain experimental;
unimplemented transfer data planes remain research. No feature is promoted to
stable merely because it has been refactored or passed a smoke test.

## Required release behavior

- Acknowledged ingestion survives restart. Memory, provenance, lineage, lifecycle
  and operation receipts share one authoritative commit. A disconnect after
  commit is an uncertain outcome; idempotent retry must resolve it.
- A rejected mutation cannot become visible. Partial external projections cannot
  masquerade as complete authoritative memory. Explicit tombstones explain
  deletions, retention and eviction.
- Exact retrieval pins scope, query/vector and embedding identity, store revision,
  policy, effective clock, filters, ordering, tokenizer and budget. Evidence
  remains data; retrieval never confers instruction authority.
- The final model input, including chat-template/tool framing and reserved output,
  fits its budget under the pinned tokenizer or fails before inference.
- A corrupt, missing initialized, unknown-version or incompatible store fails
  explicitly. No silent fallback to an older/empty store or permissive native KV
  restore is allowed.

## Guarantees implemented in this tranche

### JSONL and snapshot loading

The JSONL v1 reader requires a v1 header, count and matching checksum. It validates
record types, finite float32-compatible vectors, uniqueness and any declared
record schema version. It rejects empty/truncated files and malformed records.
Existing unversioned records under the documented v1 envelope remain readable;
new serialized records include `schema_version: 1`. Legacy JSON snapshots are
validated before runtime construction. Startup no longer swallows corruption
and continues with an empty or older DML store. Failed initialization stops workers.
A missing path on first creation remains distinct from deletion after a live
adapter has observed state. Journal identity markers also detect deletion across
restarts, provided the marker itself is retained.

### Journal v1

`JournalStateStore` uses SQLite WAL/FULL synchronous commits. Each transaction
updates live records, metadata, optional snapshot and a checksummed decision row.
`read_snapshot()` returns `(revision, detached_payload)` from one SQLite read
transaction. `save(..., expected_revision=n)` rejects a stale revision without
changing state. Adapter writes supply the observed revision and operation name.
Read and write validation checks the committed state digest and record count.
The identity sidecar binds a storage location to the database identity; ordinary
reads/writes use SQLite `mode=rw` and cannot create a missing database.

Journal hardening additionally serializes first creation with a dedicated,
persistent per-database initialization lock. Canonical absolute paths keep
subsequent operations bound to the same location. A complete existing schema-1
database is validated before a missing identity marker may be published, preserving
compatibility with the initial schema-1 implementation. An already-open handle
rejects a missing or changed marker. The database, identity and lock files must
remain together; export refuses to overwrite these coordination files.

Every constructor, load, stamp, save, decision-page read and passive checkpoint
validates the state revision, contiguous decision history, replayed record digests
and counts, and archived snapshot against its matching commit decision. Revision
zero cannot conceal prior committed state. This currently scans **the entire
decision history**, even for a stamp or one decision page: CPU and memory costs
grow with history. It is a correctness-first implementation, not a long-horizon
performance qualification. The isolated history-cost harness measures this cost;
any future validation cache must retain corruption detection before adoption.

The adapter pins startup ownership to the revision it actually loaded, even if a
peer commits before startup finishes. CLI import uses compare-and-swap revision
zero, so a competing journal client cannot have its first commit overwritten.

The journal transaction currently covers the lattice, **not** persistent RAG,
DPM/DCN files or runtime KV state. Normal JSON/RAG write compensation still exists.
Append-only client receipts are an opt-in schema-2 candidate described below.
A transactional external projection outbox is not yet implemented. Journal schema 1 is therefore a production candidate, not a completed
all-component transaction contract. Checksums detect accidental corruption; they
are not signatures against a writer with filesystem access.

### Append-only receipts (journal schema 2)

The opt-in receipt endpoint commits a memory, a scoped idempotency receipt and a
journal decision atomically. Identical retries return the historical receipt;
conflicting requests fail. Schema 2 binds embedding identity/dimension and refuses
legacy adapter mutations and lattice-only checkpoints/exports. Migration is explicit
and side-by-side; populated legacy stores cannot invent missing embedding provenance.
See the [receipt contract and recovery procedure](../receipt-hardening-2026-09-12.md)
for the supported profile, HTTP outcomes and qualification limits. External RAG
projection transactions and receipts for other lifecycle mutations remain gates.

### Disposable snapshot projection

An explicit SQLite reference projection atomically publishes live memory records
and the source identity/revision/digest. Full-snapshot reconciliation uses existing
schema-2 authority; it introduces no source migration. Pinned vector queries refuse
stale or corrupt targets and apply exact scope/lifecycle checks. This is disposable
query data, not a receipt backup or incremental outbox. External FAISS routing and
all-component atomicity remain unqualified. See the [projection contract](../projection-hardening-2026-09-14.md).

### Coalesced projection delta delivery

The reference backend also accepts versioned coalesced deltas with checked base
and target envelope digests. Changed records and deletions publish atomically with
the source cursor; stale bases are refused. Explicit receipt-adapter sync/status/query
methods and CLI `sync --incremental` integrate the backend without coupling receipt
success to projection availability. This reduces transported record payloads while
retaining full snapshot/history verification; it is not a durable operation outbox.
See the [delta contract](../projection-delta-hardening-2026-09-14.md).

### Retrieval and context

The adapter delegates lattice persistence, query caching, context compaction, lifecycle filtering and
retrieval evidence to `daystrom_dml.services`. The query cache coalesces exact-input
misses, returns immutable vectors and invalidates in-flight cache publication on
clear. Scoped retrieval runs under the adapter's cross-process ownership boundary,
with embedding I/O performed beforehand. `as_of` pins scoped ranking and expiry;
equal scores break ties by memory ID. Suppression happens before scoped top-k.
Background adapter aging uses the mutation boundary instead of mutating behind it.

Explicitly superseded/expired/quarantined/untrusted records are omitted by default,
including from recent fallback and survival-ledger admission. Trusted operator
inspection can still request quarantined material; that is not promotion. Merge
eligibility preserves trust/lifecycle/expiry and explicit claim key/value boundaries.
Merged source identifiers and child lineage remain available. The system does not
infer that two arbitrary natural-language claims are compatible or independently
verified; the adversarial corpus tests explicit policy metadata and provenance.

Legacy context compaction counts its rendered heading, sources and DPM prefix;
reported estimates are no longer clamped to hide overflow. These counts remain
estimates and are labeled as such. DCM's `admit_context_segments` accepts
`rendered_token_counter` plus `tokenizer_identity`; the callback must count the
complete rendered messages with the actual pinned tokenizer and chat template,
including any externally supplied tool framing. It rejects overflow and records
the exact result in the packet. Existing callers are not automatically upgraded
to exact counting. The harness must not append unchecked content afterward.

### Checkpoints and evidence

Checkpoint names are collision-resistant; checkpoint publication/pruning uses a
shared directory lock. Failures remain visible through `CheckpointManager.status`
and provider health. Adapter checkpoint snapshots take the memory ownership lock.
Native checkpoint compatibility gates remain enabled and experimental.

Semantic checkpoint managers now have a terminal, bounded shutdown boundary.
`close(timeout=1.0)` stops admissions and returns `True` only after manual work and
the periodic worker drain; `False` explicitly means work is still running. Python
cannot forcibly cancel a blocked provider. A provider that had not reached
publication admission cannot publish after close; a previously admitted write
may finish while close reports `False`. Retry close to observe complete drain.
`DMLAdapter` owns the same manager even with periodic checkpoints disabled and
raises `TimeoutError` on incomplete drain before closing the provider's store.

New filenames bind a publication-order number and SHA-256 content digest while
preserving the existing provider JSON shape. Retention validates all recognized
new files before deleting any, orders them by publication rather than wall-clock
mtime, and preserves legacy/unrecognized files outside the configured quota.
A corrupt recognized candidate stops pruning and degrades status. This can leave
extra files on disk until an operator inspects or repairs them. A successful write
returns its path even if later retention fails; `publication_outcome` and
`retention_error` report the two outcomes separately. A write exception after
replacement reports uncertain durability and preserves older checkpoints.

Publication order does not establish semantic freshness of an opaque provider
snapshot. A returned path acknowledges publication, not a retention lease: another
cooperating manager may subsequently prune it. These status fields are process-local
diagnostics, not a durable operator audit log. Provider exceptions are propagated
to the caller, while logs/status include only their type and stage, never their
arbitrary message or traceback. See the [checkpoint hardening record](../checkpoint-hardening-2026-09-12.md).

Journal commit decisions are durable and paginated with `dml-journal decisions`.
Retrieval responses include a deterministic context digest, returned IDs, suppressed
IDs/reasons, scope, effective time, query/vector digests and journal revision where
available. These response traces are not yet stored durably. They explicitly say
that replay inputs are incomplete: hashes alone cannot recreate the original query,
model identity, source bytes or actual rendered context. Full operator replay and
bounded audit retention/export remain release gates. No payloads are newly logged.

## Migration and recovery

Initialization has explicit process-death outcomes. Before database creation,
retry can create a new journal. After file creation but before schema commit, the
incomplete schema-0 file is preserved and rejected with `JournalSchemaError`;
inspect it and choose a new location for a fresh store, or recover from a verified
source. The schema-0 upgrade command accepts a valid legacy layout, not arbitrary
incomplete files. After the schema transaction commits, restart verifies all state
and can finish publishing the identity marker. No acknowledged memory operation
occurs before construction succeeds. Keep incomplete files for diagnosis; do not
remove markers to bypass a recovery-required state. See the [serial journal
hardening record](../journal-hardening-2026-09-12.md) for the tested boundaries.

Schema-0 journal fixtures are pinned to commit
`48b2ba21ebc4f2b479b72062557afdf9b4398be4`; they are not mislabeled as a released
semantic version. Unknown/newer journal schemas are rejected before mutation.

Stop all writers before upgrade or switching storage paths. Retain the source and
its sidecars as the rollback copy. Run:

```sh
dml-journal upgrade /path/old.sqlite3 /path/new.sqlite3
dml-journal export /path/new.sqlite3 /path/verified-export.json
dml-journal decisions /path/new.sqlite3 --after-revision 0 --limit 100
```

Upgrade reads a consistent schema-0 snapshot, writes it in one target transaction
and verifies it; it refuses an existing destination. It leaves the old database
unchanged. Only switch configuration to a verified target. A killed upgrade can
leave an incomplete target; it must not be selected based solely on file existence.
Validate by opening/loading and inspecting its commit decision; retain the source.
There is no in-place downgrade. Export validated state for older JSON tooling, or
restore the source with the old binary and explicitly account for subsequent writes.
Do not simply relaunch an old binary against schema 1.

If a journal is corrupt/missing, stop writers, preserve the database, WAL, SHM and
identity sidecar, and restore a verified backup/export into a separate location.
Never delete the identity marker to make startup appear healthy. A persistent
corruption or missing-authority error is recovery-required, not an empty result.

## Verification and remaining gates

The CI selection includes real subprocess kills at every journal hook, quota-full
SQLite commits, serialization/corruption/schema cases, legacy migration, concurrency,
context framing, cache invalidation and an eight-case adversarial corpus.
The independent baseline is SQLite + exact embeddings + fixed recency + top-k +
deterministic prompt compaction. Its 1k/10k/100k runner is an offline retrieval
smoke over a fixed 100-record corpus. It does not establish growing-store behavior,
LLM task success, TTFT or a DML advantage. Raw real-agent task outcome JSONL can be
reduced separately; absent measurements stay null, and failed-task/maintenance
costs count toward tokens per completed task.

Still required before production graduation: receipts for remaining mutation APIs; all-component
atomicity; remaining lifecycle/retrieval/persistence orchestration extraction;
complete mutation-point process-kill and platform/power-loss qualification; fully
bound model/tokenizer identities; real-agent baseline value and long-horizon
campaigns; complete migration coverage as releases accumulate; and durable replay
of all memory decisions. See the [workstream status](../production-foundations-2026-09-12.md).
