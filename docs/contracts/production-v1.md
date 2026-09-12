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

The journal transaction currently covers the lattice, **not** persistent RAG,
DPM/DCN files or runtime KV state. Normal JSON/RAG write compensation still exists.
Idempotent client receipts and a transactional projection outbox are not yet
implemented. Journal schema 1 is therefore a production candidate, not a completed
all-component transaction contract. Checksums detect accidental corruption; they
are not signatures against a writer with filesystem access.

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

Journal commit decisions are durable and paginated with `dml-journal decisions`.
Retrieval responses include a deterministic context digest, returned IDs, suppressed
IDs/reasons, scope, effective time, query/vector digests and journal revision where
available. These response traces are not yet stored durably. They explicitly say
that replay inputs are incomplete: hashes alone cannot recreate the original query,
model identity, source bytes or actual rendered context. Full operator replay and
bounded audit retention/export remain release gates. No payloads are newly logged.

## Migration and recovery

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

Still required before production graduation: idempotent receipts; all-component
atomicity; remaining lifecycle/retrieval/persistence orchestration extraction;
complete mutation-point process-kill and platform/power-loss qualification; fully
bound model/tokenizer identities; real-agent baseline value and long-horizon
campaigns; complete migration coverage as releases accumulate; and durable replay
of all memory decisions. See the [workstream status](../production-foundations-2026-09-12.md).
