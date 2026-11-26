# Coalesced projection deltas and explicit backend integration

This serial gate extends the accepted disposable SQLite projection. It delivers
changed record payloads and deletions through a narrow backend protocol while
preserving atomic target publication and verified, scoped queries. Source journal
schema 2 and projection journal schema 1 are unchanged. The existing full-snapshot
synchronization path remains available.

## What is incremental

`prepare_delta(source, backend)` compares a verified current authority snapshot
with a verified target envelope. The versioned packet transports only new/changed
record payloads, deleted IDs, the complete ordered ID manifest and final embedding
contract. Boolean, integer and floating-point metadata changes remain distinct
through canonical JSON comparisons. Source-only changes such as lineage updates
can advance the cursor without resending unchanged live records.

This is **coalesced state delivery**, not a durable per-operation outbox. Several
source revisions may collapse into one delta representing the latest observed
state. Preparation still reads and validates the full source and target. SQLite
application reconstructs and verifies the complete target envelope in memory and
uses the existing journal's changed-row writes. Full ID manifests, snapshots,
digests and whole-history validation still incur growing costs. No reduction in
end-to-end latency or 1k/10k/100k-turn qualification is claimed.

## Packet and atomic application

`dml-projection-delta-v1` packets carry a base cursor and base-envelope SHA-256,
next cursor and target-envelope SHA-256, upserts, deletions, ordered IDs, embedding
contract, source path and packet checksum. The full authority digest in the cursor
includes source metadata/lineage; the separate envelope digest verifies exactly
what the query projection must contain.

Application freezes and validates the packet before backend callbacks or I/O.
Unknown formats, malformed cursors, bad checksums, duplicate/unknown IDs,
inconsistent manifests, invalid scoped records, authority changes and rollback
are refused. The observed target must match the entire declared base. Target
records and next cursor commit together under compare-and-swap. A lost acknowledgement
is reconciled by reading the complete committed target, never by altering source
memory. A partial delta cannot be marked current.

An exact replay is acknowledged when the verified target already matches the
requested envelope, manifest, contract and upsert contents. This acknowledges
that the desired state was reached; it does not prove that each obsolete operation
was historically executed. Full synchronization may already have reached the same
state. Historical membership of an already-absent deleted ID is not re-proven on
that no-op path. First application rejects deletion of an unknown base record.

Packets with an obsolete base are rejected rather than applied partially. Regenerate
from current source/target state and retry. A delayed packet cannot overwrite a
newer target or resurrect a deleted source memory. Source ingestion can proceed
while target I/O is stalled; `matches_pinned_source` compares the target against
a new verified authority read after delivery, not against perpetual currency.

## Backend protocol

`ProjectionBackend` exposes only `path`, `read()` and `apply_delta(packet)`.
The stable path identifies a backend in a directory separate from authority.
`read()` returns a detached, internally consistent complete projection envelope.
`apply_delta()` must atomically publish records plus cursor and return the committed
cursor; its read-back must expose that durable state or a later committed revision.
The protocol does not qualify an eventually consistent backend or a backend that
cannot provide atomic cursor/data publication.

`reconcile_incremental()` freezes expected acknowledgement values before calling
the backend, validates the returned cursor, and verifies read-back. An unchanged,
behind, unbound, foreign or incorrect same-revision target cannot support a success
claim. A later revision from the same authority is allowed because another publisher
may have progressed after acknowledgement; comparison with the newly pinned source
still determines reported freshness.

The SQLite implementation is the qualified backend. Structural protocol proxies
are tested to prevent dependence on SQLite internals in orchestration. The protocol
is a trusted implementation boundary, not authenticated remote transport. Checksums
do not defeat a party able to rewrite packet contents and all linked hashes. Existing
FAISS/RAG backends are not automatically adapted or qualified.

## Explicit adapter and CLI integration

Receipt-mode adapters provide three explicit operations:

- `sync_projection(backend)` prepares, delivers and verifies one coalesced delta.
- `projection_status(backend)` compares the backend with a pinned authority snapshot.
- `query_projection(prompt, backend=..., tenant_id=..., as_of=..., ...)` embeds under
  a pinned model identity and invokes the accepted exact scoped projection query.

All three require an opt-in receipt journal and reject calls nested inside the adapter
mutation transaction. Query embedding identity is checked through preparation and again
against the pinned authoritative contract. Backend work is never introduced into
receipt ingestion, retry or the legacy compensating batch path. Backend failure
cannot revoke an acknowledged source receipt. No background delivery worker,
automatic context routing or provider HTTP endpoint is introduced.

The explicit CLI extension is:

```sh
dml-projection /absolute/authority/source.sqlite3 /absolute/projection/index.sqlite3 sync --incremental
```

Without `--incremental`, `sync` retains accepted full-snapshot behavior. Existing
status and query commands remain compatible. Errors retain sanitized JSON and exit
code 2. A stale-base error means regenerate from current state; corruption or
incomplete target creation still requires preserving the damaged target and rebuilding
into a new dedicated directory. The [snapshot projection contract](projection-hardening-2026-09-14.md)
remains the authority for recovery, query linearization and scope/lifecycle semantics.

## Independent acceptance

The independent grader rejected the first revision at **9.1/10**. Corrections
ensure that backend acknowledgements agree with observed target publication and
that JSON metadata type changes produce updates. Multi-process and competing-base
regressions strengthen the delivery oracle. Final independent grade: **9.5/10**.

The focused harness contains **52 passing regressions**, including a 256-client
run. Final local integration passed **1,213 tests**, with nine optional-dependency
skips. Ruff, 47 maintained mypy files and Hermes hygiene passed. The
[review record](artifacts/projection-delta-review-2026-09-14.json) pins the accepted
source and records both rejected and accepted reviews. Final-commit remote CI is
a separate integration gate.

## Verification and limits

The harness compares delta results with independent full synchronization, proves
unchanged text is absent from packets, and exercises additions, updates, deletions,
reordering, metadata type changes and source-only revisions. It includes actual
subprocess deaths at seven target mutation hooks, eight duplicate-delivery processes,
256 simultaneous duplicate clients, quota exhaustion, competing stale-base packets,
backend stalls, lost/false acknowledgements, immutable inputs, protocol-only proxies,
adapter restart, scoped retrieval and CLI interoperability. CI retains these results
in its portability and production-evidence gates.

This gate does not establish distributed exactly-once operation delivery, a durable
incremental changefeed, automatic retry scheduling, bounded history compaction,
FAISS integration, real-agent improvements or whole-platform production readiness.
Those remain separate serial qualification work.
