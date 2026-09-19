# Transactional operation events and ordered delivery

This serial gate adds an explicitly versioned journal outbox and a reference
transactional consumer. Each committed source revision has an immutable full-state
event. A delivery pass visits revisions in order and checks durable acceptance.
This extends P03 of the ten-area plan; it does not graduate the whole platform.

## Opt-in schema and commit contract

New stores can enable `persistence.journal=true`, `persistence.receipts=true` and
`persistence.outbox=true`. At the journal API, use
`JournalStateStore(path, receipt_mode=True, outbox_mode=True)`. This creates schema
3, adding an `outbox` table and outbox bindings to commit decisions. Default new
stores retain schemas 1/2 according to their existing configuration.

Every successful state-changing commit writes memory state, its decision, the
outbox event, and any ingestion receipt in the same SQLite transaction. A retry
that finds an existing receipt returns that receipt without a new event. A true
no-op state save also creates no revision/event; the initial save establishes the
first committed revision. Metadata-only changes, order changes, updates, deletions
and lineage changes are retained as distinct events when committed. The outbox
does not itself authorize or implement additional public lifecycle mutations.

Events use `event_format="dml-journal-outbox-v1"` and carry schema version, source
store identity, revision, source-state digest, operation, receipt binding, complete
normalized state, decision digest and event checksum. The decision binds the event
checksum; the event binds the decision excluding that checksum field. Verification
checks the entire contiguous history, source identity, every state/receipt/decision
binding and each record transition. Missing or inconsistent events prevent reads
and subsequent commits. Checksums detect damage; they are not authentication
against an attacker able to rewrite authority and all related evidence.

`outbox_events(after_revision=0, limit=100)` returns one detached page, source
identity, pinned head revision, next revision, backlog flag and a digest of the
event-checksum prefix through `after_revision`. Page limits are integers 1–1000;
future, negative, boolean and malformed cursors are rejected. Head, events and
prefix digest come from one verified read transaction. The prefix lets a consumer
prove its entire earlier history instead of merely showing a valid latest event.

## Ordered reference consumer

`SQLiteOutboxConsumer(path)` uses a dedicated schema-1 journal in a directory
separate from source authority. It atomically commits the latest complete event,
its source cursor and a contiguous ledger of accepted event checksums. A new
consumer starts unbound; accepting revision 1 binds its authority. Gaps, conflicting
duplicates and a different authority are rejected. An exact historical duplicate
returns its original acknowledgement without another consumer revision.

`deliver_outbox(source, consumer, limit=100)` prepares a bounded page after the
verified consumer position and delivers every event in order. The consumer protocol
has `path`, `read()` and `apply_event(event)`. Before delivery, the complete checksum
prefix and latest event must match authority. After each application, the expected
acknowledgement is compared with a durable read-back. A matching return value alone
does not prove acceptance. Final status rechecks authority and refuses consumer
regression. Inputs and expected acknowledgements are frozen before backend calls.

A post-commit exception can be reconciled against the exact durable ledger. If
delivery stops after some events, the committed prefix remains and the next pass
resumes there. Concurrent callers may attempt duplicate delivery; reference-consumer
CAS and checksum deduplication commit each revision once. `delivered_count` counts
events acknowledged by the pass, including duplicates committed by a concurrent
caller. It is not a count of newly executed external side effects.

The qualified consumer retains the latest full state and a historical checksum
ledger; the source retains all full events. It does not execute arbitrary callbacks,
update the existing FAISS backend, or provide distributed exactly-once side effects.
Custom consumers must implement atomic ledger/state publication and durable
read-after-write. Their claimed persistence is a trusted implementation boundary.

## Adapter, CLI and status

Receipt ingestion never invokes the consumer. Source commits remain available
while a consumer is slow or unavailable. `adapter.deliver_outbox(consumer, limit=100)`
requires an explicitly enabled schema-3 outbox and rejects execution under adapter
mutation ownership. Consumer I/O holds no source write lock. Existing scoped
retrieval, full/coalesced projection delivery and the accepted projection worker
also accept schema-3 authority while retaining receipt-mode protections.

```sh
dml-outbox /absolute/authority/dml_state.sqlite3 /absolute/consumer/inbox.sqlite3 sync --limit 100
dml-outbox /absolute/authority/dml_state.sqlite3 /absolute/consumer/inbox.sqlite3 status
```

`sync` may create a new dedicated consumer. `status` refuses a missing consumer.
Errors return sanitized exception types and exit code 2. Status reports observed
source head, consumer cursor, backlog, whether they match, and whether the consumer
is bound. An empty source/consumer can match at revision zero while remaining
unbound. Reports describe pinned observations; a later source commit can add lag.
Ordered delivery is explicit in this gate. The existing background projection
worker continues to deliver coalesced projection state.

## Migration, recovery and retained data

Existing schema-1/2 stores are never upgraded in place or implicitly. Opening them
with outbox mode enabled is rejected. An explicit migration with a defined historical
starting boundary is a separate serial gate; this implementation does not invent
events for state transitions whose old payloads were never retained. No user store
was migrated. Older binaries that support only schemas 1/2 reject schema 3. Current
readers auto-detect schema 3 and preserve its receipt and outbox guarantees even
when a reopened journal was not constructed with creation flags.

Do not downgrade by editing version fields. Receipt/outbox journals reject
lattice-only JSON export because it loses durable history. Retain a complete
consistent database backup, including receipt and outbox authority. Preserve
corrupt databases, identity markers and sidecars; repair or restore into a separate
location. A missing or damaged consumer can be rebuilt by replaying source events
into a new dedicated consumer. A bare initialized journal left before consumer
format publication is incomplete and is not silently adopted.

Source initialization, source mutation, consumer initialization and consumer
application all have explicit process-death oracles. A killed transaction leaves
either the previous complete revision or the next complete revision. No partial
memory/receipt/event or consumer state/cursor combination can be acknowledged.
POSIX tests require actual SIGKILL termination; a bounded wait avoids racing signal
delivery with a fallback exit. Windows uses abrupt process exit.

## Cost and remaining scope

This is a correctness reference with a full state per committed revision. Space
grows with the sum of historical state sizes. Page limits bound event count, not
bytes, and every page read still verifies full history. The consumer checksum ledger
also grows with accepted revisions. No throughput improvement, bounded historical
storage, compaction, retention or 1k/10k/100k-turn performance claim is made.

Historical events retain memory that was subsequently updated/deleted. A deletion
changes live state; it does not erase audit history. Secure historical erasure,
retention, schema-2 migration, ordered background scheduling and receipts for the
remaining public lifecycle operations each require their own qualification. Stable
APIs remain empty in the maturity inventory.


## Independent acceptance

Initial independent review rejected the feature at **9.1/10**: a valid last event
could hide a false historical checksum prefix. The source now returns a prefix
digest from the same verified transaction as its page, and status/delivery verify
the complete consumer prefix plus its latest event. Final delivery also rejects
ledger regression. Four independent regressions cover protocol responses and
self-consistent persisted consumer envelopes. Final independent grade: **9.5/10**.

The focused harness has **144 passing cases**: 47 journal, 32 delivery, 33 adapter
integration, 28 additional process-recovery and four independent adversarial cases.
It includes 37 actual process-kill boundaries, eight competing delivery processes,
serialization/disk-full faults, corruption with recomputed bindings, duplicate and
false acknowledgements, pinned identity/state reads, schema compatibility guards,
and continued source writes during blocked delivery.

The original stress harness exercised 256 requests through 16 workers. It was
corrected to use a bounded start barrier and 256 simultaneous worker clients; the
explicit run passed. The ordinary suite uses 32 clients, and CI separately runs
the 256-client qualification and uploads its JUnit result.

Full local integration passed **1,462 tests**, with nine optional-dependency skips
and two known warnings in 65.67 seconds. Runtime source was unchanged by the later
stress-harness correction. Ruff, 50 maintained mypy files (local Python target
3.12) and Hermes hygiene passed. The [review record](artifacts/outbox-review-2026-09-14.json)
pins accepted source hashes and rejected/accepted findings. Exact-commit remote
CI remains a separate integration gate recorded in PR #118.


The first exact-commit CI run passed the two full-suite jobs, Linux/macOS
portability and production evidence. Windows found a test resource-lifecycle
issue: SQLite's transaction context does not close a connection, so the corruption
fixture still held its database open when simulating deletion. The two test
connections now close explicitly after their transactions. Runtime code is
unchanged; the corrected harness requires another independent review and all nine
CI jobs on the corrected commit before integration acceptance.
