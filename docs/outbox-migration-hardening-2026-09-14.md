# Explicit outbox migration qualification

This serial gate extends areas 3, 7, 8 and 9 of the production plan. The feature
remains a candidate; DML is not declared production-ready. No user store is migrated
by this change. The existing schema-3 outbox format and default creation behavior
remain supported.

## Compatibility and history boundary

Schema 2 retained decisions and ingestion receipts, but not a full state for every
revision. An upgrade cannot reconstruct that absent history honestly. The explicit
`upgrade_outbox_journal(source, destination)` service therefore creates schema 4:

- The source store identity, current state, all original decision rows and all
  receipt rows are preserved. Historical rows retain their original encoded bytes,
  checksums and schema-2 decision versions.
- A checksummed origin records source schema 2, source revision R, exact state
  digest and the digest of the original decision history.
- One baseline commit at R+1 carries the complete current state, a null receipt,
  empty changed/deleted lists and operation `outbox-migration-baseline-v1`.
- Baseline and subsequent events use numeric schema version 2 and
  `event_format="dml-journal-outbox-v2"`, with the same immutable origin. New
  decisions use journal schema 4. No events are fabricated for revisions 1..R.

The decisions API continues to page original history. Outbox pages start at the
baseline for cursor zero and reject cursors inside the unavailable legacy range.
Prefix digests bind actual deliverable events. Legacy receipts remain retryable,
including receipts for a memory subsequently updated or deleted; they do not
resurrect those memories or become historical deliveries.

A new SQLite consumer atomically binds to the baseline with
`consumer_format="dml-sqlite-outbox-consumer-v2"`. Its generic state envelope stays
at schema 1; the explicit consumer format and origin define the new compatibility
boundary. An already bound v1 consumer cannot silently adopt this history. Status
reports history coverage with `legacy_operations_delivered=false`. Concurrent
delivery, checksum prefix validation and retry after a lost acknowledgement apply
from the baseline onward.

## Offline runbook

Stop every writer before migration and keep writers stopped through verification
and configuration cutover. Use an unused destination in a separate directory:

```sh
dml-journal enable-outbox /absolute/old/dml_state.sqlite3 /absolute/new/dml_state.sqlite3
dml-journal decisions /absolute/new/dml_state.sqlite3 --after-revision 0 --limit 100
dml-outbox /absolute/new/dml_state.sqlite3 /absolute/consumer/inbox.sqlite3 sync --limit 100
dml-outbox /absolute/new/dml_state.sqlite3 /absolute/consumer/inbox.sqlite3 status
```

The migration command returns a JSON report containing source identity/revision,
destination and boundary revision, preserved row counts and origin. Errors return
exit code 2 with an exception class, without exception text that might reveal store
content or paths. Switch adapter configuration explicitly to the verified target;
delivery still requires opt-in receipts/outbox settings. Existing schema-4 journals
retain receipt protections even if flags are later disabled. Projections can read
schema 4 and verify their pinned source normally.

The source is retained. SQLite backup pins a verified source read transaction;
the destination conversion and baseline commit are transactional. A fresh source
identity/revision/state check before publication detects a writer that bypassed
the advisory lock during copying. This is not an online cutover protocol: a writer
that resumes after the final check can still diverge, so stopping writers remains
mandatory.

Existing destination files, identity/migration/init-lock sidecars, WAL/SHM files
and dangling links at these paths cause rejection. Preserve failed output for
diagnosis and retry a fresh destination. Even an interruption before the migration
marker can leave a reserved initialization lock; do not reuse that path. Once a
marker exists, incomplete output fails closed on open. After publication the
destination may be complete despite the caller losing its response; verify by
opening it before deciding whether to cut over. Do not remove a marker to bypass
verification.

There is no in-place downgrade. The old reader rejects schema 4 before mutation.
The retained source is a rollback copy only until destination writes begin; later
rollback requires explicitly accounting for those writes. A snapshot export alone
does not preserve receipt history and must not be used as a receipt authority.

## Qualification and limits

The fixture was generated using receipt implementation commit
`8a08efa0976677ac813597956f87c19543be6ac8`. Its SQL and original reader are hash-pinned,
with two receipts and four revisions spanning update, deletion and typed metadata.
This is commit compatibility evidence, not a claim about a semantic release.

The recovery harness kills real child processes at seven migration boundaries:
before/after the marker, after backup, before/after the schema transaction commit,
after identity publication and after marker removal. The oracle requires unchanged
source bytes and receipt/decision rows, blocked incomplete output or a fully
verified destination. Fault-injection cases cover corruption, boundary-level
ENOSPC errors and source drift. Independent adversarial tests cover dangling paths,
mixed-version decision paging and source identity mutation within initialization.

The delivery corpus covers empty/nonempty legacy histories, bounded pages, omitted
legacy deletions, bad origins/prefixes, format mismatch, rollback, failed/lost
acknowledgements, concurrent source progress and 256 simultaneous consumers.
CI retains Linux/macOS/Windows portability coverage and adds both migrated-history
stress cases to the production evidence job.

Independent review rejected the initial implementation at 9.1 and accepted the
corrected feature at 9.5. The focused corpus passed 139 cases. The full local suite
passed 1,601 tests with nine optional-dependency skips and two known warnings in
66.63 seconds. Ruff, 51 maintained mypy files and Hermes hygiene passed; both
256-client migrated-history cases also passed separately.

The [independent review record](artifacts/outbox-migration-review-2026-09-14.json)
records the rejected and accepted assessments and exact source hashes. Final
commit CI is recorded on PR #118. Process-death tests do not establish physical
power-loss guarantees. Full-state events and full-history verification retain
unbounded byte/storage costs; event-count limits do not bound those costs. Ordered
background delivery, lifecycle receipts, retention and real-agent growing-store
value remain separate gates.
