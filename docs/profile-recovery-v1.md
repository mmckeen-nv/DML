# Supported-profile recovery contract v1

This contract applies to the candidate
[`dml-receipted-local-v1`](supported-production-profile-v1.md) profile on one host
with cooperating trusted callers. It defines milestone 4's recovery boundary;
acceptance and measured environments are recorded separately in the
[hardening record](profile-recovery-hardening-2026-09-18.md). The profile remains
candidate and `production_ready=false`.

The authority is `storage_dir/dml_state.sqlite3`, in SQLite WAL mode with
`synchronous=FULL`. Memory and its receipt, decision and optional outbox event
share one transaction. Runtime caches are rebuilt from verified authority.
A receipt is a historical commit acknowledgement, not evidence that the memory
is currently live. No cross-file transaction or physical power-loss guarantee
is introduced by this contract. SQLite documents that WAL belongs to the
database state and that durability depends on correct storage synchronization;
DML evidence must be interpreted within those prerequisites. See the official
[WAL documentation](https://sqlite.org/wal.html) and
[corruption and synchronization guidance](https://sqlite.org/howtocorrupt.html).

## Admitted mutation and component inventory

The same five mutation intents are exercised through the selected profile on
schema 2, schema 3 and explicitly migrated schema 4. Schema 2 has no outbox;
schemas 3 and 4 retain their existing outbox semantics. This stage does not change
journal, receipt, decision or outbox formats.

| Mutation | Authoritative state change | Recovery invariant |
| --- | --- | --- |
| Receipted ingestion | Append one memory with scope, provenance and embedding identity. | No partial record or receipt; retry creates at most one committed result. |
| Receipted retirement | Suppress the exact scoped record while retaining its historical content. | Tombstone, receipt and decision agree; retirement never becomes erasure. |
| Receipted supersession | Link an exact source to an exact same-scope replacement. | Only the source is suppressed; both digest preconditions remain meaningful. |
| Receipted content update | Replace text and vector in the unchanged declared embedding space. | Text/vector commit together; preserved metadata and historical receipt agree. |
| Receipted first-level promotion | Append a derived record with immutable snapshots of the exact sources. | Derived record and complete proof commit together; sources remain unchanged. |

| Component or file | Role | Recovery treatment |
| --- | --- | --- |
| SQLite records and state metadata | Current memory, lineage, lifecycle and embedding authority. | Verify complete retained authority before admitting reads or writes. |
| SQLite archived snapshot and decisions | Checksummed retained state and transaction evidence. | Validate history and its correspondence with current authority. |
| SQLite receipts | Historical scoped idempotency outcomes. | Reuse the complete original request, scope and key to resolve uncertain outcomes. |
| SQLite outbox, schemas 3/4 | Committed operation events and migration baseline. | Verify alongside authority; no external consumer or projection must be available. |
| SQLite WAL and SHM | Committed WAL state and transient coordination. | Preserve during incidents; never copy a live main database as a complete backup. |
| `dml_state.sqlite3.identity.json` | Store identity and evidence that an authority existed. | Preserve and verify against the database; never delete it to bypass missing-authority errors. |
| `dml_state.sqlite3.migration.json` | Incomplete migration evidence, when present. | Stop recovery admission and retain it for diagnosis. |
| `dml_state.sqlite3.init.lock` and `.dml_store.lock` | Cooperating initialization and mutation ownership. | Preserve paths; OS ownership ends on process death. A lock file's existence is not ownership. |
| `.dml_store.lock.json` | Diagnostic owner metadata. | Treat as diagnostic evidence, never authoritative ownership; stale or failed cleanup cannot justify deleting a live lock. |
| Runtime lattice, vector index and hydration | Disposable views of authority. | Refresh from a verified revision. Failed hydration cannot revoke an established commit. |
| Embedding provider | Preparation dependency for ingestion, update, promotion and vector retrieval. | An outage is explicit; a historical exact retry resolves before new model work. |
| HTTP acknowledgement encoding and connection | Delivery of the historical result. | A lost response leaves the caller uncertain; it does not roll back a commit. |

Retrieval, retention inspection and status calls add no logical memory mutation.
SQLite may update coordination files or checkpoint WAL while serving operations;
that does not expand the profile into multi-file atomic persistence. Selected
profile initialization does not import legacy state. Legacy JSONL/RAG persistence,
semantic checkpoints, native KV, projection consumers, model generation and
experimental automatic lifecycle paths are excluded from this inventory.
The separate exact-input companion has no durable execution state to restore.

## SQLite runtime admission

The selected profile requires the linked SQLite runtime to be **3.51.3 or newer**,
or a patched **3.44.x with patch level at least 6**, or a patched **3.50.x with
patch level at least 7**. These accepted branches account for SQLite's documented
[WAL-reset fix](https://sqlite.org/wal.html#walresetbug). Admission queries the
linked library itself with `SELECT sqlite_version(), sqlite_source_id()` before
accessing or creating profile authority. Merely installing a different SQLite
command-line binary does not change Python's linked runtime. The source ID is
recorded information, not attestation of a custom build or all dependency behavior.

Existing profile authority must already use WAL. Startup and each strict journal
connection reject changed journal mode instead of converting it silently. Legacy
non-profile readers retain their historical runtime admission. A passing version
check is a prerequisite, not certification of every SQLite build, filesystem or
platform combination. Record the actual linked version/source ID in evidence.

## Outcome contract

Classify client observations independently from the storage fault location:

| Client observation | Required restart outcome | Caller action |
| --- | --- | --- |
| Acknowledged receipt | The same historical receipt and commit are present in the retained authoritative history. Later valid lifecycle changes may alter current liveness. | Preserve the receipt outside the store and reconcile it during recovery. |
| Explicit precommit validation, capacity or lifecycle rejection | That attempted mutation has not committed. A prior conflicting request may have its own historical receipt. | Correct the intent deliberately; do not reinterpret an old key as a new request. |
| Process death, disconnect, timeout, acknowledgement failure or storage outcome unavailable | Either the complete prior state or the complete commit is allowed. Partial commit is forbidden. | Preserve the original request, full scope and key. Retry only against verified authority to establish its historical outcome. |
| Caught write error followed by verified receipt reconciliation | Return the established receipt if committed; otherwise report the explicit noncommit or unavailable outcome. | Follow the exact result; an exception alone is not proof of rollback. |
| Hydration fails after a verified commit | The receipt remains successful; runtime degradation is visible. | Repair/refresh or restart against verified authority. |
| Corrupt, missing initialized, incompatible or incompletely migrated authority | Explicit recovery-required failure. | Preserve evidence and recover separately; never substitute an empty store. |

The commit boundary includes records, metadata, archived snapshot, receipt,
decision and outbox. Kill hooks before the SQLite commit must leave the prior
state; a kill after commit retains the complete commit even if no response
arrived. Interrupted serialization and preparation fail before the transaction
can publish that mutation. The harness also crosses preparation, hydration and
acknowledgement boundaries through the actual selected adapter.

A consistent rollback is different from detectable corruption. A restored older
valid database and its matching identity can be internally consistent. Likewise,
removing committed WAL may leave a valid older main database. Local checksums and
identity alone cannot prove either loss occurred. Keep an independent durable
client receipt ledger containing complete receipt identities and original
requests. Compare that ledger with recovered authority; missing acknowledgements
are recovery discrepancies, never permission to claim all acknowledged memory
was retained. Without external evidence, recovery can prove internal consistency
only, not freshness or absence of historical loss.

## Deterministic interruption matrix

The declared matrix contains **184 child-process kill cases**. This is the
inventory size; passing results are recorded separately in the hardening record.
Each case uses the actual selected profile and synthetic pinned embeddings.
The new matrix covers Python API calls and a subprocess receipt channel. It does
not exercise an actual HTTP disconnect or transport loss; the HTTP acknowledgement
row above states the semantic contract and existing route tests remain separate.
The parent waits for a pipe barrier, kills the child with the platform process
termination primitive and reaps it under a 30-second deadline. No shutdown handler
is substituted for process death.

| Boundary | Operations and schemas | Cases |
| --- | --- | ---: |
| `after_begin`, `after_records`, `after_state`, `after_snapshot`, `after_receipt`, `after_decision`, `before_commit`, `after_commit` | Five mutations on schema 2/3/4. | 120 |
| `after_outbox` | Five mutations on schema 3/4. | 10 |
| Embedding preparation | Ingest, update and promotion on schema 2/3/4. | 9 |
| `before_hydration`, `before_response`, `after_ack` | Five mutations on schema 2/3/4. | 45 |

Only a complete returned receipt flushed to the parent is an acknowledged result.
Every other interrupted client call remains uncertain, even when the controlled
hook lets the test know a commit has not occurred. The oracle independently
checks expected records and request digests, retained prior receipts, the allowed
revision, exact retry, changed-request/key conflicts, scoped retrieval suppression
and retention counts. A second restart checks that reconciliation did not create
a duplicate commit. This is bounded hook coverage, not a claim to kill every CPU
instruction or to exercise hostile concurrent writers.

The campaign uses seeds **1729, 314159 and 7**, each on schemas 2/3/4: nine
histories, each with 12 logical public mutations and four killed attempts. Each
history includes two precommit and two postcommit lost responses, exact retries,
stale digest/key-conflict rejections, and all historical retries shuffled with
the embedding backend disabled. The aggregate schedule covers all five mutation
types. These are 108 logical mutations and 36 process kills, not 108 additional
pytest cases. Counts and final results remain separately attributed in the
hardening record. Corruption, missing/incomplete authorities, embedding outages
and SQLite quota-full cases supplement the process-kill histories.

## Fault evidence and filesystem limits

Evidence must report its class without substituting one class for another:

| Evidence class | What passing evidence establishes | What it does not establish |
| --- | --- | --- |
| Actual child-process termination at declared hooks | Atomic restart outcome and cooperating-lock release for that binary and observed environment. | Power loss, controller cache behavior or arbitrary instruction-level interruption. |
| Caught I/O, synchronization, serialization and hydration faults | Explicit error handling, receipt reconciliation and ownership cleanup for the injected failure. | Every OS/storage error or a physical full device. |
| SQLite `max_page_count` quota exhaustion | Handling of SQLite `SQLITE_FULL` under the imposed database quota. | Physical filesystem ENOSPC, inode exhaustion or unrelated files' behavior on a full device. |
| Dedicated 128 MiB ext4 volume exhausted to kernel ENOSPC | If the mandatory CI lane passes: prior-state preservation and exact retry for five mutations on three schemas after real byte-capacity exhaustion. | Every filesystem/full-disk condition, inode exhaustion, production device behavior or power loss. |
| Corrupt/truncated/missing authority fixtures | Fail-closed behavior for the tested corruption and missing-file classes. | Detection of a malicious writer rewriting all data/checksums or a consistent rollback without external receipts. |
| Offline verified backup and restore | A self-consistent captured revision can be restored into a separate location and reopened. | Recovery of commits after that revision, physical power-loss durability or authenticity against a hostile backup author. |

Capture the actual filesystem/mount observation, OS/kernel, architecture, Python,
SQLite and source identity with each run. An Ubuntu/macOS/Windows runner name
alone does not identify or qualify its filesystem. A successful `fsync` call is
not evidence that volatile devices, mount settings or controllers honor a durable
barrier. POSIX directory synchronization errors propagate; publication can already
be visible when the barrier reports failure. The Windows helper has no directory
synchronization barrier and makes no such power-loss claim.

The local development mount was observed as an overlay with `fsync=volatile`.
Its runs are process-failure and caught-error evidence only. No production
power-loss durability can be inferred from them. CI observations must be retained
as measured evidence. The dedicated qualification lane rejects an unknown,
nonlocal or unadmitted filesystem and volatile/disabled barriers. It records
actual source/runtime identities, per-case outcomes and the JUnit digest on the
same `--basetemp` volume. The separate ext4 ENOSPC lane requires all 15 cases with
zero skips; unavailable local mount capability does not qualify that failure.
See the hardening record for observed results versus pending CI.
Network/shared filesystems, noncooperating writers, multi-host
ownership and unmeasured storage combinations remain outside the profile claim.
Neither this stage nor filesystem observation qualifies physical power loss.

## Offline operator recovery procedure

Stop every application, worker and other cooperating writer for the source store.
Keep them stopped through verification, backup or incident preservation. The
operator procedures acquire the store's coordination lock, but that is not a
replacement for stopping clients that could resume after the command returns.
Only trusted operators may access the source and backup; backups contain retained
memory, historical payloads and scoped receipt keys.

Use the existing `dml-journal` entry point. Provision the destination parent
first; each backup and restore destination must be an unused directory separate
from, and neither an ancestor nor a descendant of, its source directory. The
commands never switch the running application to another store. The persistent
`.dml_backup_publish.lock` in each destination parent serializes cooperating
backup/restore publishers; retain that coordination file and never unlink it to
bypass ownership. It is outside the sealed bundle.

```sh
dml-journal backup /srv/dml/store/dml_state.sqlite3 /srv/dml-backups/recovery-001
dml-journal verify-backup /srv/dml-backups/recovery-001 --receipts /srv/dml-evidence/receipts.json
dml-journal restore /srv/dml-backups/recovery-001 /srv/dml-recovered/store-001 --receipts /srv/dml-evidence/receipts.json
dml-journal verify-backup /srv/dml-recovered/store-001 --receipts /srv/dml-evidence/receipts.json
```

The receipts file is a JSON array of the complete retained receipt objects, at
most 16 MiB. It is optional for structural verification, but without it no
external acknowledgement comparison occurs. Only supplied receipts are checked;
even an empty or incomplete file cannot establish completeness of the client
ledger. Keep original requests and exact scopes/keys separately for uncertain
operations. A supplied receipt from another store, a later revision or different
historical receipt contents rejects verification and restore. Field order and
JSON whitespace are not part of the receipt comparison.

A successful command emits JSON with `ok: true`; a caught error emits
`ok: false`, the exception class in `error`, and exit status 2. No success report
is an automatic cutover approval. Inspect these fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Exactly `dml-authority-backup-v1`. |
| `created_at_utc` | Backup creation timestamp; it is not proof of latest authority. |
| `authority.store_id`, `journal_schema_version`, `revision` | Captured identity, admitted schema 2/3/4 and recovery point. |
| `authority.state_digest`, `authority_digest`, `receipt_count` | Verified current state and complete retained table evidence, including historical receipt count. |
| `files` | SHA-256 and byte count for the database and identity sidecar. |
| `receipt_comparison` | `provided`/`matched` counts and `coverage` of `not_provided` or `provided_receipts_only`. |

The sealed bundle contains exactly `dml_state.sqlite3`,
`dml_state.sqlite3.identity.json` and `backup.json`. Verification rejects extra
files, symlinked payloads, incomplete markers, mismatched bytes, unknown manifests
and corrupt retained authority. It checks a self-contained database without WAL
or SHM dependence. Opening a restored directory as a live profile can add locks,
coordination files and new commits: it then becomes an operational store and
must no longer be treated as the original sealed backup. Keep the original bundle
unchanged, and run the final `verify-backup` command before starting the profile.

1. Preserve the original directory and all existing database, WAL, SHM, identity,
   migration and diagnostic sidecars without changing their evidence. Retain the
   exact deployed source/configuration and independent client receipt ledger.
2. For a healthy, capturable stopped source, run `backup`. It verifies complete
   authority in a SQLite
   read transaction and captures committed WAL through SQLite's backup API. A
   failure stops admission; do not remove markers, run an old writer, or declare a
   missing database empty. A read-only SQLite source may create coordination files.
   For an already corrupt or missing incident authority, preserve the raw incident
   evidence and select an earlier separately verified backup instead. `backup`
   refuses corrupt authority; it is not a salvage or corruption-repair command.
3. Run `verify-backup` and inspect its captured store identity, schema, revision and
   receipt comparison. Retain the original source; a bare live database copy and
   lattice-only export are not complete receipt-authority backups.
4. Restore a verified backup into a separate, unused store directory. Verify the
   result before changing `storage_dir`; retain the original incident directory.
5. Reconcile external acknowledged receipts and unresolved requests against the
   recovered captured revision. Account explicitly for commits newer than the
   backup; restoring a backup alone does not recover them. Do not replay new
   decisions under old keys or claim freshness from an internally consistent file.
6. Reopen with the matching profile/configuration, verify scoped reads and historical
   retries, then deliberately resume callers. Take a fresh verified backup after
   recovery and retain incident evidence according to the operator's access policy.

Backup publication is not a transaction with the original authority. Before
publication, a migration marker blocks an incomplete destination from profile
startup. Interrupted reservation can also leave a private `.dml-backup-staging-*`
sibling, which is quarantine evidence. Preserve failed outputs and retry a new
location; never remove markers or start a failed target merely because it exists.
A failure after final publication or its directory barrier may leave a complete
bundle: only successful independent verification establishes whether that bundle
is usable. Verification and restore reject incomplete or mismatched evidence. This
procedure does not perform an in-place downgrade, merge divergent histories,
automatically roll back the live source or erase retained content. Version-family
migration coverage and release installation/support policy remain milestones 5
and 11 respectively.
