# Supported-profile crash and recovery hardening

This eighteenth serial gate addresses milestone 4 of the
[finite release ledger](production-remaining-work-2026-09-18.md): crash recovery
and filesystem qualification for the existing candidate
`dml-receipted-local-v1` boundary. The [recovery contract](profile-recovery-v1.md)
inventories five admitted mutations, three authority schemas, relevant components,
outcome classes, the offline operator procedure and filesystem limits.

**Independent software review accepted 9.6/10, with no blockers. Milestone 4's
measured-environment qualification remains pending.** The independent five-module
selection passed **416 tests, with zero failures, errors or skips**, in **114.35
seconds** (JUnit 114.332 seconds). The subsequent actual six-module evidence run
passed **450 tests with zero skips**, and root integration passed **4,048 tests
with 24 skips**. Final source-hashed acceptance is recorded in the
[review artifact](artifacts/profile-recovery-review-2026-09-18.json).
Publication and measured-platform CI remain pending at this source snapshot;
software acceptance alone does not qualify the unobserved platform and real-ENOSPC
lanes. The repository remains alpha and `production_ready=false`; the original
ten workstreams and finite thirteen-milestone scope are unchanged.

## Prior-source evidence

Before this gate, stage 17's corrected source `49368a9` passed all ten jobs in
[CI run 317](https://github.com/mmckeen-nv/DML/actions/runs/35348088493). Independent
review accepted 9.6/10 and the required model-input lane passed 256 tests with zero
skips. PR #118 remains unmerged. These results establish the starting point;
they do not qualify stage 18. The stage-17 source-hashed review artifact remains
historical and is not rewritten.

## Implementation scope

The runtime fixes propagate failed POSIX directory synchronization, preserve
primary failures while always closing ownership descriptors, and require WAL
mode for selected-profile authority. The selected profile also rejects linked
SQLite runtimes without the admitted WAL-reset fix; the exact version branches
and source-ID limits are recorded in the recovery contract. The recovery harness exercises actual
selected-profile ingestion, retirement, supersession, update and promotion on
schema 2, fresh schema 3 and migrated schema 4. It records deterministic
process-kill boundaries and a reproducible seeded campaign, checking historical
receipt outcomes and complete authority after restart.

Offline verification, consistent backup and separate-destination restore use the
existing authoritative journal without introducing new runtime routes or memory
mutation APIs. The recovery bundle has its own versioned envelope. Its captured
revision bounds the recovery point; independent client receipt evidence is
required to detect consistent rollback or acknowledged WAL loss.

## Independent evidence and remaining validation

The independent five-module run recorded the following non-additive selection:

| Module | Passing cases |
| --- | ---: |
| `test_production_profile_recovery.py` | 185: 184 actual process kills plus inventory check. |
| `test_profile_recovery_campaign.py` | 55: nine seeded histories plus corruption, incomplete authority, embedding outage and SQLite quota cases. |
| `test_profile_storage_faults.py` | 76: runtime admission, file/directory synchronization barriers and ownership cleanup. |
| `test_authority_backup.py` | 60: complete-authority backup, verification, restore, receipt comparison and interrupted publication. |
| `test_crash_qualification_adversarial.py` | 40: independently authored adversarial cases. |
| **Total for this run** | **416 passed; 0 failed, errors or skipped.** |

The independent XML and log are separate from the subsequent six-module run
through the actual evidence plugin. That run passed **450 tests**, with zero
failures, errors or skips, in **114.92 seconds**, including **34 recorder unit
cases**. Its measured filesystem was overlay, device 28, with `fsync=volatile`;
the report correctly recorded `tests_passed: true`, `accepted: false` and
`power_loss_qualified: false`, with no recording errors. The report's JUnit SHA-256
matched the actual XML. This confirms honest evidence reporting, not qualification
of the local mount. The two selections overlap and are not additive.

The seeded histories use seeds 1729/314159/7 on three schemas, with 12 logical
mutations and four kills per history. The 108 logical mutations and 36 campaign
kills are workload dimensions, not extra pytest case totals.

The root full suite passed **4,048 tests**, with **24 skips** and **3 warnings**,
in **257.17 seconds** (JUnit 257.093 seconds). Its 4,072 cases contain zero failures
or errors; all **450 new regular recovery cases passed with zero skips**. The
24 skips comprise **15 dedicated ext4 ENOSPC capability skips** and **9 existing
skips**. The local container cannot mount the dedicated filesystem; those 15
skips are not filesystem-full recovery evidence. Maintained and strict new-surface
Ruff, mypy over **67 source files**, Hermes hygiene and the diff check passed.
The independent grader inspected the root XML and log separately from its own
focused run.

The [review artifact](artifacts/profile-recovery-review-2026-09-18.json) hashes
**24 source, test, workflow and documentation files**, excluding itself, and
records the final **9.6/10 ACCEPT** decision with no blockers. It preserves the
independent, actual-plugin and root integration evidence as separate observations.
The prior stage-17 review artifact remains unchanged.

| Remaining evidence | Status at this source snapshot |
| --- | --- |
| Real ext4 ENOSPC selection | Fifteen cases require the dedicated CI mount; local capability skips do not count as passing evidence. |
| Six measured-platform recovery lanes | Pending exact published-source CI. |
| Publication and all seventeen CI jobs | Pending; outcomes will be recorded in PR #118. |

Focused, full-suite, fault-campaign and remote CI selections overlap; their counts
must not be added into one test total. A simulated I/O error, SQLite quota-full
result or successful child process kill is not physical ENOSPC or power-loss
evidence.

## Continuous qualification lanes

The workflow retains its ten existing jobs and adds six `profile-recovery` jobs:
Ubuntu, macOS and Windows, each at CPython 3.10 and 3.13. Each runs all six required
modules, including the evidence-recorder tests, on an explicit disposable
`--basetemp`. The recorder measures its containing filesystem before the run,
then verifies both the actual test directory and its parent against that
observation at completion. It records runtime/source identity and per-test outcomes, and binds
the report to the JUnit SHA-256. The dedicated lane rejects missing modules,
incomplete cases, filtering, unknown/nonlocal filesystems and volatile or disabled
barriers. Its target filesystem names are ext4, XFS, Btrfs, APFS and NTFS; only an
actually observed and accepted runner result is evidence for that combination.
This is not a claim that all named families have been exercised.

The deterministic matrix and campaign permit no skips. The Windows selection may
skip only its two explicitly inapplicable POSIX directory-barrier tests, because
the Windows helper provides no directory-fsync barrier. Other skips reject the
qualification report. Local overlay runs can pass tests while recording
`target_filesystem: false`, `accepted: false` and
`power_loss_qualified: false`; that distinction must be retained.

The seventeenth workflow job, `profile-filesystem-full`, uses CPython 3.12 on a
new disposable **128 MiB ext4 loopback volume**. It checks the mount, dedicated
marker and capacity before allocating real bytes until the kernel returns ENOSPC.
For all five mutations on three schemas, the journal must actually encounter
SQLite FULL, preserve the prior authority and receipts, then succeed once on an
exact retry after capacity is freed. `REQUIRE_FILESYSTEM_FULL_TESTS=1` makes
missing capability fail the dedicated lane; its JUnit check requires exactly
15 passing cases and zero skips. No host/root filesystem is filled. The local
container lacks the required mount capability, so its skips do not qualify this
failure mode. Until the job succeeds, actual filesystem-full recovery remains
pending. A passing bounded-volume experiment still is not power-loss evidence.

All seventeen jobs probe the real linked SQLite runtime before tests. If it lacks
the admitted WAL-reset fix, the disposable CI bootstrap loads pinned official
SQLite 3.53.4 bytes after archive SHA3-256 verification and verifies the runtime
actually loaded by Python. An already admitted runtime is retained and recorded.
This is CI setup, not an automatic application-store upgrade or a test mock;
operators must provision their own admitted Python/SQLite combination.

## Qualification limits

The observed local filesystem is an overlay with `fsync=volatile`; process-kill
and caught-failure behavior on this mount cannot qualify persistent power-loss
behavior. Remote runs must capture their measured filesystem and runtime instead
of inferring storage from runner labels. No hardware, storage controller, cloud
volume class or whole filesystem family is certified.

Checksummed history detects the tested inconsistent/truncated/corrupt states.
It cannot self-detect a wholly consistent older authority or all committed WAL
being lost while the earlier main database remains valid. The independent client
receipt ledger is necessary to identify those discrepancies. A backup restores
its captured revision only. Restore makes no promise about later acknowledged
commits, maliciously rewritten checksums, or unavailable external receipt evidence.

Final software acceptance alone does not close milestone 4. Closure also requires
all six measured-platform recovery lanes and all 15 real ext4 ENOSPC cases to pass
on the exact published source, with accepted filesystem evidence. Until then the
source ledger retains **10 remaining milestones: 8 first-release and 2 deferred**.
PR #118 will record the later measured results and any resulting closure; the
next milestone's source update will carry that history forward. Power-loss
behavior remains explicitly unclaimed. Persisted-format/migration coverage,
mixed-operation concurrency, live-agent outcomes and release qualification remain
separate ledger obligations.
