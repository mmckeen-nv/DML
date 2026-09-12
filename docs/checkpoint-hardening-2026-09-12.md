# Serial hardening: semantic checkpoints

This feature follows the accepted [journal gate](journal-hardening-2026-09-12.md).
Its scope is semantic checkpoint publication, retention, shutdown and the adapter
ownership needed for those operations. Native KV compatibility and restore,
all-component transactions and durable decision replay remain separate gates.

## Review gate

A checkpoint builder agent implemented the changes; the primary agent built an
independent harness and adapter integration fixes. A separate agent reviewed the
result and ran independent interleaving oracles. The threshold is at least 9.5/10,
with any known shutdown, destructive retention or misleading-outcome defect
blocking acceptance.

Final independent grade: **9.5/10 — accepted for this feature**. The final focused
integration run passed **80 tests**, including **26 checkpoint regressions**; Ruff
and the maintained mypy selection (44 files, local target Python 3.12) also passed.
The [review record](artifacts/checkpoint-review-2026-09-12.json) preserves both
grades, category scores and exact accepted-source hashes. Remote final-commit CI
remains a separate merge gate.

The reviewer assigned **9.3/10 — rejected** to an intermediate implementation:
after releasing directory ownership, an older retention failure could overwrite
the status of a newer successful publication. The first attempted correction was
also rejected: merely acquiring directory ownership could hide an older failure
before maintenance actually succeeded. The resulting fix orders publication and
retention outcomes independently, using completed outcomes to suppress stale
updates. Explicit lock-release interleavings protect both successful recovery and
the still-degraded interval before maintenance finishes. Per-caller return/error
behavior remains intact even when an obsolete attempt cannot replace newer status.

Other corrected findings were duplicate worker starts, shutdown returning before
a provider drained, untracked manual managers when periodic checkpointing was
disabled, publication success hidden by cleanup failure, mtime-based retention
under clock rollback, and arbitrary provider exception text entering logs.

## Resulting contract

- One adapter-owned manager serves periodic and manual checkpoints, including
  interval zero. Concurrent start calls create at most one worker.
- Close is terminal. `close(timeout=1.0)` returns true only after all admitted
  operations and the worker drain. False means still draining; a blocked Python
  provider cannot be forcibly canceled. A pending provider cannot begin publication
  after close, while an already-admitted publication may finish during a false
  result. Start/checkpoint after close raise `CheckpointClosedError`.
- Adapter close drains the manager before final persistence or closing the memory
  store. Incomplete drain raises `TimeoutError`; callers can release the blocked
  dependency and retry close. This is not a guarantee about every other background
  subsystem's shutdown behavior.
- Providers return owned JSON snapshots. Their existing JSON shape, including
  JSON's numeric-key normalization, is frozen before waiting for directory
  ownership. Ambiguous duplicate keys and non-finite values are rejected.
- Names contain monotonic publication order, UUID and SHA-256 content digest.
  Publication order is independent of filesystem mtime and tolerates clock rollback.
  It does not prove semantic freshness of an opaque provider snapshot.
- Retention validates recognized new files before deletion. Corrupt, digest-invalid
  or nonregular recognized files stop all pruning and degrade status. Legacy and
  unrecognized files are preserved outside the quota for explicit operator handling.
- Once publication succeeds, later retention failure returns the published path
  and separately reports `retention_error`. An atomic-write error after replacement
  reports uncertain durability and preserves prior snapshots. The returned path is
  a publication receipt, not a lease against later retention by another caller.
- Logs/status omit provider exception messages and tracebacks. Status is local to
  the manager process; it is not durable operator replay, and checksums do not
  authenticate against a writer who can replace both filenames and content.

## Harness and evidence

The executable regression file is `dml_core/tests/test_checkpoint_hardening.py`.
It includes six actual subprocess-death boundaries, six independent publishers
producing 36 receipts under a fixed clock, blocked provider and started-write
shutdown, a real interval-zero adapter shutdown test, corrupt/legacy/symlink
retention, JSON ambiguity, directory anchoring, uncertain writes and the rejected
status race. The parent process supplies the expected publication history.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DML_SKIP_VENV_REEXEC=1 \
  python -m pytest dml_core/tests/test_checkpoint_hardening.py dml_core/tests/test_fail_closed_persistence.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DML_SKIP_VENV_REEXEC=1 \
  python -m pytest dml_core/daystrom_dml/tests dml_core/tests integrations/hermes/tests openclaw-wrapper/tests
```

Before the final status-race correction, the full local suite passed **1,005 tests**
with nine optional skips. The focused checkpoint/integration suite is rerun on
the final correction, and remote CI must pass on the published commit. The CI
portability matrix now includes this harness on Linux, macOS and Windows. A
platform-specific skipped symlink test is missing platform evidence, not proof of
symlink support. Process-kill tests do not establish hardware power-loss behavior.

The next production gate remains receipt-based ingestion and transactional
projection ownership. These two accepted hardening features do not make the
platform production-ready.
