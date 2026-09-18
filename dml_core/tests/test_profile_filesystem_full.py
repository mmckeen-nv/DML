"""Actual ext4 ENOSPC qualification on a dedicated, bounded CI loopback volume.

The base suite skips when that volume is unavailable. The dedicated CI lane sets
REQUIRE_FILESYSTEM_FULL_TESTS=1 so missing/unsafe capability is a test failure.
No host/root/shared filesystem is filled, and SQLite quota is not substituted.
"""
from __future__ import annotations

import errno
import os
from pathlib import Path
import re
import sqlite3
import sys
from tempfile import TemporaryDirectory

import pytest

from profile_crash_fixture import (
    OPERATIONS, SCHEMAS, assert_transition, invoke, make_adapter, prepare,
    request_for, sql_observation,
)
from daystrom_dml.services.receipt_ingestion import ReceiptCommitRejected, ReceiptCommitUncertain


MAX_VOLUME_BYTES = 128 * 1024 * 1024
MARKER = ".dml-ci-enospc-volume"
MARKER_BYTES = b"dml-ci-enospc-v1\n"


def unavailable(reason):
    if os.environ.get("REQUIRE_FILESYSTEM_FULL_TESTS") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def dedicated_volume():
    raw = os.environ.get("RECOVERY_ENOSPC_MOUNT")
    if not raw or not sys.platform.startswith("linux"):
        unavailable("Actual ENOSPC requires the dedicated Linux ext4 CI volume")
    candidate = Path(raw)
    if not candidate.is_absolute() or candidate.is_symlink():
        unavailable("ENOSPC volume must be an absolute non-symlink mount path")
    mount = candidate.resolve()
    if not mount.is_dir() or not os.path.ismount(mount):
        unavailable("ENOSPC volume is not a mounted directory")
    if mount.stat().st_dev == mount.parent.stat().st_dev:
        unavailable("ENOSPC volume must have a separate filesystem device")
    volume = os.statvfs(mount)
    total = volume.f_blocks * volume.f_frsize
    if not 0 < total <= MAX_VOLUME_BYTES:
        unavailable("ENOSPC volume must have a maximum capacity of 128 MiB")
    marker = mount / MARKER
    if marker.is_symlink() or not marker.is_file() or marker.read_bytes() != MARKER_BYTES:
        unavailable("ENOSPC volume lacks the explicit dedicated-volume marker")
    # Mountinfo escapes whitespace and backslashes with octal sequences.
    def unescape(value):
        return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)

    matched = []
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        separator = fields.index("-")
        if unescape(fields[4]) == str(mount):
            matched.append(fields[separator + 1])
    if matched != ["ext4"]:
        unavailable("ENOSPC qualification requires the dedicated ext4 mount")
    return mount


def fill_until_enospc(path):
    """Allocate bounded real bytes and require the kernel's ENOSPC response."""
    block = bytes(range(256)) * 4096
    assert len(block) == 1024 * 1024
    total = 0
    with path.open("xb", buffering=0) as filler:
        while total < MAX_VOLUME_BYTES:
            try:
                written = filler.write(block[:min(len(block), MAX_VOLUME_BYTES - total)])
                assert written is not None and written > 0
                total += written
                # Force ext4 delayed allocation to become an actual allocation.
                os.fsync(filler.fileno())
            except OSError as exc:
                if exc.errno != errno.ENOSPC:
                    raise
                assert total > 0, "The volume must begin with usable capacity"
                return {"errno": exc.errno, "bytes_written": total}
    pytest.fail("Bounded allocation did not produce real filesystem ENOSPC")


def contains_sqlite_full(error):
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, sqlite3.Error):
            code = getattr(current, "sqlite_errorcode", None)
            if (code is not None and code & 255 == 13) or "database or disk is full" in str(current).lower():
                return True
        pending.extend(item for item in (current.__cause__, current.__context__) if item is not None)
    return False


@pytest.fixture(autouse=True)
def isolated_configuration(tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("DML_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("schema", SCHEMAS)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_real_ext4_enospc_preserves_receipts_and_retry_commits_once(schema, operation):
    mount = dedicated_volume()
    with TemporaryDirectory(prefix="dml-enospc-case-", dir=mount) as scratch:
        case = Path(scratch)
        scenario = prepare(case / "authority", schema)
        request = request_for(operation, scenario.records)
        filler = case / "synthetic-filler.bin"
        reached = []
        adapter = make_adapter(scenario.directory, schema)

        def exhaust(point):
            if point == "after_begin":
                assert not reached, "The fault must target exactly one attempted commit"
                reached.append(fill_until_enospc(filler))

        adapter._journal._fault_hook = exhaust
        try:
            with pytest.raises((ReceiptCommitRejected, ReceiptCommitUncertain)) as observed:
                invoke(adapter, request)
            assert reached and reached[0]["errno"] == errno.ENOSPC
            assert contains_sqlite_full(observed.value), "The journal must actually encounter SQLITE_FULL"
        finally:
            # Free capacity before close/recovery, even if the operation or oracle failed.
            filler.unlink(missing_ok=True)
            adapter.close(persist=False)
        assert sql_observation(scenario.directory) == scenario.before
        reopened = make_adapter(scenario.directory, schema)
        try:
            assert sql_observation(scenario.directory) == scenario.before
            receipt = invoke(reopened, request)
            after = sql_observation(scenario.directory)
            assert receipt == assert_transition(scenario.before, after, request)
            # Replay is historical, and must not create a second revision.
            assert invoke(reopened, request) == receipt
            assert sql_observation(scenario.directory) == after
        finally:
            reopened.close(persist=False)
