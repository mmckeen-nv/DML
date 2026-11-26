"""Actual ext4 ENOSPC qualification on a dedicated, bounded CI loopback volume.

The base suite skips when that volume is unavailable. The dedicated CI lane sets
REQUIRE_FILESYSTEM_FULL_TESTS=1 so missing/unsafe capability is a test failure.
No host/root/shared filesystem is filled, and SQLite quota is not substituted.
"""
from __future__ import annotations

import errno
import json
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


def fill_until_enospc(filler, probe, mount):
    """Exhaust real ext4 allocation, including the tail below one MiB.

    Fallocate avoids mistaking a delayed-allocation write/fsync failure for an
    exhausted filesystem. Both descriptors remain open through the attempted
    SQLite commit, so close cannot release reservations before that attempt.
    """
    initial = os.statvfs(mount)
    block_size = initial.f_frsize
    assert 0 < block_size <= 1024 * 1024
    assert (1024 * 1024) % block_size == 0
    assert initial.f_bavail > 0, "The volume must begin with usable capacity"
    handles = (filler, probe)
    offsets, attempts, exhausted_attempts = [0, 0], [0, 0], [0, 0]

    def allocate(index, amount):
        if sum(offsets) + amount > MAX_VOLUME_BYTES:
            pytest.fail("Allocation reached its aggregate byte limit without stable exhaustion")
        attempts[index] += 1
        try:
            os.posix_fallocate(handles[index].fileno(), offsets[index], amount)
        except OSError as exc:
            if exc.errno != errno.ENOSPC:
                raise
            exhausted_attempts[index] += 1
            return False
        offsets[index] += amount
        return True

    while allocate(0, 1024 * 1024):
        pass
    for round_number in range(1, MAX_VOLUME_BYTES // block_size + 3):
        # Different inode layouts can still accept blocks after the first inode
        # reports ENOSPC. Consume those real allocations, advancing each offset.
        progress = [allocate(index, block_size) for index in range(2)]
        if any(progress):
            continue
        for handle in handles:
            os.fsync(handle.fileno())
        # Recheck both next extents after synchronization. Only a complete pass
        # with no successful allocation can establish the terminal condition.
        progress = [allocate(index, block_size) for index in range(2)]
        if any(progress):
            continue
        final = os.statvfs(mount)
        assert final.f_bavail == 0, "Kernel ENOSPC left usable filesystem blocks"
        physical_bytes = [os.fstat(handle.fileno()).st_blocks * 512 for handle in handles]
        assert 0 < sum(physical_bytes) <= MAX_VOLUME_BYTES
        assert sum(offsets) > 0
        return {
            "schema_version": "dml-real-enospc-evidence-v1",
            "filesystem": "ext4", "allocator": "posix_fallocate",
            "volume_bytes": initial.f_blocks * block_size,
            "block_bytes": block_size, "initial_available_blocks": initial.f_bavail,
            "allocated_bytes": physical_bytes[0], "probe_allocated_bytes": physical_bytes[1],
            "total_allocated_bytes": sum(physical_bytes),
            "confirmed_extent_bytes": offsets[0], "probe_confirmed_extent_bytes": offsets[1],
            "allocation_attempts": attempts[0], "probe_allocation_attempts": attempts[1],
            "enospc_attempts": sum(exhausted_attempts),
            "filler_enospc_attempts": exhausted_attempts[0], "probe_enospc_attempts": exhausted_attempts[1],
            "allocation_rounds": round_number,
            "filler_errno": errno.ENOSPC, "separate_inode_probe_errno": errno.ENOSPC,
            "final_free_blocks": final.f_bfree, "final_available_blocks": final.f_bavail,
            "descriptors_held_open": not filler.closed and not probe.closed,
        }
    pytest.fail("Bounded allocation passes did not produce stable filesystem ENOSPC")


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
def test_real_ext4_enospc_preserves_receipts_and_retry_commits_once(schema, operation, request):
    mount = dedicated_volume()
    with TemporaryDirectory(prefix="dml-enospc-case-", dir=mount) as scratch:
        case = Path(scratch)
        scenario = prepare(case / "authority", schema)
        operation_request = request_for(operation, scenario.records)
        filler = case / "synthetic-filler.bin"
        probe = case / "synthetic-allocation-probe.bin"
        reached = []
        adapter = make_adapter(scenario.directory, schema)
        try:
            with filler.open("xb", buffering=0) as filler_handle, probe.open("xb", buffering=0) as probe_handle:
                def exhaust(point):
                    if point == "after_begin":
                        assert not reached, "The fault must target exactly one attempted commit"
                        evidence = fill_until_enospc(filler_handle, probe_handle, mount)
                        reached.append(evidence)
                        request.node.user_properties.append(("enospc_evidence", json.dumps(evidence, sort_keys=True)))

                adapter._journal._fault_hook = exhaust
                with pytest.raises((ReceiptCommitRejected, ReceiptCommitUncertain)) as observed:
                    invoke(adapter, operation_request)
                assert reached and reached[0]["filler_errno"] == errno.ENOSPC
                assert contains_sqlite_full(observed.value), "The journal must actually encounter SQLITE_FULL"
                request.node.user_properties.append(("sqlite_full_verified", "true"))
        finally:
            # Free capacity before close/recovery, even if the operation or oracle failed.
            try:
                filler.unlink(missing_ok=True)
            finally:
                try:
                    probe.unlink(missing_ok=True)
                finally:
                    adapter.close(persist=False)
        assert sql_observation(scenario.directory) == scenario.before
        reopened = make_adapter(scenario.directory, schema)
        try:
            assert sql_observation(scenario.directory) == scenario.before
            receipt = invoke(reopened, operation_request)
            after = sql_observation(scenario.directory)
            assert receipt == assert_transition(scenario.before, after, operation_request)
            # Replay is historical, and must not create a second revision.
            assert invoke(reopened, operation_request) == receipt
            assert sql_observation(scenario.directory) == after
            request.node.user_properties.append(("recovery_verified", "true"))
        finally:
            reopened.close(persist=False)
