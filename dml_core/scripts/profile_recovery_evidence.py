"""Pytest plugin recording measured recovery evidence, never power-loss claims.

Use with an explicit disposable --basetemp and --recovery-evidence output file.
The output describes the filesystem actually holding those test stores.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone

import pytest

REQUIRED_MODULES = {
    "test_production_profile_recovery.py", "test_profile_recovery_campaign.py",
    "test_profile_storage_faults.py", "test_authority_backup.py",
    "test_crash_qualification_adversarial.py", "test_profile_recovery_evidence.py",
}
NO_SKIP_MODULES = {"test_production_profile_recovery.py", "test_profile_recovery_campaign.py"}
MINIMUM_PASSES = {"test_production_profile_recovery.py": 185, "test_profile_recovery_campaign.py": 55}
WINDOWS_SKIPS = {
    "test_directory_open_failure_reports_visible_but_unconfirmed_publication",
    "test_directory_fsync_failure_closes_descriptor_and_preserves_publication",
}
TARGET_FILESYSTEMS = {"ext4", "xfs", "btrfs", "apfs", "ntfs"}
SAFE_OPTIONS = {"rw", "ro", "sync", "async", "dirsync", "noatime", "relatime", "strictatime",
                "lazytime", "nobarrier", "volatile", "data=ordered", "data=journal", "data=writeback",
                "barrier=0", "barrier=1", "fsync=volatile", "errors=remount-ro", "errors=continue"}


def _unescape_mount(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def linux_filesystem(path: Path, mountinfo: str, *, device_id: int | None = None) -> dict:
    candidates = []
    for line in mountinfo.splitlines():
        left, separator, right = line.partition(" - ")
        before, after = left.split(), right.split()
        if not separator or len(before) < 6 or len(after) < 3:
            continue
        if device_id is not None and before[2] != f"{os.major(device_id)}:{os.minor(device_id)}":
            continue
        mount = Path(_unescape_mount(before[4]))
        if path == mount or mount in path.parents:
            options = set(before[5].split(",")) | set(after[2].split(","))
            candidates.append((len(mount.parts), after[0], sorted(options & SAFE_OPTIONS)))
    if not candidates:
        raise ValueError("Filesystem mount could not be measured")
    depth = max(item[0] for item in candidates)
    deepest = {(name, tuple(options)) for length, name, options in candidates if length == depth}
    if len(deepest) != 1:
        raise ValueError("Conflicting filesystem mount measurements")
    name, selected_options = deepest.pop()
    options = list(selected_options)
    return {"type": name, "options": options, "source": "proc-self-mountinfo", "local": name in TARGET_FILESYSTEMS}


def _darwin_filesystem(path: Path) -> dict:
    # Darwin's 64-bit statfs layout; Apple statfs(2), sys/mount.h.
    class StatFS(ctypes.Structure):
        _fields_ = [("bsize", ctypes.c_uint32), ("iosize", ctypes.c_int32),
                    *[(name, ctypes.c_uint64) for name in ("blocks", "bfree", "bavail", "files", "ffree")],
                    ("fsid", ctypes.c_int32 * 2), ("owner", ctypes.c_uint32),
                    ("type", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("subtype", ctypes.c_uint32),
                    ("name", ctypes.c_char * 16), ("mount", ctypes.c_char * 1024),
                    ("device", ctypes.c_char * 1024), ("reserved", ctypes.c_uint32 * 8)]
    library = ctypes.CDLL(None, use_errno=True)
    call = library.statfs
    call.argtypes = [ctypes.c_char_p, ctypes.POINTER(StatFS)]
    call.restype = ctypes.c_int
    result = StatFS()
    if call(os.fsencode(path), ctypes.byref(result)) != 0:
        raise OSError(ctypes.get_errno(), "Filesystem measurement failed")
    options = ["ro" if result.flags & 1 else "rw"]
    if result.flags & 2:
        options.append("sync")
    if result.flags & 0x40:
        options.append("async")
    return {"type": result.name.decode("ascii").lower(), "options": options,
            "source": "darwin-statfs", "mount_flags": result.flags, "local": bool(result.flags & 0x1000)}


def _windows_filesystem(path: Path) -> dict:
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    root, name = ctypes.create_unicode_buffer(32768), ctypes.create_unicode_buffer(256)
    volume_path = kernel.GetVolumePathNameW
    volume_path.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    volume_path.restype = wintypes.BOOL
    if not volume_path(str(path), root, len(root)):
        raise ctypes.WinError(ctypes.get_last_error())
    info = kernel.GetVolumeInformationW
    info.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD,
                     ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
                     ctypes.POINTER(wintypes.DWORD), wintypes.LPWSTR, wintypes.DWORD]
    info.restype = wintypes.BOOL
    flags = wintypes.DWORD()
    if not info(root.value, None, 0, None, None, ctypes.byref(flags), name, len(name)):
        raise ctypes.WinError(ctypes.get_last_error())
    drive_type = kernel.GetDriveTypeW
    drive_type.argtypes = [wintypes.LPCWSTR]
    drive_type.restype = wintypes.UINT
    return {"type": name.value.lower(), "options": [], "source": "windows-volume-information",
            "volume_flags": flags.value, "local": drive_type(root.value) in {2, 3, 6}}


def filesystem(path: Path) -> dict:
    path = path.resolve(strict=True)
    if sys.platform == "linux":
        result = linux_filesystem(path, Path("/proc/self/mountinfo").read_text(), device_id=path.stat().st_dev)
    elif sys.platform == "darwin":
        result = _darwin_filesystem(path)
    elif sys.platform == "win32":
        result = _windows_filesystem(path)
    else:
        raise ValueError("Unsupported filesystem measurement platform")
    result["device_id"] = path.stat().st_dev
    return result


def target_filesystem(observed: dict) -> bool:
    return (observed.get("type") in TARGET_FILESYSTEMS and observed.get("local") is True
            and observed.get("source") in {"proc-self-mountinfo", "darwin-statfs", "windows-volume-information"}
            and type(observed.get("device_id")) is int and type(observed.get("options")) is list
            and all(type(option) is str and option in SAFE_OPTIONS for option in observed["options"])
            and not {"ro", "volatile", "fsync=volatile", "nobarrier", "barrier=0"}.intersection(observed.get("options", [])))


def pytest_addoption(parser):
    group = parser.getgroup("DML recovery evidence")
    group.addoption("--recovery-evidence", help="Write measured recovery evidence to this JSON file")
    group.addoption("--require-recovery-filesystem", action="store_true", help="Reject unmeasured or non-target test filesystems")


def pytest_configure(config):
    output = config.getoption("--recovery-evidence")
    if output:
        config.pluginmanager.register(RecoveryEvidence(config, Path(output)), "dml-recovery-recorder")


class RecoveryEvidence:
    def __init__(self, config, output: Path):
        self.config, self.output = config, output.resolve()
        self.cases: dict[str, dict] = {}
        self.errors: list[str] = []
        self.initial = None

    def pytest_sessionstart(self, session):
        from daystrom_dml.journal import require_patched_sqlite
        require_patched_sqlite()
        basetemp = self.config.option.basetemp
        if not basetemp:
            raise pytest.UsageError("Recovery evidence requires an explicit disposable --basetemp")
        if (self.config.option.keyword or self.config.option.markexpr or
                getattr(self.config.option, "lf", False) or getattr(self.config.option, "ff", False) or
                any("::" in arg for arg in self.config.args)):
            raise pytest.UsageError("Recovery evidence requires complete module selection without filters")
        # pytest owns/removes basetemp. Its parent is the measured mount.
        self.basetemp = Path(basetemp).absolute()
        self.parent = self.basetemp.resolve().parent
        self.parent.mkdir(parents=True, exist_ok=True)
        self.initial = filesystem(self.parent)
        if self.config.getoption("--require-recovery-filesystem") and not target_filesystem(self.initial):
            raise pytest.UsageError("Recovery qualification requires a measured local target filesystem without volatile/disabled barriers")
        self.started = datetime.now(timezone.utc).isoformat()
        self.commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.config.rootpath, text=True).strip()
        self.source_tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=self.config.rootpath, text=True).strip()
        self.dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=self.config.rootpath, text=True).strip())
        with sqlite3.connect(":memory:") as connection:
            self.sqlite = connection.execute("select sqlite_version(), sqlite_source_id()").fetchone()

    def pytest_runtest_logreport(self, report):
        case = self.cases.setdefault(report.nodeid, {"phases": {}, "duration_seconds": 0.0})
        case["phases"][report.when] = report.outcome
        case["duration_seconds"] += report.duration
        if report.skipped:
            case["skip_reason"] = str(report.longrepr)[-1000:]

    def pytest_deselected(self, items):
        if items:
            self.errors.append("Recovery cases were deselected")

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        if self.initial is None:
            return
        observed = filesystem(self.parent)
        if self.basetemp.is_symlink() or not self.basetemp.is_dir() or filesystem(self.basetemp) != self.initial:
            self.errors.append("Actual test directory does not match measured filesystem")
        modules = {Path(node.split("::")[0]).name for node in self.cases}
        if not REQUIRED_MODULES.issubset(modules):
            self.errors.append("Missing required recovery test modules")
        if observed != self.initial:
            self.errors.append("Test filesystem changed during the campaign")
        if len(self.cases) != session.testscollected:
            self.errors.append("Not every collected test produced evidence")
        passed_by_module = {}
        for node, case in self.cases.items():
            phases = case["phases"]
            if "failed" in phases.values():
                self.errors.append("Test failure: " + node)
            if "skipped" in phases.values():
                module = Path(node.split("::")[0]).name
                function = node.split("::")[-1].split("[")[0]
                if not (os.name == "nt" and module == "test_profile_storage_faults.py" and function in WINDOWS_SKIPS
                        and "Windows helper has no directory fsync barrier" in case.get("skip_reason", "")):
                    self.errors.append("Unqualified recovery case skipped: " + node)
            elif phases != {"setup": "passed", "call": "passed", "teardown": "passed"}:
                self.errors.append("Incomplete case: " + node)
            else:
                module = Path(node.split("::")[0]).name
                passed_by_module[module] = passed_by_module.get(module, 0) + 1
        for module in REQUIRED_MODULES:
            if passed_by_module.get(module, 0) < MINIMUM_PASSES.get(module, 1):
                self.errors.append("Insufficient passing coverage: " + module)
        if exitstatus != 0:
            self.errors.append("Pytest did not exit successfully")
        junit = Path(self.config.option.xmlpath) if getattr(self.config.option, "xmlpath", None) else None
        if junit is None or not junit.is_file():
            self.errors.append("Required JUnit report is missing")
        result = {
            "schema_version": 1, "kind": "dml-profile-recovery-evidence-v1",
            "started_at_utc": self.started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": self.commit, "source_tree": self.source_tree, "working_tree_dirty": self.dirty,
            "python": platform.python_version(), "platform": platform.system(),
            "kernel": platform.release(), "architecture": platform.machine(),
            "sqlite_version": self.sqlite[0], "sqlite_source_id": self.sqlite[1],
            "filesystem": observed, "target_filesystem": target_filesystem(observed),
            "directory_sync_available": os.name != "nt", "power_loss_qualified": False,
            "scope": "process termination, caught faults and verified recovery on this measured runner",
            "collected_cases": session.testscollected, "cases": self.cases,
            "junit_sha256": hashlib.sha256(junit.read_bytes()).hexdigest() if junit and junit.is_file() else None,
            "tests_passed": not self.errors and exitstatus == 0,
            "accepted": not self.errors and exitstatus == 0 and target_filesystem(observed), "errors": self.errors,
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        if self.errors:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
