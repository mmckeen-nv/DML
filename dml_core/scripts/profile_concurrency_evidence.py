"""Fail-closed CI evidence for the finite supported-profile concurrency matrix.

The JUnit properties contain counts and timings only. Submitted memory content,
receipts, HTTP credentials, captured output and private paths are never copied.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import gzip
import io
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import sqlite3
import subprocess
import xml.etree.ElementTree as ET

import pytest

from scripts.profile_recovery_evidence import filesystem, target_filesystem

SCHEMA = "dml-concurrency-evidence-v1"
SCALES = {"full": (1, 16, 64, 256), "bounded": (1, 16)}
TRANSPORTS = ("threads-shared", "threads-separate", "processes", "http")
REQUIRED_MODULES = {
    "test_profile_concurrency", "test_profile_concurrency_http",
    "test_profile_concurrency_runtime", "test_profile_concurrency_adversarial",
    "test_profile_concurrency_evidence",
}
COMMON_FIELDS = {
    "schema_version", "schema", "transport", "clients", "measured_ready_clients",
    "process_count", "committed_revisions", "duration_seconds", "first_progress_seconds",
    "history_verified", "history_events", "max_operation_seconds",
    "history_file", "history_sha256",
    "ownership_rejections", "max_attempts",
}
HTTP_FIELDS = {"max_http_in_flight", "max_active_workers", "worker_threads",
               "client_keepalive_expiry_seconds", "server_keepalive_timeout_seconds"}
HTTP_LIBRARIES = ("httpx", "httpcore", "h11", "uvicorn", "fastapi", "starlette")
FORK_CASE = "test_inherited_profile_rejects_all_authority_paths_before_work"
FORK_REASON = "POSIX fork inheritance is unavailable on this platform"


def _number(value, low, high):
    return type(value) in (int, float) and low <= value <= high and math.isfinite(value)


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Repeated JSON key")
        result[key] = value
    return result


def _json(value):
    if len(value) > 8192:
        raise ValueError("Oversized evidence property")
    return json.loads(value, object_pairs_hook=_pairs,
                      parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def validate_campaign(value, mode):
    """Return a strictly typed, content-free campaign or reject its claim."""
    if type(value) is not dict:
        raise ValueError("Campaign is not an object")
    transport = value.get("transport")
    required = COMMON_FIELDS | (HTTP_FIELDS if transport == "http" else set())
    if set(value) != required or value.get("schema_version") != SCHEMA:
        raise ValueError("Campaign fields or version differ from the contract")
    if type(value["schema"]) is not int or value["schema"] not in (2, 3, 4):
        raise ValueError("Unsupported journal schema")
    clients = value["clients"]
    if transport not in TRANSPORTS or type(clients) is not int or clients not in SCALES[mode]:
        raise ValueError("Campaign is outside the declared matrix")
    if not _integer(value["measured_ready_clients"], clients, clients):
        raise ValueError("Configured callers were not all measured ready")
    if not _integer(value["process_count"], 1, min(4, clients) if transport == "processes" else 1):
        raise ValueError("Invalid measured authority process count")
    if transport == "processes" and clients > 1 and value["process_count"] < 2:
        raise ValueError("Process campaign did not overlap multiple processes")
    if not _integer(value["committed_revisions"], 1, 40):
        raise ValueError("Unique commits exceed the bounded workload")
    if value["history_verified"] is not True or not _integer(value["history_events"], clients, 10000):
        raise ValueError("Missing independent complete-history verification")
    if not _integer(value["max_attempts"], 1, 3) or not _integer(value["ownership_rejections"], 0, 2 * value["history_events"]):
        raise ValueError("Invalid bounded ownership retry measurements")
    duration = value["duration_seconds"]
    if not _number(duration, 0, 90):
        raise ValueError("Campaign exceeded the completion deadline")
    if not _number(value["first_progress_seconds"], 0, min(duration, 10)):
        raise ValueError("Campaign exceeded the first-progress deadline")
    # HTTP client attempts begin before the server's Nth-arrival release gate.
    # Their elapsed time is bounded by the original campaign budget, not by the
    # shorter server-release-to-completion duration.
    if not _number(value["max_operation_seconds"], 0, 90):
        raise ValueError("Invalid maximum operation duration")
    if transport == "http":
        if not _integer(value["max_http_in_flight"], clients, value["history_events"]):
            raise ValueError("HTTP transport did not overlap all configured callers")
        if not _integer(value["max_active_workers"], 1, clients):
            raise ValueError("Invalid measured active HTTP workers")
        if not _integer(value["worker_threads"], value["max_active_workers"], clients * 8):
            raise ValueError("Invalid measured HTTP worker thread count")
        if (not _number(value["client_keepalive_expiry_seconds"], 1, 1)
                or not _number(value["server_keepalive_timeout_seconds"], 5, 5)):
            raise ValueError("HTTP transport differs from the declared idle connection policy")
    expected_file = f"schema{value['schema']}-{transport}-{clients}.json.gz"
    if value["history_file"] != expected_file or not isinstance(value["history_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["history_sha256"]):
        raise ValueError("Invalid history artifact identity")
    return dict(value)


def validate_history(campaign, directory):
    """Reopen a bounded synthetic artifact and independently replay its history."""
    from profile_concurrency_fixture import check_history

    directory = Path(directory).resolve(strict=True)
    path = directory / campaign["history_file"]
    if path.is_symlink() or path.resolve(strict=True).parent != directory or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Invalid history artifact path or size")
    compressed = path.read_bytes()
    if hashlib.sha256(compressed).hexdigest() != campaign["history_sha256"]:
        raise ValueError("History artifact digest mismatch")
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        raw = stream.read(32 * 1024 * 1024 + 1)
    if len(raw) > 32 * 1024 * 1024:
        raise ValueError("Decompressed history is oversized")
    envelope = json.loads(raw, object_pairs_hook=_pairs,
                          parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    if (type(envelope) is not dict or set(envelope) != {"schema_version", "before", "events", "after", "clients"}
            or envelope["schema_version"] != "dml-concurrency-history-v1"
            or type(envelope["clients"]) is not int or envelope["clients"] != campaign["clients"]
            or envelope["before"]["schema"] != campaign["schema"] or envelope["after"]["schema"] != campaign["schema"]):
        raise ValueError("History envelope does not identify this campaign")
    validate_attempt_transport(envelope["events"], campaign["transport"])
    checked = check_history(envelope["before"], envelope["events"], envelope["after"], clients=envelope["clients"])
    if (checked["history_verified"] is not True or checked["history_events"] != campaign["history_events"]
            or checked["unique_commits"] != campaign["committed_revisions"]
            or checked["ownership_rejections"] != campaign["ownership_rejections"]
            or checked["max_attempts"] != campaign["max_attempts"]
            or checked["max_operation_seconds"] != campaign["max_operation_seconds"]):
        raise ValueError("Persisted history disagrees with campaign measurements")


def validate_attempt_transport(events, transport):
    """A recorded HTTP history cannot silently shed its transport outcomes."""
    if type(events) is not list or not events:
        raise ValueError("Missing recorded events")
    common = {"started_ns", "finished_ns", "request_digest", "error_code"}
    for event in events:
        attempts = event["attempts"]
        if type(attempts) is not list or not 1 <= len(attempts) <= 3:
            raise ValueError("Missing bounded recorded attempts")
        for number, attempt in enumerate(attempts):
            final = number == len(attempts) - 1
            fields = common | ({"status_code"} if transport == "http" else set())
            if transport == "http" and not final:
                fields |= {"error_detail"}
            if type(attempt) is not dict or set(attempt) != fields:
                raise ValueError("Recorded attempt differs from its transport schema")
            if transport == "http":
                if type(attempt["status_code"]) is not int or attempt["status_code"] != (200 if final else 503):
                    raise ValueError("Missing actual HTTP attempt outcome")
                if not final:
                    detail = attempt["error_detail"]
                    operation = event.get("request", {}).get("operation")
                    if operation in {"ingest", "update", "promote", "supersede", "retire"}:
                        expected = {"code": "receipt_ownership_unavailable", "retry_same_key": True}
                    elif operation == "retrieve":
                        expected = {"code": "retrieval_outcome_unavailable", "reason": "store_ownership_timeout"}
                    else:
                        raise ValueError("This operation has no qualified HTTP ownership retry")
                    if (type(detail) is not dict or detail != expected
                            or "retry_same_key" in detail and detail["retry_same_key"] is not True):
                        raise ValueError("Retry was not an explicit HTTP ownership rejection")


def validate_lock_evidence(value):
    required = {"schema_version", "schema", "clients", "timing_domain", "successful_acquisitions",
                "blocked_attempts", "unfinished_acquisitions", "max_wait_seconds", "injected_owner_delay_seconds"}
    if (type(value) is not dict or set(value) != required
            or value["schema_version"] != "dml-ownership-wait-evidence-v1"
            or type(value["schema"]) is not int or value["schema"] != 3
            or type(value["clients"]) is not int or value["clients"] != 16
            or value["timing_domain"] != "first_os_lock_attempt_to_acquired"
            or not _integer(value["successful_acquisitions"], 1, 10000)
            or not _integer(value["blocked_attempts"], 1, 1000000)
            or not _integer(value["unfinished_acquisitions"], 0, 0)
            or not _number(value["max_wait_seconds"], 0, 32)
            or not _number(value["injected_owner_delay_seconds"], 0.002, 0.002)):
        raise ValueError("Missing bounded mixed-contention OS ownership measurements")
    return dict(value)


def validate_junit(payload, *, mode, system, history_directory=None):
    """Validate every case and the exact matrix without exporting JUnit content."""
    if mode not in SCALES:
        raise ValueError("Unknown qualification mode")
    if len(payload) > 32 * 1024 * 1024 or b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
        raise ValueError("Oversized or entity-bearing JUnit report")
    root = ET.fromstring(payload)
    if root.tag not in {"testsuites", "testsuite"}:
        raise ValueError("Invalid JUnit report root")
    errors, campaigns, identities, modules, skips, waits, releases = [], {}, set(), set(), [], [], []
    outcomes, locks = {}, []
    cases = list(root.iter("testcase"))
    for case in cases:
        identity = (case.get("classname", ""), case.get("name", ""))
        module, function = identity[0].rsplit(".", 1)[-1], identity[1].split("[", 1)[0]
        if not all(identity) or identity in identities:
            errors.append("Missing or repeated JUnit case identity")
        identities.add(identity)
        if case.find("failure") is not None or case.find("error") is not None:
            errors.append("JUnit contains a failed case")
        skip = case.find("skipped")
        if skip is not None:
            if (system == "Windows" and module == "test_profile_concurrency_runtime"
                    and function == FORK_CASE and skip.get("message") == FORK_REASON):
                skips.append({"module": module, "case": function, "reason": FORK_REASON})
            else:
                errors.append("JUnit contains an unqualified skip")
        else:
            modules.add(module)
        outcomes[identity] = "skipped" if skip is not None else "failed" if case.find("failure") is not None or case.find("error") is not None else "passed"
        seen_properties = set()
        for prop in case.findall("properties/property"):
            name = prop.get("name")
            if name not in {"concurrency_evidence", "actual_ownership_timeout_seconds", "ownership_release_progress_seconds", "concurrency_lock_evidence"}:
                continue
            if name in seen_properties:
                errors.append("Repeated evidence property within one case")
            seen_properties.add(name)
            try:
                value = _json(prop.get("value", ""))
                if name == "concurrency_evidence":
                    campaign = validate_campaign(value, mode)
                    expected_module = "test_profile_concurrency_http" if campaign["transport"] == "http" else "test_profile_concurrency"
                    if module != expected_module or skip is not None:
                        raise ValueError("Campaign evidence came from the wrong case module")
                    key = (campaign["schema"], campaign["transport"], campaign["clients"])
                    if key in campaigns:
                        raise ValueError("Duplicate campaign cell")
                    if history_directory is not None:
                        validate_history(campaign, history_directory)
                    campaigns[key] = campaign
                elif name == "concurrency_lock_evidence":
                    if module != "test_profile_concurrency" or skip is not None:
                        raise ValueError("Mixed-contention evidence came from the wrong module")
                    locks.append(validate_lock_evidence(value))
                elif name == "actual_ownership_timeout_seconds":
                    if (module != "test_profile_concurrency_runtime" or type(value) is not list
                            or len(value) != 6 or not all(_number(item, 29.8, 32) for item in value)):
                        raise ValueError("Missing actual default ownership timeout measurements")
                    waits.append(value)
                else:
                    if module != "test_profile_concurrency_runtime" or not _number(value, 0, 10):
                        raise ValueError("Missing bounded ownership release progress")
                    releases.append(value)
            except (ValueError, TypeError, KeyError, AssertionError, OSError, EOFError, IndexError):
                # Never copy the untrusted property or exception text into artifacts.
                errors.append("Invalid or duplicate qualification property")
    expected = {(schema, transport, clients) for schema in (2, 3, 4)
                for transport in TRANSPORTS for clients in SCALES[mode]}
    if set(campaigns) != expected:
        errors.append("Missing or unexpected concurrency matrix cells")
    if not REQUIRED_MODULES.issubset(modules):
        errors.append("Missing passing required test modules")
    if len(waits) != 1 or len(releases) != 1:
        errors.append("Missing or repeated default ownership deadline qualification")
    if len(locks) != 1:
        errors.append("Missing or repeated mixed-contention ownership qualification")
    return {"campaigns": [campaigns[key] for key in sorted(campaigns)], "errors": sorted(set(errors)), "_case_outcomes": outcomes,
            "case_count": len(cases), "platform_skips": skips,
            "mixed_contention_ownership": locks[0] if len(locks) == 1 else None,
            "actual_ownership_timeout_seconds": waits[0] if len(waits) == 1 else None,
            "ownership_release_progress_seconds": releases[0] if len(releases) == 1 else None}


def source_identity(root, excluded=()):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root).decode("utf-8")

    status = git("status", "--porcelain=v1", "-z", "--untracked-files=all").split("\0")
    # Only new test outputs are excluded. Tracked modifications are always dirty.
    dirty = any(item and (not item.startswith("?? ") or not any(
        (root / item[3:]).resolve() == path or path in (root / item[3:]).resolve().parents
        for path in excluded)) for item in status)
    tracked = git("ls-files", "-z").split("\0")
    hashes = {}
    for name in tracked:
        if (name.startswith(("dml_core/daystrom_dml/", ".github/")) and name.endswith((".py", ".yml", ".yaml"))
                or "concurrency" in name and name.endswith(".py")
                or name in {"pyproject.toml", "dml_core/tests/profile_crash_fixture.py", "dml_core/scripts/profile_recovery_evidence.py"}):
            hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    return {"commit": git("rev-parse", "HEAD").strip(), "tree": git("rev-parse", "HEAD^{tree}").strip(),
            "dirty": dirty, "sha256": hashes}


def http_library_identity():
    """Record installed transport implementations rather than assume versions."""
    versions = {}
    for name in HTTP_LIBRARIES:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError("Required HTTP runtime distribution is missing") from error
        if type(version) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+!_-]{0,127}", version):
            raise ValueError("Invalid HTTP runtime distribution version")
        versions[name] = version
    return versions


def pytest_addoption(parser):
    group = parser.getgroup("DML concurrency evidence")
    group.addoption("--concurrency-evidence", help="Write measured qualification JSON")
    group.addoption("--concurrency-mode", choices=tuple(SCALES), default="full")


def pytest_configure(config):
    output = config.getoption("--concurrency-evidence")
    if output:
        config.pluginmanager.register(ConcurrencyEvidence(config, Path(output)), "dml-concurrency-recorder")


class ConcurrencyEvidence:
    def __init__(self, config, output):
        self.config, self.output = config, output.resolve()
        self.errors, self.phases = [], {}
        self.collected = set()
        self.started = None

    def pytest_sessionstart(self, session):
        from daystrom_dml.journal import require_patched_sqlite
        require_patched_sqlite()
        option = self.config.option
        if (not option.basetemp or not getattr(option, "xmlpath", None) or option.keyword or option.markexpr
                or getattr(option, "lf", False) or getattr(option, "ff", False)
                or any("::" in arg for arg in self.config.args)):
            raise pytest.UsageError("Concurrency evidence requires complete modules, explicit --basetemp and --junitxml")
        self.mode = self.config.getoption("--concurrency-mode")
        if os.environ.get("DML_CONCURRENCY_SCALES", "1,16") != ",".join(map(str, SCALES[self.mode])):
            raise pytest.UsageError("DML_CONCURRENCY_SCALES must exactly match --concurrency-mode")
        self.basetemp = Path(option.basetemp).absolute()
        self.parent = self.basetemp.resolve().parent
        self.parent.mkdir(parents=True, exist_ok=True)
        self.junit = Path(option.xmlpath).resolve()
        history_directory = os.environ.get("DML_CONCURRENCY_HISTORY_DIR")
        if not history_directory:
            raise pytest.UsageError("Concurrency evidence requires DML_CONCURRENCY_HISTORY_DIR")
        self.history_directory = Path(history_directory).resolve()
        self.excluded = (self.basetemp.resolve(), self.junit, self.output)
        self.initial_fs = filesystem(self.parent)
        self.initial_source = source_identity(self.config.rootpath, self.excluded)
        self.http_libraries = http_library_identity()
        with sqlite3.connect(":memory:") as connection:
            self.sqlite = connection.execute("select sqlite_version(), sqlite_source_id()").fetchone()
        self.started = datetime.now(timezone.utc).isoformat()

    def pytest_collection_finish(self, session):
        self.collected = {item.nodeid for item in session.items}
        if len(self.collected) != len(session.items):
            self.errors.append("Repeated collected test identity")

    def pytest_runtest_logreport(self, report):
        phases = self.phases.setdefault(report.nodeid, {})
        if report.when in phases:
            self.errors.append("Repeated test phase")
        phases[report.when] = report.outcome

    def pytest_deselected(self, items):
        if items:
            self.errors.append("Concurrency cases were deselected")

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        if self.started is None:
            return
        errors = list(self.errors)
        measured, current_source, current_fs, payload = {}, None, None, None
        try:
            payload = self.junit.read_bytes()
            measured = validate_junit(payload, mode=self.mode, system=platform.system(), history_directory=self.history_directory)
            errors.extend(measured["errors"])
        except (OSError, ValueError, ET.ParseError):
            errors.append("JUnit report could not be validated")
        try:
            current_source = source_identity(self.config.rootpath, self.excluded)
            if self.initial_source["dirty"] or current_source["dirty"] or current_source != self.initial_source:
                errors.append("Source identity changed or source tree is dirty")
        except (OSError, subprocess.SubprocessError, UnicodeError):
            errors.append("Source identity could not be measured")
        try:
            if http_library_identity() != self.http_libraries:
                errors.append("HTTP runtime distributions changed during qualification")
        except ValueError:
            errors.append("HTTP runtime distributions could not be measured")
        try:
            current_fs = filesystem(self.parent)
            if (self.basetemp.is_symlink() or not self.basetemp.is_dir()
                    or filesystem(self.basetemp) != self.initial_fs or current_fs != self.initial_fs
                    or not target_filesystem(current_fs)):
                errors.append("Actual test filesystem is changed or unqualified")
        except (OSError, ValueError):
            errors.append("Actual test filesystem could not be measured")
        if exitstatus != 0:
            errors.append("Pytest did not exit successfully")
        if len(self.phases) != session.testscollected or measured.get("case_count") != session.testscollected:
            errors.append("Collected, executed and JUnit case counts differ")
        if set(self.phases) != self.collected:
            errors.append("Collected and executed case identities differ")
        actual_outcomes = {}
        for node, phases in self.phases.items():
            path, bracket, parameters = node.partition("[")
            names = path.split("::")
            names[0] = names[0].replace("/", ".").removesuffix(".py")
            names[-1] += bracket + parameters
            prefix = getattr(self.config.option, "junitprefix", None)
            if prefix:
                names.insert(0, prefix)
            identity = (".".join(names[:-1]), names[-1])
            actual_outcomes[identity] = "failed" if "failed" in phases.values() else "skipped" if "skipped" in phases.values() else "passed"
        if actual_outcomes != measured.get("_case_outcomes"):
            errors.append("Executed case identities or outcomes differ from JUnit")
        for phases in self.phases.values():
            expected = {"setup": "passed", "call": "passed", "teardown": "passed"}
            if phases != expected and not (phases == {"setup": "skipped", "teardown": "passed"}
                                          or phases == {"setup": "passed", "call": "skipped", "teardown": "passed"}):
                errors.append("Incomplete or failed test phases")
        result = {
            "schema_version": 1, "kind": SCHEMA, "mode": self.mode,
            "scope": "finite measured mixed-operation histories; no arbitrary-scheduler or production latency guarantee",
            "schemas": [2, 3, 4], "transports": list(TRANSPORTS), "scales": list(SCALES[self.mode]),
            "started_at_utc": self.started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": self.initial_source["commit"], "source_tree": self.initial_source["tree"],
            "source_sha256": self.initial_source["sha256"],
            "source_identity_verified": current_source == self.initial_source and not self.initial_source["dirty"],
            "python": platform.python_version(), "platform": platform.system(), "kernel": platform.release(),
            "architecture": platform.machine(), "sqlite_version": self.sqlite[0], "sqlite_source_id": self.sqlite[1],
            "http_libraries": self.http_libraries,
            "filesystem": current_fs, "power_loss_qualified": False,
            "junit_sha256": hashlib.sha256(payload).hexdigest() if payload is not None else None,
            **{key: value for key, value in measured.items() if key != "errors" and not key.startswith("_")},
            "accepted": not errors, "errors": sorted(set(errors)),
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if errors:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
