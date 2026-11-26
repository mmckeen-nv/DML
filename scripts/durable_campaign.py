#!/usr/bin/env python3
"""Single-use systemd campaign supervisor. No generation retry or resume."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def publish(path, value, *, replace=False):
    path = Path(path)
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    tmp = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        tmp.unlink(missing_ok=True)


def verify_freeze(root):
    freeze = json.loads((root / "freeze.json").read_text())
    for name, expected in freeze["files"].items():
        if not Path(name).is_absolute() or digest(name) != expected:
            raise ValueError("Frozen file differs: " + name)
    if not freeze["command"] or not all(type(v) is str for v in freeze["command"]):
        raise ValueError("Invalid command")
    return freeze


def finalize(root):
    if not (root / "terminal.json").exists():
        try:
            publish(root / "terminal.json", {
                "status": "interrupted", "time": time.time(),
                "reason": "supervisor_exited_without_terminal_receipt",
                "service_result": os.environ.get("SERVICE_RESULT", "unknown"),
                "exit_code": os.environ.get("EXIT_CODE", "unknown"),
                "exit_status": os.environ.get("EXIT_STATUS", "unknown"),
                "usage_unknown": True, "resumable": False,
            })
        except FileExistsError:
            pass


def launch(root, unit):
    verify_freeze(root)
    publish(root / "launch.json", {"time": time.time(), "unit": unit,
            "freeze_sha256": digest(root / "freeze.json"), "resumable": False})
    script = str(Path(__file__).resolve())
    command = ["systemd-run", "--user", "--unit=" + unit,
               "--property=Restart=no", "--property=KillMode=control-group",
               "--property=TimeoutStopSec=15", "--property=UMask=0077",
               "--property=ExecStopPost=" + sys.executable + " " + script + " finalize " + str(root),
               sys.executable, script, "worker", str(root)]
    result = subprocess.run(command, check=False)
    if result.returncode:
        publish(root / "terminal.json", {"status": "launch_failed", "time": time.time(),
                "returncode": result.returncode, "resumable": False})
    return result.returncode


def worker(root):
    os.umask(0o077)
    with (root / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch_receipt = json.loads((root / "launch.json").read_text())
        if digest(root / "freeze.json") != launch_receipt["freeze_sha256"]:
            raise ValueError("Freeze differs from permanent launch receipt")
        # Both a permanent claim and a kernel lock: even a finished run cannot restart.
        publish(root / "start.json", {"time": time.time(), "pid": os.getpid(),
                "parent_pid": os.getppid(), "freeze_sha256": digest(root / "freeze.json"),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()})
        process = None
        interrupted = False

        def stop(signum, frame):
            nonlocal interrupted
            interrupted = True
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            freeze = verify_freeze(root)
            env = {**os.environ, **freeze.get("environment", {})}
            with (root / "worker.log").open("xb", buffering=0) as log:
                if interrupted:
                    raise InterruptedError("Stopped before worker launch")
                process = subprocess.Popen(freeze["command"], cwd=freeze["cwd"],
                    env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                started = time.monotonic()
                while process.poll() is None:
                    reports = sorted(root.glob("episodes/**/report.json"))
                    publish(root / "progress.json", {"time": time.time(), "child_pid": process.pid,
                        "elapsed_seconds": time.monotonic() - started,
                        "completed_task_reports": len(reports)}, replace=True)
                    if interrupted or time.monotonic() - started > freeze.get("max_seconds", 3600):
                        stop(signal.SIGTERM, None)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        break
                    time.sleep(1)
                code = process.wait()
                os.fsync(log.fileno())
            campaign = root / "campaign.json"
            status = "completed" if code in (0, 1) and campaign.exists() else "failed"
            if interrupted or code < 0:
                status = "interrupted"
            if freeze.get("synthetic") and code == 0:
                status = "completed_synthetic"
            publish(root / "terminal.json", {"status": status, "time": time.time(),
                "returncode": code, "resumable": False, "usage_unknown": status == "interrupted",
                "campaign_sha256": digest(campaign) if campaign.exists() else None,
                "retained_reports": {str(p.relative_to(root)): digest(p)
                    for p in sorted(root.glob("episodes/**/report.json"))}})
            return code if code >= 0 else 128 - code
        except BaseException as exc:
            if process is not None and process.poll() is None:
                stop(signal.SIGTERM, None)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            publish(root / "terminal.json", {"status": "supervisor_error", "time": time.time(),
                "error_type": type(exc).__name__, "resumable": False, "usage_unknown": True})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("launch", "worker", "finalize", "reconcile"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--unit")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.operation == "launch":
        if not args.unit or not all(c.isalnum() or c in "-_" for c in args.unit):
            parser.error("A simple unique --unit is required")
        return launch(root, args.unit)
    if args.operation == "worker":
        return worker(root)
    if args.operation == "reconcile":
        if (root / "terminal.json").exists():
            return 0
        start = json.loads((root / "start.json").read_text())
        current_boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if start["boot_id"] == current_boot:
            with (root / "worker.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                finalize(root)
        else:
            finalize(root)
        return 0
    finalize(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
