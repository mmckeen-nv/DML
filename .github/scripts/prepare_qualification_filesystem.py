"""Prepare a measured filesystem for finite CI recovery/concurrency tests.

Only an ephemeral GitHub-hosted Linux runner may use this helper. If its actual
root ext4 filesystem disables barriers, enable them on that same filesystem and
verify the readback. Never create a filesystem that conceals rejected backing
storage. The ordinary qualification recorders still check the test filesystem
before and after their campaigns; this setup makes no power-loss claim.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

from scripts.profile_recovery_evidence import filesystem, target_filesystem

DISABLED_BARRIERS = {"nobarrier", "barrier=0"}
BARRIER_OPTIONS = DISABLED_BARRIERS | {"barrier", "barrier=1"}
KINDS = ("recovery", "concurrency")


def require_hosted_linux():
    if (sys.platform != "linux" or os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("RUNNER_OS") != "Linux"
            or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"):
        raise RuntimeError("Filesystem setup is restricted to GitHub-hosted Linux runners")


def observe(path):
    path = path.resolve(strict=True)
    row = {"path": str(path)}
    try:
        if not path.is_dir():
            raise ValueError("Candidate test filesystem parent is not a directory")
        row["filesystem"] = filesystem(path)
        row["admitted"] = target_filesystem(row["filesystem"])
        row["free_bytes"] = shutil.disk_usage(path).free
        row["findmnt"] = json.loads(subprocess.check_output([
            "findmnt", "--json", "--target", str(path),
            "--output", "TARGET,SOURCE,FSTYPE,OPTIONS"], text=True))
        mounts = row["findmnt"]["filesystems"]
        if len(mounts) != 1 or not all(isinstance(mounts[0].get(key), str)
                                      for key in ("target", "source", "fstype", "options")):
            raise ValueError("Expected one complete measured mount")
        row["mount"] = mounts[0]
        mount = Path(row["mount"]["target"]).resolve(strict=True)
        if (mount != path and mount not in path.parents
                or mount.stat().st_dev != row["filesystem"]["device_id"]
                or row["mount"]["fstype"] != row["filesystem"]["type"]):
            raise ValueError("Filesystem and mount measurements disagree")
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def admitted(row):
    return "error" not in row and target_filesystem(row.get("filesystem", {}))


def repairable(row):
    if "error" in row:
        return False
    measured = row.get("filesystem", {})
    options = measured.get("options", [])
    return (measured.get("type") == "ext4" and bool(DISABLED_BARRIERS.intersection(options))
            and target_filesystem({**measured, "options": [value for value in options
                                                           if value not in DISABLED_BARRIERS]}))


def identity(row):
    return (row["filesystem"]["device_id"], row["filesystem"]["type"],
            row["mount"]["source"], row["mount"]["target"], row["mount"]["fstype"])


def require_root_block(row):
    if (row["mount"]["target"] != "/" or row["mount"]["fstype"] != "ext4"
            or not row["mount"]["source"].startswith("/dev/")
            or Path(row["mount"]["source"]).name.startswith("loop")):
        raise RuntimeError("Barrier repair requires the actual existing root ext4 block filesystem")
    source = Path(row["mount"]["source"]).resolve(strict=True)
    device = source.stat()
    if (source.name.startswith("loop") or os.major(device.st_rdev) == 7
            or not stat.S_ISBLK(device.st_mode) or device.st_rdev != row["filesystem"]["device_id"]):
        raise RuntimeError("Root block device does not match the measured filesystem")


def enable_root_barriers(candidate, evidence):
    require_hosted_linux()
    before = observe(Path("/"))
    repair = evidence["barrier_repair"] = {"before": before, "after": None, "accepted": False}
    if not repairable(candidate) or not repairable(before) or identity(candidate) != identity(before):
        raise RuntimeError("Only a stable root filesystem rejected solely for disabled barriers may be repaired")
    require_root_block(before)
    command = ["sudo", "mount", "-o", "remount,barrier=1", "/"]
    repair["command"] = command
    try:
        result = subprocess.run(command, check=False, text=True, capture_output=True, timeout=60)
        repair.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.SubprocessError) as error:
        repair["command_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        repair["after"] = observe(Path("/"))
    after = repair["after"]
    if result.returncode:
        raise RuntimeError("Enabling root filesystem barriers failed")
    if (not admitted(after) or identity(before) != identity(after)
            or set(before["mount"]["options"].split(",")) - BARRIER_OPTIONS
            != set(after["mount"]["options"].split(",")) - BARRIER_OPTIONS
            or DISABLED_BARRIERS.intersection(after["mount"]["options"].split(","))):
        raise RuntimeError("Actual root filesystem readback did not confirm the bounded barrier repair")
    require_root_block(after)
    repair["accepted"] = True


def candidates():
    rows = []
    for value in (os.environ["RUNNER_TEMP"], "/mnt"):
        try:
            row = observe(Path(value))
        except OSError as error:
            row = {"error": f"{type(error).__name__}: {error}"}
        rows.append({"requested_path": value, **row})
    return rows


def prepare(kind):
    require_hosted_linux()
    output = Path(f"profile-{kind}-artifacts/filesystem-selection.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence = {"schema_version": "dml-ci-qualification-filesystem-v1", "kind": kind,
                "scope": "actual existing filesystem setup for finite CI qualification",
                "power_loss_qualified": False, "candidates": [], "selected": None}
    # Reserve the evidence path before any mount or directory mutation.
    with output.open("x", encoding="utf-8") as report:
        try:
            evidence["candidates"] = candidates()
            selected = next((row for row in evidence["candidates"] if admitted(row)), None)
            if selected is None:
                eligible = next((row for row in evidence["candidates"] if repairable(row)), None)
                if eligible is None:
                    raise RuntimeError("No existing candidate passes or is eligible for the bounded barrier repair")
                enable_root_barriers(eligible, evidence)
                evidence["candidates_after_repair"] = candidates()
                selected = next((row for row in evidence["candidates_after_repair"] if admitted(row)), None)
                if selected is None:
                    raise RuntimeError("No candidate passes after the actual root barrier repair")
            parent = Path(selected["path"])
            owned = Path(subprocess.check_output([
                "sudo", "mktemp", "-d", str(parent / f"dml-{kind}.XXXXXX")], text=True).strip())
            if (owned.parent != parent or not owned.name.startswith(f"dml-{kind}.")
                    or owned.is_symlink() or any(value in str(owned) for value in ("\n", "\r", "\0"))):
                raise RuntimeError("Disposable test directory identity differs")
            subprocess.run(["sudo", "chown", f"{os.getuid()}:{os.getgid()}", str(owned)], check=True)
            (owned / f".dml-ci-{kind}").write_text(f"dml-ci-{kind}-v1\n", encoding="utf-8")
            with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as environment:
                prefix = f"DML_{kind.upper()}"
                environment.write(f"{prefix}_PARENT={parent}\n{prefix}_OWNED={owned}\n")
                environment.write(f"{prefix}_BASETEMP={owned / 'stores'}\n")
            allocated = observe(owned)
            evidence["allocated"] = allocated
            if not admitted(allocated) or allocated["filesystem"] != selected["filesystem"]:
                raise RuntimeError("Allocated test directory is not on the selected admitted filesystem")
            evidence["selected"] = {"parent": str(parent), "directory": str(owned),
                                    "filesystem": allocated["filesystem"], "mount": allocated["mount"]}
        except BaseException as error:
            evidence["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            report.write(json.dumps(evidence, sort_keys=True, indent=2) + "\n")
            report.flush()
            os.fsync(report.fileno())


def cleanup(kind):
    require_hosted_linux()
    prefix = f"DML_{kind.upper()}"
    value = os.environ.get(f"{prefix}_OWNED")
    if value:
        owned = Path(value)
        parent = Path(os.environ[f"{prefix}_PARENT"]).resolve(strict=True)
        marker = owned / f".dml-ci-{kind}"
        if (owned.is_symlink() or owned.parent != parent or not owned.name.startswith(f"dml-{kind}.")
                or marker.is_symlink() or marker.read_text(encoding="utf-8") != f"dml-ci-{kind}-v1\n"):
            raise RuntimeError("Refusing cleanup of an unowned test directory")
        shutil.rmtree(owned)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "cleanup"))
    parser.add_argument("--kind", choices=KINDS, required=True)
    args = parser.parse_args()
    (prepare if args.action == "prepare" else cleanup)(args.kind)


if __name__ == "__main__":
    main()
