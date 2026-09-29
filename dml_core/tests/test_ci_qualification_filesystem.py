"""Fault controls for actual hosted-runner filesystem setup; no real remounts."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

if sys.platform != "linux":
    pytest.skip("Hosted ext4 setup controls require Linux block device identifiers", allow_module_level=True)


@pytest.fixture
def helper(monkeypatch):
    path = Path(__file__).resolve().parents[2] / ".github/scripts/prepare_qualification_filesystem.py"
    spec = importlib.util.spec_from_file_location("ci_qualification_filesystem", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.sys, "platform", "linux")
    for name, value in {"GITHUB_ACTIONS": "true", "RUNNER_OS": "Linux",
                        "RUNNER_ENVIRONMENT": "github-hosted"}.items():
        monkeypatch.setenv(name, value)
    return module


def observation(path="/", options=None):
    if options is None:
        options = ["data=writeback", "errors=remount-ro", "nobarrier", "relatime", "rw"]
    measured = {"device_id": 2049, "local": True, "options": options,
                "source": "proc-self-mountinfo", "type": "ext4"}
    mount = {"target": "/", "source": "/dev/sda1", "fstype": "ext4",
             "options": ",".join(options + ["discard", "journal_async_commit", "commit=30"])}
    return {"path": str(path), "filesystem": measured, "mount": mount,
            "findmnt": {"filesystems": [mount]}, "free_bytes": 1024**3}


def repaired(path="/"):
    return observation(path, ["data=writeback", "errors=remount-ro", "relatime", "rw"])


@pytest.mark.parametrize("name,value", [("GITHUB_ACTIONS", "false"), ("RUNNER_OS", "Windows"),
                                       ("RUNNER_ENVIRONMENT", "self-hosted")])
def test_refuses_non_hosted_environment_before_mutation(helper, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    command = Mock()
    monkeypatch.setattr(helper.subprocess, "run", command)
    with pytest.raises(RuntimeError, match="GitHub-hosted Linux"):
        helper.prepare("concurrency")
    command.assert_not_called()


@pytest.mark.parametrize("options", [["nobarrier", "ro"], ["nobarrier", "volatile", "rw"],
                                      ["barrier=0", "fsync=volatile", "rw"], ["nobarrier", "unknown", "rw"]])
def test_other_disqualifiers_cannot_be_repaired(helper, options):
    assert not helper.repairable(observation(options=options))


def test_guard_itself_is_unchanged_and_rejects_disabled_barriers(helper):
    assert not helper.target_filesystem(observation()["filesystem"])
    assert helper.target_filesystem(repaired()["filesystem"])
    assert helper.repairable(observation(options=["barrier=0", "rw"]))


def setup_repair(helper, monkeypatch, after=None, result=None):
    before = observation()
    after = copy.deepcopy(after if after is not None else repaired())
    observe = Mock(side_effect=[before, after])
    monkeypatch.setattr(helper, "observe", observe)
    monkeypatch.setattr(helper, "require_root_block", Mock())
    command = Mock(return_value=result or subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(helper.subprocess, "run", command)
    return before, after, observe, command


def test_real_root_repair_requires_actual_readback_without_fabricated_barrier_flag(helper, monkeypatch):
    before, after, observe, command = setup_repair(helper, monkeypatch)
    evidence = {}
    helper.enable_root_barriers(before, evidence)
    command.assert_called_once_with(["sudo", "mount", "-o", "remount,barrier=1", "/"],
                                    check=False, text=True, capture_output=True, timeout=60)
    assert observe.call_count == 2
    assert evidence["barrier_repair"]["accepted"]
    assert evidence["barrier_repair"]["before"] == before
    assert evidence["barrier_repair"]["after"] == after
    assert "barrier=1" not in after["filesystem"]["options"]


@pytest.mark.parametrize("field,value", [("device_id", 2050), ("type", "xfs"), ("local", False)])
def test_actual_filesystem_drift_rejects(helper, monkeypatch, field, value):
    after = repaired()
    after["filesystem"][field] = value
    before, after, _, _ = setup_repair(helper, monkeypatch, after)
    evidence = {}
    with pytest.raises(RuntimeError, match="readback"):
        helper.enable_root_barriers(before, evidence)
    assert evidence["barrier_repair"]["after"] == after
    assert not evidence["barrier_repair"]["accepted"]


@pytest.mark.parametrize("field,value", [("source", "/dev/sdb1"), ("target", "/mnt"),
                                       ("options", "rw,nobarrier"), ("options", "rw,relatime")])
def test_mount_source_target_or_unrelated_option_drift_rejects(helper, monkeypatch, field, value):
    after = repaired()
    after["mount"][field] = value
    before, _, _, _ = setup_repair(helper, monkeypatch, after)
    with pytest.raises(RuntimeError, match="readback"):
        helper.enable_root_barriers(before, {})


def test_command_failure_retains_after_readback(helper, monkeypatch):
    before, after, _, _ = setup_repair(helper, monkeypatch,
                                      result=subprocess.CompletedProcess([], 32, "", "mount refused"))
    evidence = {}
    with pytest.raises(RuntimeError, match="barriers failed"):
        helper.enable_root_barriers(before, evidence)
    repair = evidence["barrier_repair"]
    assert repair["returncode"] == 32 and repair["stderr"] == "mount refused"
    assert repair["after"] == after and not repair["accepted"]


def test_command_exception_retains_after_readback(helper, monkeypatch):
    before, after, _, command = setup_repair(helper, monkeypatch)
    command.side_effect = subprocess.TimeoutExpired("mount", 60)
    evidence = {}
    with pytest.raises(subprocess.TimeoutExpired):
        helper.enable_root_barriers(before, evidence)
    repair = evidence["barrier_repair"]
    assert "TimeoutExpired" in repair["command_error"]
    assert repair["after"] == after and not repair["accepted"]


def test_candidate_and_actual_root_must_match_before_any_command(helper, monkeypatch):
    before, _, _, command = setup_repair(helper, monkeypatch)
    candidate = copy.deepcopy(before)
    candidate["filesystem"]["device_id"] = 2050
    with pytest.raises(RuntimeError, match="stable root"):
        helper.enable_root_barriers(candidate, {})
    command.assert_not_called()


@pytest.mark.parametrize("target,source", [("/mnt", "/dev/sda1"), ("/", "overlay"),
                                           ("/", "/dev/loop0")])
def test_unexpected_backing_mount_refused(helper, target, source):
    row = observation()
    row["mount"].update(target=target, source=source)
    with pytest.raises(RuntimeError, match="actual existing root"):
        helper.require_root_block(row)


@pytest.mark.parametrize("resolved_name,mode,device", [("loop0", stat.S_IFBLK, 2049),
                                                      ("alias", stat.S_IFBLK, os.makedev(7, 0)),
                                                      ("sda1", stat.S_IFREG, 2049),
                                                      ("sda1", stat.S_IFBLK, 2050)])
def test_resolved_block_device_identity_and_loop_aliases_refused(helper, monkeypatch, resolved_name, mode, device):
    resolved = SimpleNamespace(name=resolved_name, stat=lambda: SimpleNamespace(st_mode=mode, st_rdev=device))
    monkeypatch.setattr(helper, "Path", lambda value: SimpleNamespace(
        name="by-uuid", resolve=lambda strict: resolved))
    with pytest.raises(RuntimeError, match="Root block device"):
        helper.require_root_block(observation())


def test_prepare_refusal_retains_complete_candidate_observation(helper, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    rejected = observation(options=["nobarrier", "ro"])
    monkeypatch.setattr(helper, "candidates", lambda: [rejected])
    command = Mock()
    monkeypatch.setattr(helper.subprocess, "run", command)
    with pytest.raises(RuntimeError, match="No existing candidate"):
        helper.prepare("recovery")
    evidence = json.loads((tmp_path / "profile-recovery-artifacts/filesystem-selection.json").read_text())
    assert evidence["candidates"] == [rejected]
    assert evidence["selected"] is None and not evidence["power_loss_qualified"]
    command.assert_not_called()


def test_existing_evidence_refuses_any_mount_or_directory_mutation(helper, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "profile-recovery-artifacts/filesystem-selection.json"
    output.parent.mkdir()
    output.write_text("prior evidence")
    observed = Mock()
    monkeypatch.setattr(helper, "candidates", observed)
    with pytest.raises(FileExistsError):
        helper.prepare("recovery")
    observed.assert_not_called()
    assert output.read_text() == "prior evidence"


@pytest.mark.parametrize("kind", ["recovery", "concurrency"])
def test_admitted_existing_storage_allocates_records_and_cleans_without_remount(helper, monkeypatch, tmp_path, kind):
    monkeypatch.chdir(tmp_path)
    parent = tmp_path / "parent"
    parent.mkdir()
    selected = repaired(parent)
    monkeypatch.setattr(helper, "candidates", lambda: [selected])
    owned = parent / f"dml-{kind}.ABCDEF"
    owned.mkdir()
    monkeypatch.setattr(helper.subprocess, "check_output", Mock(return_value=str(owned) + "\n"))
    command = Mock()
    monkeypatch.setattr(helper.subprocess, "run", command)
    monkeypatch.setattr(helper, "observe", lambda path: repaired(path))
    environment = tmp_path / "github-env"
    monkeypatch.setenv("GITHUB_ENV", str(environment))
    helper.prepare(kind)
    assert command.call_count == 1 and command.call_args.args[0][:2] == ["sudo", "chown"]
    evidence = json.loads((tmp_path / f"profile-{kind}-artifacts/filesystem-selection.json").read_text())
    assert evidence["selected"]["directory"] == str(owned)
    assert "barrier_repair" not in evidence
    for line in environment.read_text().splitlines():
        name, value = line.split("=", 1)
        monkeypatch.setenv(name, value)
    helper.cleanup(kind)
    assert not owned.exists()


@pytest.mark.parametrize("fault", ["marker", "marker_symlink", "directory_symlink", "parent"])
def test_cleanup_refuses_unowned_or_redirected_directory(helper, monkeypatch, tmp_path, fault):
    owned = tmp_path / "dml-recovery.ABCDEF"
    owned.mkdir()
    marker = owned / ".dml-ci-recovery"
    marker.write_text("dml-ci-recovery-v1\n")
    monkeypatch.setenv("DML_RECOVERY_OWNED", str(owned))
    monkeypatch.setenv("DML_RECOVERY_PARENT", str(tmp_path))
    if fault == "marker":
        marker.write_text("unowned")
    elif fault == "marker_symlink":
        original = tmp_path / "original-marker"
        marker.rename(original)
        marker.symlink_to(original)
    elif fault == "directory_symlink":
        original = tmp_path / "original-directory"
        owned.rename(original)
        owned.symlink_to(original, target_is_directory=True)
    else:
        monkeypatch.setenv("DML_RECOVERY_PARENT", str(tmp_path.parent))
    with pytest.raises(RuntimeError, match="unowned"):
        helper.cleanup("recovery")
    assert owned.exists()
