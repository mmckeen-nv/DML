"""Evidence gate unit regressions plus unmocked native filesystem/SQLite probes.

Synthetic reports below test recorder rejection logic only. They are not actual
qualification campaigns and are never emitted as accepted runner evidence.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace
import zipfile

import pytest

from scripts import profile_recovery_evidence as evidence

REPO = Path(__file__).resolve().parents[2]
GOOD_FS = {"type": "ext4", "options": ["relatime", "rw"], "source": "proc-self-mountinfo",
           "local": True, "device_id": 1}
MODULES = {
    "test_production_profile_recovery.py", "test_profile_recovery_campaign.py",
    "test_profile_storage_faults.py", "test_authority_backup.py",
    "test_crash_qualification_adversarial.py", "test_profile_recovery_evidence.py",
}


@pytest.fixture
def bootstrap():
    spec = importlib.util.spec_from_file_location("sqlite_bootstrap_under_test", REPO / ".github/scripts/prepare_sqlite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_linux_mount_probe_chooses_containing_mount_and_omits_private_paths():
    observed = evidence.linux_filesystem(Path("/runner/stores/case"),
        "1 0 8:0 / / rw,relatime - ext4 /dev/private-device rw,errors=remount-ro\n"
        "2 1 0:42 / /runner rw,relatime - overlay overlay rw,lowerdir=/PRIVATE/lower,upperdir=/PRIVATE/upper\n"
        "3 2 8:1 /PRIVATE/source /runner/stores rw,sync - xfs /dev/PRIVATE rw,attr2,inode64\n")
    assert observed["type"] == "xfs"
    assert observed["options"] == ["rw", "sync"]
    raw = json.dumps(observed)
    assert "PRIVATE" not in raw and "/dev/" not in raw and "/runner" not in raw
    assert observed["source"] == "proc-self-mountinfo"


def test_linux_mount_probe_decodes_spaces_and_avoids_path_prefix_confusion():
    observed = evidence.linux_filesystem(Path("/volume space/store/case"),
        "1 0 8:0 / / rw - ext4 /dev/root rw\n"
        r"2 1 8:1 / /volume\040space/store rw - xfs /dev/disk rw" + "\n"
        r"3 1 8:2 / /volume\040space/store-other rw - btrfs /dev/other rw" + "\n")
    assert observed["type"] == "xfs"


def test_linux_mount_probe_rejects_unmeasured_path():
    with pytest.raises(ValueError, match="could not be measured"):
        evidence.linux_filesystem(Path("/actual/store"), "malformed\n1 0 8:0 / /elsewhere rw - ext4 /dev/root rw")


@pytest.mark.parametrize("observed", [
    {}, {"type": "ext4"}, {"type": "ext4", "options": ["rw"], "source": "unverified"},
    {**GOOD_FS, "type": "overlay"}, {**GOOD_FS, "type": "nfs4"},
    {**GOOD_FS, "remote": True, "local": False}, {**GOOD_FS, "options": ["rw", "barrier=0"]},
    {**GOOD_FS, "options": ["ro"]},
], ids=["missing", "type-only", "unknown-source", "overlay", "network", "remote", "barrier-disabled", "readonly"])
def test_incomplete_or_non_target_filesystem_never_qualifies(observed):
    assert evidence.target_filesystem(observed) is False


def test_native_filesystem_probe_reports_actual_host_without_path_disclosure(tmp_path):
    # This call is deliberately unmocked on every CI platform. Linux independently
    # compares against the kernel's mount inventory read in this test.
    observed = evidence.filesystem(tmp_path)
    assert isinstance(observed["type"], str) and observed["type"]
    assert observed["source"] == {
        "linux": "proc-self-mountinfo", "darwin": "darwin-statfs", "win32": "windows-volume-information",
    }[sys.platform]
    if sys.platform == "linux":
        assert observed == {**evidence.linux_filesystem(tmp_path.resolve(), Path("/proc/self/mountinfo").read_text()),
                            "device_id": tmp_path.stat().st_dev}
    assert str(tmp_path) not in json.dumps(observed)
    assert all(option in evidence.SAFE_OPTIONS for option in observed["options"])


def recorder(tmp_path, monkeypatch):
    """Synthetic successful reports are explicitly unit-test inputs."""
    options = SimpleNamespace(basetemp=str(tmp_path / "cases"), xmlpath=str(tmp_path / "junit.xml"),
                              keyword="", markexpr="", lf=False)
    Path(options.xmlpath).write_bytes(b'<testsuites><testsuite tests="1"/></testsuites>')
    arguments = [str(REPO / "dml_core/tests" / module) for module in sorted(MODULES)]
    config = SimpleNamespace(option=options, args=arguments, rootpath=REPO,
                             getoption=lambda name: name == "--require-recovery-filesystem")
    instance = evidence.RecoveryEvidence(config, tmp_path / "unit-recorder.json")
    monkeypatch.setattr(evidence, "filesystem", lambda _path: dict(GOOD_FS))
    instance.pytest_sessionstart(SimpleNamespace())
    Path(options.basetemp).mkdir()
    nodes = []
    # Exercise a full deterministic inventory; other modules need positive cases.
    # This is synthetic acceptance-gate data, not a substitute for their execution.
    for module in sorted(MODULES):
        for number in range({"test_production_profile_recovery.py": 185,
                             "test_profile_recovery_campaign.py": 55}.get(module, 1)):
            node = f"dml_core/tests/{module}::test_unit_report_{number}"
            nodes.append(node)
            for phase in ("setup", "call", "teardown"):
                instance.pytest_runtest_logreport(SimpleNamespace(nodeid=node, when=phase, outcome="passed",
                    duration=0.001, skipped=False, longrepr=None))
    session = SimpleNamespace(testscollected=len(nodes), exitstatus=0)
    return instance, session, nodes


def finish(instance, session, status=0):
    instance.pytest_sessionfinish(session, status)
    return json.loads(instance.output.read_text(encoding="utf-8"))


def test_complete_unit_reports_bind_runtime_junit_and_never_claim_power_loss(tmp_path, monkeypatch):
    instance, session, _ = recorder(tmp_path, monkeypatch)
    result = finish(instance, session)
    assert result["accepted"] is True
    assert result["source_commit"]
    with sqlite3.connect(":memory:") as connection:
        assert result["sqlite_version"] == connection.execute("select sqlite_version()").fetchone()[0]
    assert result["junit_sha256"] == hashlib.sha256(Path(instance.config.option.xmlpath).read_bytes()).hexdigest()
    assert result["power_loss_qualified"] is False


def test_mandatory_skip_cannot_produce_accepted_evidence(tmp_path, monkeypatch):
    instance, session, nodes = recorder(tmp_path, monkeypatch)
    node = next(node for node in nodes if "test_profile_recovery_campaign.py" in node)
    instance.cases[node]["phases"]["call"] = "skipped"
    assert finish(instance, session)["accepted"] is False
    assert session.exitstatus != 0


def test_partial_deterministic_inventory_cannot_produce_accepted_evidence(tmp_path, monkeypatch):
    instance, session, nodes = recorder(tmp_path, monkeypatch)
    removed = [node for node in nodes if "test_production_profile_recovery.py" in node][1:]
    for node in removed:
        del instance.cases[node]
    session.testscollected -= len(removed)
    assert finish(instance, session)["accepted"] is False


def test_missing_required_module_cannot_produce_accepted_evidence(tmp_path, monkeypatch):
    instance, session, nodes = recorder(tmp_path, monkeypatch)
    node = next(node for node in nodes if "test_authority_backup.py" in node)
    del instance.cases[node]
    session.testscollected -= 1
    assert finish(instance, session)["accepted"] is False


def test_incomplete_test_phases_cannot_produce_accepted_evidence(tmp_path, monkeypatch):
    instance, session, nodes = recorder(tmp_path, monkeypatch)
    instance.cases[nodes[0]]["phases"].pop("teardown")
    assert finish(instance, session)["accepted"] is False


def test_failed_pytest_exit_cannot_produce_accepted_evidence(tmp_path, monkeypatch):
    instance, session, _ = recorder(tmp_path, monkeypatch)
    assert finish(instance, session, status=1)["accepted"] is False


def test_changed_filesystem_cannot_produce_accepted_evidence(tmp_path, monkeypatch):
    instance, session, _ = recorder(tmp_path, monkeypatch)
    monkeypatch.setattr(evidence, "filesystem", lambda _path: {**GOOD_FS, "type": "xfs"})
    assert finish(instance, session)["accepted"] is False


def test_unqualified_filesystem_never_receives_acceptance_even_without_require_flag(tmp_path, monkeypatch):
    instance, session, _ = recorder(tmp_path, monkeypatch)
    instance.initial = {**GOOD_FS, "type": "overlay"}
    instance.config.getoption = lambda _name: False
    monkeypatch.setattr(evidence, "filesystem", lambda _path: dict(instance.initial))
    assert finish(instance, session)["accepted"] is False


def test_skipped_only_noncritical_module_cannot_count_as_positive_coverage(tmp_path, monkeypatch):
    instance, session, nodes = recorder(tmp_path, monkeypatch)
    node = next(node for node in nodes if "test_authority_backup.py" in node)
    instance.cases[node]["phases"] = {"setup": "skipped", "teardown": "passed"}
    assert finish(instance, session)["accepted"] is False


def test_bootstrap_archive_pins_are_exact_and_fixed(bootstrap):
    assert bootstrap.VERSION == "3.53.4"
    assert bootstrap.PRODUCT == "3530400"
    assert bootstrap.ARCHIVES == {
        "source": ("sqlite-amalgamation-3530400.zip", "628a44cfe82c66aed1ccbbe85a562d2e33ebe64b3288981ed76285612227934e"),
        "windows": ("sqlite-dll-win-x64-3530400.zip", "deddee963c810d1eeac3ce5e15c7c41da21a1c54d7a39cf54fbf577d2f50de3a"),
    }


@pytest.mark.parametrize("tampered", [False, True], ids=["valid-sha3", "corrupt-sha3"])
def test_bootstrap_archive_integrity_precedes_use(bootstrap, monkeypatch, tampered):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as bundle:
        bundle.writestr("fixture", "unit test only")
    original = payload.getvalue()
    monkeypatch.setitem(bootstrap.ARCHIVES, "source", ("unit-fixture.zip", hashlib.sha3_256(original).hexdigest()))
    delivered = original + (b"corruption" if tampered else b"")
    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", lambda *_args, **_kwargs: io.BytesIO(delivered))
    if tampered:
        with pytest.raises(RuntimeError, match="integrity mismatch"):
            bootstrap.archive("source")
    else:
        with bootstrap.archive("source") as bundle:
            assert bundle.read("fixture") == b"unit test only"


def test_bootstrap_probes_actual_linked_sqlite_and_preserves_already_patched_host(bootstrap, capsys):
    with sqlite3.connect(":memory:") as connection:
        expected = list(connection.execute("select sqlite_version(), sqlite_source_id()").fetchone())
    assert bootstrap.probe() == expected
    assert bootstrap.patched(expected[0]) is True
    bootstrap.main()
    observed = json.loads(capsys.readouterr().out)
    assert observed == {"sqlite_version": expected[0], "sqlite_source_id": expected[1], "ci_replacement": False}


def test_bootstrap_refuses_automatic_replacement_outside_disposable_ci(bootstrap, monkeypatch):
    monkeypatch.setattr(bootstrap, "probe", lambda *_args: ["3.51.2", "old fixture"])
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("RUNNER_TEMP", raising=False)
    with pytest.raises(RuntimeError, match="limited to disposable GitHub CI"):
        bootstrap.main()


@pytest.mark.parametrize("version", ["3.051.3", "3.51.3.1", "3.51.3-custom"])
def test_bootstrap_malformed_versions_do_not_bypass_upgrade(bootstrap, version):
    assert bootstrap.patched(version) is False


def test_bootstrap_security_backport_boundaries_are_exact(bootstrap):
    for version, expected in (("3.44.5", False), ("3.44.6", True), ("3.50.6", False),
                              ("3.50.7", True), ("3.51.2", False), ("3.51.3", True)):
        assert bootstrap.patched(version) is expected


def test_filtered_campaign_arguments_reject_before_execution(tmp_path, monkeypatch):
    instance, _, _ = recorder(tmp_path, monkeypatch)
    for option in ("keyword", "markexpr", "lf"):
        original = getattr(instance.config.option, option)
        setattr(instance.config.option, option, True if option == "lf" else "only_one")
        with pytest.raises(pytest.UsageError, match="complete module selection"):
            instance.pytest_sessionstart(SimpleNamespace())
        setattr(instance.config.option, option, original)
    instance.config.args[0] += "::test_one"
    with pytest.raises(pytest.UsageError, match="complete module selection"):
        instance.pytest_sessionstart(SimpleNamespace())


def test_deselection_cannot_be_hidden_by_matching_collected_count(tmp_path, monkeypatch):
    instance, session, _ = recorder(tmp_path, monkeypatch)
    instance.pytest_deselected([SimpleNamespace(nodeid="unreported-required-case")])
    assert finish(instance, session)["accepted"] is False


def test_missing_junit_or_actual_test_directory_rejects_evidence(tmp_path, monkeypatch):
    instance, session, _ = recorder(tmp_path, monkeypatch)
    Path(instance.config.option.xmlpath).unlink()
    instance.basetemp.rmdir()
    result = finish(instance, session)
    assert result["accepted"] is False
    assert "Required JUnit report is missing" in result["errors"]
    assert "Actual test directory does not match measured filesystem" in result["errors"]


def test_conflicting_deepest_mounts_reject_until_actual_device_identifies_one():
    mountinfo = (
        "1 0 8:0 / / rw - ext4 /dev/root rw\n"
        "2 1 8:1 / /qualification rw - xfs /dev/first rw\n"
        "3 1 8:2 / /qualification rw - ext4 /dev/second rw\n"
    )
    with pytest.raises(ValueError, match="[Aa]mbiguous|[Cc]onflicting"):
        evidence.linux_filesystem(Path("/qualification/store"), mountinfo)
    import os
    if hasattr(os, "makedev"):
        observed = evidence.linux_filesystem(Path("/qualification/store"), mountinfo,
                                             device_id=os.makedev(8, 2))
        assert observed["type"] == "ext4"
        assert observed["local"] is True
