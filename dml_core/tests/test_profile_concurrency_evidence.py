"""Synthetic evidence mutations are gate unit tests, never real qualification."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import gzip
import json
import subprocess
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from scripts import profile_concurrency_evidence as evidence


def campaign(schema=2, transport="threads-shared", clients=1):
    value = {"schema_version": evidence.SCHEMA, "schema": schema, "transport": transport,
             "clients": clients, "measured_ready_clients": clients,
             "process_count": min(clients, 4) if transport == "processes" else 1,
             "committed_revisions": 7, "duration_seconds": 2.0, "first_progress_seconds": 0.01,
             "max_operation_seconds": 1.0, "history_verified": True, "history_events": clients * 3,
             "history_file": f"schema{schema}-{transport}-{clients}.json.gz", "history_sha256": "a" * 64,
             "ownership_rejections": 0, "max_attempts": 1}
    if transport == "http":
        value.update(max_http_in_flight=clients, max_active_workers=min(clients, 40), worker_threads=min(clients, 40),
                     client_keepalive_expiry_seconds=1.0, server_keepalive_timeout_seconds=5.0)
    return value


def junit(mode="bounded"):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    for module in sorted(evidence.REQUIRED_MODULES):
        case = ET.SubElement(suite, "testcase", classname=f"tests.{module}", name="test_synthetic_unit_coverage")
        if module == "test_profile_concurrency_runtime":
            properties = ET.SubElement(case, "properties")
            ET.SubElement(properties, "property", name="actual_ownership_timeout_seconds", value=json.dumps([30.0] * 6))
            ET.SubElement(properties, "property", name="ownership_release_progress_seconds", value="0.25")
        elif module == "test_profile_concurrency":
            properties = ET.SubElement(case, "properties")
            ET.SubElement(properties, "property", name="concurrency_lock_evidence", value=json.dumps(lock_evidence()))
    for schema in (2, 3, 4):
        for transport in evidence.TRANSPORTS:
            for clients in evidence.SCALES[mode]:
                module = "test_profile_concurrency_http" if transport == "http" else "test_profile_concurrency"
                case = ET.SubElement(suite, "testcase", classname=f"tests.{module}", name=f"test_synthetic[{schema}-{transport}-{clients}]")
                properties = ET.SubElement(case, "properties")
                ET.SubElement(properties, "property", name="concurrency_evidence", value=json.dumps(campaign(schema, transport, clients)))
    return root


def validate(root, mode="bounded", system="Linux"):
    return evidence.validate_junit(ET.tostring(root), mode=mode, system=system)


def test_complete_matrices_are_explicit_and_do_not_expand_bounded_claims():
    for mode, cells in (("bounded", 24), ("full", 48)):
        result = validate(junit(mode), mode)
        assert result["errors"] == []
        assert len(result["campaigns"]) == cells
        assert {item["clients"] for item in result["campaigns"]} == set(evidence.SCALES[mode])
    assert validate(junit(), "full")["errors"]


@pytest.mark.parametrize("field,value", [
    ("schema", True), ("clients", True), ("measured_ready_clients", 0),
    ("measured_ready_clients", 2), ("history_verified", 1), ("history_events", 0),
    ("committed_revisions", 41), ("committed_revisions", 0), ("duration_seconds", 90.001),
    ("duration_seconds", float("nan")), ("duration_seconds", float("inf")),
    ("duration_seconds", True), ("duration_seconds", -1), ("first_progress_seconds", 10.001),
    ("first_progress_seconds", 3), ("max_operation_seconds", 90.001), ("process_count", 2),
    ("max_attempts", 4), ("ownership_rejections", -1), ("history_file", "../private.gz"),
    ("history_sha256", "not-a-hash"),
    ("schema_version", "future"), ("transport", "imaginary"), ("memory", "PRIVATE_SECRET_SENTINEL"),
])
def test_invalid_types_bounds_and_content_never_qualify(field, value):
    item = campaign()
    item[field] = value
    with pytest.raises(ValueError):
        evidence.validate_campaign(item, "bounded")


def test_one_process_cannot_claim_multi_process_qualification():
    item = campaign(transport="processes", clients=16)
    item["process_count"] = 1
    with pytest.raises(ValueError, match="multiple processes"):
        evidence.validate_campaign(item, "bounded")


@pytest.mark.parametrize("field,value", [("max_http_in_flight", 15), ("max_active_workers", 17), ("worker_threads", 0),
    ("client_keepalive_expiry_seconds", 5.0), ("server_keepalive_timeout_seconds", 10.0),
    ("client_keepalive_expiry_seconds", True), ("server_keepalive_timeout_seconds", float("nan"))])
def test_http_transport_and_worker_measurements_must_be_consistent(field, value):
    item = campaign(transport="http", clients=16)
    item[field] = value
    with pytest.raises(ValueError):
        evidence.validate_campaign(item, "bounded")


@pytest.mark.parametrize("mutation", ["missing-cell", "duplicate-cell", "duplicate-property", "failure", "skip", "missing-module", "unknown-json-key", "duplicate-json-key", "nonfinite-json", "missing-property"])
def test_junit_mutations_reject_incomplete_or_fabricated_qualification(mutation):
    root = junit()
    suite = root.find("testsuite")
    case = next(case for case in suite if case.find("properties/property[@name='concurrency_evidence']") is not None)
    prop = case.find("properties/property")
    if mutation == "missing-cell":
        suite.remove(case)
    elif mutation == "duplicate-cell":
        duplicate = deepcopy(case)
        duplicate.set("name", "test_duplicated_cell")
        suite.append(duplicate)
    elif mutation == "duplicate-property":
        case.find("properties").append(deepcopy(prop))
    elif mutation == "failure":
        ET.SubElement(case, "failure").text = "PRIVATE_SECRET_SENTINEL"
    elif mutation == "skip":
        ET.SubElement(case, "skipped", message="PRIVATE_SECRET_SENTINEL")
    elif mutation == "missing-module":
        suite.remove(next(case for case in suite if case.get("classname").endswith("test_profile_concurrency_adversarial")))
    elif mutation == "unknown-json-key":
        value = json.loads(prop.get("value"))
        value["private_text"] = "PRIVATE_SECRET_SENTINEL"
        prop.set("value", json.dumps(value))
    elif mutation == "duplicate-json-key":
        prop.set("value", prop.get("value").replace('"schema": 2', '"schema": 2, "schema": 2'))
    elif mutation == "nonfinite-json":
        prop.set("value", prop.get("value").replace('"duration_seconds": 2.0', '"duration_seconds": NaN'))
    elif mutation == "missing-property":
        case.find("properties").remove(prop)
    result = validate(root)
    assert result["errors"]
    assert "PRIVATE_SECRET_SENTINEL" not in json.dumps({key: value for key, value in result.items() if not key.startswith("_")})


@pytest.mark.parametrize("system,reason,accepted", [
    ("Windows", evidence.FORK_REASON, True), ("Linux", evidence.FORK_REASON, False),
    ("Windows", "generic unsupported platform", False),
])
def test_only_exact_declared_windows_fork_skip_is_allowed(system, reason, accepted):
    root = junit()
    case = ET.SubElement(root.find("testsuite"), "testcase", classname="tests.test_profile_concurrency_runtime", name=evidence.FORK_CASE)
    ET.SubElement(case, "skipped", message=reason)
    assert (not validate(root, system=system)["errors"]) is accepted


@pytest.mark.parametrize("name,value", [
    ("actual_ownership_timeout_seconds", [30.0] * 5),
    ("actual_ownership_timeout_seconds", [30.0] * 5 + [0.1]),
    ("actual_ownership_timeout_seconds", [30.0] * 5 + [32.01]),
    ("ownership_release_progress_seconds", 10.01),
])
def test_actual_default_deadlines_cannot_be_replaced_by_shortened_unit_budgets(name, value):
    root = junit()
    root.find(f".//property[@name='{name}']").set("value", json.dumps(value))
    assert validate(root)["errors"]


@pytest.mark.parametrize("mutation", ["drop-status", "drop-detail", "bool-retry", "unknown-outcome", "python-with-http-status"])
def test_attempt_transport_cannot_be_downgraded_by_removing_http_evidence(mutation):
    attempt = {"started_ns": 1, "finished_ns": 2, "request_digest": "a" * 64,
               "error_code": "store_lock_timeout", "status_code": 503,
               "error_detail": {"code": "receipt_ownership_unavailable", "retry_same_key": True}}
    final = {"started_ns": 3, "finished_ns": 4, "request_digest": "a" * 64, "error_code": None, "status_code": 200}
    events = [{"request": {"operation": "ingest"}, "attempts": [attempt, final]}]
    evidence.validate_attempt_transport(events, "http")
    transport = "http"
    if mutation == "drop-status":
        attempt.pop("status_code")
        final.pop("status_code")
    elif mutation == "drop-detail":
        attempt.pop("error_detail")
    elif mutation == "bool-retry":
        attempt["error_detail"]["retry_same_key"] = 1
    elif mutation == "unknown-outcome":
        attempt["status_code"] = 504
    else:
        transport = "threads-shared"
    with pytest.raises(ValueError):
        evidence.validate_attempt_transport(events, transport)



def test_xml_entity_expansion_is_rejected_without_parsing():
    payload = b'<!DOCTYPE foo [<!ENTITY secret "PRIVATE_SECRET_SENTINEL">]><testsuites/>'
    with pytest.raises(ValueError, match="entity-bearing"):
        evidence.validate_junit(payload, mode="bounded", system="Linux")


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    root = junit()
    junit_path = tmp_path / "junit.xml"
    junit_path.write_bytes(ET.tostring(root))
    basetemp = tmp_path / "stores"
    basetemp.mkdir()
    config = SimpleNamespace(rootpath=tmp_path, option=SimpleNamespace(junitprefix=None))
    instance = evidence.ConcurrencyEvidence(config, tmp_path / "evidence.json")
    instance.started = "synthetic-unit-fixture"
    instance.mode, instance.junit = "bounded", junit_path
    instance.parent, instance.basetemp, instance.excluded = tmp_path, basetemp, ()
    instance.history_directory = tmp_path / "histories"
    instance.initial_fs = {"type": "ext4", "options": ["rw"], "source": "proc-self-mountinfo", "local": True, "device_id": 1}
    instance.initial_source = {"commit": "a" * 40, "tree": "b" * 40, "dirty": False, "sha256": {"unit.py": "c" * 64}}
    instance.sqlite = ("3.53.4", "synthetic unit identity")
    instance.http_libraries = {name: "1.0.synthetic" for name in evidence.HTTP_LIBRARIES}
    for case in root.iter("testcase"):
        node = case.get("classname").replace(".", "/") + ".py::" + case.get("name")
        instance.phases[node] = {"setup": "passed", "call": "passed", "teardown": "passed"}
    instance.collected = set(instance.phases)
    monkeypatch.setattr(evidence, "filesystem", lambda _path: dict(instance.initial_fs))
    monkeypatch.setattr(evidence, "source_identity", lambda *_args: deepcopy(instance.initial_source))
    monkeypatch.setattr(evidence, "validate_history", lambda *_args: None)
    monkeypatch.setattr(evidence, "http_library_identity", lambda: dict(instance.http_libraries))
    return instance, SimpleNamespace(testscollected=len(instance.phases), exitstatus=0)


def finish(instance, session, status=0):
    instance.pytest_sessionfinish(session, status)
    return json.loads(instance.output.read_text())


def test_recorder_binds_actual_junit_bytes_and_explicit_matrix(recorder):
    instance, session = recorder
    result = finish(instance, session)
    assert result["accepted"] is True
    assert result["junit_sha256"] == hashlib.sha256(instance.junit.read_bytes()).hexdigest()
    assert result["scales"] == [1, 16]
    assert result["power_loss_qualified"] is False
    assert result["http_libraries"] == instance.http_libraries


@pytest.mark.parametrize("mutation", ["dirty", "changed-source", "filesystem", "counts", "incomplete", "failed-exit", "missing-junit", "changed-http-library"])
def test_recorder_rejects_unbound_source_runner_and_incomplete_execution(recorder, monkeypatch, mutation):
    instance, session = recorder
    status = 0
    if mutation == "dirty":
        instance.initial_source["dirty"] = True
    elif mutation == "changed-source":
        monkeypatch.setattr(evidence, "source_identity", lambda *_args: {**instance.initial_source, "tree": "d" * 40})
    elif mutation == "filesystem":
        monkeypatch.setattr(evidence, "filesystem", lambda _path: {**instance.initial_fs, "type": "overlay"})
    elif mutation == "counts":
        session.testscollected += 1
    elif mutation == "incomplete":
        next(iter(instance.phases.values())).pop("teardown")
    elif mutation == "failed-exit":
        status = 1
    elif mutation == "missing-junit":
        instance.junit.unlink()
    elif mutation == "changed-http-library":
        monkeypatch.setattr(evidence, "http_library_identity", lambda: {**instance.http_libraries, "httpx": "different"})
    assert finish(instance, session, status)["accepted"] is False
    assert session.exitstatus != 0


def test_source_identity_detects_untracked_sources_and_never_excludes_tracked_edits(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    git("init", "--quiet")
    source = tmp_path / "pyproject.toml"
    source.write_text("# synthetic source\n")
    git("add", "pyproject.toml")
    git("-c", "user.name=Synthetic Test", "-c", "user.email=test@example.invalid", "commit", "--quiet", "-m", "fixture")
    observed = evidence.source_identity(tmp_path)
    assert observed["dirty"] is False
    assert observed["sha256"]["pyproject.toml"] == hashlib.sha256(source.read_bytes()).hexdigest()
    untracked = tmp_path / "new_runtime.py"
    untracked.write_text("# source must be tracked for qualification\n")
    assert evidence.source_identity(tmp_path)["dirty"] is True
    untracked.unlink()
    source.write_text("# modified tracked source\n")
    assert evidence.source_identity(tmp_path, (source,))["dirty"] is True


@pytest.mark.parametrize("mutation", ["junit-replacement", "executed-replacement", "junit-outcome", "collected-replacement"])
def test_same_count_substitution_cannot_discard_a_noncampaign_test(recorder, mutation):
    instance, session = recorder
    if mutation.startswith("junit"):
        root = ET.fromstring(instance.junit.read_bytes())
        case = root.find("testsuite/testcase")
        if mutation == "junit-replacement":
            case.set("name", "test_unexecuted_replacement")
        else:
            ET.SubElement(case, "skipped", message="invented")
        instance.junit.write_bytes(ET.tostring(root))
    elif mutation == "executed-replacement":
        node, phases = instance.phases.popitem()
        instance.phases[node + "_uncollected"] = phases
    else:
        instance.collected.remove(next(iter(instance.collected)))
        instance.collected.add("tests/test_fake.py::test_unexecuted")
    assert finish(instance, session)["accepted"] is False


@pytest.fixture
def raw_history(tmp_path, monkeypatch):
    import profile_concurrency_fixture
    item = campaign()
    envelope = {"schema_version": "dml-concurrency-history-v1", "before": {"schema": 2},
                "after": {"schema": 2}, "events": [{"attempts": [{"started_ns": 1, "finished_ns": 2,
                    "request_digest": "a" * 64, "error_code": None}]}], "clients": 1}
    path = tmp_path / item["history_file"]
    payload = gzip.compress(json.dumps(envelope).encode(), mtime=0)
    path.write_bytes(payload)
    item["history_sha256"] = hashlib.sha256(payload).hexdigest()
    checked = {"history_verified": True, "history_events": 3, "unique_commits": 7, "max_operation_seconds": 1.0,
               "ownership_rejections": 0, "max_attempts": 1}
    def checker(before, events, after, *, clients):
        assert before == after == {"schema": 2}
        assert events == envelope["events"] and clients == 1
        return checked
    monkeypatch.setattr(profile_concurrency_fixture, "check_history", checker)
    return item, path, envelope, checked


def test_persisted_history_is_reopened_and_replayed(raw_history):
    item, path, _envelope, _checked = raw_history
    evidence.validate_history(item, path.parent)


@pytest.mark.parametrize("mutation", ["missing", "changed-bytes", "changed-clients", "changed-schema", "extra-content", "changed-event-count", "changed-commits", "checker-failure"])
def test_raw_history_mutations_never_receive_replay_acceptance(raw_history, monkeypatch, mutation):
    import profile_concurrency_fixture
    item, path, envelope, checked = raw_history
    if mutation == "missing":
        path.unlink()
    elif mutation == "changed-bytes":
        path.write_bytes(path.read_bytes() + b"tampered")
    elif mutation in {"changed-clients", "changed-schema", "extra-content"}:
        if mutation == "changed-clients":
            envelope["clients"] = 16
        elif mutation == "changed-schema":
            envelope["after"]["schema"] = 3
        else:
            envelope["private_data"] = "PRIVATE_SECRET_SENTINEL"
        payload = gzip.compress(json.dumps(envelope).encode(), mtime=0)
        path.write_bytes(payload)
        item["history_sha256"] = hashlib.sha256(payload).hexdigest()
    elif mutation == "changed-event-count":
        checked["history_events"] += 1
    elif mutation == "changed-commits":
        checked["unique_commits"] += 1
    else:
        def reject(*_args, **_kwargs):
            raise AssertionError("Synthetic incorrect history")
        monkeypatch.setattr(profile_concurrency_fixture, "check_history", reject)
    with pytest.raises((ValueError, OSError, AssertionError)):
        evidence.validate_history(item, path.parent)


def lock_evidence():
    return {"schema_version": "dml-ownership-wait-evidence-v1", "schema": 3, "clients": 16,
            "timing_domain": "first_os_lock_attempt_to_acquired", "successful_acquisitions": 8,
            "blocked_attempts": 2, "unfinished_acquisitions": 0,
            "max_wait_seconds": 0.25, "injected_owner_delay_seconds": 0.002}


@pytest.mark.parametrize("field,value", [
    ("clients", True), ("blocked_attempts", 0), ("successful_acquisitions", 0),
    ("unfinished_acquisitions", 1), ("max_wait_seconds", 32.001),
    ("timing_domain", "entire_request"), ("injected_owner_delay_seconds", 0),
    ("schema", 2), ("schema_version", "future"), ("max_wait_seconds", float("nan")),
])
def test_mixed_contention_measurements_cannot_claim_unmeasured_lock_work(field, value):
    value_map = lock_evidence()
    value_map[field] = value
    with pytest.raises(ValueError):
        evidence.validate_lock_evidence(value_map)


def test_missing_mixed_contention_record_rejects_otherwise_complete_junit():
    root = junit()
    for properties in root.iter("properties"):
        for item in list(properties):
            if item.get("name") == "concurrency_lock_evidence":
                properties.remove(item)
    assert validate(root)["errors"]


def retrieval_retry(detail, *, operation="retrieve"):
    return [{"request": {"operation": operation}, "attempts": [
        {"started_ns": 1, "finished_ns": 2, "request_digest": "a" * 64,
         "error_code": "store_lock_timeout", "status_code": 503, "error_detail": detail},
        {"started_ns": 3, "finished_ns": 4, "request_digest": "a" * 64,
         "error_code": None, "status_code": 200},
    ]}]


def test_only_positive_typed_retrieval_ownership_evidence_qualifies_read_retry():
    events = retrieval_retry({"code": "retrieval_outcome_unavailable", "reason": "store_ownership_timeout"})
    evidence.validate_attempt_transport(events, "http")


@pytest.mark.parametrize("detail,operation", [
    ({"code": "retrieval_outcome_unavailable"}, "retrieve"),
    ({"code": "retrieval_outcome_unavailable", "reason": None}, "retrieve"),
    ({"code": "retrieval_outcome_unavailable", "reason": True}, "retrieve"),
    ({"code": "retrieval_outcome_unavailable", "reason": "backend_timeout"}, "retrieve"),
    ({"code": "retrieval_outcome_unavailable", "reason": "store_ownership_timeout", "retry_same_key": True}, "retrieve"),
    ({"code": "receipt_ownership_unavailable", "retry_same_key": True}, "retrieve"),
    ({"code": "retention_outcome_unavailable", "reason": "store_ownership_timeout"}, "retention"),
    ({"code": "retrieval_outcome_unavailable", "reason": "store_ownership_timeout"}, "ingest"),
    ({"code": "retrieval_outcome_unavailable", "reason": "store_ownership_timeout"}, "retention"),
    ({"code": "retrieval_outcome_unavailable", "reason": "store_ownership_timeout"}, "unknown"),
])
def test_uncertain_read_outcomes_cannot_be_reclassified_as_ownership_retries(detail, operation):
    with pytest.raises(ValueError):
        evidence.validate_attempt_transport(retrieval_retry(detail, operation=operation), "http")


def test_http_library_identity_measures_installed_versions():
    import importlib.metadata
    observed = evidence.http_library_identity()
    assert set(observed) == set(evidence.HTTP_LIBRARIES)
    assert observed == {name: importlib.metadata.version(name) for name in evidence.HTTP_LIBRARIES}


@pytest.mark.parametrize("version", [None, "", "1.0\nPRIVATE_PATH", "x" * 129])
def test_invalid_http_library_versions_never_become_runner_identity(monkeypatch, version):
    monkeypatch.setattr(evidence.importlib.metadata, "version", lambda _name: version)
    with pytest.raises(ValueError, match="Invalid HTTP runtime"):
        evidence.http_library_identity()


def test_missing_http_library_cannot_receive_qualification_identity(monkeypatch):
    def missing(_name):
        raise evidence.importlib.metadata.PackageNotFoundError("synthetic missing distribution")
    monkeypatch.setattr(evidence.importlib.metadata, "version", missing)
    with pytest.raises(ValueError, match="missing"):
        evidence.http_library_identity()
