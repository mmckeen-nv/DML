"""Independent scope, lifecycle, evidence and CLI projection regressions."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from daystrom_dml.services.projection import (
    ProjectionError, ProjectionStale, prepare_source, projection_status, reconcile,
)
from test_projection import IDENTITY, SCOPE, query, write
from daystrom_dml.journal import JournalStateStore
from daystrom_dml.services.projection import SQLiteProjection


@pytest.fixture
def pair(tmp_path):
    source = JournalStateStore(tmp_path / "authority" / "source.sqlite3", receipt_mode=True)
    write(source)
    return source, SQLiteProjection(tmp_path / "projection" / "index.sqlite3")


def test_exact_scope_tuple_does_not_leak_optional_scope_memories(pair):
    source, target = pair
    receipts = []
    for member in SCOPE:
        scoped = {**SCOPE, member: "other"}
        receipts.append((scoped, write(source, member, scope=scoped)))
    reconcile(source, target)
    assert [hit["memory"]["id"] for hit in query(source, target)["results"]] == [0]
    for scoped, receipt in receipts:
        answer = query(source, target, scope=scoped)
        assert [hit["memory"]["id"] for hit in answer["results"]] == [receipt["result"]["memory"]["id"]]
        assert answer["suppressed"] == []


@pytest.mark.parametrize("meta,reason", [
    ({"source_trust": "untrusted"}, "untrusted_source"),
    ({"memory_state": "quarantined"}, "state_quarantined"),
    ({"namespace": "quarantine"}, "quarantine_namespace"),
    ({"memory_state": "superseded"}, "state_superseded"),
    ({"superseded_by": 0}, "explicitly_superseded"),
    ({"expires_at": 100.0}, "expired"),
    ({"expires_at": "tomorrow"}, "invalid_expiry"),
])
def test_lifecycle_suppression_has_reason_and_never_changes_authority(pair, meta, reason):
    source, target = pair
    receipt = write(source, "suppressed", meta=meta)
    before = source.read_snapshot()
    reconcile(source, target)
    answer = query(source, target)
    assert [hit["memory"]["id"] for hit in answer["results"]] == [0]
    assert answer["suppressed"] == [{"id": receipt["result"]["memory"]["id"], "reason": reason}]
    assert source.read_snapshot() == before


def test_expiry_uses_explicit_query_timestamp(pair):
    source, target = pair
    receipt = write(source, "expiring", meta={"expires_at": 100.0})
    reconcile(source, target)
    identity = receipt["result"]["memory"]["id"]
    assert identity in [hit["memory"]["id"] for hit in query(source, target, as_of=99.0)["results"]]
    assert query(source, target, as_of=100.0)["suppressed"] == [{"id": identity, "reason": "expired"}]


def test_cosine_ties_use_memory_id_and_top_k_is_repeatable(pair):
    source, target = pair
    for key in ("two", "three", "four"):
        write(source, key)
    revision, state = source.read_snapshot()
    state["items"].reverse()
    source.save(state, expected_revision=revision, operation="test-reorder")
    reconcile(source, target)
    first = query(source, target, top_k=2)
    assert [hit["memory"]["id"] for hit in first["results"]] == [0, 1]
    assert [hit["score"] for hit in first["results"]] == [1.0, 1.0]
    assert query(source, target, top_k=2) == first


@pytest.mark.parametrize("vector", [[1e308, 1e308], [1e-300, 0.0]])
def test_unrepresentable_query_norm_fails_closed(pair, vector):
    source, target = pair
    reconcile(source, target)
    before = source.read_snapshot()
    with pytest.raises(ProjectionError):
        query(source, target, vector=vector)
    assert source.read_snapshot() == before


def test_large_finite_stored_vectors_have_finite_correct_cosine(pair):
    source, target = pair
    revision, state = source.read_snapshot()
    state["items"][0]["embedding"] = [3e38, 0.0]
    source.save(state, expected_revision=revision, operation="test-extreme-vector")
    reconcile(source, target)
    assert query(source, target)["results"][0]["score"] == 1.0


def test_publication_rejects_changed_state_with_original_digest(pair):
    source, target = pair
    snapshot = prepare_source(source)
    forged = deepcopy(snapshot)
    forged["source_state"]["items"][0]["text"] = "forged"
    with pytest.raises(ProjectionError, match="digest"):
        target.publish(forged)
    assert target.read()["cursor"] is None
    assert prepare_source(source) == snapshot


def test_same_revision_different_digest_cannot_replace_published_snapshot(pair):
    source, target = pair
    snapshot = prepare_source(source)
    target.publish(snapshot)
    before = target.read()
    forged = deepcopy(snapshot)
    forged["source_state"]["items"][0]["text"] = "forged"
    encoded = json.dumps(forged["source_state"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    forged["source_digest"] = hashlib.sha256(encoded.encode()).hexdigest()
    with pytest.raises(ProjectionError):
        target.publish(forged)
    assert target.read() == before
    assert prepare_source(source) == snapshot


def test_forged_unpublished_snapshot_is_detected_by_authority_check(pair):
    source, target = pair
    snapshot = prepare_source(source)
    forged = deepcopy(snapshot)
    forged["source_state"]["items"][0]["text"] = "forged"
    encoded = json.dumps(forged["source_state"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    forged["source_digest"] = hashlib.sha256(encoded.encode()).hexdigest()
    target.publish(forged)
    assert projection_status(source, target)["matches_pinned_source"] is False
    with pytest.raises(ProjectionStale):
        query(source, target)
    assert prepare_source(source) == snapshot


def test_query_evidence_binds_results_to_verified_revision_and_digest(pair):
    source, target = pair
    reconcile(source, target)
    snapshot = prepare_source(source)
    answer = query(source, target)
    expected = {key: snapshot[key] for key in ("source_store_id", "source_revision", "source_digest")}
    assert answer["source"] == expected
    assert answer["scope"] == SCOPE
    assert answer["as_of"] == 100.0
    assert answer["results"][0]["memory"] == snapshot["source_state"]["items"][0]
    write(source, "new")
    status = projection_status(source, target)
    assert status["matches_pinned_source"] is False
    assert status["projection"] == expected
    assert status["source"]["source_revision"] == expected["source_revision"] + 1
    with pytest.raises(ProjectionStale):
        query(source, target)


def test_cli_sync_status_query_and_strict_unknown_fields(pair, tmp_path):
    source, target = pair
    script = Path(__file__).resolve().parents[1] / "scripts" / "dml_projection.py"
    def cli(*args):
        return subprocess.run([sys.executable, str(script), str(source.path), str(target.path), *map(str, args)],
                              capture_output=True, text=True, timeout=30)
    before = source.read_snapshot()
    sync = cli("sync")
    assert sync.returncode == 0, sync.stderr + sync.stdout
    assert json.loads(sync.stdout)["matches_pinned_source"] is True
    status = cli("status")
    assert status.returncode == 0, status.stderr + status.stdout
    assert json.loads(status.stdout)["matches_pinned_source"] is True
    request = {"vector": [1.0, 0.0], "embedding_identity": IDENTITY, "scope": SCOPE, "as_of": 100.0}
    path = tmp_path / "query.json"
    path.write_text(json.dumps(request))
    answer = cli("query", path)
    assert answer.returncode == 0, answer.stderr + answer.stdout
    assert [hit["memory"]["id"] for hit in json.loads(answer.stdout)["results"]] == [0]
    path.write_text(json.dumps({**request, "private_injected_field": "private-content"}))
    rejected = cli("query", path)
    assert rejected.returncode == 2
    assert json.loads(rejected.stdout) == {"ok": False, "error": "ValueError"}
    assert "private" not in rejected.stdout + rejected.stderr
    assert source.read_snapshot() == before


def test_query_freezes_vector_identity_and_scope_before_target_io(pair, monkeypatch):
    source, target = pair
    reconcile(source, target)
    vector = [1.0, 0.0]
    identity = dict(IDENTITY)
    scope = dict(SCOPE)
    original_read = target.read
    def mutate_during_read():
        vector[:] = [0.0, 1.0]
        identity["revision"] = "unrelated"
        scope["tenant_id"] = "other"
        return original_read()
    monkeypatch.setattr(target, "read", mutate_during_read)
    answer = query(source, target, vector=vector, embedding_identity=identity, scope=scope)
    assert answer["scope"] == SCOPE
    assert answer["results"][0]["memory"]["id"] == 0
    assert answer["results"][0]["score"] == 1.0
    assert vector == [0.0, 1.0]
    assert identity["revision"] == "unrelated"
    assert scope["tenant_id"] == "other"


def test_query_linearizes_at_verified_authority_read(pair, monkeypatch):
    source, target = pair
    reconcile(source, target)
    original = prepare_source(source)
    original_read = target.read
    def advance_before_projection_read():
        write(source, "new-during-query")
        return original_read()
    monkeypatch.setattr(target, "read", advance_before_projection_read)
    answer = query(source, target)
    assert answer["source"]["source_revision"] == original["source_revision"]
    assert answer["source"]["source_digest"] == original["source_digest"]
    assert [hit["memory"]["id"] for hit in answer["results"]] == [0]
    assert source.read_snapshot()[0] == original["source_revision"] + 1
    monkeypatch.setattr(target, "read", original_read)
    with pytest.raises(ProjectionStale):
        query(source, target)
