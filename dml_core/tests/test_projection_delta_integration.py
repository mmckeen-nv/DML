"""Independent adapter and CLI oracles for explicit incremental projections."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from daystrom_dml.services.projection import ProjectionError, ProjectionStale, SQLiteProjection
from daystrom_dml.services.receipt_ingestion import ReceiptEmbeddingCompatibilityError
from daystrom_dml.store_lock import store_write_lock
from test_receipt_adapter import CountingEmbedder, append, factory as receipt_factory

factory = receipt_factory


class ProtocolProxy:
    """Expose only the public backend protocol, without SQLite implementation state."""

    def __init__(self, target, observe=None):
        self.path = target.path
        self._read = target.read
        self._apply = target.apply_delta
        self.observe = observe or (lambda: None)
        self.calls = []

    def read(self):
        self.observe()
        self.calls.append("read")
        return self._read()

    def apply_delta(self, delta):
        self.observe()
        self.calls.append("apply_delta")
        return self._apply(delta)


def answer(adapter, backend, **kwargs):
    return adapter.query_projection("my preference", backend=backend,
                                    tenant_id="owner", as_of=100.0, **kwargs)


@pytest.fixture
def target(tmp_path):
    return SQLiteProjection(tmp_path / "separate-projection" / "index.sqlite3")


def test_adapter_receipt_incremental_sync_query_and_restart(factory, target):
    adapter = factory()
    receipt = append(adapter)
    backend = ProtocolProxy(target)
    assert adapter.projection_status(backend)["matches_pinned_source"] is False
    assert adapter.sync_projection(backend)["matches_pinned_source"] is True
    projected = answer(adapter, backend)
    assert [hit["memory"]["id"] for hit in projected["results"]] == [receipt["result"]["memory"]["id"]]
    assert projected["results"][0]["memory"]["text"] == receipt["result"]["memory"]["text"]
    assert "apply_delta" in backend.calls
    adapter.close(persist=False)
    restarted = factory()
    assert append(restarted) == receipt
    reopened = ProtocolProxy(SQLiteProjection(target.path))
    assert answer(restarted, reopened) == projected
    assert restarted.sync_projection(reopened)["matches_pinned_source"] is True


def test_adapter_stale_projection_fails_until_explicit_resync(factory, target):
    adapter = factory()
    append(adapter)
    backend = ProtocolProxy(target)
    adapter.sync_projection(backend)
    before = target.read()
    append(adapter, text="Second preference", key="second")
    assert target.read() == before
    with pytest.raises(ProjectionStale):
        answer(adapter, backend)
    assert adapter.sync_projection(backend)["matches_pinned_source"] is True
    assert len(answer(adapter, backend)["results"]) == 2


@pytest.mark.parametrize("failing_method", ["read", "apply_delta"])
def test_backend_outage_never_undoes_or_blocks_receipt_retry(factory, target, failing_method):
    adapter = factory()
    receipt = append(adapter)
    backend = ProtocolProxy(target)

    def unavailable(*_args):
        raise OSError("private backend credential")

    setattr(backend, failing_method, unavailable)
    source_before = adapter._journal.read_snapshot()
    with pytest.raises(OSError):
        adapter.sync_projection(backend)
    assert adapter._journal.read_snapshot() == source_before
    assert append(adapter) == receipt
    later = append(adapter, text="Stored while backend unavailable", key="later")
    assert later["revision"] > receipt["revision"]
    recovered = ProtocolProxy(target)
    assert adapter.sync_projection(recovered)["matches_pinned_source"] is True
    assert len(answer(adapter, recovered)["results"]) == 2


@pytest.mark.parametrize("operation", ["sync", "status", "query"])
def test_projection_operations_reject_nested_source_ownership(factory, target, operation):
    adapter = factory()
    append(adapter)
    backend = ProtocolProxy(target)
    calls_before = adapter.embedder.calls
    with adapter.mutation_transaction("test-nested-projection"):
        with pytest.raises(ValueError, match="ownership"):
            if operation == "sync":
                adapter.sync_projection(backend)
            elif operation == "status":
                adapter.projection_status(backend)
            else:
                answer(adapter, backend)
    assert backend.calls == []
    assert adapter.embedder.calls == calls_before


@pytest.mark.parametrize("operation", ["sync", "status", "query"])
def test_projection_integration_requires_explicit_receipt_opt_in(factory, target, operation):
    adapter = factory(persistence={"journal": True, "receipts": False, "enable": False})
    backend = ProtocolProxy(target)
    with pytest.raises(ValueError, match="receipt"):
        if operation == "sync":
            adapter.sync_projection(backend)
        elif operation == "status":
            adapter.projection_status(backend)
        else:
            answer(adapter, backend)
    assert backend.calls == []
    assert adapter.embedder.calls == 0


def test_backend_io_runs_after_releasing_source_write_lock(factory, target):
    adapter = factory()
    append(adapter)
    observations = []

    def independent_lock_probe():
        with store_write_lock(adapter._journal.path.parent, operation="backend-lock-probe", timeout_ms=100):
            observations.append(True)

    backend = ProtocolProxy(target, observe=independent_lock_probe)
    adapter.sync_projection(backend)
    adapter.projection_status(backend)
    answer(adapter, backend)
    assert len(observations) == len(backend.calls) > 0


def test_query_embedding_identity_switch_fails_before_backend_read(factory, target):
    embedder = CountingEmbedder()
    adapter = factory(embedder=embedder)
    append(adapter)
    backend = ProtocolProxy(target)
    adapter.sync_projection(backend)
    backend.calls.clear()
    embedder.callback = lambda: setattr(embedder, "receipt_embedding_identity", "changed-revision")
    with pytest.raises(ReceiptEmbeddingCompatibilityError):
        answer(adapter, backend)
    assert backend.calls == []


def test_query_changed_embedding_dimension_fails_closed(factory, target, monkeypatch):
    import numpy as np

    adapter = factory()
    append(adapter)
    backend = ProtocolProxy(target)
    adapter.sync_projection(backend)
    monkeypatch.setattr(adapter.embedder, "embed", lambda _text: np.ones(7, dtype=np.float32))
    with pytest.raises(ReceiptEmbeddingCompatibilityError):
        answer(adapter, backend)


def test_backend_false_acknowledgement_is_not_reported_as_success(factory, target):
    adapter = factory()
    append(adapter)
    backend = ProtocolProxy(target)
    real_apply = backend.apply_delta
    backend.apply_delta = lambda _delta: {"source_revision": 1, "invented": True}
    with pytest.raises(ProjectionError):
        adapter.sync_projection(backend)
    assert target.read()["cursor"] is None
    backend.apply_delta = real_apply
    assert adapter.sync_projection(backend)["matches_pinned_source"] is True


def test_cli_incremental_and_default_full_sync_are_compatible(factory, target):
    adapter = factory()
    append(adapter)
    script = Path(__file__).resolve().parents[1] / "scripts" / "dml_projection.py"

    def cli(*args):
        return subprocess.run([sys.executable, str(script), str(adapter._journal.path), str(target.path), *args],
                              capture_output=True, text=True, timeout=30)

    for flags in [("sync", "--incremental"), ("sync",), ("sync", "--incremental")]:
        response = cli(*flags)
        assert response.returncode == 0, response.stdout + response.stderr
        assert json.loads(response.stdout)["matches_pinned_source"] is True
    append(adapter, text="new memory", key="new")
    response = cli("sync", "--incremental")
    assert response.returncode == 0, response.stdout + response.stderr
    assert json.loads(response.stdout)["record_count"] == 2


def test_cli_incremental_backend_error_is_sanitized(factory, target, monkeypatch, capsys):
    from scripts import dml_projection
    from daystrom_dml.services import projection_delta

    adapter = factory()
    append(adapter)

    def unavailable(*_args, **_kwargs):
        raise OSError("private tenant content and credentials")

    monkeypatch.setattr(projection_delta, "reconcile_incremental", unavailable)
    result = dml_projection.main([str(adapter._journal.path), str(target.path), "sync", "--incremental"])
    captured = capsys.readouterr()
    assert result == 2
    assert json.loads(captured.out) == {"ok": False, "error": "OSError"}
    assert "private" not in captured.out + captured.err
    assert append(adapter)["revision"] == 1
