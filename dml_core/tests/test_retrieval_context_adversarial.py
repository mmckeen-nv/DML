"""Independent adversarial checks for the owned scoped-context service boundary.

These checks use hostile capabilities and controlled interleavings rather than
reimplementing the ranking, rendering, or evidence algorithms.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import FrozenInstanceError
import inspect
from threading import Event

import numpy as np
import pytest

from daystrom_dml import dml_adapter as adapter_module
from daystrom_dml import store_lock
from daystrom_dml.dml_adapter import DMLAdapter
from daystrom_dml.memory_store import MemoryItem
from daystrom_dml.services import context, scoped_retrieval
from daystrom_dml.services.outbox_migration import upgrade_outbox_journal
from daystrom_dml.services.receipt_ingestion import ReceiptEmbeddingCompatibilityError
from daystrom_dml.services.receipt_lifecycle import memory_digest
from daystrom_dml.services.scoped_retrieval import (
    ScopedRetrievalRequest, recent_context_items, select_scoped_context,
    suppressed_context_items, survival_ledger_for_scope,
)


SCOPE = ("owner", None, None, None)


def item(ident, *, stamp=10.0, **metadata):
    return MemoryItem(id=ident, text=f"Private payload {ident}",
        embedding=np.array([1.0, 0.0], dtype=np.float32), timestamp=stamp,
        salience=1.0, fidelity=1.0, level=0,
        meta={"tenant_id": "owner", "kind": "note", **metadata})


def request(**overrides):
    return ScopedRetrievalRequest(**{
        "scope": SCOPE, "kinds": None, "phase": None, "top_k": 2,
        "as_of": 100.0, **overrides,
    })


def forbidden_read():
    pytest.fail("Recent candidate authority was accessed without empty ranked success")


def test_ranked_capability_is_narrow_and_success_never_reads_recent_state():
    # The ranker is a function, with no owner, store, lock, or private state to
    # discover. Its unusual result order must survive orchestration unchanged.
    source = [item(80, stamp=1.0), item(1, stamp=99.0)]
    vector = np.array([1.0, 0.0], dtype=np.float32)
    observed = []

    def ranked(query, **kwargs):
        observed.append(kwargs)
        assert query is vector
        assert set(kwargs) == {"tenant_id", "client_id", "session_id", "instance_id",
            "kinds", "top_k", "strict_scope", "as_of", "eligible"}
        assert kwargs["strict_scope"] is True
        assert kwargs["as_of"] == 100.0
        assert kwargs["kinds"] == ("note",)
        assert kwargs["eligible"](item(3, expires_at=100.001))
        assert not kwargs["eligible"](item(4, expires_at=100.0))
        assert not kwargs["eligible"](item(5, source_trust="untrusted"))
        return source

    result = select_scoped_context(retrieve_filtered=ranked,
        recent_candidates=forbidden_read, query_embedding=vector,
        request=request(kinds=("note",), phase="debug"))
    assert len(observed) == 1
    assert result is not source
    assert result[0] is source[0] and result[1] is source[1]
    result.clear()
    assert len(source) == 2


@pytest.mark.parametrize("error", [OSError, ValueError, RuntimeError])
def test_rank_failure_preserves_exception_and_never_falls_back(error):
    failure = error("rank authority failed")

    def ranked(*_args, **_kwargs):
        raise failure

    with pytest.raises(error) as caught:
        select_scoped_context(retrieve_filtered=ranked, recent_candidates=forbidden_read,
            query_embedding=np.ones(2), request=request())
    assert caught.value is failure


def test_recent_authority_is_read_after_empty_semantic_result():
    candidates = [item(1, stamp=10.0)]
    calls = []

    def ranked(*_args, **_kwargs):
        calls.append("rank")
        candidates[:] = [item(2, stamp=11.0)]
        return []

    def recent():
        calls.append("recent")
        return candidates

    result = select_scoped_context(retrieve_filtered=ranked, recent_candidates=recent,
        query_embedding=np.ones(2), request=request())
    assert calls == ["rank", "recent"]
    assert [record.id for record in result] == [2]


def test_request_detaches_collection_inputs_and_keeps_empty_kind_identity():
    scope = list(SCOPE)
    kinds = []
    frozen = request(scope=scope, kinds=kinds)
    scope[0] = "other"
    kinds.append("error")
    assert frozen.scope == SCOPE
    assert frozen.kinds == ()
    with pytest.raises(FrozenInstanceError):
        frozen.as_of = 500.0
    # Empty kinds still permit legacy recent fallback outside execute/debug.
    assert [record.id for record in recent_context_items([item(7)], request=frozen)] == [7]


@pytest.mark.parametrize("field", ["tenant_id", "client_id", "session_id", "instance_id"])
def test_suppression_evidence_cannot_leak_foreign_scope_and_ignores_rank_filters(field):
    foreign = item(999, **{field: "foreign"}, memory_state="deleted")
    candidates = [item(20, kind="error", phase="execute", memory_state="quarantined"),
        foreign, item(3, kind="action", phase="debug", expires_at=100.0)]
    found = suppressed_context_items(candidates,
        request=request(kinds=("note",), phase="plan", top_k=1))
    assert found == [
        {"id": "20", "reason": "state_quarantined"},
        {"id": "3", "reason": "expired"},
    ]
    assert suppressed_context_items(candidates,
        request=request(include_quarantined=True)) == []


def test_latest_ledger_is_chosen_before_suppression_without_reviving_old_state():
    older = item(2, kind="survival_ledger", tenant_id=8, session_id=5)
    newest = item(3, stamp=20.0, kind="survival_ledger", tenant_id=8, session_id=5,
        memory_state="deleted")
    tied = item(4, stamp=20.0, kind="survival_ledger", tenant_id=8, session_id=5)
    selected = survival_ledger_for_scope([older, newest, tied],
        scope=("8", None, "5", None), enabled=True)
    assert selected is newest


@pytest.mark.parametrize("module", [context, scoped_retrieval])
def test_services_do_not_reach_adapter_or_mutable_store_ownership(module):
    tree = ast.parse(inspect.getsource(module))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Attribute):
            assert node.attr not in {"_items", "_lock", "_journal", "_mutation_transaction",
                "_last_observed_state", "_score_candidates", "_select_top_items"}
    assert not any(name.split(".")[-1] in {
        "dml_adapter", "store_lock", "journal", "threading", "persistence",
    } for name in imports)


class FixedEmbedder:
    receipt_embedding_identity = "adversarial-retrieval-v1"

    def embed(self, _text):
        return np.array([1.0, 0.0], dtype=np.float32)


@pytest.fixture
def factory(tmp_path):
    adapters = []

    def build(*, schema=2, directory=None, **overrides):
        directory = directory or tmp_path / f"store-{len(adapters)}"
        existing_journal = (directory / "dml_state.sqlite3").exists()
        config = {"storage_dir": str(directory), "model_name": "dummy",
            "embedding_model": None, "checkpoint_interval_seconds": 0,
            "metrics_enabled": False, "persistence": {"journal": schema is not None,
                "receipts": schema in (2, 3, 4),
                "outbox": schema == 3 or (schema == 4 and existing_journal), "enable": False},
            "rag_store": {"enable": False},
            "dpm": {"enable": False, "include_in_context": False}, **overrides}
        adapter = DMLAdapter(config_overrides=config, embedder=FixedEmbedder(),
            start_aging_loop=False)
        adapters.append(adapter)
        if schema == 4 and adapter._journal.schema_version != 4:
            adapter.close(persist=False)
            target = directory.parent / f"{directory.name}-v4" / "dml_state.sqlite3"
            upgrade_outbox_journal(adapter._journal.path, target)
            return build(schema=4, directory=target.parent, **overrides)
        return adapter

    yield build
    for adapter in reversed(adapters):
        adapter.close(persist=False)


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_writer_cannot_change_revision_until_context_evidence_is_materialized(factory, monkeypatch, schema):
    reader = factory(schema=schema)
    initial = reader.ingest_memory_receipted("Borrowed memory must stay stable",
        tenant_id="owner", idempotency_key="original")
    memory = initial["result"]["memory"]
    writer = factory(schema=schema, directory=reader._journal.path.parent)
    report_ready, release, writer_blocked = Event(), Event(), Event()
    build_report = adapter_module.build_context_report
    acquire_lock = store_lock.acquire_file_lock

    def pinned_report(**kwargs):
        assert getattr(reader._mutation_local, "depth", 0) == 1
        report = build_report(**kwargs)
        report_ready.set()
        assert release.wait(10)
        return report

    def observed_acquire(handle):
        try:
            return acquire_lock(handle)
        except BlockingIOError:
            writer_blocked.set()
            raise

    monkeypatch.setattr(adapter_module, "build_context_report", pinned_report)
    monkeypatch.setattr(store_lock, "acquire_file_lock", observed_acquire)
    with ThreadPoolExecutor(max_workers=2) as executor:
        reading = executor.submit(reader.retrieve_context, "query", tenant_id="owner", as_of=100.0)
        try:
            assert report_ready.wait(10)
            writing = executor.submit(writer.retire_memory_receipted, memory["id"],
                expected_memory_digest=memory_digest(memory), reason="retire after read",
                idempotency_key="retirement", tenant_id="owner")
            assert writer_blocked.wait(10)
            assert not writing.done()
            assert reader._journal.read_snapshot()[0] == initial["revision"]
        finally:
            release.set()
        report = reading.result(timeout=10)
        retirement = writing.result(timeout=10)
    assert report["decision"]["store_revision"] == initial["revision"]
    assert report["decision"]["returned_ids"] == [str(memory["id"])]
    assert report["decision"]["suppressed"] == []
    monkeypatch.setattr(adapter_module, "build_context_report", build_report)
    later = reader.retrieve_context("query", tenant_id="owner", as_of=100.0)
    assert later["decision"]["store_revision"] == retirement["revision"]
    assert later["items"] == []
    assert later["decision"]["suppressed"] == [{"id": str(memory["id"]), "reason": "state_deleted"}]


def test_embedding_identity_change_at_ownership_boundary_fails_before_ranking(factory, monkeypatch):
    adapter = factory()
    adapter.ingest_memory_receipted("Pinned identity", tenant_id="owner", idempotency_key="seed")
    transaction = adapter._mutation_transaction

    @contextmanager
    def changed_before_ownership(operation):
        if operation == "retrieve-context":
            adapter.embedder.receipt_embedding_identity = "changed-after-query"
        with transaction(operation):
            yield

    def forbidden_rank(*_args, **_kwargs):
        pytest.fail("Identity-mismatched query reached ranking")

    monkeypatch.setattr(adapter, "_mutation_transaction", changed_before_ownership)
    monkeypatch.setattr(adapter.store, "retrieve_filtered", forbidden_rank)
    with pytest.raises(ReceiptEmbeddingCompatibilityError):
        adapter.retrieve_context("query", tenant_id="owner")


def test_public_report_detaches_caller_kinds_and_borrowed_personality_overlay(factory, monkeypatch):
    adapter = factory(schema=None, dpm={"enable": False, "include_in_context": True})
    adapter.ingest_memory("Stable source metadata", tenant_id="owner", kind="note",
        meta={"nested": {"tags": ["store-owned"]}})
    kinds = ["note"]
    overlay = {"traits": {"tags": ["personality-owned"]}}
    monkeypatch.setattr(adapter, "personality_overlay", lambda **_: overlay)
    monkeypatch.setattr(adapter.personality_matrix, "render_context_block", lambda _: "Persona")
    report = adapter.retrieve_context("query", tenant_id="owner", kinds=kinds, as_of=100.0)
    untouched = deepcopy(report)
    kinds.append("action")
    overlay["traits"]["tags"].append("later")
    assert report == untouched
    report["kinds"].append("error")
    report["personality_overlay"]["traits"]["tags"].clear()
    report["items"][0]["meta"]["nested"]["tags"].clear()
    assert kinds == ["note", "action"]
    assert overlay["traits"]["tags"] == ["personality-owned", "later"]
    assert adapter.store.items()[0].meta["nested"]["tags"] == ["store-owned"]


@pytest.mark.parametrize("mutation_stage", ["ranking", "personality"])
def test_report_describes_frozen_kinds_even_when_callbacks_mutate_original(factory, monkeypatch, mutation_stage):
    adapter = factory(dpm={"enable": False, "include_in_context": True})
    note = adapter.ingest_memory_receipted("Selected note", tenant_id="owner", kind="note",
        idempotency_key="note")["result"]["memory"]
    adapter.ingest_memory_receipted("Unselected action", tenant_id="owner", kind="action",
        idempotency_key="action")
    kinds = ["note"]
    ranked = adapter.store.retrieve_filtered

    def mutating_rank(*args, **kwargs):
        kinds[:] = ["action"]
        return ranked(*args, **kwargs)

    def personality(**_kwargs):
        if mutation_stage == "personality":
            kinds[:] = ["action"]
        return None

    if mutation_stage == "ranking":
        monkeypatch.setattr(adapter.store, "retrieve_filtered", mutating_rank)
    monkeypatch.setattr(adapter, "personality_overlay", personality)
    report = adapter.retrieve_context("query", tenant_id="owner", kinds=kinds, as_of=100.0)
    assert kinds == ["action"]
    assert report["kinds"] == report["decision"]["kinds"] == ["note"]
    assert report["decision"]["returned_ids"] == [str(note["id"])]
