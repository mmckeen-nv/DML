"""Behavioral boundaries for owned, scoped context selection."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from daystrom_dml.memory_store import MemoryItem, MemoryStore
from daystrom_dml.services.scoped_retrieval import (
    ScopedRetrievalRequest,
    recent_context_items,
    select_scoped_context,
    suppressed_context_items,
    survival_ledger_for_scope,
)
from daystrom_dml.summarizer import DummySummarizer


def record(identity, *, timestamp=99.0, vector=(1.0, 0.0), salience=1.0, **meta):
    return MemoryItem(
        id=identity, text=f"memory {identity}", embedding=np.array(vector, dtype=np.float32),
        timestamp=timestamp, salience=salience, fidelity=1.0, level=0,
        meta={"tenant_id": "owner", "kind": "note", **meta},
    )


def selection_request(**overrides):
    return ScopedRetrievalRequest(**{
        "scope": ("owner", None, None, None), "kinds": None, "phase": None,
        "top_k": 2, "as_of": 100.0, **overrides,
    })


def ids(items):
    return [item.id for item in items]


@pytest.fixture
def store_factory():
    stores = []

    def build(records, **overrides):
        store = MemoryStore(**{
            "summarizer": DummySummarizer(), "beta_a": 0.08, "beta_r": 0.2,
            "eta": 0.15, "gamma": 0.02, "kappa": 0.5, "tau_s": 0.3,
            "theta_merge": 0.92, "K": 4, "capacity": 100,
            "start_aging_loop": False, "similarity_threshold": 0.0, **overrides,
        })
        store.import_state({"items": [item.to_dict() for item in records], "lineage": []})
        stores.append(store)
        return store

    yield build
    for store in stores:
        store.close()


def select(store, request, vector=(1.0, 0.0)):
    return select_scoped_context(
        retrieve_filtered=store.retrieve_filtered, recent_candidates=store.items,
        query_embedding=np.array(vector, dtype=np.float32), request=request,
    )


def test_request_copies_mutable_scope_and_kind_collections():
    scope = ["owner", None, None, None]
    kinds = ["action"]
    request = selection_request(scope=scope, kinds=kinds)
    scope[0] = "other"
    kinds.clear()
    assert request.scope == ("owner", None, None, None)
    assert request.kinds == ("action",)
    with pytest.raises(FrozenInstanceError):
        request.phase = "debug"
    with pytest.raises(ValueError, match="four fields"):
        selection_request(scope=("owner",))


def test_reader_receives_pinned_policy_and_success_never_reads_fallback():
    query = np.array([1.0, 0.0], dtype=np.float32)
    selected = record(4)
    request = selection_request(
        scope=("owner", "client", "session", "instance"), kinds=("action",),
        phase="execute", top_k=7, as_of=50.0,
    )
    calls = []

    def retrieve(vector, **kwargs):
        calls.append(vector)
        eligible = kwargs.pop("eligible")
        assert kwargs == {
            "tenant_id": "owner", "client_id": "client", "session_id": "session",
            "instance_id": "instance", "kinds": ("action",), "top_k": 7,
            "strict_scope": True, "as_of": 50.0,
        }
        assert eligible(record(1, expires_at=51.0))
        assert not eligible(record(2, expires_at=50.0))
        assert not eligible(record(3, memory_state="quarantined"))
        return [selected]

    def no_fallback():
        pytest.fail("A successful ranked hit must not read recent candidates")

    result = select_scoped_context(
        retrieve_filtered=retrieve, recent_candidates=no_fallback,
        query_embedding=query, request=request,
    )
    assert len(calls) == 1 and calls[0] is query
    assert result == [selected] and result[0] is selected


def test_empty_success_reads_fallback_once_after_ranking():
    events = []
    recent = record(2)

    def retrieve(*_args, **_kwargs):
        events.append("rank")
        return []

    def candidates():
        events.append("recent")
        return [recent]

    assert select_scoped_context(
        retrieve_filtered=retrieve, recent_candidates=candidates,
        query_embedding=np.ones(2), request=selection_request(),
    ) == [recent]
    assert events == ["rank", "recent"]


def test_reader_error_propagates_without_fallback_read():
    failure = RuntimeError("reader failed")

    def retrieve(*_args, **_kwargs):
        raise failure

    def no_fallback():
        pytest.fail("A reader error must not become recent-memory success")

    with pytest.raises(RuntimeError) as caught:
        select_scoped_context(
            retrieve_filtered=retrieve, recent_candidates=no_fallback,
            query_embedding=np.ones(2), request=selection_request(),
        )
    assert caught.value is failure


@pytest.mark.parametrize("private_field", ["tenant_id", "client_id", "session_id", "instance_id"])
def test_real_store_suppresses_before_top_k_and_preserves_scope_and_ties(store_factory, private_field):
    store = store_factory([
        record(4), record(2),
        record(1, salience=1000.0, memory_state="quarantined"),
        record(3, salience=100.0, **{private_field: "private"}),
    ])
    before = store.export_state()
    request = selection_request()
    assert ids(select(store, request)) == [2, 4]
    assert suppressed_context_items(store.items(), request=request) == [
        {"id": "1", "reason": "state_quarantined"},
    ]
    assert store.export_state() == before


@pytest.mark.parametrize("as_of,expected", [(7200.0, [2]), (36000.0, [1])])
def test_real_store_scoring_retains_weights_similarity_floor_and_pinned_time(store_factory, as_of, expected):
    store = store_factory([
        record(1, timestamp=0.0, salience=4.0),
        record(2, timestamp=7200.0, salience=0.0),
        record(3, timestamp=7200.0, salience=1000.0, vector=(-1.0, 0.0)),
    ], eta=1.0, gamma=0.1, kappa=0.0, similarity_threshold=0.5)
    assert ids(select(store, selection_request(top_k=1, as_of=as_of))) == expected


def test_dimension_mismatch_falls_back_with_phase_scope_and_recent_ties(store_factory):
    store = store_factory([
        record(4, timestamp=99.0, kind="action", phase="execute"),
        record(2, timestamp=99.0, kind="action", phase=" Execute "),
        record(1, timestamp=100.0, kind="action", phase="plan"),
        record(3, timestamp=101.0, kind="action", phase="execute", client_id="private"),
    ])
    before = store.export_state()
    request = selection_request(kinds=("action",), phase="execute")
    assert ids(select(store, request, vector=(1.0, 0.0, 0.0))) == [2, 4]
    assert store.export_state() == before


@pytest.mark.parametrize("kinds,expected", [(None, [2]), ((), [1])])
def test_empty_kinds_preserves_semantic_vs_recent_fallback_distinction(store_factory, kinds, expected):
    store = store_factory([
        record(1, timestamp=100.0, salience=0.0),
        record(2, timestamp=99.0, salience=100.0),
    ])
    assert ids(select(store, selection_request(kinds=kinds, top_k=1))) == expected


@pytest.mark.parametrize("vector,expected", [((1.0, 0.0), [1]), ((0.0, 1.0), [2])])
def test_phase_filter_is_preserved_only_for_recent_scoped_fallback(store_factory, vector, expected):
    store = store_factory([
        record(1, kind="action", phase="plan"),
        record(2, kind="action", phase="execute", vector=(-1.0, 0.0)),
    ], similarity_threshold=0.5)
    request = selection_request(kinds=("action",), phase="execute")
    assert ids(select(store, request, vector)) == expected


@pytest.mark.parametrize("phase", ["execute", "debug"])
def test_empty_kinds_recent_fallback_retains_agentic_defaults(phase):
    candidates = [
        record(1, timestamp=101.0, kind="note", phase=phase),
        record(2, timestamp=100.0, kind="ACTION", phase=phase),
        record(3, timestamp=99.0, kind="observation"),
        record(4, timestamp=102.0, kind="error", phase="plan"),
    ]
    assert ids(recent_context_items(
        candidates, request=selection_request(kinds=(), phase=phase),
    )) == [2, 3]


def test_unscoped_recent_reads_are_global_unless_explicitly_restricted():
    candidates = [record(1, timestamp=101.0), record(2, tenant_id=None)]
    request = selection_request(scope=(None, None, None, None))
    assert ids(recent_context_items(candidates, request=request)) == [1, 2]
    assert ids(recent_context_items(candidates, request=request, require_unscoped=True)) == [2]


def test_suppression_evidence_preserves_source_order_and_ignores_kind_and_phase():
    candidates = [
        record(8, kind="plan", phase="plan", memory_state="superseded"),
        record(2, kind="note", expires_at=100.0),
        record(1, client_id="private", memory_state="deleted"),
        record(3, source_trust="untrusted"),
    ]
    request = selection_request(kinds=("action",), phase="execute", top_k=1)
    assert suppressed_context_items(candidates, request=request) == [
        {"id": "8", "reason": "state_superseded"},
        {"id": "2", "reason": "expired"},
        {"id": "3", "reason": "untrusted_source"},
    ]


@pytest.mark.parametrize("as_of,expected", [(99.0, [1]), (100.0, [])])
def test_expiry_uses_the_same_boundary_for_recent_selection_and_evidence(as_of, expected):
    candidates = [record(1, expires_at=100.0)]
    request = selection_request(as_of=as_of)
    assert ids(recent_context_items(candidates, request=request)) == expected
    evidence = suppressed_context_items(candidates, request=request)
    assert evidence == ([] if expected else [{"id": "1", "reason": "expired"}])


def test_operator_inspection_retains_all_suppression_bypass(store_factory):
    store = store_factory([
        record(1, memory_state="deleted"), record(2, source_trust="untrusted"),
        record(3, expires_at=0.0), record(4, namespace="quarantined"),
    ])
    request = selection_request(include_quarantined=True, top_k=4)
    assert ids(select(store, request)) == [1, 2, 3, 4]
    assert ids(recent_context_items(store.items(), request=request)) == [1, 2, 3, 4]
    assert suppressed_context_items(store.items(), request=request) == []


def test_latest_ledger_uses_string_scope_and_source_order_ties_before_suppression():
    candidates = [
        record(8, timestamp=100.0, kind="survival_ledger", session_id=2),
        record(1, timestamp=100.0, kind="survival_ledger", session_id="2"),
        record(2, timestamp=101.0, kind="survival_ledger", session_id="2", client_id="private"),
    ]
    scope = ("owner", None, "2", None)
    assert survival_ledger_for_scope(candidates, scope=scope, enabled=True) is candidates[0]
    suppressed_latest = record(
        9, timestamp=102.0, kind="survival_ledger", session_id="2", memory_state="quarantined",
    )
    candidates.append(suppressed_latest)
    assert survival_ledger_for_scope(candidates, scope=scope, enabled=True) is suppressed_latest


@pytest.mark.parametrize("scope,enabled", [
    (("owner", None, "session", None), False),
    ((None, None, "session", None), True),
    (("", None, "session", None), True),
    (("owner", None, None, None), True),
    (("owner", None, "", None), True),
])
def test_ledger_requires_enabled_tenant_and_session(scope, enabled):
    candidates = [record(1, kind="survival_ledger", session_id="session")]
    assert survival_ledger_for_scope(candidates, scope=scope, enabled=enabled) is None
