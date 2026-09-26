"""Public retrieval oracles captured before extracting scoped context services.

The small literal responses below intentionally describe observable behavior,
including the different phase rules of exact selection and recent fallback.
All stores are temporary; receipt scenarios exercise their real journals.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from daystrom_dml.agent_schema import MemoryPhase
from daystrom_dml.dml_adapter import DMLAdapter
from daystrom_dml.services.context import ContextBudgetError
from daystrom_dml.services.outbox_migration import upgrade_outbox_journal
from daystrom_dml.services.receipt_ingestion import ReceiptEmbeddingCompatibilityError


NOW = 1_700_000_000.0
SCOPE = {"tenant_id": "owner", "client_id": None, "session_id": None, "instance_id": None}
PROFILES = ["legacy", 1, 2, 3, 4]


class FixedEmbedder:
    receipt_embedding_identity = "retrieval-characterization-v1"

    def __init__(self):
        self.calls = []
        self.vectors = {}
        self.callback = None

    def embed(self, text):
        self.calls.append(text)
        if self.callback:
            self.callback(text)
        return np.array(self.vectors.get(text, [1, 0, 0, 0]), dtype=np.float32)


class LiteralSummarizer:
    def summarize(self, text, max_len=256):
        return text[:max_len]


@pytest.fixture
def factory(tmp_path, monkeypatch):
    monkeypatch.setattr("time.time", lambda: NOW)
    monkeypatch.setattr("time.perf_counter", lambda: 100.0)
    adapters = []

    def build(*, profile="legacy", directory=None, embedder=None, **overrides):
        directory = directory or tmp_path / f"store-{len(adapters)}"
        config = {
            "storage_dir": str(directory), "model_name": "dummy", "embedding_model": None,
            "checkpoint_interval_seconds": 0, "metrics_enabled": False,
            "persistence": {"enable": False, "journal": profile != "legacy",
                            "receipts": profile in (2, 3, 4), "outbox": profile == 3},
            "rag_store": {"enable": False},
            "dpm": {"enable": False, "include_in_context": False},
            "theta_merge": 2.0, "eta": 0.0, "gamma": 0.0, "kappa": 0.0,
            "similarity_threshold": 0.0, "enable_quality_on_retrieval": False,
            "ann_min_items": 0, "token_budget": 600, "dml_context_max_items": 10,
            **overrides,
        }
        adapter = DMLAdapter(config_overrides=config, embedder=embedder or FixedEmbedder(),
                             summarizer=LiteralSummarizer(), start_aging_loop=False)
        adapters.append(adapter)
        if profile == 4:
            adapter.close(persist=False)
            destination = directory.parent / f"{directory.name}-migrated" / "dml_state.sqlite3"
            upgrade_outbox_journal(adapter._journal.path, destination)
            config["storage_dir"] = str(destination.parent)
            config["persistence"]["outbox"] = True
            adapter = DMLAdapter(config_overrides=config, embedder=adapter.embedder,
                                 summarizer=LiteralSummarizer(), start_aging_loop=False)
            adapters.append(adapter)
        if profile != "legacy":
            assert adapter._journal.schema_version == profile
        return adapter

    yield build
    for adapter in reversed(adapters):
        adapter.close(persist=False)


def remember(adapter, text, *, meta=None, kind="note", **scope):
    arguments = {**SCOPE, **scope, "kind": kind, "meta": {"source": "fixture", **(meta or {})}}
    if adapter._journal is not None and adapter._journal.schema_version in (2, 3, 4):
        receipt = adapter.ingest_memory_receipted(text, idempotency_key=text, **arguments)
        return receipt["result"]["memory"]["id"]
    return adapter.ingest_memory(text, **arguments).id


def public_digest(value):
    """Independent canonical JSON check for the public evidence contract."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@pytest.mark.parametrize("profile", PROFILES)
def test_exact_scoped_response_ranking_and_evidence(factory, profile):
    adapter = factory(profile=profile)
    first = remember(adapter, "First scoped memory.")
    second = remember(adapter, "Second scoped memory.")
    adapter.embedder.vectors["Lower similarity memory."] = [0, 1, 0, 0]
    remember(adapter, "Lower similarity memory.")
    for member in SCOPE:
        remember(adapter, f"Excluded {member} memory.", **{member: "other"})
    before = deepcopy(adapter.store.export_state())
    records = {record["id"]: record for record in before["items"]}
    revision = adapter._journal.revision if adapter._journal else None
    calls = len(adapter.embedder.calls)

    report = adapter.retrieve_context("scoped query", tenant_id="owner", top_k=2, as_of=NOW)

    context = ("=== Retrieved Context ===\n"
               "- (2023-11-14) [source=fixture]\n  First scoped memory.\n"
               "- (2023-11-14) [source=fixture]\n  Second scoped memory.")
    entries = []
    for ident, text in [(first, "First scoped memory."), (second, "Second scoped memory.")]:
        record = records[ident]
        entries.append({"id": str(ident), "text": text, "summary": text,
                        "meta": record["meta"], "timestamp": NOW, "level": 0,
                        "fidelity": 1.0, "salience": record["salience"], "tokens": 5})
    evidence = {
        "schema_version": 1, "policy_version": "scoped-exact-v1",
        "query_digest": public_digest("scoped query"), "scope": SCOPE,
        "store_revision": revision, "effective_time": NOW, "top_k": 2, "kinds": None,
        "query_embedding_digest": public_digest([1.0, 0.0, 0.0, 0.0]),
        "returned_ids": [str(first), str(second)],
        "source_digests": [public_digest(records[ident]["meta"]) for ident in [first, second]],
        "suppressed": [], "context_digest": public_digest(context),
        "replay_inputs_complete": False, "deterministic_scoped_ranking": True,
    }
    assert report == {
        "decision": {**evidence, "decision_digest": public_digest(evidence)},
        "token_count_kind": "estimate", "raw_context": context,
        "context_tokens": 34, "top_k": 2, "kinds": None, "phase": None,
        "include_quarantined": False, "items": entries,
        "survival_ledger_included": False, "personality_overlay": None, "latency_ms": 0,
    }
    assert adapter.embedder.calls[calls:] == ["scoped query"]
    assert adapter.store.export_state() == before


@pytest.mark.parametrize("profile", ["legacy", 2])
def test_lifecycle_filters_before_top_k_and_reports_only_exact_scope(factory, profile):
    adapter = factory(profile=profile)
    cases = [
        ({"memory_state": "quarantined"}, "state_quarantined"),
        ({"lifecycle_state": "deleted"}, "state_deleted"),
        ({"namespace": "quarantine"}, "quarantine_namespace"),
        ({"source_trust": "untrusted"}, "untrusted_source"),
        ({"superseded_by": 99}, "explicitly_superseded"),
        ({"expires_at": NOW}, "expired"),
    ]
    suppressed = []
    for index, (metadata, reason) in enumerate(cases):
        ident = remember(adapter, f"Hidden memory {index}.", meta=metadata)
        suppressed.append({"id": str(ident), "reason": reason})
    active = remember(adapter, "Available scoped memory.", meta={"expires_at": NOW + 1})
    other_kind = remember(adapter, "Hidden different kind.", kind="plan",
                          meta={"memory_state": "suppressed"})
    suppressed.append({"id": str(other_kind), "reason": "state_suppressed"})
    remember(adapter, "Hidden other session.", session_id="other",
             meta={"memory_state": "quarantined"})
    before = deepcopy(adapter.store.export_state())

    normal = adapter.retrieve_context("query", tenant_id="owner", kinds=["note"], top_k=1,
                                      as_of=NOW)
    inspected = adapter.retrieve_context("query", tenant_id="owner", kinds=["note"], top_k=10,
                                         include_quarantined=True, as_of=NOW)

    assert [item["id"] for item in normal["items"]] == [str(active)]
    assert normal["decision"]["suppressed"] == suppressed
    assert "Hidden" not in normal["raw_context"]
    assert [item["id"] for item in inspected["items"]] == [
        *[row["id"] for row in suppressed[:-1]], str(active)]
    assert inspected["decision"]["suppressed"] == []
    assert inspected["include_quarantined"] is True
    assert adapter.store.export_state() == before, "inspection cannot promote lifecycle authority"


@pytest.mark.parametrize("profile", ["legacy", 2])
def test_recent_fallback_keeps_available_memories_and_filters_phase_scope_and_lifecycle(factory, profile):
    adapter = factory(profile=profile, similarity_threshold=0.9)
    rows = [
        ("First fallback action.", "action", {"phase": "execute"}, {}),
        ("Second fallback observation.", "observation", {}, {}),
        ("Wrong phase action.", "action", {"phase": "plan"}, {}),
        ("Wrong kind plan.", "plan", {"phase": "execute"}, {}),
        ("Quarantined action.", "action", {"memory_state": "quarantined"}, {}),
        ("Other session action.", "action", {}, {"session_id": "elsewhere"}),
    ]
    ids = []
    for text, kind, metadata, scope in rows:
        adapter.embedder.vectors[text] = [0, 1, 0, 0]
        ids.append(remember(adapter, text, kind=kind, meta=metadata, **scope))

    report = adapter.retrieve_context("unrelated query", tenant_id="owner", phase=" EXECUTE ",
                                       top_k=5, as_of=NOW)

    assert report["phase"] == "execute"
    assert report["kinds"] == ["action", "observation", "error"]
    assert [entry["id"] for entry in report["items"]] == [str(ids[0]), str(ids[1])]
    assert report["raw_context"] == (
        "=== Retrieved Context ===\n"
        "- (2023-11-14) [source=fixture]\n  First fallback action.\n"
        "- (2023-11-14) [source=fixture]\n  Second fallback observation.")
    assert report["decision"]["suppressed"] == [{"id": str(ids[4]), "reason": "state_quarantined"}]


def test_exact_scoped_phase_uses_kind_defaults_without_recent_phase_filter(factory):
    adapter = factory()
    plan_action = remember(adapter, "Action recorded during planning.", kind="action", meta={"phase": "plan"})
    remember(adapter, "Planning note.", kind="plan")
    result = adapter.retrieve_context("query", tenant_id="owner", phase="execute")
    # Compatibility: exact scoped ranking does not apply the recent fallback's
    # metadata-phase filter. Extraction must not silently alter that contract.
    assert [item["id"] for item in result["items"]] == [str(plan_action)]
    assert result["phase"] == "execute"
    assert result["kinds"] == ["action", "observation", "error"]


@pytest.mark.parametrize("profile", ["legacy", 2])
def test_recent_fallback_orders_newest_first_then_lower_id(factory, monkeypatch, profile):
    adapter = factory(profile=profile, similarity_threshold=0.9)
    for text in ["Older available memory.", "Newer first memory.", "Newer second memory."]:
        adapter.embedder.vectors[text] = [0, 1, 0, 0]
    older = remember(adapter, "Older available memory.")
    monkeypatch.setattr("time.time", lambda: NOW + 10)
    newer_first = remember(adapter, "Newer first memory.")
    newer_second = remember(adapter, "Newer second memory.")

    report = adapter.retrieve_context("query", tenant_id="owner", top_k=3, as_of=NOW + 10)

    assert [entry["id"] for entry in report["items"]] == [
        str(newer_first), str(newer_second), str(older)]
    assert [entry["timestamp"] for entry in report["items"]] == [NOW + 10, NOW + 10, NOW]


def test_empty_kind_list_remains_distinct_from_omitted_kinds_in_exact_then_fallback(factory):
    adapter = factory()
    first = remember(adapter, "Available with empty kinds.")
    report = adapter.retrieve_context("query", tenant_id="owner", kinds=[])
    assert report["kinds"] == []
    assert [item["id"] for item in report["items"]] == [str(first)]


def test_legacy_unscoped_literal_failure_still_returns_recent_available_context(factory, monkeypatch):
    adapter = factory(similarity_threshold=0.9)
    adapter.embedder.vectors["Available legacy observation."] = [0, 1, 0, 0]
    memory = remember(adapter, "Available legacy observation.", kind="observation")
    adapter.embedder.vectors["Hidden legacy observation."] = [0, 1, 0, 0]
    remember(adapter, "Hidden legacy observation.", kind="observation", meta={"memory_state": "quarantined"})

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("literal index unavailable")

    monkeypatch.setattr(adapter.literal_retriever, "retrieve", unavailable)
    report = adapter.retrieve_context("unrelated query", phase="execute", as_of=NOW)
    assert [entry["id"] for entry in report["items"]] == [str(memory)]
    assert "Available legacy observation." in report["raw_context"]
    assert "Hidden" not in report["raw_context"]
    assert report["decision"]["deterministic_scoped_ranking"] is False


def test_router_arguments_precedence_and_public_top_k_cap(factory):
    adapter = factory(dml_top_k=7)
    observation = remember(adapter, "Router observation.", kind="observation")
    note = remember(adapter, "Explicit note.", kind="note")
    decisions = []

    def decide(**kwargs):
        decisions.append(kwargs)
        return SimpleNamespace(selected_profile="fixture", overrides=SimpleNamespace(
            top_k=1, allowed_kinds=["observation"]))

    adapter.agentic_mode_enabled = True
    adapter.agentic_router = SimpleNamespace(decide=decide)
    routed = adapter.retrieve_context("x" * 130, tenant_id="owner", phase="debug")
    explicit = adapter.retrieve_context("query", tenant_id="owner", phase="unknown", kinds=["note"], top_k=99)

    assert decisions == [
        {"meta": {"prompt": "x" * 100}, "phase": MemoryPhase.DEBUG, "token_pressure": 0.0},
        {"meta": {"prompt": "query"}, "phase": None, "token_pressure": 0.0},
    ]
    assert (routed["top_k"], routed["kinds"]) == (1, ["observation"])
    assert [entry["id"] for entry in routed["items"]] == [str(observation)]
    assert (explicit["top_k"], explicit["kinds"], explicit["phase"]) == (10, ["note"], None)
    assert [entry["id"] for entry in explicit["items"]] == [str(note)]


def test_dpm_prefix_budget_and_scope_arguments_preserve_pipeline_order(factory, monkeypatch):
    adapter = factory(token_budget=30, dpm={"enable": False, "include_in_context": True})
    ident = remember(adapter, "abcdefghijklmnopqrstuvwx", session_id="session")
    remember(adapter, "Second memory must not fit.", session_id="session")
    events = []
    original_transaction = adapter._mutation_transaction
    original_rank = adapter.store.retrieve_filtered
    overlay = {"overlay": {"rendered_text": "Be concise."}}

    @contextmanager
    def owned(operation):
        events.append(("acquire", operation))
        with original_transaction(operation):
            events.append(("owned", operation))
            yield
        events.append(("release", operation))

    def rank(*args, **kwargs):
        events.append(("rank", kwargs["strict_scope"], kwargs["as_of"]))
        return original_rank(*args, **kwargs)

    def personality(**kwargs):
        events.append(("dpm", kwargs))
        return deepcopy(overlay)

    adapter.embedder.callback = lambda text: events.append(("embed", text))
    monkeypatch.setattr(adapter, "_mutation_transaction", owned)
    monkeypatch.setattr(adapter.store, "retrieve_filtered", rank)
    monkeypatch.setattr(adapter, "personality_overlay", personality)
    report = adapter.retrieve_context("query", tenant_id="owner", session_id="session",
        dpm_project_id="project", top_k=2, as_of=NOW)

    assert events == [
        ("embed", "query"), ("acquire", "retrieve-context"), ("owned", "retrieve-context"),
        ("rank", True, NOW), ("dpm", {"prompt": "query", "thread_id": "session",
         "project_id": "project", "relationship_id": "owner"}), ("release", "retrieve-context"),
    ]
    assert report["raw_context"] == (
        "=== Personality Matrix ===\nBe concise.\n=== Retrieved Context ===\n"
        "- (2023-11-14) [source=fixture]\n  abcdefghijklmnopqrstuvwx")
    assert report["context_tokens"] == 30
    assert [entry["id"] for entry in report["items"]] == [str(ident)]
    assert report["personality_overlay"] == overlay


@pytest.mark.parametrize("budget, expected_text, expected_tokens", [(27, "abcdefghijkl", 27), (9, None, 9)])
def test_prefix_survives_first_item_clipping_and_empty_context(factory, monkeypatch, budget, expected_text, expected_tokens):
    adapter = factory(token_budget=budget, dpm={"enable": False, "include_in_context": True})
    remember(adapter, "abcdefghijklmnopqrstuvwxyz0123456789abcdefghijkl")
    monkeypatch.setattr(adapter, "personality_overlay",
                        lambda **_: {"overlay": {"rendered_text": "Be concise."}})
    report = adapter.retrieve_context("query", tenant_id="owner")
    prefix = "=== Personality Matrix ===\nBe concise."
    assert report["context_tokens"] == expected_tokens
    if expected_text is None:
        assert report["items"] == []
        assert report["raw_context"] == prefix
    else:
        assert [entry["text"] for entry in report["items"]] == [expected_text]
        assert report["raw_context"] == (prefix + "\n=== Retrieved Context ===\n"
                                        "- (2023-11-14) [source=fixture]\n  " + expected_text)


def test_oversized_dpm_prefix_fails_without_returning_overbudget_context(factory, monkeypatch):
    adapter = factory(token_budget=8, dpm={"enable": False, "include_in_context": True})
    monkeypatch.setattr(adapter, "personality_overlay",
                        lambda **_: {"overlay": {"rendered_text": "Be concise."}})
    with pytest.raises(ContextBudgetError, match="prefix exceeds"):
        adapter.retrieve_context("query", tenant_id="owner")


def test_survival_ledger_is_prepended_deduplicated_and_cannot_reintroduce_quarantine(factory):
    adapter = factory()
    active = remember(adapter, "Available action.", session_id="session", kind="action")
    ledger = remember(adapter, "Continuity ledger.", session_id="session", kind="survival_ledger")
    remember(adapter, "Other-session ledger.", session_id="elsewhere", kind="survival_ledger")
    result = adapter.retrieve_context("query", tenant_id="owner", session_id="session", top_k=1,
                                      kinds=["action"])
    assert result["survival_ledger_included"] is True
    assert [entry["id"] for entry in result["items"]] == [str(ledger), str(active)]
    already_selected = adapter.retrieve_context("query", tenant_id="owner", session_id="session",
                                                kinds=["survival_ledger"], top_k=1)
    assert [entry["id"] for entry in already_selected["items"]] == [str(ledger)]
    next(item for item in adapter.store.items() if item.id == ledger).meta["memory_state"] = "quarantined"
    suppressed = adapter.retrieve_context("query", tenant_id="owner", session_id="session", top_k=1,
                                          kinds=["action"])
    assert suppressed["survival_ledger_included"] is False
    assert [entry["id"] for entry in suppressed["items"]] == [str(active)]
    assert "ledger" not in suppressed["raw_context"]


def test_latest_quarantined_ledger_does_not_fall_back_to_older_ledger(factory, monkeypatch):
    adapter = factory()
    remember(adapter, "Older eligible ledger.", session_id="session", kind="survival_ledger")
    active = remember(adapter, "Available action.", session_id="session", kind="action")
    monkeypatch.setattr("time.time", lambda: NOW + 10)
    newest = remember(adapter, "Newer quarantined ledger.", session_id="session", kind="survival_ledger",
                      meta={"memory_state": "quarantined"})

    report = adapter.retrieve_context("query", tenant_id="owner", session_id="session", kinds=["action"])

    assert report["survival_ledger_included"] is False
    assert [entry["id"] for entry in report["items"]] == [str(active)]
    assert report["decision"]["suppressed"] == [{"id": str(newest), "reason": "state_quarantined"}]
    assert "ledger" not in report["raw_context"]


def test_ledger_selection_flag_survives_compaction_omission(factory):
    adapter = factory(token_budget=1)
    remember(adapter, "Continuity ledger.", session_id="session", kind="survival_ledger")
    report = adapter.retrieve_context("query", tenant_id="owner", session_id="session", kinds=["action"])
    assert report["survival_ledger_included"] is True
    assert report["items"] == report["decision"]["returned_ids"] == []
    assert report["raw_context"] == ""
    assert report["context_tokens"] == 0


@pytest.mark.parametrize("profile", ["legacy", 2])
def test_response_metadata_and_decision_lists_are_detached_from_memory_and_next_call(factory, profile):
    adapter = factory(profile=profile)
    ident = remember(adapter, "Stable memory.", meta={"provenance": {"tags": ["original"]}})
    before = deepcopy(adapter.store.export_state())
    report = adapter.retrieve_context("query", tenant_id="owner")
    expected = deepcopy(report)
    report["items"][0]["meta"]["provenance"]["tags"].append("response mutation")
    report["items"][0]["text"] = "response mutation"
    report["decision"]["returned_ids"].append("invented")
    report["decision"]["scope"]["tenant_id"] = "other"
    assert adapter.store.export_state() == before
    assert adapter.retrieve_context("query", tenant_id="owner") == expected
    stored = next(item for item in adapter.store.items() if item.id == ident)
    stored.meta["provenance"]["tags"].append("later source mutation")
    assert report["items"][0]["meta"]["provenance"]["tags"] == ["original", "response mutation"]


def test_receipt_retrieval_refreshes_revision_after_query_embedding_without_blocking_writer(factory, tmp_path):
    reader = factory(profile=2, directory=tmp_path / "shared")
    first = remember(reader, "Initial committed memory.")
    writer = factory(profile=2, directory=tmp_path / "shared")
    entered, release = Event(), Event()

    def delayed(text):
        if text == "racing query":
            entered.set()
            assert release.wait(5), "query embedding was not released"

    reader.embedder.callback = delayed
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = executor.submit(reader.retrieve_context, "racing query", tenant_id="owner", top_k=3)
        try:
            assert entered.wait(5)
            second = executor.submit(remember, writer, "Committed during embedding.").result(timeout=5)
            committed_revision = writer._journal.revision
        finally:
            release.set()
        report = pending.result(timeout=5)
    assert [entry["id"] for entry in report["items"]] == [str(first), str(second)]
    assert report["decision"]["store_revision"] == committed_revision == 2
    assert "Committed during embedding." in report["raw_context"]
    assert reader._journal.revision == 2


def test_receipt_identity_change_during_embedding_cannot_rank_or_publish_context(factory, monkeypatch):
    adapter = factory(profile=2)
    remember(adapter, "Committed under original identity.")
    before = adapter._journal.read_snapshot()
    entered, release = Event(), Event()

    def delayed(text):
        if text == "racing query":
            entered.set()
            assert release.wait(5), "query embedding was not released"

    def forbidden(*_args, **_kwargs):
        pytest.fail("ranking ran after the embedding identity changed")

    adapter.embedder.callback = delayed
    monkeypatch.setattr(adapter.store, "retrieve_filtered", forbidden)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(adapter.retrieve_context, "racing query", tenant_id="owner")
        try:
            assert entered.wait(5)
            adapter.embedder.receipt_embedding_identity = "retrieval-characterization-v2"
        finally:
            release.set()
        with pytest.raises(ReceiptEmbeddingCompatibilityError, match="identity changed"):
            pending.result(timeout=5)
    assert adapter._journal.read_snapshot() == before


@pytest.mark.parametrize("as_of", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_effective_time_rejected_before_ranking(factory, monkeypatch, as_of):
    adapter = factory()
    monkeypatch.setattr(adapter.store, "retrieve_filtered", lambda *_args, **_kwargs: pytest.fail("ranked"))
    with pytest.raises(ValueError, match="as_of must be finite"):
        adapter.retrieve_context("query", tenant_id="owner", as_of=as_of)
