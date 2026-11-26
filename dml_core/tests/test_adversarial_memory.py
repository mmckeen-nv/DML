"""Explicit semantic-policy regressions, separate from model task success."""
import copy
import json
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from daystrom_dml.dml_adapter import DMLAdapter
from daystrom_dml.journal import JournalStateStore, RevisionConflict
from daystrom_dml.services.context import compact_context, ContextBudgetError
from daystrom_dml.services.retrieval import QueryEmbeddingCache
from daystrom_dml.services.telemetry import digest
from daystrom_dml.context.admission import admit_context_segments
from daystrom_dml.context.schema import ContextSegment, ContextAuthority
from daystrom_dml.api_contracts import DaystromScope, ContractError

CORPUS = json.loads((Path(__file__).parent / "fixtures/memory_adversarial_v1.json").read_text())


class FixedEmbedder:
    def embed(self, text):
        return np.ones(4, dtype=np.float32)


def adapter(path):
    return DMLAdapter(config_overrides={"storage_dir": str(path), "model_name": "dummy", "embedding_model": None,
        "persistence": {"journal": True, "enable": False}, "rag_store": {"enable": False},
        "token_budget": 1200, "survival_ledger_enabled": False, "similarity_threshold": 0.0,
        "dpm": {"enable": False, "include_in_context": False}}, embedder=FixedEmbedder(), start_aging_loop=False)


@pytest.mark.parametrize("case", CORPUS["cases"], ids=lambda case: case["id"])
def test_adversarial_corpus_before_and_after_restart(tmp_path, case):
    instance = adapter(tmp_path)
    for record in case["records"]:
        item = instance.ingest_memory(record["text"], tenant_id="owner", session_id="s", meta=record["meta"])
        if "salience" in record:
            with instance.atomic_batch("salience-fixture"):
                item.salience = record["salience"]
    assert instance.memory_count() == case["memory_count"]
    # Every scenario carries an otherwise matching cross-scope canary.
    instance.ingest_memory("PRIVATE OTHER TENANT", tenant_id="other", session_id="s", meta={"no_merge": True})
    if case.get("concurrent_update"):
        peer = JournalStateStore(instance._journal.path)
        old = peer.load()
        revision = peer._revision
        instance.ingest_memory("Committed related change", tenant_id="owner", session_id="s", meta={"no_merge": True})
        with pytest.raises(RevisionConflict):
            peer.save(old, expected_revision=revision)
    for restart in (False, True):
        if restart:
            instance.close()
            instance = adapter(tmp_path)
        before = copy.deepcopy(instance.store.export_state())
        report = instance.retrieve_context("evidence", tenant_id="owner", session_id="s", top_k=case.get("top_k", 16), as_of=CORPUS["effective_time"])
        text = report["raw_context"]
        assert all(value in text for value in case["visible"])
        assert all(value not in text for value in case["hidden"] + ["PRIVATE OTHER TENANT"])
        assert instance.store.export_state() == before, "retrieval must not raise source authority or salience"
        assert report["decision"]["context_digest"] == digest(text)
        assert report["decision"]["store_revision"] == instance._journal.stamp()[0]
        if case["hidden"]:
            assert report["decision"]["suppressed"]
        if case.get("source_ids"):
            assert report["items"][0]["meta"]["source_ids"] == case["source_ids"]
    instance.close()


def test_context_budget_counts_heading_source_prefix_and_multibyte_text(tmp_path):
    instance = adapter(tmp_path)
    item = instance.ingest_memory("漢字 🌍 " * 100, tenant_id="owner", meta={"source":"long-source-name"})
    def count(text):
        return len(text.encode("utf-8"))
    entries, context, tokens = compact_context([item], budget=120, item_limit=2, summary_chars=200,
        ledger_chars=200, count_tokens=count, prefix="required prefix")
    assert entries
    assert tokens == count(context) <= 120
    with pytest.raises(ContextBudgetError):
        compact_context([item], budget=2, item_limit=1, summary_chars=200, ledger_chars=200, count_tokens=count, prefix="too long")
    instance.close()


def test_cache_clear_during_inflight_request_does_not_publish_old_generation():
    cache = QueryEmbeddingCache()
    entered, release = threading.Event(), threading.Event()
    def old(text):
        entered.set()
        assert release.wait(5)
        return np.ones(2)
    with ThreadPoolExecutor(max_workers=1) as executor:
        request = executor.submit(cache.get, "query", old)
        assert entered.wait(5)
        cache.clear()
        np.testing.assert_equal(cache.get("query", lambda _: np.zeros(2)), np.zeros(2))
        release.set()
        np.testing.assert_equal(request.result(), np.ones(2))
    np.testing.assert_equal(cache.get("query", old), np.zeros(2))
    with pytest.raises(ValueError):
        cache.values["query"][0] = 999


def test_exact_rendered_context_boundary_fails_before_returning_an_overflow():
    scope = DaystromScope(session_id="s")
    segment = ContextSegment(segment_id="pinned", kind="instruction", content="required", scope=scope,
        authority=ContextAuthority.IMMUTABLE, estimated_tokens=1)
    kwargs = dict(scope=scope, segments=[segment], model_id="m", runtime_id="r", model_limit_tokens=20,
                  output_reserved_tokens=5, runtime_reserved_tokens=2, tokenizer_identity="pinned-tokenizer/template-sha")
    with pytest.raises(ContractError, match="rendered context exceeds"):
        admit_context_segments(**kwargs, rendered_token_counter=lambda _: 14)
    packet = admit_context_segments(**kwargs, rendered_token_counter=lambda messages: 13)
    assert packet.manifest.exact_input_tokens == 13
    assert packet.budget.admitted_input_tokens == 13
    assert packet.decisions["tokenizer_identity"] == kwargs["tokenizer_identity"]
    with pytest.raises(ContractError, match="non-negative integer"):
        admit_context_segments(**kwargs, rendered_token_counter=lambda _: True)
