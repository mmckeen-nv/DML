"""Context bytes, evidence, budgets, and ownership at the pure service boundary."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from daystrom_dml.memory_store import MemoryItem
from daystrom_dml.services.context import (
    ContextBudgetError,
    build_context_report,
    compact_context,
)


GOLDEN_CONTEXT = (
    "  pinned Ω  \n\n=== Retrieved Context ===\n"
    "- (1970-01-02) [source=tool]\n  Café 漢字 🌍\nnext\n"
    "- (1970-01-01) [source=unknown]\n  cached ✓"
)
GOLDEN_DECISION = {
    "schema_version": 1,
    "policy_version": "scoped-exact-v1",
    "query_digest": "d45b5fbf35c44338709289a4180ef5484bbeabc5046d81ec29a3bdb4ff5a173e",
    "scope": {"tenant_id": "owner", "client_id": None, "session_id": "s", "instance_id": None},
    "store_revision": 19,
    "effective_time": 1000.0,
    "top_k": 3,
    "kinds": ["note", "action"],
    "query_embedding_digest": "f69eaaeba8c4c00660da4f9de82140709ede72107fc5e9436ed0b9e75e52c98a",
    "returned_ids": ["7", "3"],
    "source_digests": [
        "d926b9c714c1de2134e6ee2757096eeb407e97d71ad528764cfe7441b9cc1aa0",
        "f83073c4b020cc9a8a3694eae28f2f326cba98a857fe5b98a5d5cb85bf2e7be1",
    ],
    "suppressed": [{"id": "12", "reason": "expired"}, {"id": "9", "reason": "untrusted_source"}],
    "context_digest": "1fba48594a5a1be92b0d771731bea830d33a5b256d2ccb974b70952f802d04fa",
    "replay_inputs_complete": False,
    "deterministic_scoped_ranking": True,
    "decision_digest": "fbaf8ae12f4becf902b5d362359b2a96bd67cfdb52b292e6e28f0e02d35a1a2c",
}


def byte_count(text):
    return len(text.encode("utf-8"))


def memory(ident=1, text="memory", *, meta=None, timestamp=0.0, level=0,
           fidelity=1.0, salience=0.25):
    return MemoryItem(id=ident, text=text, embedding=np.array([1.0, 0.0]),
                      timestamp=timestamp, level=level, fidelity=fidelity,
                      salience=salience, meta=meta)


def compact(items, **overrides):
    options = {"budget": 4096, "item_limit": 4, "summary_chars": 200,
               "ledger_chars": 300, "count_tokens": byte_count}
    options.update(overrides)
    return compact_context(items, **options)


@pytest.fixture
def source_items():
    return [
        memory(7, "  Café 漢字 🌍\nnext  ", timestamp=86400.0, level=2,
               fidelity=0.625, salience=0.75,
               meta={"source": "tool", "source_ids": [91, 92],
                     "nested": {"typed": [True, 1, 1.0, None]}}),
        memory(3, "uncached body", meta={"summary": "  cached ✓  ", "kind": "note"}),
    ]


@pytest.fixture
def report_inputs(source_items):
    entries, context, tokens = compact(source_items, prefix="  pinned Ω  \n")
    return {
        "entries": entries,
        "context": context,
        "tokens_used": tokens,
        "prompt": "Find Ω",
        "scope": {"tenant_id": "owner", "client_id": None, "session_id": "s", "instance_id": None},
        "revision": 19,
        "as_of": 1000.0,
        "top_k": 3,
        "kinds": ["note", "action"],
        "embedding": np.array([0.25, -0.5, 1.0], dtype=np.float32),
        "suppressed": [{"id": "12", "reason": "expired"}, {"id": "9", "reason": "untrusted_source"}],
        "replayable": True,
        "phase": "execute",
        "include_quarantined": False,
        "survival_ledger_included": True,
        "personality_overlay": {"overlay": {"rendered_text": "pinned Ω"}, "source_ids": [8]},
        "latency_ms": 17,
    }


def test_context_has_exact_bytes_and_counts_every_rendered_character(source_items):
    entries, context, tokens = compact(source_items, prefix="  pinned Ω  \n")

    assert context.encode("utf-8") == GOLDEN_CONTEXT.encode("utf-8")
    assert tokens == 139 == byte_count(context)
    assert entries == [
        {"id": "7", "text": "Café 漢字 🌍\nnext", "summary": "Café 漢字 🌍\nnext",
         "meta": {"source": "tool", "source_ids": [91, 92],
                  "nested": {"typed": [True, 1, 1.0, None]}},
         "timestamp": 86400.0, "level": 2, "fidelity": 0.625, "salience": 0.75, "tokens": 22},
        {"id": "3", "text": "cached ✓", "summary": "cached ✓",
         "meta": {"summary": "  cached ✓  ", "kind": "note"},
         "timestamp": 0.0, "level": 0, "fidelity": 1.0, "salience": 0.25, "tokens": 10},
    ]
    assert tokens > sum(entry["tokens"] for entry in entries)
    assert all("embedding" not in entry for entry in entries)


def test_report_preserves_public_fields_order_and_exact_evidence(report_inputs):
    report = build_context_report(**report_inputs)
    expected = {
        "decision": GOLDEN_DECISION,
        "token_count_kind": "estimate",
        "raw_context": GOLDEN_CONTEXT,
        "context_tokens": 139,
        "top_k": 3,
        "kinds": ["note", "action"],
        "phase": "execute",
        "include_quarantined": False,
        "items": report_inputs["entries"],
        "survival_ledger_included": True,
        "personality_overlay": {"overlay": {"rendered_text": "pinned Ω"}, "source_ids": [8]},
        "latency_ms": 17,
    }
    assert report == expected
    assert list(report) == list(expected)
    assert list(report["decision"]) == list(GOLDEN_DECISION)


def test_compaction_detaches_nested_metadata_without_mutating_source(source_items):
    before = copy.deepcopy([item.to_dict() for item in source_items])
    entries, _, _ = compact(source_items)
    assert [item.to_dict() for item in source_items] == before

    entries[0]["meta"]["nested"]["typed"].append("report edit")
    entries[0]["meta"]["source_ids"].clear()
    assert [item.to_dict() for item in source_items] == before
    source_items[1].meta["summary"] = "source edit"
    assert entries[1]["meta"]["summary"] == "  cached ✓  "


def test_report_does_not_retain_caller_owned_metadata_or_provenance(report_inputs):
    report = build_context_report(**report_inputs)
    before = copy.deepcopy(report)
    report_inputs["entries"][0]["meta"]["nested"]["typed"].append("caller edit")
    report_inputs["entries"].clear()
    report_inputs["scope"]["tenant_id"] = "other"
    report_inputs["kinds"].append("other")
    report_inputs["suppressed"][0]["reason"] = "caller edit"
    report_inputs["suppressed"].reverse()
    report_inputs["personality_overlay"]["overlay"]["rendered_text"] = "caller edit"
    report_inputs["personality_overlay"]["source_ids"].append(99)
    report_inputs["embedding"][:] = 99

    assert report == before
    assert report["decision"] == GOLDEN_DECISION


def test_report_mutation_cannot_modify_inputs_or_store_metadata(report_inputs, source_items):
    inputs_before = copy.deepcopy({key: value for key, value in report_inputs.items() if key != "embedding"})
    source_before = copy.deepcopy([item.to_dict() for item in source_items])
    report = build_context_report(**report_inputs)
    report["items"][0]["meta"]["nested"]["typed"].clear()
    report["decision"]["scope"]["tenant_id"] = "other"
    report["kinds"].clear()
    report["decision"]["suppressed"][0]["reason"] = "report edit"
    report["personality_overlay"]["overlay"]["rendered_text"] = "report edit"
    report["personality_overlay"]["source_ids"].clear()

    assert {key: value for key, value in report_inputs.items() if key != "embedding"} == inputs_before
    assert [item.to_dict() for item in source_items] == source_before


def test_report_uses_supplied_time_counts_and_flags_without_runtime_access(report_inputs, monkeypatch):
    import daystrom_dml.services.context as context_module

    def unexpected_clock():
        raise AssertionError("Context report construction must not read a clock")

    monkeypatch.setattr(context_module.time, "time", unexpected_clock)
    monkeypatch.setattr(context_module.time, "perf_counter", unexpected_clock)
    report_inputs.update(tokens_used=73, as_of=42.5, latency_ms=6,
                         include_quarantined=True, replayable=False,
                         survival_ledger_included=False)
    report_inputs["embedding"].setflags(write=False)
    report = build_context_report(**report_inputs)

    assert report["context_tokens"] == 73
    assert report["decision"]["effective_time"] == 42.5
    assert report["latency_ms"] == 6
    assert report["include_quarantined"] is True
    assert report["survival_ledger_included"] is False
    assert report["decision"]["deterministic_scoped_ranking"] is False
    assert report["decision"]["suppressed"] == GOLDEN_DECISION["suppressed"]


@pytest.mark.parametrize("kinds", [None, []])
def test_report_preserves_unspecified_and_explicitly_empty_kinds(report_inputs, kinds):
    report_inputs["kinds"] = kinds
    report = build_context_report(**report_inputs)
    assert report["kinds"] == report["decision"]["kinds"] == kinds
    if kinds is not None:
        kinds.append("later caller edit")
        assert report["kinds"] == report["decision"]["kinds"] == []


@pytest.mark.parametrize("prefix", ["", "  required Ω\n"])
def test_empty_and_prefix_only_context_remain_exact(prefix, report_inputs):
    entries, context, tokens = compact([], budget=byte_count(prefix), prefix=prefix)
    assert entries == []
    assert context == prefix
    assert tokens == byte_count(prefix)
    report_inputs.update(entries=entries, context=context, tokens_used=tokens,
                         kinds=None, phase=None, revision=None, personality_overlay=None)
    report = build_context_report(**report_inputs)

    assert report["raw_context"] == prefix
    assert report["context_tokens"] == tokens
    assert report["items"] == report["decision"]["returned_ids"] == report["decision"]["source_digests"] == []
    assert report["kinds"] is report["phase"] is report["personality_overlay"] is None
    assert report["decision"]["kinds"] is report["decision"]["store_revision"] is None
    # This flag describes the caller's ledger selection, even if no entry fits.
    assert report["survival_ledger_included"] is True


def test_first_summary_halves_until_the_actual_framed_bytes_fit():
    frame = "=== Retrieved Context ===\n- (1970-01-01) [source=unknown]\n  "
    entries, context, tokens = compact([memory(text="abcdefghijklmno")],
                                       budget=byte_count(frame + "abc"))
    assert entries[0]["summary"] == "abc"
    assert entries[0]["tokens"] == 3
    assert context == frame + "abc"
    assert tokens == byte_count(context)


def test_first_unicode_summary_counts_bytes_without_splitting_characters():
    frame = "=== Retrieved Context ===\n- (1970-01-01) [source=unknown]\n  "
    entries, context, tokens = compact([memory(text="漢🌍" * 8)],
                                       budget=byte_count(frame + "漢🌍"))
    assert entries[0]["summary"] == "漢🌍"
    assert entries[0]["tokens"] == 7
    assert context.encode("utf-8") == (frame + "漢🌍").encode("utf-8")
    assert tokens == byte_count(context)


def test_later_overflow_stops_selection_without_truncating_or_skipping_it():
    first_context = "=== Retrieved Context ===\n- (1970-01-01) [source=unknown]\n  A"
    short_following_line = "\n- (1970-01-01) [source=unknown]\n  C"
    entries, context, tokens = compact([
        memory(1, "A"), memory(2, "B" * 200), memory(3, "C"),
    ], budget=byte_count(first_context + short_following_line))

    assert [entry["id"] for entry in entries] == ["1"]
    assert context == first_context
    assert tokens == byte_count(first_context)


def test_item_limit_applies_before_unrenderable_items_are_skipped():
    items = [memory(1, "A", meta={"source": "S" * 100}), memory(2, "B")]
    assert compact(items, budget=100, item_limit=1) == ([], "", 0)
    entries, context, tokens = compact(items, budget=100, item_limit=2)
    assert [entry["id"] for entry in entries] == ["2"]
    assert context.endswith("  B")
    assert tokens == byte_count(context) <= 100


@pytest.mark.parametrize("ledger_chars, expected_ledger", [(5, "LLLLLLL..."), (24, "L" * 21 + "...")])
def test_ledger_summary_uses_larger_limit_without_changing_normal_summaries(ledger_chars, expected_ledger):
    entries, _, _ = compact([
        memory(1, "L" * 40, meta={"kind": "survival_ledger"}),
        memory(2, "N" * 40), memory(3, "excluded"),
    ], summary_chars=10, ledger_chars=ledger_chars, item_limit=2)
    assert [entry["id"] for entry in entries] == ["1", "2"]
    assert [entry["summary"] for entry in entries] == [expected_ledger, "NNNNNNN..."]


def test_receipted_content_update_uses_corrected_text_and_retains_old_provenance():
    item = memory(text="corrected fact", meta={
        "summary": "obsolete cached claim",
        "content_update_decision": {"schema_version": "dml-content-update-decision-v1", "prior_digest": "before"},
    })
    entries, context, _ = compact([item])
    assert entries[0]["summary"] == "corrected fact"
    assert "corrected fact" in context
    assert "obsolete cached claim" not in context
    assert entries[0]["meta"] == item.meta


def test_required_prefix_exceeding_budget_preserves_public_error():
    with pytest.raises(ContextBudgetError, match="^Context prefix exceeds token budget$"):
        compact([], prefix="Ω", budget=1)
    assert compact([], prefix="Ω", budget=2) == ([], "Ω", 2)


@pytest.mark.parametrize("invalid_count", [True, -1, 1.0, "1", None])
def test_invalid_token_counter_preserves_public_error(invalid_count):
    with pytest.raises(ContextBudgetError, match="^Token counter must return a nonnegative integer$"):
        compact([], prefix="prefix", count_tokens=lambda _: invalid_count)


def test_invalid_summary_token_count_is_not_hidden_by_valid_total_count():
    def counter(text):
        return True if text == "memory" else byte_count(text)

    with pytest.raises(ContextBudgetError, match="^Token counter must return a nonnegative integer$"):
        compact([memory()], count_tokens=counter)


def test_empty_context_does_not_call_counter():
    def counter(text):
        raise AssertionError("An empty context has zero tokens without counter work")

    assert compact([], budget=0, count_tokens=counter) == ([], "", 0)


@pytest.mark.parametrize("invalid_input", ["metadata", "embedding"])
def test_nonfinite_evidence_preserves_json_failure(report_inputs, invalid_input):
    if invalid_input == "metadata":
        report_inputs["entries"][0]["meta"]["confidence"] = float("nan")
    else:
        report_inputs["embedding"][0] = float("nan")
    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        build_context_report(**report_inputs)
