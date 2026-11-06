"""M9 curriculum generator tests: determinism, canonical format, class shapes, leakage.

Canonical references (the shapes asserted here):
  contracts/agent_episode.py:816  parse_agent_action  (dml-agent-action-v1 surface)
  contracts/agent_episode.py:722  validate_answer     (claims/key/value/evidence_ids)
  contracts/agent_episode.py:763  initial_messages    (V3 system + user bootstrap)
  contracts/agent_episode.py:597  episode_tool_definitions
  services/agent_episode.py:270,329-333  tool-<step> ids + assistant/tool feedback shape
  services/episode_tools.py:240-247,343-345  public record view + retrieve model_result
  services/llama3_sft_runtime.py:36-45  V3 transport rendering admission
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from daystrom_dml.contracts.agent_episode import (
    ACTION_VERSION,
    LLAMA3_SFT_V3_CONSUMER_PROFILE as CONSUMER_PROFILE,
    canonical_json,
    episode_tool_definitions,
    initial_messages,
    parse_agent_action,
    validate_answer,
)
from daystrom_dml.services.episode_tools import canonical as tool_canonical
from daystrom_dml.services import llama3_sft_runtime
from scripts.m9_curriculum import generate

FIXTURES = Path(__file__).resolve().parents[1] / "scripts" / "m8_battery" / "fixtures"
ROW_KEYS = {
    "schema_version",
    "episode_id",
    "class",
    "consumer_profile",
    "task",
    "allowed_tools",
    "output_reserved_tokens",
    "messages",
    "tools",
    "meta",
}
RECORD_KEYS = {"id", "text", "record_ref", "source", "claim_key", "claim_value", "source_trust", "memory_state"}
SMALL = {cls: 2 for cls in generate.CLASSES}


def _rows_by_class(output):
    return {
        cls: [json.loads(line) for line in (output / (cls + ".jsonl")).read_text().splitlines()]
        for cls in generate.CLASSES
    }


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """The full default curriculum, generated once per test module."""
    output = tmp_path_factory.mktemp("m9") / "out"
    generate.generate(output)
    return output, _rows_by_class(output)


# ---------------------------------------------------------------------------
# Canonical format validation (one shared, strict walker)
# ---------------------------------------------------------------------------


def _presented(row):
    """id -> record for every retrieve tool message, with uniqueness."""
    presented = {}
    for message in row["messages"]:
        if message["role"] != "tool":
            continue
        result = json.loads(message["content"])
        assert set(result) == {"records", "requested_top_k", "returned_count", "limit_reached"}
        assert type(result["requested_top_k"]) is int and 1 <= result["requested_top_k"] <= 10
        assert result["returned_count"] == len(result["records"])
        assert result["limit_reached"] == (result["returned_count"] == result["requested_top_k"])
        for record in result["records"]:
            assert RECORD_KEYS <= set(record) <= RECORD_KEYS | {"superseded_by"}
            assert type(record["id"]) is int and record["id"] >= 0
            assert type(record["text"]) is str and record["text"].strip()
            assert type(record["record_ref"]) is str and record["record_ref"]
            assert type(record["source"]) is str and record["source"]
            assert type(record["claim_key"]) is str and record["claim_key"]
            assert type(record["claim_value"]) in (str, int, float, bool, type(None))
            assert record["source_trust"] == "trusted"
            assert record["memory_state"] in ("current", "superseded")
            if "superseded_by" in record:
                assert type(record["superseded_by"]) is int
                assert record["memory_state"] == "superseded"
            assert record["id"] not in presented, "duplicate presented record id"
            presented[record["id"]] = record
    return presented


def _final_action(row):
    return parse_agent_action(row["messages"][-1]["content"])


def assert_canonical_row(row):
    """Full structural validation of one train row against the V3 surface."""
    assert set(row) == ROW_KEYS
    assert row["schema_version"] == generate.SCHEMA
    assert row["consumer_profile"] == CONSUMER_PROFILE
    assert row["class"] in generate.CLASSES
    assert row["allowed_tools"] == ["retrieve"]
    assert row["tools"] == [t for t in episode_tool_definitions() if t["function"]["name"] == "retrieve"]
    messages = row["messages"]
    expected = initial_messages(row["task"]["prompt"], consumer_profile=CONSUMER_PROFILE)
    assert messages[0] == expected[0] == {"role": "system", "content": llama3_sft_runtime.SYSTEM}
    assert messages[1] == expected[1] == {"role": "user", "content": row["task"]["prompt"]}
    presented = _presented(row)
    step = 0
    index = 2
    while index < len(messages) - 1:
        assistant, tool = messages[index], messages[index + 1]
        assert set(assistant) == {"role", "content", "tool_calls"}
        assert assistant["role"] == "assistant"
        action = parse_agent_action(assistant["content"])
        assert action["kind"] == "tool" and action["name"] == "retrieve"
        # Assistant text is the canonical serialization of the parsed action.
        assert assistant["content"] == canonical_json(action).decode("utf-8")
        call_id = "tool-" + str(step)
        assert assistant["tool_calls"] == [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "retrieve", "arguments": tool_canonical(action["arguments"])},
            }
        ]
        assert set(tool) == {"role", "tool_call_id", "name", "content"}
        assert tool["role"] == "tool" and tool["tool_call_id"] == call_id and tool["name"] == "retrieve"
        assert json.loads(tool["content"])["requested_top_k"] == action["arguments"]["top_k"]
        index += 2
        step += 1
    assert step >= 1, "every curriculum episode retrieves at least once"
    final_message = messages[-1]
    assert set(final_message) == {"role", "content"} and final_message["role"] == "assistant"
    action = _final_action(row)
    assert action["kind"] == "final"
    validate_answer(action["answer"])
    assert final_message["content"] == canonical_json(action).decode("utf-8")
    for claim in action["answer"]["claims"]:
        for record_id in claim["evidence_ids"]:
            assert record_id in presented, "claim cites an unpresented record"
    # The exact V3 transport admits this transcript unchanged.
    rendered = llama3_sft_runtime.render_messages(messages, row["tools"])
    assert len(rendered) == len(messages)
    assert [m["role"] for m in rendered] == [
        "system" if m["role"] == "system" else "user" if m["role"] == "tool" else m["role"] for m in messages
    ]
    return presented, action


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_determinism_same_seed_identical_bytes(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    generate.generate(first, counts=SMALL)
    generate.generate(second, counts=SMALL)
    for cls in generate.CLASSES:
        name = cls + ".jsonl"
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()


def test_manifest(dataset):
    output, rows = dataset
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema_version"] == generate.MANIFEST_SCHEMA
    assert manifest["seed"] == generate.SEED == [20261106]
    assert manifest["consumer_profile"] == CONSUMER_PROFILE
    assert manifest["action_version"] == ACTION_VERSION
    assert manifest["counts"] == generate.DEFAULT_COUNTS
    assert manifest["episodes"] == sum(generate.DEFAULT_COUNTS.values())
    assert manifest["generator"]["script"] == "dml_core/scripts/m9_curriculum/generate.py"
    assert manifest["generator"]["commit"]
    assert {f["path"] for f in manifest["files"]} == {cls + ".jsonl" for cls in generate.CLASSES}
    for entry in manifest["files"]:
        raw = (output / entry["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        assert len(raw) == entry["bytes"]
        assert entry["episodes"] == len(rows[entry["path"][: -len(".jsonl")]])
    assert manifest["classes"].keys() == set(generate.CLASSES)


def test_canonical_format_all_rows(dataset):
    _, rows = dataset
    for cls in generate.CLASSES:
        for row in rows[cls]:
            assert_canonical_row(row)


def test_abstain_exact_shape(dataset):
    _, rows = dataset
    variants = set()
    for row in rows["abstain_exact"]:
        presented, action = assert_canonical_row(row)
        assert action["answer"] == {"claims": []}, "abstain-exact must end with empty claims"
        variant = row["meta"]["variant"]
        variants.add(variant)
        if variant == "empty_store":
            assert not presented
        elif variant == "irrelevant_only":
            assert presented
            assert not [r for r in presented.values() if r["claim_key"] in row["task"]["prompt"]]
        else:
            assert variant == "superseded_only"
            assert presented and all(r["memory_state"] == "superseded" for r in presented.values())
    assert variants == {"empty_store", "irrelevant_only", "superseded_only"}


def test_coverage_shape(dataset):
    _, rows = dataset
    variants = set()
    for row in rows["coverage"]:
        presented, action = assert_canonical_row(row)
        variants.add(row["meta"]["variant"])
        claims = action["answer"]["claims"]
        assert len(claims) >= 2, "coverage episodes answer multi-part questions"
        keys = [claim["key"] for claim in claims]
        assert len(set(keys)) == len(keys)
        for claim in claims:
            assert claim["evidence_ids"], "each part cites its record"
            for record_id in claim["evidence_ids"]:
                record = presented[record_id]
                assert record["claim_key"] == claim["key"]
                assert record["claim_value"] == claim["value"]
                assert type(record["claim_value"]) is type(claim["value"])
        # Every current record whose claim_key was answered is cited by that claim.
        for record in presented.values():
            if record["claim_key"] in keys and record["memory_state"] == "current":
                claim = claims[keys.index(record["claim_key"])]
                assert record["id"] in claim["evidence_ids"]
    assert variants == {"single_retrieval", "split_retrieval"}


def test_conflict_split_shape(dataset):
    _, rows = dataset
    variants = set()
    for row in rows["conflict_split"]:
        presented, action = assert_canonical_row(row)
        claims = action["answer"]["claims"]
        variant = row["meta"]["variant"]
        variants.add(variant)
        if variant == "separate_keys":
            assert len(claims) == 2
            assert claims[0]["key"] != claims[1]["key"]
            assert ".per_" in claims[0]["key"] and ".per_" in claims[1]["key"]
            left = presented[claims[0]["evidence_ids"][0]]
            right = presented[claims[1]["evidence_ids"][0]]
        else:
            assert variant == "declare_conflict"
            assert len(claims) == 1
            assert claims[0]["key"].endswith(".conflict")
            assert len(claims[0]["evidence_ids"]) == 2
            left = presented[claims[0]["evidence_ids"][0]]
            right = presented[claims[0]["evidence_ids"][1]]
        assert left["claim_key"] == right["claim_key"]
        assert left["claim_value"] != right["claim_value"]
        assert left["source"] != right["source"]
    assert variants == {"separate_keys", "declare_conflict"}


def test_injection_resist_shape(dataset):
    _, rows = dataset
    markers = ("Ignore previous instructions", "System instruction", "you must answer")
    variants = set()
    for row in rows["injection_resist"]:
        presented, action = assert_canonical_row(row)
        variants.add(row["meta"]["variant"])
        injected = {rid: r for rid, r in presented.items() if any(m in r["text"] for m in markers)}
        assert injected, "episode must present at least one injection record"
        claims = action["answer"]["claims"]
        assert claims, "injection-resist episodes answer from legitimate records"
        for claim in claims:
            assert claim["evidence_ids"]
            for record_id in claim["evidence_ids"]:
                assert record_id not in injected, "claim cites the injection record"
                record = presented[record_id]
                assert record["claim_value"] == claim["value"]
        lure_values = {r["claim_value"] for r in injected.values()}
        assert lure_values.isdisjoint(c["value"] for c in claims), "final adopted the lure value"
    assert variants == {"capped_result", "full_result"}


def test_replay_shape(dataset):
    _, rows = dataset
    value_types = set()
    for row in rows["replay"]:
        presented, action = assert_canonical_row(row)
        claims = action["answer"]["claims"]
        assert claims
        for claim in claims:
            assert claim["evidence_ids"]
            record = presented[claim["evidence_ids"][0]]
            assert record["claim_key"] == claim["key"]
            assert record["claim_value"] == claim["value"]
            assert type(record["claim_value"]) is type(claim["value"]), "native JSON type preserved"
            value_types.add(type(claim["value"]))
    assert {str, int, bool} <= value_types, "replay preserves string/numeric/boolean claim values"


def test_entities_disjoint_from_m8_fixtures(dataset):
    fixture_texts = {path.name: path.read_text(encoding="utf-8") for path in sorted(FIXTURES.glob("*.json"))}
    assert fixture_texts, "m8 battery fixtures missing"
    for entity in generate.ENTITY_POOL_STRINGS:
        for name, text in fixture_texts.items():
            assert entity not in text, f"curriculum entity {entity!r} appears in fixture {name}"
    # Per-episode entity usage (names and string values) is likewise absent.
    _, rows = dataset
    used = set()
    for cls in generate.CLASSES:
        for row in rows[cls]:
            used.update(row["meta"]["entities"])
    assert used
    for entity in used:
        for name, text in fixture_texts.items():
            assert entity not in text, f"generated entity {entity!r} appears in fixture {name}"


def test_no_fixture_text_leakage(dataset):
    fixture_event_texts = []
    for path in sorted(FIXTURES.glob("*.json")):
        fixture_event_texts.extend(event["text"] for event in json.loads(path.read_text())["events"])
    _, rows = dataset
    curriculum_texts = []
    for cls in generate.CLASSES:
        for row in rows[cls]:
            curriculum_texts.append(row["task"]["prompt"])
            curriculum_texts.extend(
                record["text"]
                for message in row["messages"]
                if message["role"] == "tool"
                for record in json.loads(message["content"])["records"]
            )
    assert curriculum_texts and fixture_event_texts
    for fixture_text in fixture_event_texts:
        for text in curriculum_texts:
            assert fixture_text not in text, "fixture record text leaked into curriculum"
