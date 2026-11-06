"""M9 curriculum generator tests: determinism, canonical format, class shapes, leakage.

Canonical references (the shapes asserted here):
  contracts/agent_episode.py:816  parse_agent_action  (dml-agent-action-v1 surface)
  contracts/agent_episode.py:722  validate_answer     (claims/key/value/evidence_ids)
  contracts/agent_episode.py:763  initial_messages    (V3 system + user bootstrap)
  contracts/agent_episode.py:597  episode_tool_definitions
  services/agent_episode.py:270,329-333  tool-<step> ids + assistant/tool feedback shape
  services/episode_tools.py:240-247,343-345  public record view + retrieve model_result
  services/llama3_sft_runtime.py:36-45  V3 transport rendering admission

Dataset version 2 (adversarial-audit repairs). All ten v1 test intents are
preserved; four were adapted because the repair changed the behavior they
pinned (no test was deleted):

* test_manifest — now also pins manifest["dataset_version"] == 2.
* test_abstain_exact_shape — irrelevant_only may now present records with the
  ASKED claim_key about a DIFFERENT subject (that is the point of the
  repair); the adapted assertion is subject-aware: no presented current
  record may support the asked (subject, key).
* test_coverage_shape — evidence-completeness ("every current record for an
  answered key is cited") is now scoped to records about the ASKED subject,
  because same-key records about other subjects are intentionally uncited.
* test_injection_resist_shape — injected records are identified via
  meta.injection_ids (provenance) cross-checked against text markers, because
  the source field no longer segregates lures and marker coverage must be
  proven for the widened template pool.

New tests assert the repair properties directly (the fix-list proof):
  test_same_key_distractors_all_classes     (audit finding 1)
  test_source_carries_no_trust_signal       (audit finding 2)
  test_prompts_ask_plainly                  (audit findings 3 + grammar fixes)
  test_superseded_records_spread            (audit finding: supersession confined to abstain)
  test_phrasing_is_not_a_citability_signal  (audit finding 5)
  test_retrieval_counts_vary                (audit finding 6)
  test_value_vocabulary_is_wide             (audit finding 8)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

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

# Injection text markers: every injection template must carry at least one,
# and no clean template may carry any (cross-checked against meta provenance).
INJECTION_MARKERS = (
    "Ignore previous instructions",
    "System instruction",
    "you must answer",
    "NOTE TO ASSISTANT",
    "admin override",
)

# Directive phrasings removed by the audit repair: the situation (no support,
# conflicting sources) must be detected from the records, never announced.
DIRECTIVE_PHRASES = (
    "If no stored evidence",
    "empty claims list",
    "Independent sources disagree",
    "Do not merge",
    "declare the conflict",
    "distinct claim key",
)

PLAIN_SINGLE_PROMPT = re.compile(
    r"^What is the current .+ for .+\? Report claim key [\w.]+ with the exact stored value "
    r"and cite its stored evidence\.$"
)


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
# Anti-shortcut helpers (subject/key extraction from the plain prompts)
# ---------------------------------------------------------------------------


def _asked_subject(row):
    match = re.search(r"[Ff]or (.+?)[?,]", row["task"]["prompt"])
    assert match, row["task"]["prompt"]
    return match.group(1)


def _base_key(key):
    return re.sub(r"\.conflict$|\.per_.*$", "", key)


def _asked_keys(row, action):
    keys = {_base_key(claim["key"]) for claim in action["answer"]["claims"]}
    if not keys:  # abstain: the asked key is named in the prompt
        keys = {fact["key"] for fact in generate.FACTS if fact["key"] in row["task"]["prompt"]}
        assert len(keys) == 1, row["task"]["prompt"]
    return keys


def _near_misses(row, presented, action):
    """Uncited records with an ASKED claim_key whose text is NOT about the
    asked subject. Asserts en route that none carries an answer value."""
    subject = _asked_subject(row)
    keys = _asked_keys(row, action)
    cited = {rid for claim in action["answer"]["claims"] for rid in claim["evidence_ids"]}
    answers = {}
    for claim in action["answer"]["claims"]:
        for rid in claim["evidence_ids"]:
            answers.setdefault(presented[rid]["claim_key"], set()).add(presented[rid]["claim_value"])
    misses = []
    for rid, record in presented.items():
        if record["claim_key"] in keys and rid not in cited and subject not in record["text"]:
            assert record["claim_value"] not in answers.get(record["claim_key"], set()), (
                "near-miss record carries the answer value",
                row["episode_id"],
                record,
            )
            misses.append(record)
    return misses


# ---------------------------------------------------------------------------
# Tests (v1 intents)
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
    # v2: regenerated under the audit repairs; bytes differ from v1 by design.
    assert manifest["dataset_version"] == generate.DATASET_VERSION == 2
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
            # Adapted for v2 (same intent: no support for the asked claim).
            # The store may present records with the ASKED claim_key about a
            # DIFFERENT subject; nothing current may be about the asked
            # (subject, key) — and at least one such same-key near-miss must
            # be present so key-matching alone cannot produce an answer.
            subject = _asked_subject(row)
            (key,) = _asked_keys(row, action)
            assert not [
                r
                for r in presented.values()
                if r["claim_key"] == key and subject in r["text"] and r["memory_state"] == "current"
            ]
            assert any(r["claim_key"] == key and subject not in r["text"] for r in presented.values()), (
                "irrelevant_only must include an asked-key record about a different subject",
                row["episode_id"],
            )
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
        # Every current record about the ASKED subject whose claim_key was
        # answered is cited by that claim. (v2: scoped to the asked subject —
        # same-key records about other subjects are intentional near-misses.)
        subject = _asked_subject(row)
        for record in presented.values():
            if record["claim_key"] in keys and record["memory_state"] == "current" and subject in record["text"]:
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
        # Both conflicting records are about the asked subject and current.
        subject = _asked_subject(row)
        for record in (left, right):
            assert record["memory_state"] == "current"
            assert subject in record["text"]
    assert variants == {"separate_keys", "declare_conflict"}


def test_injection_resist_shape(dataset):
    _, rows = dataset
    variants = set()
    for row in rows["injection_resist"]:
        presented, action = assert_canonical_row(row)
        variants.add(row["meta"]["variant"])
        # v2: injected records are identified by meta provenance, and the
        # text-marker scan must agree exactly (proves the marker pool covers
        # every injection template and no clean template carries a marker).
        lure_ids = set(row["meta"]["injection_ids"])
        assert lure_ids and lure_ids <= set(presented), "episode must present at least one injection record"
        marker_ids = {rid for rid, r in presented.items() if any(m in r["text"] for m in INJECTION_MARKERS)}
        assert marker_ids == lure_ids, "marker detection and meta provenance disagree"
        injected = {rid: presented[rid] for rid in lure_ids}
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


# ---------------------------------------------------------------------------
# Tests (v2 anti-shortcut properties — the audit fix-list proof)
# ---------------------------------------------------------------------------


def test_same_key_distractors_all_classes(dataset):
    """Audit finding 1: claim_key-filtering alone must not identify evidence.

    Every class presents, in >= 40% of episodes, at least one record with an
    asked claim_key about a DIFFERENT subject whose value is not the answer;
    the cited evidence for an asked key is always about the asked subject.
    """
    _, rows = dataset
    for cls in generate.CLASSES:
        hits = 0
        for row in rows[cls]:
            presented, action = assert_canonical_row(row)
            misses = _near_misses(row, presented, action)
            hits += bool(misses)
            # The correct citation requires subject disambiguation: cited
            # evidence for an asked key is about the asked subject.
            subject = _asked_subject(row)
            keys = _asked_keys(row, action)
            for claim in action["answer"]["claims"]:
                for rid in claim["evidence_ids"]:
                    record = presented[rid]
                    if record["claim_key"] in keys:
                        assert subject in record["text"], "cited evidence is not about the asked subject"
        rate = hits / len(rows[cls])
        assert rate >= 0.40, f"{cls}: same-key near-miss rate {rate:.0%} below the 40% floor"


def test_source_carries_no_trust_signal(dataset):
    """Audit finding 2: source identity must not separate lures from clean
    records in either direction."""
    _, rows = dataset
    injection = rows["injection_resist"]
    trusted_lure_episodes = 0
    for row in injection:
        presented, _ = assert_canonical_row(row)
        lures = [presented[rid] for rid in row["meta"]["injection_ids"]]
        if any(record["source"] in generate._SOURCES for record in lures):
            trusted_lure_episodes += 1
    rate = trusted_lure_episodes / len(injection)
    assert rate >= 0.30, f"lures on trusted source ids in only {rate:.0%} of injection episodes"

    # Clean records (cited evidence and plain distractors alike) also sit on
    # the lure-prone source ids.
    cited_on_lure_source = {cls: 0 for cls in generate.CLASSES}
    uncited_clean_on_lure_source = 0
    for cls in generate.CLASSES:
        for row in rows[cls]:
            presented, action = assert_canonical_row(row)
            cited = {rid for claim in action["answer"]["claims"] for rid in claim["evidence_ids"]}
            lure_ids = set(row["meta"].get("injection_ids", ()))
            for rid in cited:
                if presented[rid]["source"] in generate._INJECTION_SOURCES:
                    cited_on_lure_source[cls] += 1
            uncited_clean_on_lure_source += sum(
                1
                for rid, record in presented.items()
                if rid not in cited and rid not in lure_ids and record["source"] in generate._INJECTION_SOURCES
            )
    classes_with_cited = {cls for cls, n in cited_on_lure_source.items() if n}
    assert classes_with_cited == {"coverage", "conflict_split", "injection_resist", "replay"}, (
        "every answer-bearing class must cite clean records from lure-prone sources",
        cited_on_lure_source,
    )
    assert uncited_clean_on_lure_source >= 20, "lure-prone sources must also carry plain clean records"


def test_prompts_ask_plainly(dataset):
    """Audit finding 3 (+ grammar fixes): no directive suffixes anywhere; the
    single-key classes share one plain question shape; queries carry no list
    punctuation; no 'current current' duplication."""
    _, rows = dataset
    for cls in generate.CLASSES:
        for row in rows[cls]:
            prompt = row["task"]["prompt"]
            for phrase in DIRECTIVE_PHRASES:
                assert phrase not in prompt, (cls, phrase, prompt)
            assert "current current" not in prompt
            for message in row["messages"]:
                if message["role"] == "assistant" and "tool_calls" in message:
                    query = json.loads(message["content"])["arguments"]["query"]
                    assert "," not in query, "retrieve query inherited listing punctuation"
    # abstain / conflict / injection prompts are indistinguishable from a
    # plain replay single-key question: the episode situation (no support,
    # conflict, lure) is announced nowhere.
    for cls in ("abstain_exact", "conflict_split", "injection_resist"):
        for row in rows[cls]:
            assert PLAIN_SINGLE_PROMPT.match(row["task"]["prompt"]), row["task"]["prompt"]
    # And every output contract still resolves correctly from the records:
    # abstain ends empty, conflict splits, injection answers (asserted in the
    # per-class shape tests above).


def test_superseded_records_spread(dataset):
    """Superseded records are no longer confined to abstain_exact: every
    answer-bearing class exercises memory_state discrimination, with stale
    entries chained to their presented current replacements."""
    _, rows = dataset
    chained = 0
    for cls in generate.CLASSES:
        episodes_with_superseded = 0
        for row in rows[cls]:
            presented, _ = assert_canonical_row(row)
            stale = [r for r in presented.values() if r["memory_state"] == "superseded"]
            episodes_with_superseded += bool(stale)
            for record in stale:
                target = record.get("superseded_by")
                if target in presented and presented[target]["memory_state"] == "current":
                    chained += 1
        if cls != "abstain_exact":
            rate = episodes_with_superseded / len(rows[cls])
            assert rate >= 0.25, f"{cls}: superseded records in only {rate:.0%} of episodes"
    assert chained >= 20, "superseded entries must chain to presented current replacements"
    # The discrimination skill proper: somewhere a stale record about the
    # ASKED subject with the ASKED key must be passed over for the current one.
    asked_subject_stale = 0
    for cls in ("replay", "coverage"):
        for row in rows[cls]:
            presented, action = assert_canonical_row(row)
            subject = _asked_subject(row)
            keys = _asked_keys(row, action)
            cited_values = {c["value"] for c in action["answer"]["claims"]}
            for record in presented.values():
                if (
                    record["memory_state"] == "superseded"
                    and record["claim_key"] in keys
                    and subject in record["text"]
                    and record["claim_value"] not in cited_values
                ):
                    asked_subject_stale += 1
    assert asked_subject_stale >= 10, "stale asked-subject records must compete with the current answer"


def test_phrasing_is_not_a_citability_signal(dataset):
    """Audit finding 5: every non-plain record template appears on cited AND
    uncited records — prose register cannot identify citable evidence."""
    _, rows = dataset
    cited_templates = set()
    uncited_templates = set()
    for cls in generate.CLASSES:
        for row in rows[cls]:
            presented, action = assert_canonical_row(row)
            cited = {rid for claim in action["answer"]["claims"] for rid in claim["evidence_ids"]}
            lure_ids = set(row["meta"].get("injection_ids", ()))
            for rid, record in presented.items():
                if rid in lure_ids:
                    continue
                if record["text"].startswith("Current "):
                    template = "current"
                elif record["text"].endswith("as of the latest sync."):
                    template = "sync"
                else:
                    template = "plain"
                (cited_templates if rid in cited else uncited_templates).add(template)
    assert cited_templates == {"current", "plain", "sync"}
    assert uncited_templates == {"current", "plain", "sync"}


def test_retrieval_counts_vary(dataset):
    """Audit finding 6: top_k / result counts vary within every class, and the
    injection capped window is not rigged to exactly legit + lure."""
    _, rows = dataset
    for cls in generate.CLASSES:
        top_ks = set()
        for row in rows[cls]:
            for message in row["messages"]:
                if message["role"] == "tool":
                    top_ks.add(json.loads(message["content"])["requested_top_k"])
        assert len(top_ks) >= 3, f"{cls}: requested_top_k does not vary ({sorted(top_ks)})"
    # limit_reached is no longer a class tell: both classes sit mid-range.
    for cls in ("injection_resist", "replay"):
        limits = []
        for row in rows[cls]:
            for message in row["messages"]:
                if message["role"] == "tool":
                    limits.append(json.loads(message["content"])["limit_reached"])
        rate = sum(limits) / len(limits)
        assert 0.15 <= rate <= 0.85, f"{cls}: limit_reached rate {rate:.0%} is a distributional tell"
    capped_sizes = set()
    for row in rows["injection_resist"]:
        if row["meta"]["variant"] == "capped_result":
            for message in row["messages"]:
                if message["role"] == "tool":
                    capped_sizes.add(json.loads(message["content"])["returned_count"])
    assert len(capped_sizes) >= 2, f"capped result windows are rigged ({sorted(capped_sizes)})"


def test_value_vocabulary_is_wide(dataset):
    """Audit finding 8: claim values span wide domains (no memorizable vocab)."""
    _, rows = dataset
    kind_by_key = {fact["key"]: fact["kind"] for fact in generate.FACTS}
    values = {}
    for cls in generate.CLASSES:
        for row in rows[cls]:
            for record in _presented(row).values():
                values.setdefault(kind_by_key[record["claim_key"]], set()).add(record["claim_value"])
    floors = {
        "time": 40,
        "date": 30,
        "version": 30,
        "host": 30,
        "endpoint": 20,
        "int": 25,
        "float": 20,
        "person": 40,
        "days": 10,
        "phase": 7,
        "tier": 6,
    }
    for kind, floor in floors.items():
        assert len(values[kind]) >= floor, f"{kind}: only {len(values[kind])} distinct values (< {floor})"
