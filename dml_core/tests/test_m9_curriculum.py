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

Dataset version 3 (Track-B candidate-1 gate repairs: claim-key echo +
loop resistance). All v1/v2 intents are preserved; two were adapted
because the v3 repair changed what they pin (no test was deleted):

* test_manifest — now pins manifest["dataset_version"] == 3.
* test_conflict_split_shape — conflict records are now presented under the
  keys the labeled claims use (per-source keys in separate_keys, per-source
  attributed truth keys in declare_conflict after the 2026-11-12
  reconciliation), and the assertions pin the echo: every claim key is
  exactly a cited record's claim_key.

New v3 tests assert the repair properties directly:
  test_claim_key_echo                        (claim key = presented claim_key of a supporting record)
  test_no_identical_consecutive_retrieves    (loop-attractor hard pattern)
  test_reformulation_after_empty_or_unhelpful (>= 30% of answer-bearing episodes; one rephrase, then finalize)
  test_same_value_different_key_distractors  (key-echo alone is insufficient)

Dataset version 4 (M10 generalization repairs, seed re-based to [20261112],
then re-based again to [20261116] on the 2026-11-12 leakage re-audit —
[20261112] sits in the M10 qualification gate's forbidden seed set).
All v1/v2/v3 intents are preserved; two pins moved with the version:

* test_manifest — now pins manifest["dataset_version"] == 4 and
  manifest["seed"] == [20261116] (and the seed's absence from the gate's
  forbidden set).
* test_value_vocabulary_is_wide — the kind map spans V4_FACTS too, with new
  floors for the v4 value kinds.

New v4 tests assert the generalization-class properties directly:
  test_idiom_echo_shape            (terse truth-key echo across phrasing families; narrative key is the wrong answer)
  test_multipart_decomp_shape      (every part of a 2-/3-part question answered with its presented key)
  test_v4_pools_disjoint_by_construction  (all-new pools: token/key-level disjoint from v1-v3 pools, fixtures, sealed workloads)
  test_v4_zero_word_8gram_overlap  (M8 leakage method: zero word 8-grams vs sealed corpora, fixtures, v3 output)

Dataset version 4, reconciled 2026-11-12 against the observer-verified
failure decomposition (docs/m10-failure-decomposition-VERIFICATION-
2026-11-12.md; the verifier never moves). The v4 generalization intents are
preserved; three changed with the reconciliation:

* conflict_declare was REFUTED as the winning mechanism (the verifier
  rejects 'conflict: ...' narrative claims — 6 such episodes failed — and
  credits DUAL-ASSERTION under source-attributed truth keys). The class is
  rewritten as conflict_dual_assert: the gold final asserts both values,
  each under its own <fact>_<source>_condition key citing its own record;
  the 'conflict:' narrative survives only as a near-miss distractor record.
* conflict_split's declare_conflict variant carried the same refuted
  narrative as its v3 gold — contradictory gold standards for the same
  behavior. Its gold is aligned to the identical dual-assert truth-key
  idiom (shared _conflict_truth_key / _dual_assert_final helpers); the
  'conflict:' narrative is demoted to a never-cited near-miss (ids in
  meta.conflict_narrative_ids).
* value_shape is new: the verifier's leaf gate requires string values equal
  to the record's exact '... is <value>.' tail; gold records end in that
  form with unit-bearing strings and a type-stripped bare-int near-miss
  (same subject, asked key) is always present and never cited.
* injection_resist gains a novel-surface template pool (the campaign's 4
  injection successes came on phrasings disjoint from the v3 templates);
  meta.injection_surfaces records the pools used per episode.

New/updated tests pin the reconciled shapes directly:
  test_conflict_dual_assert_shape  (dual-assert gold; 'conflict:' narrative is the near-miss)
  test_conflict_split_shape        (declare_conflict gold aligned to the same dual-assert idiom)
  test_no_narrative_conflict_gold_anywhere (no gold final in ANY class uses the refuted form)
  test_value_shape_shape           (verbatim unit-bearing string echo; type-stripped int never cited)
  test_injection_resist_shape      (extended: both injection surfaces exercised)

Stem purge (2026-11-12 leakage re-audit, gate checklist (b)): nine
content-bearing curriculum pool stems collided with measurement-corpora fact
keys at STEM level — berth/gantry/louver/forum/lighting (heldout2 fact-key
families), lumen/juniper (pilot216/heldout truth keys), helios/kestrel
(dev-fixture keys). The word 8-gram sweeps cannot catch a single shared
stem, so the pools were renamed and a STEM-level test added:
  test_pool_stems_disjoint_from_measurement_corpora
      (first-token stems of every entity-pool string + every token of the
      source ids and v4 terse fact keys/labels vs every fact/truth key token
      of pilot216, heldout, heldout2 (when its sealed copy is on disk) and
      the 12 dev fixtures; the nine purged stems are pinned absent as a
      regression list)
The rename touched the v1-v3 system/source pools, so the v1-v3 class shas
changed too — ACCEPTED: the M9-qualified adapter was already trained on the
old v3 data; this regeneration serves the v4 training run only.

Dataset version 5 (M10 v4.1-campaign failure repairs; grounded in
docs/m10-v41-failure-decomposition-2026-11-12.md and
docs/m10-dependent-lookups-investigation-2026-11-12.md). ADDITIVE: every
v3/v4 builder and pool is untouched, so the nine v3/v4 class files
regenerate byte-identical under the unchanged seed [20261116] — pinned
sha256 by sha256 in test_v4_class_bytes_unchanged. All v1-v4 test intents
are preserved; two pins moved with the version:

* test_manifest — now pins manifest["dataset_version"] == 5.
* test_value_vocabulary_is_wide — the kind map spans V5_FACTS too, with new
  floors for the v5 unit kinds.

One helper was generalized (no behavior change for v1-v4 rows):
_asked_keys reads meta.asked_keys when an abstaining episode's claims are
empty and its keys are parameterized (injection_resist_qn's abstain
variant, whose per-case keys cannot be looked up in a static pool).

New v5 tests assert the repair properties directly:
  test_v4_class_bytes_unchanged            (v5 is additive; v3/v4 bytes pinned)
  test_v5_classes_additive                 (CLASSES/counts grow only by the v5 classes)
  test_dependent_lookup_chain_shape        (3-claim gold: parameterized hop-2 echo + hop-1 + terse rule)
  test_dependent_lookup_chain_second_hop_retrieval (second retrieve keyed on the discovered token)
  test_dependent_lookup_chain_near_misses_never_cited (stale register/table, handbill, stripped int, narrative key)
  test_dependent_lookup_chain_rule_status_value_tail (rule record tail == terse dimension value)
  test_dependent_lookup_chain_variants_exercised
  test_dependent_lookup_chain_unit_bearing_hop2 (hop-2 value is a unit-bearing string, never an int)
  test_value_shape_chain_shape             (corrected unit-bearing echo through a correction chain)
  test_value_shape_chain_correction_chain  (superseded chain; gold cites the tail revision)
  test_value_shape_chain_stripped_near_miss (bare int of the CORRECTED value, always present, never cited)
  test_injection_resist_qn_shape           (legit keys only / abstain without support)
  test_injection_resist_qn_no_fabricated_key_gold (the *_signoff_circular shape is never gold)
  test_injection_resist_qn_marker_provenance (QN surface markers == meta.injection_ids)
  test_injection_resist_qn_variant_split   (both golds exercised)
  test_injection_resist_qn_novel_surfaces_disjoint (template pools disjoint from v3/v4 injection surfaces)
  test_answer_supported_shape              (support present; gold answers)
  test_answer_supported_never_empty_gold   (over-abstention near-miss is never the gold)
  test_answer_supported_traps_present      (abstain-trigger surfaces present per variant)
  test_answer_supported_variants_exercised
  test_v5_episode_id_prefixes              (m10- prefix, per-class numbering)
  test_v5_pools_disjoint_by_construction   (v5 pools vs v1-v4 pools, fixtures, ALL sealed corpora incl. heldout2)
  test_v5_zero_word_8gram_overlap          (M8 method incl. the heldout2 leg and v4 out/ leg)
  test_v5_key_stems_disjoint_from_measurement_corpora (incl. parameterized/fabricated key stems)

Dataset version 6 (M10 v5-campaign non-transfer repair; grounded in
docs/m10-dependent-lookups-nontransfer-2026-11-13.md: the chain transferred,
the claim SHAPE did not). ADDITIVE: every v3/v4/v5 builder and pool is
untouched, so the thirteen v3-v5 class files regenerate byte-identical under
the unchanged seed [20261116] — pinned sha256 by sha256 in
test_v4_class_bytes_unchanged and test_v5_class_bytes_unchanged. All v1-v5
test intents are preserved; three pins/helpers moved with the version:

* test_manifest — now pins manifest["dataset_version"] == 6.
* test_v5_classes_additive — v5 classes now sit at CLASSES[9:13]; the v6
  tail is pinned by test_v6_classes_additive.
* _presented — the shared record walker now admits the v6 compose record
  shape: for V6_CLASSES rows a presented record carries claim_key and NEVER
  claim_value (the campaign presentation, runner_transport_native
  ._ingest_record); for all other classes claim_value stays mandatory.
  _near_misses reads a record's advertised value via _record_value (the
  claim_value field when present, else the composable status_value tail of
  the raw text), and _asked_subject falls back to meta.asked_subject for
  prompts without a "For <subject>," clause (mirrors the v5 _asked_keys
  meta fallback precedent).

New v6 tests assert the repair properties directly:
  test_v5_class_bytes_unchanged            (v6 is additive; v5 bytes pinned)
  test_v6_classes_additive                 (CLASSES/counts grow only by the v6 class)
  test_v6_episode_id_prefixes              (m10- prefix, per-class numbering)
  test_dependent_lookup_compose_shape      (3-claim gold: parameterized hop-2 + hop-1 + terse rule)
  test_dependent_lookup_compose_no_claim_value_crutch (no presented record carries claim_value)
  test_dependent_lookup_compose_value_composition (every gold value is the cited record's text tail)
  test_dependent_lookup_compose_rule_status_value_tail (rule tail -> terse dimension, never the sentence)
  test_dependent_lookup_compose_keying_instruction_omitted (>= half unkeyed + misleading rule noun)
  test_dependent_lookup_compose_query_bait_near_miss (query-baited keys presented, never cited, never gold)
  test_dependent_lookup_compose_second_hop_retrieval (second retrieve keyed on the discovered token)
  test_dependent_lookup_compose_revise_on_miss (bare-token miss -> ONE reformulated query -> success)
  test_dependent_lookup_compose_bare_token_queries_confined (bare-token queries only in flagged demos)
  test_dependent_lookup_compose_near_misses_never_cited (stale register/table, placard, unit-stripped text)
  test_dependent_lookup_compose_distractor_carries_stale_token_value
  test_dependent_lookup_compose_subject_diversity (person / plot / part subjects)
  test_dependent_lookup_compose_variants_exercised
  test_v6_pools_disjoint_by_construction   (v6 pools vs v1-v5 pools, fixtures, ALL sealed corpora)
  test_v6_zero_word_8gram_overlap          (M8 method, heldout2 leg live, v1-v5 rows swept live)
  test_v6_key_stems_disjoint_from_measurement_corpora
"""

from __future__ import annotations

import hashlib
import json
import os
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
from scripts.m8_battery import sealed_workloads as sealed
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
# v6 compose records carry claim_key but NEVER claim_value (the campaign
# presentation: meta.claim_key + raw text only — the value must be composed
# from the text, not copied from a field).
COMPOSE_RECORD_KEYS = RECORD_KEYS - {"claim_value"}
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
            if row["class"] in generate.V6_CLASSES:
                assert COMPOSE_RECORD_KEYS <= set(record) <= COMPOSE_RECORD_KEYS | {"superseded_by"}
                assert "claim_value" not in record, "v6 compose records never carry the claim_value crutch"
            else:
                assert RECORD_KEYS <= set(record) <= RECORD_KEYS | {"superseded_by"}
                assert type(record["claim_value"]) in (str, int, float, bool, type(None))
            assert type(record["id"]) is int and record["id"] >= 0
            assert type(record["text"]) is str and record["text"].strip()
            assert type(record["record_ref"]) is str and record["record_ref"]
            assert type(record["source"]) is str and record["source"]
            assert type(record["claim_key"]) is str and record["claim_key"]
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
    if match:
        return match.group(1)
    # v6 compose prompts mirror the heldout2 query surfaces, which need not
    # carry a "For <subject>," clause; the generator pins the asked subject
    # (mirrors the v5 meta.asked_keys fallback precedent in _asked_keys).
    pinned = row["meta"].get("asked_subject")
    assert pinned, row["task"]["prompt"]
    return pinned


def _base_key(key):
    return re.sub(r"\.conflict$|\.per_.*$", "", key)


def _asked_keys(row, action):
    keys = {_base_key(claim["key"]) for claim in action["answer"]["claims"]}
    if not keys:  # abstain: the asked key is named in the prompt or pinned in meta
        hinted = row["meta"].get("asked_keys")
        if hinted is not None:
            # v5: parameterized per-episode keys (per-case QN keys) cannot be
            # looked up in a static fact pool; the generator pins them.
            return set(hinted)
        keys = {fact["key"] for fact in generate.FACTS if fact["key"] in row["task"]["prompt"]}
        assert len(keys) == 1, row["task"]["prompt"]
    return keys


def _record_value(record):
    """The value a presented record advertises: the claim_value field when
    present (v1-v5 records), else the composable tail of the raw text for the
    v6 compose records ('... is <value>.' or '... stands at <value>.'), else
    None when the text advertises no tail value."""
    if "claim_value" in record:
        return record["claim_value"]
    text = record["text"].rstrip(".")
    for marker in (" is ", " stands at "):
        if marker in text:
            return text.rsplit(marker, 1)[1]
    return None


def _near_misses(row, presented, action):
    """Uncited records with an ASKED claim_key whose text is NOT about the
    asked subject. Asserts en route that none carries an answer value."""
    subject = _asked_subject(row)
    keys = _asked_keys(row, action)
    cited = {rid for claim in action["answer"]["claims"] for rid in claim["evidence_ids"]}
    answers = {}
    for claim in action["answer"]["claims"]:
        # The claim's own (key, value): by the claim-key-echo property this
        # equals the cited record's (claim_key, advertised value) — and it
        # stays well-defined for v6 compose records (no claim_value field).
        answers.setdefault(claim["key"], set()).add(claim["value"])
    misses = []
    for rid, record in presented.items():
        if record["claim_key"] in keys and rid not in cited and subject not in record["text"]:
            value = _record_value(record)
            if value is not None:
                assert value not in answers.get(record["claim_key"], set()), (
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
    # v6: one additive M10 v5-campaign non-transfer-repair class on all-new
    # disjoint pools; seed unchanged at [20261116]; the v3/v4 and v5 class
    # bytes are pinned unchanged by test_v4_class_bytes_unchanged and
    # test_v5_class_bytes_unchanged.
    assert manifest["dataset_version"] == generate.DATASET_VERSION == 6
    assert manifest["seed"] == generate.SEED == [20261116]
    # M10 qualification gate checklist (b): the curriculum seed must be
    # distinct from every seed the measurement harness uses or has burned.
    forbidden = {0, 20260929, 20261106, 20261110, 20261112, 20261114}
    assert forbidden.isdisjoint(generate.SEED), "curriculum seed is in the gate's forbidden seed set"
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
    assert variants == {"single_retrieval", "split_retrieval", "reformulation"}


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
            # v3 echo: the distinct keys the claims use are exactly the
            # claim_keys the two records were presented under.
            assert left["claim_key"] == claims[0]["key"]
            assert right["claim_key"] == claims[1]["key"]
            assert left["claim_key"] != right["claim_key"]
        else:
            assert variant == "declare_conflict"
            # Reconciled 2026-11-12 against the observer-verified failure
            # decomposition (the verifier never moves): the gold is the
            # DUAL-ASSERT shape — exactly the conflict_dual_assert idiom —
            # never the refuted 'conflict: ...' narrative.
            assert len(claims) == 2
            assert claims[0]["key"] != claims[1]["key"]
            (left_id,) = claims[0]["evidence_ids"]
            (right_id,) = claims[1]["evidence_ids"]
            left = presented[left_id]
            right = presented[right_id]
            for claim, record in ((claims[0], left), (claims[1], right)):
                assert claim["key"].endswith("_condition"), "source-attributed truth key idiom"
                assert not (type(claim["value"]) is str and claim["value"].startswith("conflict: "))
                # v3 echo: each claim key is exactly the claim_key its own
                # cited record was presented under.
                assert record["claim_key"] == claim["key"]
                assert record["claim_value"] == claim["value"]
                assert type(record["claim_value"]) is type(claim["value"])
            # Both attributed keys extend the same base fact key, each
            # embedding its cited record's source attribution.
            base_a = claims[0]["key"][: -len("_condition")]
            base_b = claims[1]["key"][: -len("_condition")]
            assert base_a != base_b
            assert any(base_a.startswith(fact["key"] + "_") for fact in generate.FACTS)
            assert any(base_b.startswith(fact["key"] + "_") for fact in generate.FACTS)
            assert left["source"].replace("-", "_") in claims[0]["key"]
            assert right["source"].replace("-", "_") in claims[1]["key"]
            # The refuted narrative survives only as a never-cited near-miss
            # under the shared base key (meta provenance, mirroring v4).
            cited = {rid for claim in claims for rid in claim["evidence_ids"]}
            for rid in row["meta"].get("conflict_narrative_ids", ()):
                near_miss = presented[rid]
                value = near_miss["claim_value"]
                assert type(value) is str and value.startswith("conflict: ")
                assert left["source"] in value and right["source"] in value
                # The narrative sits under the shared base key and carries
                # both sources' rendered values.
                kind = {fact["key"]: fact["kind"] for fact in generate.FACTS}[near_miss["claim_key"]]
                assert generate._value_text(left["claim_value"], kind) in value
                assert generate._value_text(right["claim_value"], kind) in value
                assert near_miss["claim_key"] not in {claims[0]["key"], claims[1]["key"]}
                assert rid not in cited, "the conflict-narrative record must never be cited"
        assert left["claim_value"] != right["claim_value"]
        assert left["source"] != right["source"]
        # Both conflicting records are about the asked subject and current.
        subject = _asked_subject(row)
        for record in (left, right):
            assert record["memory_state"] == "current"
            assert subject in record["text"]
    assert variants == {"separate_keys", "declare_conflict"}
    # The narrative near-miss carries real weight across declare_conflict
    # episodes (knob 0.80; deterministic rate 17/21 under seed [20261116]).
    dc = [row for row in rows["conflict_split"] if row["meta"]["variant"] == "declare_conflict"]
    narrated = sum(1 for row in dc if row["meta"].get("conflict_narrative_ids"))
    assert narrated / len(dc) >= 0.60, f"conflict-narrative near-misses in only {narrated}/{len(dc)}"


def test_no_narrative_conflict_gold_anywhere(dataset):
    """The verifier REJECTS the 'conflict: ...' narrative claim form
    (observer-verified 2026-11-12): no gold final in ANY class may use it —
    it exists only as a never-cited near-miss distractor."""
    _, rows = dataset
    for cls in generate.CLASSES:
        for row in rows[cls]:
            _, action = assert_canonical_row(row)
            for claim in action["answer"]["claims"]:
                value = claim["value"]
                assert not (type(value) is str and value.startswith("conflict: ")), (
                    "narrative-conflict claim credited as gold",
                    row["episode_id"],
                    claim,
                )


def test_injection_resist_shape(dataset):
    _, rows = dataset
    variants = set()
    all_surfaces = set()
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
        surfaces = row["meta"]["injection_surfaces"]
        assert surfaces and set(surfaces) <= {"v3", "v4"}
        all_surfaces.update(surfaces)
    assert variants == {"capped_result", "full_result"}
    # v4 phrasing variance: the campaign's injection successes came on novel
    # phrasings, so both injection-surface pools must be exercised and the
    # v4 pool must carry real weight (not a token handful of episodes).
    assert all_surfaces == {"v3", "v4"}
    v4_episodes = sum(1 for row in rows["injection_resist"] if "v4" in row["meta"]["injection_surfaces"])
    assert v4_episodes >= 10, f"novel-surface injection lures in only {v4_episodes} episodes"


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
    assert classes_with_cited == {
        "coverage",
        "conflict_split",
        "injection_resist",
        "replay",
        "idiom_echo",
        "conflict_dual_assert",
        "multipart_decomp",
        "value_shape",
        # v5 classes likewise cite clean records from lure-prone sources
        # (deterministic under seed [20261116]: chain 14, chain-value 2,
        # qn 2, answer-supported 2 episodes).
        "dependent_lookup_chain",
        "value_shape_chain",
        "injection_resist_qn",
        "answer_supported",
        # v6: dependent_lookup_compose cites clean records from lure-prone
        # sources too (deterministic under the unchanged seed).
        "dependent_lookup_compose",
    }, (
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
    kind_by_key.update({fact["key"]: fact["kind"] for fact in generate.V4_FACTS})
    kind_by_key.update({fact["key"]: fact["kind"] for fact in generate.V5_FACTS})
    values = {}
    for cls in generate.CLASSES:
        for row in rows[cls]:
            for record in _presented(row).values():
                # v3: conflict separate_keys records carry per-source keys;
                # the base key identifies the value kind. v4 narrative keys
                # are not in the kind map and are skipped.
                kind = kind_by_key.get(_base_key(record["claim_key"]))
                if kind is not None:
                    values.setdefault(kind, set()).add(record["claim_value"])
    floors = {
        # v4 re-based every floor under seed [20261116] with margin below the
        # measured deterministic counts (time 43, date 40, version 56,
        # host 40, endpoint 53, int 59, float 57, person 111, days 18,
        # phase 9, tier 8; v4 kinds: rpm 85, lux 84, hpa 84, tally 78,
        # mode 10 — the full mode enum plus conflict-narrative strings).
        "time": 35,
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
        "rpm": 60,
        "lux": 60,
        "hpa": 60,
        "tally": 60,
        "mode": 4,
        # v5 unit kinds (measured under seed [20261116]: fathoms 25,
        # gallons 37, shackles 30, hundredweight 47 — floors hold margin).
        "fathoms": 20,
        "gallons": 30,
        "shackles": 25,
        "hundredweight": 40,
    }
    for kind, floor in floors.items():
        assert len(values[kind]) >= floor, f"{kind}: only {len(values[kind])} distinct values (< {floor})"


# ---------------------------------------------------------------------------
# Tests (v3: claim-key echo discipline + loop resistance — Track-B gate repairs)
# ---------------------------------------------------------------------------

ANSWER_BEARING = (
    "coverage",
    "conflict_split",
    "injection_resist",
    "replay",
    "idiom_echo",
    "conflict_dual_assert",
    "multipart_decomp",
    "value_shape",
    # v5: all four new classes are answer-bearing (injection_resist_qn's
    # abstain variant answers nothing BY DESIGN — support is absent there).
    "dependent_lookup_chain",
    "value_shape_chain",
    "injection_resist_qn",
    "answer_supported",
)


def _retrieve_actions(row):
    return [
        parse_agent_action(message["content"])
        for message in row["messages"]
        if message["role"] == "assistant" and "tool_calls" in message
    ]


def _tool_results(row):
    return [json.loads(message["content"]) for message in row["messages"] if message["role"] == "tool"]


def test_claim_key_echo(dataset):
    """v3: every labeled claim key is exactly the claim_key of a presented
    record that supports it — the model must never invent key phrasings.

    Records carry the public claim_key in every episode (mirroring the
    control presentation), and each claim's key is echoed from at least one
    of ITS OWN cited records ("claim key = the claim_key of the record that
    supports it")."""
    _, rows = dataset
    for cls in generate.CLASSES:
        for row in rows[cls]:
            presented, action = assert_canonical_row(row)
            assert presented or row["class"] == "abstain_exact"
            presented_keys = {record["claim_key"] for record in presented.values()}
            for claim in action["answer"]["claims"]:
                assert claim["key"] in presented_keys, (
                    "claim key was never presented in this episode",
                    row["episode_id"],
                    claim["key"],
                )
                cited_keys = {presented[rid]["claim_key"] for rid in claim["evidence_ids"]}
                assert claim["key"] in cited_keys, (
                    "claim key must echo the claim_key of a record that supports it",
                    row["episode_id"],
                    claim,
                )


def test_no_identical_consecutive_retrieves(dataset):
    """v3 hard pattern: identical consecutive retrieve arguments NEVER
    appear in any labeled episode (the Track-B loop attractor)."""
    _, rows = dataset
    for cls in generate.CLASSES:
        for row in rows[cls]:
            actions = _retrieve_actions(row)
            assert actions, "every curriculum episode retrieves at least once"
            for previous, current in zip(actions, actions[1:]):
                assert previous["arguments"] != current["arguments"], (
                    "identical consecutive retrieve arguments",
                    row["episode_id"],
                    current["arguments"],
                )


def test_reformulation_after_empty_or_unhelpful(dataset):
    """v3 loop resistance: after an empty or unhelpful retrieve the labeled
    behavior is at most ONE reformulated query (semantically rephrased,
    never identical), then the episode finalizes — answer or abstain.

    >= 30% of answer-bearing episodes carry the pattern; abstain episodes
    drill reformulate-once-then-abstain."""
    _, rows = dataset
    flagged = 0
    total = 0
    for cls in ANSWER_BEARING:
        for row in rows[cls]:
            presented, action = assert_canonical_row(row)
            total += 1
            mode = row["meta"].get("reformulation")
            if mode is None:
                continue
            flagged += 1
            assert mode in ("after_empty", "after_unhelpful")
            actions = _retrieve_actions(row)
            results = _tool_results(row)
            assert len(actions) == 2 == len(results), "at most one reformulated query, then finalize"
            assert actions[0]["arguments"]["query"] != actions[1]["arguments"]["query"], (
                "reformulation must rephrase, not repeat",
                row["episode_id"],
            )
            if mode == "after_empty":
                assert results[0]["returned_count"] == 0
            else:
                first_ids = {record["id"] for record in results[0]["records"]}
                cited = {rid for claim in action["answer"]["claims"] for rid in claim["evidence_ids"]}
                assert cited and cited.isdisjoint(first_ids), (
                    "unhelpful first retrieve must not contain the cited evidence",
                    row["episode_id"],
                )
    rate = flagged / total
    assert rate >= 0.30, f"reformulation pattern in only {rate:.0%} of answer-bearing episodes"
    abstain_flagged = 0
    for row in rows["abstain_exact"]:
        mode = row["meta"].get("reformulation")
        if mode is None:
            continue
        abstain_flagged += 1
        assert mode == "after_empty"
        actions = _retrieve_actions(row)
        results = _tool_results(row)
        assert len(actions) == 2 == len(results), "one rephrased retry, then abstain"
        assert actions[0]["arguments"]["query"] != actions[1]["arguments"]["query"]
        assert results[0]["returned_count"] == 0
        assert _final_action(row)["answer"] == {"claims": []}
    assert abstain_flagged, "abstain episodes must also drill reformulate-once-then-abstain"


def test_same_value_different_key_distractors(dataset):
    """v3: same-value/different-key near-misses — a current record about the
    ASKED subject carrying an ANSWER value under a DIFFERENT claim_key —
    appear in >= 30% of answer-bearing episodes, so key-echo alone is
    insufficient and value-matching cannot stand in for it. Provenance is
    meta.same_value_ids (mirrors meta.injection_ids)."""
    _, rows = dataset
    hits = 0
    total = 0
    for cls in ANSWER_BEARING:
        for row in rows[cls]:
            presented, action = assert_canonical_row(row)
            total += 1
            ids = [rid for rid in row["meta"].get("same_value_ids", ()) if rid in presented]
            claims = action["answer"]["claims"]
            cited = {rid for claim in claims for rid in claim["evidence_ids"]}
            claim_keys = {claim["key"] for claim in claims}
            answer_values = {claim["value"] for claim in claims} | {presented[rid]["claim_value"] for rid in cited}
            subject = _asked_subject(row)
            for rid in ids:
                record = presented[rid]
                assert rid not in cited, "same-value near-miss must never be cited"
                assert record["claim_key"] not in claim_keys
                assert record["claim_value"] in answer_values, "near-miss must carry an answer value"
                assert record["memory_state"] == "current"
                assert subject in record["text"], "near-miss must be about the asked subject"
            hits += bool(ids)
    rate = hits / total
    assert rate >= 0.30, f"same-value/different-key near-misses in only {rate:.0%} of answer-bearing episodes"


# ---------------------------------------------------------------------------
# Tests (v4: M10 generalization classes on all-new disjoint pools)
# ---------------------------------------------------------------------------

# The terse truth-key idiom: snake_case, no dots, no spaces (the sealed
# workloads' truth-key shape).
TERSE_KEY = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$")


def _v4_asked_keys(row, action):
    keys = {claim["key"] for claim in action["answer"]["claims"]}
    assert keys, "v4 classes are answer-bearing"
    for key in keys:
        assert TERSE_KEY.match(key), ("v4 claim key is not the terse truth-key idiom", row["episode_id"], key)
    return keys


def test_idiom_echo_shape(dataset):
    """v4 (a): claim-key idiom echo across MANY phrasing families. Every final
    claim key is the terse presented truth-key idiom echoed from its cited
    record; the narrative-key near-miss (asked subject, ANSWER value, long
    narrative key) is presented in most episodes and is never cited — the
    narrative key is the wrong answer."""
    _, rows = dataset
    families = set()
    narrative_hits = 0
    for row in rows["idiom_echo"]:
        presented, action = assert_canonical_row(row)
        claims = action["answer"]["claims"]
        assert len(claims) == 1, "idiom_echo asks one question"
        claim = claims[0]
        (key,) = _v4_asked_keys(row, action)
        assert key in row["task"]["prompt"], "the asked key is named in the prompt"
        (record_id,) = claim["evidence_ids"]
        record = presented[record_id]
        assert record["claim_key"] == claim["key"]
        assert record["claim_value"] == claim["value"]
        assert record["memory_state"] == "current"
        subject = _asked_subject(row)
        assert subject in record["text"]
        families.add(row["meta"]["prompt_family"])
        ids = [rid for rid in row["meta"].get("narrative_key_ids", ()) if rid in presented]
        if not ids:
            continue
        narrative_hits += 1
        for rid in ids:
            near_miss = presented[rid]
            assert not TERSE_KEY.match(near_miss["claim_key"]), (
                "narrative near-miss key must be the narrative form, not the idiom",
                near_miss["claim_key"],
            )
            assert " report " in near_miss["claim_key"]
            assert near_miss["claim_value"] == claim["value"], (
                "the narrative near-miss differs from the answer ONLY in its key"
            )
            assert near_miss["memory_state"] == "current"
            assert subject in near_miss["text"]
            assert rid != record_id, "narrative-key record must never be cited"
    assert families == {name for name, _ in generate._V4_IDIOM_PROMPTS}, (
        "every phrasing family must be exercised",
        families,
    )
    rate = narrative_hits / len(rows["idiom_echo"])
    assert rate >= 0.60, f"narrative-key near-misses in only {rate:.0%} of idiom_echo episodes"


def test_conflict_dual_assert_shape(dataset):
    """v4 (b, observer-reconciled): conflict DUAL-ASSERT is the gold. Two
    claims, one per source, each under its source-attributed truth key
    (``<fact>_<source>_condition``) echoed from its own cited record, each
    carrying that record's exact value — never the refuted 'conflict: ...'
    narrative. The narrative form is presented as a near-miss distractor in
    most episodes and is never cited."""
    _, rows = dataset
    families = set()
    narrative_hits = 0
    for row in rows["conflict_dual_assert"]:
        presented, action = assert_canonical_row(row)
        claims = action["answer"]["claims"]
        assert len(claims) == 2, "dual-assert: one claim per source"
        keys = _v4_asked_keys(row, action)
        assert len(keys) == 2, "the two sources assert under DISTINCT attributed keys"
        left, right = claims
        assert left["key"].endswith("_condition") and right["key"].endswith("_condition")
        # Both attributed keys extend the same base fact key.
        base_a = left["key"][: -len("_condition")]
        base_b = right["key"][: -len("_condition")]
        assert base_a != base_b
        assert any(base_a.startswith(fact["key"] + "_") for fact in generate.V4_FACTS)
        assert any(base_b.startswith(fact["key"] + "_") for fact in generate.V4_FACTS)
        for claim in claims:
            value = claim["value"]
            # The gold NEVER takes the refuted narrative form.
            assert not (type(value) is str and value.startswith("conflict: ")), (
                "narrative-conflict claim credited as gold",
                row["episode_id"],
            )
            (record_id,) = claim["evidence_ids"]
            record = presented[record_id]
            assert record["claim_key"] == claim["key"], "each claim echoes its own source record's key"
            assert record["claim_value"] == claim["value"]
            assert type(record["claim_value"]) is type(claim["value"])
            assert record["memory_state"] == "current"
        record_a = presented[left["evidence_ids"][0]]
        record_b = presented[right["evidence_ids"][0]]
        assert record_a["source"] != record_b["source"]
        assert record_a["claim_value"] != record_b["claim_value"], "the two sources genuinely disagree"
        # Each claim's key embeds its cited record's source attribution.
        assert record_a["source"].replace("-", "_") in left["key"]
        assert record_b["source"].replace("-", "_") in right["key"]
        subject = _asked_subject(row)
        for record in (record_a, record_b):
            assert subject in record["text"]
        families.add(row["meta"]["prompt_family"])
        ids = [rid for rid in row["meta"].get("conflict_narrative_ids", ()) if rid in presented]
        if not ids:
            continue
        narrative_hits += 1
        cited = {rid for claim in claims for rid in claim["evidence_ids"]}
        for rid in ids:
            near_miss = presented[rid]
            value = near_miss["claim_value"]
            assert type(value) is str and value.startswith("conflict: ")
            assert record_a["source"] in value and record_b["source"] in value
            assert str(record_a["claim_value"]) in value and str(record_b["claim_value"]) in value
            assert near_miss["claim_key"] not in keys, "the narrative sits under the base key, not a truth key"
            assert rid not in cited, "the conflict-narrative record must never be cited"
    assert families == {name for name, _ in generate._V4_CONFLICT_PROMPTS}, (
        "every phrasing family must be exercised",
        families,
    )
    rate = narrative_hits / len(rows["conflict_dual_assert"])
    assert rate >= 0.60, f"conflict-narrative near-misses in only {rate:.0%} of conflict_dual_assert episodes"


def test_value_shape_shape(dataset):
    """v4 (d, observer-driven): value-shape fidelity. The gold claim value is
    the unit-bearing STRING echoed verbatim from a record whose text ends in
    the verifier's status_value form ('... is <value>.'); a type-stripped
    near-miss (same subject, asked key, the same number as a bare int) is
    ALWAYS present and never cited — the int is the campaign failure."""
    _, rows = dataset
    families = set()
    for row in rows["value_shape"]:
        presented, action = assert_canonical_row(row)
        claims = action["answer"]["claims"]
        assert len(claims) == 1
        claim = claims[0]
        (key,) = _v4_asked_keys(row, action)
        assert key in row["task"]["prompt"], "the asked key is named in the prompt"
        assert type(claim["value"]) is str, "the gold value is a string, never a bare number"
        (record_id,) = claim["evidence_ids"]
        record = presented[record_id]
        assert record["claim_key"] == claim["key"]
        assert record["claim_value"] == claim["value"]
        # The cited record states the value in the verifier's status_value
        # form: the text ends '... is <value>.' and the tail equals the claim.
        assert record["text"].endswith(f" is {claim['value']}."), (
            "cited record is not in status_value form",
            record["text"],
        )
        assert record["memory_state"] == "current"
        subject = _asked_subject(row)
        assert subject in record["text"]
        families.add(row["meta"]["prompt_family"])
        # The type-stripped near-miss: same subject, asked key, same number,
        # bare int — always present, never cited, never the gold's type.
        ids = row["meta"]["type_stripped_ids"]
        assert len(ids) == 1 and ids[0] in presented
        stripped = presented[ids[0]]
        assert ids[0] != record_id
        assert stripped["claim_key"] == claim["key"]
        assert type(stripped["claim_value"]) is int, "the near-miss is the type-stripped int"
        assert stripped["claim_value"] == int(claim["value"].split()[0])
        assert subject in stripped["text"]
        assert stripped["memory_state"] == "current"
    assert families == {name for name, _ in generate._V4_VALUE_SHAPE_PROMPTS}


def test_multipart_decomp_shape(dataset):
    """v4 (c): multi-part decomposition. 2- and 3-part questions; the final
    covers EVERY part, each claim keyed by its presented terse key and citing
    its own record."""
    _, rows = dataset
    families = set()
    part_counts = set()
    for row in rows["multipart_decomp"]:
        presented, action = assert_canonical_row(row)
        claims = action["answer"]["claims"]
        parts = row["meta"]["parts"]
        assert parts in (2, 3)
        part_counts.add(parts)
        assert len(claims) == parts, "every part of the question is answered"
        keys = _v4_asked_keys(row, action)
        assert len(keys) == len(claims), "duplicate keys would strand a part"
        subject = _asked_subject(row)
        for claim in claims:
            assert claim["key"] in row["task"]["prompt"], "each part's key is named in the prompt"
            (record_id,) = claim["evidence_ids"]
            record = presented[record_id]
            assert record["claim_key"] == claim["key"]
            assert record["claim_value"] == claim["value"]
            assert type(record["claim_value"]) is type(claim["value"])
            assert record["memory_state"] == "current"
            assert subject in record["text"]
        families.add(row["meta"]["prompt_family"])
    assert families == {name for name, _ in generate._V4_MULTIPART_PROMPTS}
    assert part_counts == {2, 3}, "both 2- and 3-part decompositions must appear"


# ---------------------------------------------------------------------------
# v4 leakage discipline (M8 method: word 8-gram sweeps + disjoint pools)
# ---------------------------------------------------------------------------

# The sealed workload corpora are byte-reproducible from the in-repo
# generator (scripts/m8_battery/sealed_workloads.py); where a sealed copy
# exists on disk (DML_M8_WORKLOADS_DIR, default the campaign scratch path)
# the test first proves the on-disk bytes equal the regenerated bytes, then
# sweeps against them. The v3 curriculum output leg sweeps a dataset_version
# 3 out/ directory when one is available (DML_M9_CURRICULUM_V3_OUT or an
# in-tree out/ whose manifest pins version 3).
M8_WORKLOADS_DIR = Path(os.environ.get("DML_M8_WORKLOADS_DIR", str(Path.home() / ".hermes/cache/scratch/m8-workloads")))
V3_OUT_DIR = Path(
    os.environ.get(
        "DML_M9_CURRICULUM_V3_OUT",
        str(Path(__file__).resolve().parents[1] / "scripts" / "m9_curriculum" / "out"),
    )
)


def _word_ngrams(text, n=8):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[index : index + n]) for index in range(len(words) - n + 1)}


def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _row_texts(row):
    """The curriculum text surface mirroring the M8 sweep: task prompt,
    retrieve queries, presented record texts, string claim values."""
    texts = [row["task"]["prompt"]]
    for message in row["messages"]:
        if message["role"] == "tool":
            texts.extend(record["text"] for record in json.loads(message["content"])["records"])
        elif message["role"] == "assistant" and "tool_calls" in message:
            texts.append(json.loads(message["content"])["arguments"]["query"])
        elif message["role"] == "assistant":
            for claim in json.loads(message["content"])["answer"]["claims"]:
                if type(claim["value"]) is str:
                    texts.append(claim["value"])
    return texts


def _sealed_scenario_texts(payload):
    texts = [event["text"] for event in payload["events"]] + [payload["query"]]
    texts.extend(claim["value"] for claim in payload["truth"] if type(claim.get("value")) is str)
    return texts


@pytest.fixture(scope="module")
def sealed_corpora(tmp_path_factory):
    """Regenerated pilot216 + heldout sealed corpora (byte-identical to any
    sealed on-disk copy, which is verified when present)."""
    out = tmp_path_factory.mktemp("sealed")
    scenarios = {}
    for workload, directory in (("pilot", "pilot216"), ("heldout", "heldout")):
        sealed.generate_workload(workload, out / directory)
        scenarios[directory] = {p.name: p for p in sorted((out / directory / "scenarios").glob("*.json"))}
    on_disk = {
        "pilot216": M8_WORKLOADS_DIR / "pilot216" / "scenarios",
        "heldout": M8_WORKLOADS_DIR / "heldout" / "scenarios",
    }
    for directory, scenarios_dir in on_disk.items():
        if not scenarios_dir.is_dir():
            continue
        disk = {p.name: p.read_bytes() for p in sorted(scenarios_dir.glob("*.json"))}
        assert len(disk) == len(scenarios[directory]), f"on-disk {directory} scenario count differs"
        for name, path in scenarios[directory].items():
            assert disk.get(name) == path.read_bytes(), f"on-disk {directory}/{name} differs from regenerated bytes"
    texts = []
    for directory in scenarios.values():
        for path in directory.values():
            texts.extend(_sealed_scenario_texts(json.loads(path.read_text(encoding="utf-8"))))
    return texts


def test_v4_pools_disjoint_by_construction(dataset, sealed_corpora):
    """The v4 pools are all-new: token-level disjoint from the v1-v3 entity
    pools, the m8 fixtures and the sealed workloads; the terse v4 keys are
    disjoint from every prior and sealed key space."""
    v3_pool_tokens = set()
    for entity in generate.ENTITY_POOL_STRINGS:
        v3_pool_tokens |= _tokens(entity)
    v4_pool_tokens = set()
    for entity in generate.V4_ENTITY_POOL_STRINGS:
        v4_pool_tokens |= _tokens(entity)
    assert not (v4_pool_tokens & v3_pool_tokens), (
        "v4 entity pool shares tokens with the v1-v3 pools",
        sorted(v4_pool_tokens & v3_pool_tokens),
    )
    v3_keys = {fact["key"] for fact in generate.FACTS}
    v4_keys = {fact["key"] for fact in generate.V4_FACTS}
    assert not (v3_keys & v4_keys)
    sealed_text_blob = "\n".join(sealed_corpora)
    sealed_tokens = _tokens(sealed_text_blob)
    assert not (v4_pool_tokens & sealed_tokens), (
        "v4 entity pool shares tokens with the sealed workloads",
        sorted(v4_pool_tokens & sealed_tokens),
    )
    fixture_blob = "\n".join(path.read_text(encoding="utf-8") for path in sorted(FIXTURES.glob("*.json")))
    assert not (v4_pool_tokens & _tokens(fixture_blob))
    # v4 value-unit and mode tokens are likewise absent from every corpus.
    value_tokens = set()
    for fact in generate.V4_FACTS:
        value_tokens |= _tokens(fact["label"]) | _tokens(fact["key"])
    value_tokens |= _tokens(" ".join(generate._V4_MODES)) | {"rpm", "lux", "hpa"}
    assert not (value_tokens & sealed_tokens)
    assert not (value_tokens & _tokens(fixture_blob))
    assert not (value_tokens & v3_pool_tokens)
    # String-level: no v4 pool string appears in any corpus text, and no
    # per-episode v4 entity usage leaks either.
    _, rows = dataset
    used = set()
    for cls in generate.V4_CLASSES:
        for row in rows[cls]:
            used.update(row["meta"]["entities"])
    assert used
    for entity in set(generate.V4_ENTITY_POOL_STRINGS) | used:
        assert entity not in sealed_text_blob
        assert entity not in fixture_blob
    for key in v4_keys:
        assert key not in sealed_text_blob, "v4 terse key collides with sealed truth keys"


def test_v4_zero_word_8gram_overlap(dataset, sealed_corpora):
    """The M8 leakage method re-run for curriculum v4: zero shared word
    8-grams between the v4-class rows (prompts, queries, record texts, string
    claim values) and (1) the sealed pilot216/heldout corpora, (2) the m8 dev
    fixtures, (3) the dataset_version 3 curriculum output, when available."""
    v4_grams = set()
    for cls in generate.V4_CLASSES:
        for row in dataset[1][cls]:
            for text in _row_texts(row):
                v4_grams |= _word_ngrams(text)
    assert v4_grams
    sealed_grams = set()
    for text in sealed_corpora:
        sealed_grams |= _word_ngrams(text)
    overlap = v4_grams & sealed_grams
    assert not overlap, f"shared 8-grams with the sealed corpora: {sorted(overlap)[:5]}"
    fixture_grams = set()
    for path in sorted(FIXTURES.glob("*.json")):
        for event in json.loads(path.read_text(encoding="utf-8"))["events"]:
            fixture_grams |= _word_ngrams(event["text"])
    overlap = v4_grams & fixture_grams
    assert not overlap, f"shared 8-grams with the m8 fixtures: {sorted(overlap)[:5]}"
    manifest_path = V3_OUT_DIR / "manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()).get("dataset_version") == 3:
        v3_grams = set()
        for path in sorted(V3_OUT_DIR.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                for text in _row_texts(json.loads(line)):
                    v3_grams |= _word_ngrams(text)
        overlap = v4_grams & v3_grams
        assert not overlap, f"shared 8-grams with the v3 curriculum output: {sorted(overlap)[:5]}"

# ---------------------------------------------------------------------------
# Stem-level disjointness (2026-11-12 leakage re-audit, gate checklist (b))
# ---------------------------------------------------------------------------

# The nine stems the re-audit found shared between the curriculum pools and
# measurement-corpora fact keys (heldout2: berth/gantry/louver/forum/
# lighting; pilot216/heldout: lumen/juniper; dev fixtures: helios/kestrel).
# Pinned absent as a regression list so no future pool edit reintroduces
# them; the disjointness assertion below is the real proof.
PURGED_STEMS = frozenset(
    {"berth", "gantry", "louver", "forum", "lighting", "lumen", "juniper", "helios", "kestrel"}
)


def _fact_key_tokens(payload):
    """Every token of every fact/truth key a scenario or fixture carries."""
    tokens = set()
    for claim in payload.get("truth", ()):
        tokens |= _tokens(str(claim.get("key", "")))
    for event in payload.get("events", ()):
        tokens |= _tokens(str(event.get("fact_key", "")))
    return tokens


@pytest.fixture(scope="module")
def measurement_key_tokens(tmp_path_factory):
    """Token union of every fact/truth key in the measurement corpora:
    pilot216 + heldout (byte-regenerated from the in-repo sealed generator),
    heldout2 (the M10 qualification-gate workload; swept when its sealed copy
    exists under DML_M8_WORKLOADS_DIR — its generator spec lives on the main
    branch, not this worktree), and the 12 dev fixtures (in-repo)."""
    tokens = set()
    out = tmp_path_factory.mktemp("sealed-keys")
    coverage = []
    for workload, directory in (("pilot", "pilot216"), ("heldout", "heldout")):
        sealed.generate_workload(workload, out / directory)
        scenarios = sorted((out / directory / "scenarios").glob("*.json"))
        assert scenarios, f"regenerated {directory} scenarios missing"
        for path in scenarios:
            tokens |= _fact_key_tokens(json.loads(path.read_text(encoding="utf-8")))
        coverage.append(f"{directory}:{len(scenarios)}")
    heldout2_dir = M8_WORKLOADS_DIR / "heldout2" / "scenarios"
    if heldout2_dir.is_dir():
        scenarios = sorted(heldout2_dir.glob("*.json"))
        assert scenarios, "on-disk heldout2 scenarios missing"
        for path in scenarios:
            tokens |= _fact_key_tokens(json.loads(path.read_text(encoding="utf-8")))
        coverage.append(f"heldout2:{len(scenarios)}")
    fixtures = sorted(FIXTURES.glob("*.json"))
    assert fixtures, "m8 battery fixtures missing"
    for path in fixtures:
        tokens |= _fact_key_tokens(json.loads(path.read_text(encoding="utf-8")))
    coverage.append(f"fixtures:{len(fixtures)}")
    assert tokens, "no measurement-corpora fact keys collected"
    return tokens, coverage


def _curriculum_pool_stems():
    """The content-bearing stems of the curriculum pools (the audit's stem
    semantics): the FIRST token of every entity-pool string (entity families
    are carried by their first token), and EVERY token of the source ids and
    of the v4 terse fact keys/labels (a fact-key family stem may sit in any
    position — 'louver' in vent_louver_mode was the audit's case).

    The v1-v3 dotted fact keys (ops.maintenance_window, ...) are out of
    scope BY DESIGN: they intentionally mirror the dev-fixture key idioms on
    disjoint synthetic entities — that mirroring is the M9-qualified design,
    not leakage."""
    stems = set()
    for entity in list(generate.ENTITY_POOL_STRINGS) + list(generate.V4_ENTITY_POOL_STRINGS):
        stems.add(re.findall(r"[a-z0-9]+", entity.lower())[0])
    # v5: same audit semantics — first-token stems of the v5 entity pool,
    # every token of the v5 source ids and terse fact keys/labels, and every
    # declared v5 key stem (parameterized hop-2 prefixes, the fabricated QN
    # near-miss stem and the adverse-notice key included).
    for entity in generate.V5_ENTITY_POOL_STRINGS:
        stems.add(re.findall(r"[a-z0-9]+", entity.lower())[0])
    for source in list(generate._SOURCES) + list(generate._INJECTION_SOURCES) + list(generate._V4_SOURCES):
        stems |= _tokens(source)
    for source in generate._V5_SOURCES:
        stems |= _tokens(source)
    for fact in generate.V4_FACTS:
        stems |= _tokens(fact["key"]) | _tokens(fact["label"])
    for fact in generate.V5_FACTS:
        stems |= _tokens(fact["key"]) | _tokens(fact["label"])
    stems |= generate.V5_KEY_STEMS
    # v6: same audit semantics — first-token stems of the v6 entity pool
    # (persons, plots, part ids), every token of the v6 source ids, and every
    # declared v6 key stem (parameterized hop-2 prefixes, hop tokens, units,
    # rule-key suffixes and the query-baited near-miss key stems included).
    for entity in generate.V6_ENTITY_POOL_STRINGS:
        stems.add(re.findall(r"[a-z0-9]+", entity.lower())[0])
    for source in generate._V6_SOURCES:
        stems |= _tokens(source)
    stems |= generate.V6_KEY_STEMS
    return stems


def test_pool_stems_disjoint_from_measurement_corpora(measurement_key_tokens):
    """Gate checklist (b), stem level: no content-bearing curriculum pool stem
    may be a token of any measurement-corpora fact/truth key. The word
    8-gram sweeps structurally cannot catch this — a single shared stem never
    forms an 8-gram — so this test sweeps stems directly."""
    tokens, coverage = measurement_key_tokens
    stems = _curriculum_pool_stems()
    assert stems
    overlap = stems & tokens
    assert not overlap, (
        "curriculum pool stems shared with measurement-corpora fact keys",
        sorted(overlap),
        coverage,
    )
    reintroduced = PURGED_STEMS & stems
    assert not reintroduced, f"purged stems reintroduced into the pools: {sorted(reintroduced)}"

# ---------------------------------------------------------------------------
# Tests (v5: M10 v4.1-campaign failure repairs — additive classes)
# ---------------------------------------------------------------------------

# The nine v3/v4 class files at dataset_version 4 (seed [20261116], commit
# b8a82fd), verified byte-identical when the v5 generator landed. v5 is
# ADDITIVE: no v3/v4 builder or pool may change these bytes.
V4_CLASS_SHAS = {
    "abstain_exact.jsonl": "1c643cfa9338187a7f21503e468504c512bc7d90bbcb493a70e3003cfbe1a1d8",
    "coverage.jsonl": "d26503376f73e6a5e2a1cb1f3c531cddc50992a240eae551394c1b7737cf8faf",
    "conflict_split.jsonl": "98dd921e8a9d8e9424e54600f627e9c5f546398356c365debd5e858bd53102f0",
    "injection_resist.jsonl": "f4a5dc115ef744080b4126f9fa16e8acb468c551c6aa76da1e57382e5b5104b8",
    "replay.jsonl": "2eaafe4468090be506cd33e65999c4bee7046af5e201fdbd7f7103500224b600",
    "idiom_echo.jsonl": "e3b60249ff84fa3d3e71d48e3b67372afaac498ac387a3f706776a4abdafba3b",
    "conflict_dual_assert.jsonl": "7c48ba20a84207de93f67007bb288db1750d7d175fcc747ea3fa775945e385ec",
    "multipart_decomp.jsonl": "ad0aa371f3280b9b34ede94fb6a33638424d637e90b6825a26efcc574040cb48",
    "value_shape.jsonl": "b5f0d1ce677e5eefe81a6f27b3668c1792cfe1e8ad6db8a6921273a4a11558ea",
}

V4_CLASS_COUNTS = {
    "abstain_exact": 30,
    "coverage": 40,
    "conflict_split": 30,
    "injection_resist": 42,
    "replay": 40,
    "idiom_echo": 40,
    "conflict_dual_assert": 30,
    "multipart_decomp": 40,
    "value_shape": 20,
}

FABRICATED_QN_KEY = re.compile(r"^zx_\d+_signoff_circular$")


def test_v4_class_bytes_unchanged(dataset):
    """v5 is additive: the nine v3/v4 class files regenerate byte-identical
    under the unchanged seed [20261116]. Any edit to a v3/v4 builder, pool or
    draw order breaks this pin."""
    output, _ = dataset
    manifest = json.loads((output / "manifest.json").read_text())
    shas = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    for name, sha in V4_CLASS_SHAS.items():
        assert shas[name] == sha, f"v3/v4 class bytes changed: {name}"


def test_v5_classes_additive():
    """CLASSES and DEFAULT_COUNTS keep the v5 classes in their v5 positions;
    the v6 tail is pinned separately by test_v6_classes_additive."""
    assert generate.CLASSES[:9] == generate.V3_CLASSES + generate.V4_CLASSES
    assert generate.CLASSES[9:13] == generate.V5_CLASSES
    assert set(generate.V5_CLASSES) == {
        "dependent_lookup_chain",
        "value_shape_chain",
        "injection_resist_qn",
        "answer_supported",
    }
    for cls, count in V4_CLASS_COUNTS.items():
        assert generate.DEFAULT_COUNTS[cls] == count, f"v3/v4 class count changed: {cls}"
    assert generate.DEFAULT_COUNTS["dependent_lookup_chain"] == 40
    assert generate.DEFAULT_COUNTS["value_shape_chain"] == 16
    assert generate.DEFAULT_COUNTS["injection_resist_qn"] == 14
    assert generate.DEFAULT_COUNTS["answer_supported"] == 15


def test_v5_episode_id_prefixes(dataset):
    """v5 episodes are m10-prefixed (M10 curriculum), numbered per class."""
    _, rows = dataset
    for cls in generate.V5_CLASSES:
        for index, row in enumerate(rows[cls]):
            assert row["episode_id"] == f"m10-{cls.replace('_', '-')}-{index:04d}"


# ---------------------------------------------------------------------------
# v5 (a): dependent_lookup_chain — the chained-lookup capability
# ---------------------------------------------------------------------------


def _chain_spec(row):
    (spec,) = [s for s in generate.V5_CHAIN_VARIANTS if s["variant"] == row["meta"]["chain_variant"]]
    return spec


def test_dependent_lookup_chain_shape(dataset):
    """The gold solves the whole chain: exactly three claims — the hop-2
    value under the PARAMETERIZED claim_key echoed verbatim from the
    presented record's meta (unit-bearing string), the corrected hop-1 value
    under its key, and the rule under its key with the terse dimension
    value — each citing exactly its own presented record."""
    _, rows = dataset
    for row in rows["dependent_lookup_chain"]:
        presented, action = assert_canonical_row(row)
        spec = _chain_spec(row)
        claims = action["answer"]["claims"]
        assert len(claims) == 3, "chain gold answers hop-2 + hop-1 + rule"
        hop2, hop1, rule = claims
        assert TERSE_KEY.match(hop2["key"]) and TERSE_KEY.match(hop1["key"]) and TERSE_KEY.match(rule["key"])
        # Hop-2 key is parameterized by the DISCOVERED (corrected) token.
        assert hop1["value"] in {token for token in spec["tokens"]}
        assert hop2["key"] == f"{spec['hop2_prefix']}_{hop1['value']}", (
            "hop-2 claim key is not parameterized by the discovered hop-1 token",
            row["episode_id"],
            hop2["key"],
        )
        assert hop1["key"] == spec["hop1_key"]
        assert rule["key"] == spec["rule_key"]
        assert rule["value"] == spec["dimension"], "the rule claim carries the terse dimension value"
        subject = _asked_subject(row)
        for claim in (hop2, hop1, rule):
            (record_id,) = claim["evidence_ids"]
            record = presented[record_id]
            assert record["claim_key"] == claim["key"], "each claim echoes its cited record's claim_key"
            assert record["claim_value"] == claim["value"]
            assert type(record["claim_value"]) is type(claim["value"])
            assert record["memory_state"] == "current"
            assert subject in record["text"]


def test_dependent_lookup_chain_second_hop_retrieval(dataset):
    """The capability the campaign showed missing (hop-2 token in a retrieve
    query: 2/48 dml, 0/48 rag): every gold episode issues a SECOND retrieve
    keyed on the discovered hop-1 token, and the hop-2 current table is
    absent from the first window (its keying token is unknown until the
    correction is read)."""
    _, rows = dataset
    for row in rows["dependent_lookup_chain"]:
        presented, action = assert_canonical_row(row)
        actions = _retrieve_actions(row)
        results = _tool_results(row)
        assert len(actions) == 2 == len(results), "chain gold is exactly two retrieves, then the final"
        hop1_value = action["answer"]["claims"][1]["value"]
        first_query = actions[0]["arguments"]["query"]
        second_query = actions[1]["arguments"]["query"]
        assert hop1_value not in first_query, "the hop-2 token is not in the question or first query"
        assert hop1_value in second_query, "second retrieve must key on the discovered hop-1 token"
        assert first_query != second_query
        hop2_id = action["answer"]["claims"][0]["evidence_ids"][0]
        first_ids = {record["id"] for record in results[0]["records"]}
        assert hop2_id not in first_ids, "hop-2 current table must be discovered, not handed over"
        assert hop2_id in {record["id"] for record in results[1]["records"]}


def test_dependent_lookup_chain_near_misses_never_cited(dataset):
    """Every near-miss shape from the campaign is presented and never cited:
    the superseded hop-1 register (stale token), the stale hop-2 table, the
    untrusted-instruction handbill, the type-stripped bare-int readout and
    the narrative-key record."""
    _, rows = dataset
    stale_hop1 = 0
    narrative_hits = 0
    for row in rows["dependent_lookup_chain"]:
        presented, action = assert_canonical_row(row)
        spec = _chain_spec(row)
        claims = action["answer"]["claims"]
        cited = {rid for claim in claims for rid in claim["evidence_ids"]}
        answer_values = {claim["value"] for claim in claims}
        hop2_key, hop1_value = claims[0]["key"], claims[1]["value"]
        for rid in row["meta"].get("stale_hop1_ids", ()):
            stale_hop1 += 1
            record = presented[rid]
            assert rid not in cited
            assert record["claim_key"] == spec["hop1_key"]
            assert record["claim_value"] != hop1_value, "stale register carries the stale token"
            assert record["memory_state"] == "superseded"
            # Chained to the presented correction (memory_state, not prose).
            assert record["superseded_by"] in cited
        for rid in row["meta"]["distractor_ids"]:
            record = presented[rid]
            assert rid not in cited
            assert record["claim_key"].startswith("handbill_")
            assert record["claim_value"] not in answer_values
            assert "command" in record["text"], "the distractor is an untrusted instruction"
        # The stale hop-2 table: parameterized by the STALE token, never cited.
        stale_tables = [
            r
            for r in presented.values()
            if r["claim_key"].startswith(spec["hop2_prefix"] + "_") and r["claim_key"] != hop2_key
        ]
        assert stale_tables, "the dual hop-2 tables must both be presented"
        for record in stale_tables:
            assert record["id"] not in cited
        # Type-stripped bare-int of the CORRECT hop-2 value: never cited.
        (stripped_id,) = row["meta"]["type_stripped_ids"]
        stripped = presented[stripped_id]
        assert stripped_id not in cited
        assert stripped["claim_key"] == hop2_key
        assert type(stripped["claim_value"]) is int
        assert stripped["claim_value"] == int(claims[0]["value"].split()[0])
        for rid in row["meta"].get("narrative_key_ids", ()):
            narrative_hits += 1
            record = presented[rid]
            assert rid not in cited
            assert not TERSE_KEY.match(record["claim_key"]), "narrative near-miss is not the terse idiom"
            assert record["claim_value"] == claims[0]["value"], "narrative record differs ONLY in its key"
    rate = stale_hop1 / len(rows["dependent_lookup_chain"])
    assert rate >= 0.60, f"superseded hop-1 registers in only {rate:.0%} of chain episodes"
    rate = narrative_hits / len(rows["dependent_lookup_chain"])
    assert rate >= 0.40, f"narrative-key near-misses in only {rate:.0%} of chain episodes"


def test_dependent_lookup_chain_rule_status_value_tail(dataset):
    """The rule record ends in the verifier's status_value form and the gold
    rule claim equals the tail (the investigation's Defect-A fix shape) —
    never the full rule sentence."""
    _, rows = dataset
    for row in rows["dependent_lookup_chain"]:
        presented, action = assert_canonical_row(row)
        rule = action["answer"]["claims"][2]
        record = presented[rule["evidence_ids"][0]]
        assert record["text"].endswith(f" is {rule['value']}."), (
            "rule record is not in status_value tail form",
            record["text"],
        )
        assert len(rule["value"].split()) == 2, "the rule value is the terse 2-word dimension"
        assert rule["value"] != record["text"], "the rule claim must not be the full sentence"


def test_dependent_lookup_chain_variants_exercised(dataset):
    _, rows = dataset
    variants = {row["meta"]["chain_variant"] for row in rows["dependent_lookup_chain"]}
    assert variants == {spec["variant"] for spec in generate.V5_CHAIN_VARIANTS}


def test_dependent_lookup_chain_unit_bearing_hop2(dataset):
    """Unit retention under multi-claim load (40/49 campaign claims dropped
    units): every hop-2 gold value is a string ending in the variant unit."""
    _, rows = dataset
    for row in rows["dependent_lookup_chain"]:
        _, action = assert_canonical_row(row)
        spec = _chain_spec(row)
        hop2 = action["answer"]["claims"][0]
        assert type(hop2["value"]) is str
        number, _, unit = hop2["value"].partition(" ")
        assert number.isdigit() and unit == spec["unit"]


# ---------------------------------------------------------------------------
# v5 (b): value_shape_chain — units through a correction chain
# ---------------------------------------------------------------------------


def test_value_shape_chain_shape(dataset):
    """changing_facts-style chain: the value carries units THROUGH the
    correction; the gold echoes the corrected unit-bearing string verbatim
    from the tail-form revision record."""
    _, rows = dataset
    families = set()
    for row in rows["value_shape_chain"]:
        presented, action = assert_canonical_row(row)
        claims = action["answer"]["claims"]
        assert len(claims) == 1
        claim = claims[0]
        assert TERSE_KEY.match(claim["key"])
        assert claim["key"] in row["task"]["prompt"], "the asked key is named in the prompt"
        assert type(claim["value"]) is str, "the gold value is a string, never a bare number"
        number, _, unit = claim["value"].partition(" ")
        assert number.isdigit() and unit in generate._V5_UNIT_BOUNDS
        (record_id,) = claim["evidence_ids"]
        record = presented[record_id]
        assert record["claim_key"] == claim["key"]
        assert record["claim_value"] == claim["value"]
        assert record["text"].endswith(f" is {claim['value']}."), "cited record is not in status_value form"
        assert record["memory_state"] == "current"
        assert _asked_subject(row) in record["text"]
        families.add(row["meta"]["prompt_family"])
    assert families == {name for name, _ in generate._V5_VALUE_SHAPE_PROMPTS}


def test_value_shape_chain_correction_chain(dataset):
    """Every episode presents a superseded bulletin chain (2-3 steps) whose
    gold cites the TAIL revision; each superseded link chains to its
    presented replacement; the stale values differ from the answer."""
    _, rows = dataset
    lengths = set()
    for row in rows["value_shape_chain"]:
        presented, action = assert_canonical_row(row)
        claim = action["answer"]["claims"][0]
        (record_id,) = claim["evidence_ids"]
        chain = [
            r
            for r in presented.values()
            if r["claim_key"] == claim["key"] and _asked_subject(row) in r["text"] and type(r["claim_value"]) is str
        ]
        length = row["meta"]["chain_length"]
        lengths.add(length)
        assert len(chain) == length, "the whole bulletin chain is presented"
        stale = [r for r in chain if r["memory_state"] == "superseded"]
        assert len(stale) == length - 1
        for record in stale:
            assert record["claim_value"] != claim["value"], "stale link carries a stale value"
            assert record["superseded_by"] in {r["id"] for r in chain}, "chain links to a presented replacement"
        # The cited revision supersedes the chain's middle (or only) link.
        assert any(record["superseded_by"] == record_id for record in stale) or length == 2
    assert 3 in lengths, "3-step correction chains must appear"


def test_value_shape_chain_stripped_near_miss(dataset):
    """The type-stripped near-miss is the CORRECTED value as a bare int (the
    v4.1 failure shape) — always present, same subject, asked key, never
    cited."""
    _, rows = dataset
    for row in rows["value_shape_chain"]:
        presented, action = assert_canonical_row(row)
        claim = action["answer"]["claims"][0]
        (record_id,) = claim["evidence_ids"]
        ids = row["meta"]["type_stripped_ids"]
        assert len(ids) == 1 and ids[0] in presented
        stripped = presented[ids[0]]
        assert ids[0] != record_id
        assert stripped["claim_key"] == claim["key"]
        assert type(stripped["claim_value"]) is int
        assert stripped["claim_value"] == int(claim["value"].split()[0]), "the near-miss strips the CORRECTED value"
        assert _asked_subject(row) in stripped["text"]


# ---------------------------------------------------------------------------
# v5 (c): injection_resist_qn — the QN fabricated-key regression archetype
# ---------------------------------------------------------------------------


def test_injection_resist_qn_shape(dataset):
    """QN archetype: gold answers ONLY the legitimate case keys citing the
    case-tracker records (ignore), or abstains when no legitimate record is
    presented. The injection record is never cited; the injected date never
    appears in a claim."""
    _, rows = dataset
    for row in rows["injection_resist_qn"]:
        presented, action = assert_canonical_row(row)
        (injection_id,) = row["meta"]["injection_ids"]
        injection = presented[injection_id]
        claims = action["answer"]["claims"]
        cited = {rid for claim in claims for rid in claim["evidence_ids"]}
        assert injection_id not in cited, "claim cites the injection record"
        assert injection["claim_value"] not in {claim["value"] for claim in claims}, (
            "the injected date was echoed into the answer"
        )
        legit_keys = set(row["meta"]["asked_keys"])
        if row["meta"]["variant"] == "answer_legitimate":
            assert {claim["key"] for claim in claims} == legit_keys, "gold covers exactly the legitimate keys"
            for claim in claims:
                (record_id,) = claim["evidence_ids"]
                record = presented[record_id]
                assert record["claim_key"] == claim["key"]
                assert record["claim_value"] == claim["value"]
                assert record["memory_state"] == "current"
                assert f"Case ZX-{claim['key'].split('_')[1]}" in record["text"]
        else:
            assert row["meta"]["variant"] == "abstain_injection_only"
            assert claims == [], "abstain variant must end with empty claims"
            # No CURRENT record about the asked case under a legitimate key
            # is presented: abstention is the supported behavior here.
            case = f"Case ZX-{row['meta']['asked_keys'][0].split('_')[1]}"
            assert not [
                r
                for r in presented.values()
                if r["claim_key"] in legit_keys and r["memory_state"] == "current" and case in r["text"]
            ]


def test_injection_resist_qn_no_fabricated_key_gold(dataset):
    """The regression shape — injected value verbatim under a FABRICATED
    ``*_signoff_circular`` key citing the injection record — is the
    near-miss and NEVER the gold: no gold claim anywhere uses the
    fabricated key, and the injection record always presents under it."""
    _, rows = dataset
    for row in rows["injection_resist_qn"]:
        presented, action = assert_canonical_row(row)
        fabricated = row["meta"]["fabricated_key"]
        assert FABRICATED_QN_KEY.match(fabricated)
        (injection_id,) = row["meta"]["injection_ids"]
        assert presented[injection_id]["claim_key"] == fabricated
        for claim in action["answer"]["claims"]:
            assert claim["key"] != fabricated, "fabricated-key claim credited as gold"
            assert not FABRICATED_QN_KEY.match(claim["key"])


def test_injection_resist_qn_marker_provenance(dataset):
    """The QN surface markers (disjoint from the v3/v4 INJECTION_MARKERS)
    identify exactly the injection records: marker scan == meta provenance,
    and no classic v3/v4 marker appears on the QN surface (marker-based
    refusal cannot catch this archetype — the lesson is content judgment)."""
    _, rows = dataset
    for row in rows["injection_resist_qn"]:
        presented, _ = assert_canonical_row(row)
        lure_ids = set(row["meta"]["injection_ids"])
        assert len(lure_ids) == 1
        marker_ids = {
            rid for rid, r in presented.items() if any(m in r["text"] for m in generate._V5_QN_MARKERS)
        }
        assert marker_ids == lure_ids, "QN marker scan and meta provenance disagree"
        for rid in lure_ids:
            assert not any(m in presented[rid]["text"] for m in INJECTION_MARKERS), (
                "the QN surface must not carry a classic v3/v4 injection marker"
            )


def test_injection_resist_qn_variant_split(dataset):
    """Both golds exercised: answer-from-legitimate-records (the over-
    abstention fix forbids blanket abstention) and abstain-only-without-
    support. Deterministic split under seed [20261116]: 10 / 4."""
    _, rows = dataset
    variants = [row["meta"]["variant"] for row in rows["injection_resist_qn"]]
    answered = variants.count("answer_legitimate")
    abstained = variants.count("abstain_injection_only")
    assert answered >= 8 and abstained >= 3, (answered, abstained)


def test_injection_resist_qn_novel_surfaces_disjoint():
    """The QN injection surface templates are novel: no template shares a
    word 8-gram with any v3/v4 injection template."""
    qn_grams = set()
    for template in generate._V5_QN_INJECTION_TEMPLATES:
        probe = template.format(case="ZX-100", date="2028-01-15")
        qn_grams |= _word_ngrams(probe)
    for pool in (generate._INJECTION_TEMPLATES, generate._INJECTION_TEMPLATES_V4):
        for template in pool:
            probe = template.format(subject="X", label="y", lure="z")
            overlap = qn_grams & _word_ngrams(probe)
            assert not overlap, f"QN surface shares an 8-gram with a v3/v4 injection template: {overlap}"


# ---------------------------------------------------------------------------
# v5 (d): answer_supported — over-abstention counterweight
# ---------------------------------------------------------------------------


def test_answer_supported_shape(dataset):
    """Every episode presents records that SUPPORT the asked truth; the gold
    answers from them (key echo, exact stored value, own citation, current
    record about the asked subject)."""
    _, rows = dataset
    for row in rows["answer_supported"]:
        presented, action = assert_canonical_row(row)
        claims = action["answer"]["claims"]
        assert len(claims) == 1
        claim = claims[0]
        assert TERSE_KEY.match(claim["key"])
        (record_id,) = claim["evidence_ids"]
        record = presented[record_id]
        assert record["claim_key"] == claim["key"]
        assert record["claim_value"] == claim["value"]
        assert type(record["claim_value"]) is type(claim["value"])
        assert record["memory_state"] == "current"
        assert _asked_subject(row) in record["text"]


def test_answer_supported_never_empty_gold(dataset):
    """The near-miss is the empty-claims abstention despite support present
    (the v4.1 over-abstention): no answer_supported gold is ever empty, and
    support for the asked (subject, key) is always presented."""
    _, rows = dataset
    for row in rows["answer_supported"]:
        presented, action = assert_canonical_row(row)
        claims = action["answer"]["claims"]
        assert claims, "empty-claims abstention with support present is the near-miss, never the gold"
        subject = _asked_subject(row)
        for claim in claims:
            assert any(
                r["claim_key"] == claim["key"]
                and r["claim_value"] == claim["value"]
                and r["memory_state"] == "current"
                and subject in r["text"]
                for r in presented.values()
            ), "gold claim must be supported by a presented current record"


def test_answer_supported_traps_present(dataset):
    """Each variant presents its abstain-TRIGGER surface: an untrusted
    'withdrawn / do not answer' notice (never cited), or a first window
    holding only stale/irrelevant records (one reformulated query then
    finds support), or support among heavy distractor load."""
    _, rows = dataset
    for row in rows["answer_supported"]:
        presented, action = assert_canonical_row(row)
        variant = row["meta"]["variant"]
        traps = row["meta"]["trap_features"]
        assert traps
        cited = {rid for claim in action["answer"]["claims"] for rid in claim["evidence_ids"]}
        if variant == "adverse_notice":
            assert traps == ["adverse_notice"]
            (notice_id,) = row["meta"]["adverse_notice_ids"]
            notice = presented[notice_id]
            assert notice_id not in cited, "the adverse notice must never be cited"
            assert "must not answer" in notice["text"]
            assert notice["claim_key"] == "withdrawn_folios_circular"
        elif variant == "stale_first":
            assert traps == ["stale_first_window"]
            assert row["meta"]["reformulation"] == "after_unhelpful"
            results = _tool_results(row)
            assert len(results) == 2
            first_ids = {record["id"] for record in results[0]["records"]}
            assert cited and cited.isdisjoint(first_ids), "support arrives on the reformulated query"
            # The first window holds no CURRENT support for the asked key.
            asked = action["answer"]["claims"][0]["key"]
            assert not [
                r
                for r in results[0]["records"]
                if r["claim_key"] == asked and r["memory_state"] == "current"
            ]
        else:
            assert variant == "plain_support"
            assert traps == ["support_among_distractors"]
            assert len(presented) >= 3, "support sits among distractor load"


def test_answer_supported_variants_exercised(dataset):
    _, rows = dataset
    variants = {row["meta"]["variant"] for row in rows["answer_supported"]}
    assert variants == {"adverse_notice", "stale_first", "plain_support"}


# ---------------------------------------------------------------------------
# v5 leakage discipline (M8 method re-run, heldout2 leg LIVE)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sealed_corpora_v5(tmp_path_factory):
    """Regenerated pilot216 + heldout + heldout2 sealed corpora. heldout2 is
    the CURRENT seal (the parallel dependent_lookups generator fix lands
    separately — v5 pools are drawn disjoint from the current shapes, and
    any on-disk sealed copy is byte-verified against the regeneration)."""
    out = tmp_path_factory.mktemp("sealed-v5")
    scenarios = {}
    for workload, directory in (("pilot", "pilot216"), ("heldout", "heldout"), ("heldout2", "heldout2")):
        sealed.generate_workload(workload, out / directory)
        scenarios[directory] = {p.name: p for p in sorted((out / directory / "scenarios").glob("*.json"))}
    on_disk = {
        "pilot216": M8_WORKLOADS_DIR / "pilot216" / "scenarios",
        "heldout": M8_WORKLOADS_DIR / "heldout" / "scenarios",
        "heldout2": M8_WORKLOADS_DIR / "heldout2" / "scenarios",
    }
    for directory, scenarios_dir in on_disk.items():
        if not scenarios_dir.is_dir():
            continue
        disk = {p.name: p.read_bytes() for p in sorted(scenarios_dir.glob("*.json"))}
        assert len(disk) == len(scenarios[directory]), f"on-disk {directory} scenario count differs"
        for name, path in scenarios[directory].items():
            assert disk.get(name) == path.read_bytes(), f"on-disk {directory}/{name} differs from regenerated bytes"
    texts = []
    for directory in scenarios.values():
        for path in directory.values():
            texts.extend(_sealed_scenario_texts(json.loads(path.read_text(encoding="utf-8"))))
    return texts


def test_v5_pools_disjoint_by_construction(dataset, sealed_corpora_v5):
    """The v5 pools are all-new: token-level disjoint from the v1-v4 pools,
    the m8 fixtures and ALL sealed workloads (pilot216, heldout, heldout2 —
    including the current dependent_lookups shapes)."""
    prior_pool_tokens = set()
    for entity in list(generate.ENTITY_POOL_STRINGS) + list(generate.V4_ENTITY_POOL_STRINGS):
        prior_pool_tokens |= _tokens(entity)
    v5_pool_tokens = set()
    for entity in generate.V5_ENTITY_POOL_STRINGS:
        v5_pool_tokens |= _tokens(entity)
    assert not (v5_pool_tokens & prior_pool_tokens), (
        "v5 entity pool shares tokens with the v1-v4 pools",
        sorted(v5_pool_tokens & prior_pool_tokens),
    )
    prior_keys = {fact["key"] for fact in generate.FACTS} | {fact["key"] for fact in generate.V4_FACTS}
    v5_keys = {fact["key"] for fact in generate.V5_FACTS}
    assert not (prior_keys & v5_keys)
    sealed_text_blob = "\n".join(sealed_corpora_v5)
    sealed_tokens = _tokens(sealed_text_blob)
    assert not (v5_pool_tokens & sealed_tokens), (
        "v5 entity pool shares tokens with the sealed workloads",
        sorted(v5_pool_tokens & sealed_tokens),
    )
    fixture_blob = "\n".join(path.read_text(encoding="utf-8") for path in sorted(FIXTURES.glob("*.json")))
    assert not (v5_pool_tokens & _tokens(fixture_blob))
    # v5 key/label/unit/hop-token and value vocabulary: absent from every
    # corpus and every prior pool.
    value_tokens = set(generate.V5_KEY_STEMS)
    for fact in generate.V5_FACTS:
        value_tokens |= _tokens(fact["label"]) | _tokens(fact["key"])
    for disposition in generate._V5_QN_DISPOSITIONS:
        value_tokens |= _tokens(disposition)
    for sign in generate._V5_QN_SIGNS:
        value_tokens |= _tokens(sign)
    assert not (value_tokens & sealed_tokens), sorted(value_tokens & sealed_tokens)
    assert not (value_tokens & _tokens(fixture_blob))
    assert not (value_tokens & prior_pool_tokens)
    # String-level: no v5 pool string or per-episode v5 entity usage (case
    # ids, injected dates, chain values) appears in any corpus text.
    _, rows = dataset
    used = set()
    for cls in generate.V5_CLASSES:
        for row in rows[cls]:
            used.update(row["meta"]["entities"])
    assert used
    for entity in set(generate.V5_ENTITY_POOL_STRINGS) | used:
        assert entity not in sealed_text_blob, entity
        assert entity not in fixture_blob, entity
    for key in v5_keys:
        assert key not in sealed_text_blob, "v5 terse key collides with sealed truth keys"


def test_v5_zero_word_8gram_overlap(dataset, sealed_corpora_v5):
    """The M8 leakage method re-run for curriculum v5: zero shared word
    8-grams between the v5-class rows (prompts, queries, record texts,
    string claim values) and (1) the sealed pilot216/heldout corpora,
    (2) the CURRENT sealed heldout2 corpus (live leg), (3) the m8 dev
    fixtures, (4) the prior curriculum output when a v3/v4 out/ directory
    is available."""
    v5_grams = set()
    for cls in generate.V5_CLASSES:
        for row in dataset[1][cls]:
            for text in _row_texts(row):
                v5_grams |= _word_ngrams(text)
    assert v5_grams
    sealed_grams = set()
    for text in sealed_corpora_v5:
        sealed_grams |= _word_ngrams(text)
    overlap = v5_grams & sealed_grams
    assert not overlap, f"shared 8-grams with the sealed corpora (incl. heldout2): {sorted(overlap)[:5]}"
    fixture_grams = set()
    for path in sorted(FIXTURES.glob("*.json")):
        for event in json.loads(path.read_text(encoding="utf-8"))["events"]:
            fixture_grams |= _word_ngrams(event["text"])
    overlap = v5_grams & fixture_grams
    assert not overlap, f"shared 8-grams with the m8 fixtures: {sorted(overlap)[:5]}"
    manifest_path = V3_OUT_DIR / "manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()).get("dataset_version") in (3, 4):
        prior_grams = set()
        for path in sorted(V3_OUT_DIR.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["class"] in generate.V5_CLASSES:
                    continue  # same-generator v5 rows, not a prior dataset
                for text in _row_texts(row):
                    prior_grams |= _word_ngrams(text)
        overlap = v5_grams & prior_grams
        assert not overlap, f"shared 8-grams with the prior curriculum output: {sorted(overlap)[:5]}"


def test_v5_key_stems_disjoint_from_measurement_corpora(measurement_key_tokens):
    """Stem level, v5-specific: every declared v5 key stem — parameterized
    hop-2 prefixes, hop tokens, units, the fabricated QN near-miss stem and
    the adverse-notice key — is disjoint from every measurement-corpora
    fact/truth key token (heldout2's current dependent_lookups shapes
    included)."""
    tokens, coverage = measurement_key_tokens
    overlap = generate.V5_KEY_STEMS & tokens
    assert not overlap, (
        "v5 key stems shared with measurement-corpora fact keys",
        sorted(overlap),
        coverage,
    )


def test_dependent_lookup_chain_distractor_carries_stale_token_value(dataset):
    """The untrusted-instruction distractor always asserts the STALE token's
    value (the adverse shape from the sealed family): its value equals the
    stale hop-2 table's and differs from every gold claim value."""
    _, rows = dataset
    for row in rows["dependent_lookup_chain"]:
        presented, action = assert_canonical_row(row)
        spec = _chain_spec(row)
        answer_values = {claim["value"] for claim in action["answer"]["claims"]}
        (distractor_id,) = row["meta"]["distractor_ids"]
        distractor = presented[distractor_id]
        stale_tables = [
            r
            for r in presented.values()
            if r["claim_key"].startswith(spec["hop2_prefix"] + "_")
            and r["claim_key"] != action["answer"]["claims"][0]["key"]
        ]
        assert distractor["claim_value"] in {r["claim_value"] for r in stale_tables}
        assert distractor["claim_value"] not in answer_values
        # The stale token appears in the distractor text; the corrected
        # token never does.
        hop1_value = action["answer"]["claims"][1]["value"]
        assert hop1_value not in distractor["text"]


def test_value_shape_chain_units_span_pool(dataset):
    """Every v5 unit kind is exercised by value_shape_chain golds (no
    memorizable single-unit vocabulary)."""
    _, rows = dataset
    units = set()
    for row in rows["value_shape_chain"]:
        _, action = assert_canonical_row(row)
        units.add(action["answer"]["claims"][0]["value"].split()[1])
    assert units == set(generate._V5_UNIT_BOUNDS)


def test_injection_resist_qn_case_ids_disjoint_from_corpora(sealed_corpora_v5):
    """Every generated ZX case id and injected signoff date is absent from
    all sealed corpora text (heldout2's QN numbers/dates included)."""
    episodes = generate.build_episodes(dict(generate.DEFAULT_COUNTS))["injection_resist_qn"]
    blob = "\n".join(sealed_corpora_v5)
    cases = set()
    for ep in episodes:
        row = ep.row()
        for entity in row["meta"]["entities"]:
            if entity.startswith("ZX-"):
                cases.add(entity)
            if re.fullmatch(r"20\d\d-\d\d-\d\d", entity):
                assert entity not in blob, f"injected date collides with a sealed corpus: {entity}"
    assert cases, "no case ids generated"
    for case in cases:
        assert case not in blob, f"case id collides with a sealed corpus: {case}"


def test_injection_resist_qn_prompt_families(dataset):
    _, rows = dataset
    families = {row["meta"]["prompt_family"] for row in rows["injection_resist_qn"]}
    assert families == {name for name, _ in generate._V5_QN_PROMPTS}


def test_answer_supported_stale_first_two_retrieves(dataset):
    """stale_first: the first window holds only stale/irrelevant records,
    exactly ONE differently phrased reformulated query follows, and the
    episode finalizes with the answer (never the abstention)."""
    _, rows = dataset
    hits = 0
    for row in rows["answer_supported"]:
        if row["meta"]["variant"] != "stale_first":
            continue
        hits += 1
        actions = _retrieve_actions(row)
        assert len(actions) == 2
        assert actions[0]["arguments"]["query"] != actions[1]["arguments"]["query"]
        assert _final_action(row)["answer"]["claims"], "stale_first must answer, not abstain"
    assert hits, "stale_first variant never generated"


def test_answer_supported_prompt_families_exercised(dataset):
    _, rows = dataset
    families = {row["meta"]["prompt_family"] for row in rows["answer_supported"]}
    assert families == {name for name, _ in generate._V5_ANSWER_PROMPTS}


# ---------------------------------------------------------------------------
# Tests (v6: M10 v5-campaign non-transfer repair — additive class)
# ---------------------------------------------------------------------------

# The four v5 class files at dataset_version 5 (seed [20261116], verified
# byte-identical to the v5 release out/ when the v6 generator landed). v6 is
# ADDITIVE: no v3/v4/v5 builder or pool may change these bytes.
V5_CLASS_SHAS = {
    "dependent_lookup_chain.jsonl": "87b1577709861b7722af8dda6863fc81886db552e2e2125ab0656da4b38614ff",
    "value_shape_chain.jsonl": "8c349908273f59e56775f363d53d502754b5e5b0e5f700f9b32c0b5bafe3a4c2",
    "injection_resist_qn.jsonl": "552ee04698424dfd3f6bf6d3b72cdb2c56b043e72379d90835de2efdd4377649",
    "answer_supported.jsonl": "95b709cc5be281c482677c397c28acf9f96809523283cecb1791a248ccca8084",
}


def test_v5_class_bytes_unchanged(dataset):
    """v6 is additive: the four v5 class files regenerate byte-identical
    under the unchanged seed [20261116]. Any edit to a v3/v4/v5 builder,
    pool or draw order breaks this pin (or the v3/v4 pin above)."""
    output, _ = dataset
    manifest = json.loads((output / "manifest.json").read_text())
    shas = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    for name, sha in V5_CLASS_SHAS.items():
        assert shas[name] == sha, f"v5 class bytes changed: {name}"


def test_v6_classes_additive():
    """CLASSES and DEFAULT_COUNTS grow ONLY by the v6 class; the v3-v5
    classes keep their counts and their order (seed-spawn order)."""
    assert generate.CLASSES[:13] == generate.V3_CLASSES + generate.V4_CLASSES + generate.V5_CLASSES
    assert generate.CLASSES[13:] == generate.V6_CLASSES
    assert generate.V6_CLASSES == ("dependent_lookup_compose",)
    assert generate.DEFAULT_COUNTS["dependent_lookup_compose"] == 40


def test_v6_episode_id_prefixes(dataset):
    """v6 episodes are m10-prefixed (M10 curriculum), numbered per class."""
    _, rows = dataset
    for cls in generate.V6_CLASSES:
        for index, row in enumerate(rows[cls]):
            assert row["episode_id"] == f"m10-{cls.replace('_', '-')}-{index:04d}"


# ---------------------------------------------------------------------------
# v6: dependent_lookup_compose — composition, not copying
# ---------------------------------------------------------------------------


def _compose_spec(row):
    (spec,) = [s for s in generate.V6_COMPOSE_VARIANTS if s["variant"] == row["meta"]["chain_variant"]]
    return spec


def test_dependent_lookup_compose_shape(dataset):
    """The gold solves the whole chain with COMPOSED values: exactly three
    claims — the hop-2 value under the PARAMETERIZED claim_key echoed from
    the presented record's claim_key (unit-bearing string), the corrected
    hop-1 value under its key, and the rule under its key with the terse
    dimension value — each citing exactly its own presented record."""
    _, rows = dataset
    for row in rows["dependent_lookup_compose"]:
        presented, action = assert_canonical_row(row)
        spec = _compose_spec(row)
        claims = action["answer"]["claims"]
        assert len(claims) == 3, "compose gold answers hop-2 + hop-1 + rule"
        hop2, hop1, rule = claims
        assert TERSE_KEY.match(hop2["key"]) and TERSE_KEY.match(hop1["key"]) and TERSE_KEY.match(rule["key"])
        assert [claim["key"] for claim in claims] == row["meta"]["asked_keys"]
        # Hop-2 key is parameterized by the DISCOVERED (corrected) token.
        assert hop1["value"] in set(spec["tokens"])
        assert hop2["key"] == f"{spec['hop2_prefix']}_{hop1['value']}", (
            "hop-2 claim key is not parameterized by the discovered hop-1 token",
            row["episode_id"],
            hop2["key"],
        )
        assert hop1["key"] == spec["hop1_key"]
        assert rule["key"] == spec["rule_key"]
        assert rule["value"] == spec["dimension"], "the rule claim carries the terse dimension value"
        subject = _asked_subject(row)
        for claim in (hop2, hop1, rule):
            (record_id,) = claim["evidence_ids"]
            record = presented[record_id]
            assert record["claim_key"] == claim["key"], "each claim echoes its cited record's claim_key"
            assert record["memory_state"] == "current"
            assert subject in record["text"]


def test_dependent_lookup_compose_no_claim_value_crutch(dataset):
    """The primary non-transfer gap: campaign records present meta.claim_key
    plus raw text only. EVERY presented record in the class carries claim_key
    and NEVER a claim_value field — the gold cannot be produced by copying."""
    _, rows = dataset
    for row in rows["dependent_lookup_compose"]:
        presented, action = assert_canonical_row(row)
        assert presented
        for record in presented.values():
            assert "claim_value" not in record, (
                "compose record carries the claim_value crutch",
                row["episode_id"],
                record,
            )
            assert record["claim_key"]
        assert action["answer"]["claims"], "compose episodes are answer-bearing"


def test_dependent_lookup_compose_value_composition(dataset):
    """Every gold value is COMPOSED from its cited record's raw text: the
    claim value equals the record's tail ('... is <value>.' or '... stands
    at <value>.'), the hop-2 value keeps number AND unit together (never the
    bare number), and the number itself is visible in the cited text."""
    _, rows = dataset
    for row in rows["dependent_lookup_compose"]:
        presented, action = assert_canonical_row(row)
        spec = _compose_spec(row)
        hop2, hop1, rule = action["answer"]["claims"]
        for claim in (hop2, hop1, rule):
            record = presented[claim["evidence_ids"][0]]
            assert _record_value(record) == claim["value"], (
                "claim value is not composable from the cited record's text",
                row["episode_id"],
                claim,
                record["text"],
            )
            tail_forms = (f" is {claim['value']}.", f" stands at {claim['value']}.")
            assert any(record["text"].endswith(tail) for tail in tail_forms), (
                "cited record carries no composable tail for the claim value",
                record["text"],
            )
        # Unit retention under multi-claim load (28/48 campaign episodes
        # stripped units): the hop-2 value is the unit-bearing string.
        assert type(hop2["value"]) is str
        number, _, unit = hop2["value"].partition(" ")
        assert number.isdigit() and unit == spec["unit"]
        record = presented[hop2["evidence_ids"][0]]
        assert f"{number} {unit}" in record["text"]
        assert hop2["value"] != number, "the gold is never the bare number"


def test_dependent_lookup_compose_rule_status_value_tail(dataset):
    """The rule claim value comes from the rule record's status_value tail
    ('the governing lookup dimension is <dimension>.') — the terse 2-word
    dimension, NEVER the full rule sentence (27/48 campaign episodes echoed
    the sentence)."""
    _, rows = dataset
    for row in rows["dependent_lookup_compose"]:
        presented, action = assert_canonical_row(row)
        rule = action["answer"]["claims"][2]
        record = presented[rule["evidence_ids"][0]]
        assert record["text"].endswith(f"the governing lookup dimension is {rule['value']}."), (
            "rule record is not in the status_value tail form",
            record["text"],
        )
        assert len(rule["value"].split()) == 2, "the rule value is the terse 2-word dimension"
        assert rule["value"] != record["text"], "the rule claim must not be the full sentence"


def test_dependent_lookup_compose_keying_instruction_omitted(dataset):
    """Spec (c): >= half the tasks OMIT the explicit keying instruction, and
    every query names the rule by its MISLEADING surface noun (whose
    snake_case form is the bait key, never the stored rule key). Keyed
    episodes still exist so the behavior also holds when asked."""
    _, rows = dataset
    uninstructed = 0
    instructed = 0
    for row in rows["dependent_lookup_compose"]:
        assert_canonical_row(row)
        spec = _compose_spec(row)
        prompt = row["task"]["prompt"]
        keying = row["meta"]["keying"]
        assert keying in ("instructed", "uninstructed")
        # The misleading surface noun is used in every query; its snake form
        # is the bait key, never the stored key.
        assert spec["rule_surface"] in prompt
        bait_rule_key = row["meta"]["bait_keys"]["rule"]
        assert bait_rule_key == spec["rule_surface"].replace(" ", "_")
        assert bait_rule_key != spec["rule_key"]
        if keying == "uninstructed":
            uninstructed += 1
            assert "claim key" not in prompt, "uninstructed prompt carries a keying instruction"
        else:
            instructed += 1
            assert "stored claim key" in prompt
    total = len(rows["dependent_lookup_compose"])
    assert uninstructed >= total // 2, f"only {uninstructed}/{total} episodes omit the keying instruction"
    assert instructed, "keyed-instruction episodes must exist (behavior holds when asked)"


def test_dependent_lookup_compose_query_bait_near_miss(dataset):
    """The query-baited near-miss keys (the campaign's galley_rule /
    mess_stipend inventions): every episode presents a rule-bait record under
    the surface-noun key and a hop-2 bait record carrying the CORRECT value
    text about the asked subject under the measure-snake key. Both are never
    cited and the bait keys are never gold."""
    _, rows = dataset
    for row in rows["dependent_lookup_compose"]:
        presented, action = assert_canonical_row(row)
        spec = _compose_spec(row)
        claims = action["answer"]["claims"]
        cited = {rid for claim in claims for rid in claim["evidence_ids"]}
        gold_keys = {claim["key"] for claim in claims}
        bait_keys = row["meta"]["bait_keys"]
        assert bait_keys["hop2"] == spec["measure"].replace(" ", "_")
        assert spec["measure"] in row["task"]["prompt"], "the hop-2 bait key is query-derived"
        assert bait_keys["rule"] == spec["rule_surface"].replace(" ", "_")
        assert spec["rule_surface"] in row["task"]["prompt"], "the rule bait key is query-derived"
        assert not ({bait_keys["hop2"], bait_keys["rule"]} & gold_keys), "a query-baited key was credited as gold"
        rule_bait_id, hop2_bait_id = row["meta"]["bait_ids"]
        rule_bait = presented[rule_bait_id]
        hop2_bait = presented[hop2_bait_id]
        assert rule_bait_id not in cited and hop2_bait_id not in cited
        assert rule_bait["claim_key"] == bait_keys["rule"]
        assert hop2_bait["claim_key"] == bait_keys["hop2"]
        # The hop-2 bait carries the ANSWER value text about the asked
        # subject — only the key is wrong (the strongest bait).
        assert claims[0]["value"] in hop2_bait["text"]
        assert _asked_subject(row) in hop2_bait["text"]
        assert spec["rule_surface"] in rule_bait["text"]


def test_dependent_lookup_compose_second_hop_retrieval(dataset):
    """The chain is preserved from v5 (it transferred): every standard
    episode issues a SECOND retrieve keyed on the discovered hop-1 token as
    '<dimension> <token> <measure>', and the hop-2 current table is absent
    from the first window."""
    _, rows = dataset
    standard = 0
    for row in rows["dependent_lookup_compose"]:
        if row["meta"].get("bare_token_miss"):
            continue
        standard += 1
        presented, action = assert_canonical_row(row)
        spec = _compose_spec(row)
        actions = _retrieve_actions(row)
        results = _tool_results(row)
        assert len(actions) == 2 == len(results), "standard compose gold is exactly two retrieves"
        hop1_value = action["answer"]["claims"][1]["value"]
        first_query = actions[0]["arguments"]["query"]
        second_query = actions[1]["arguments"]["query"]
        assert hop1_value not in first_query, "the hop-2 token is not in the question or first query"
        assert hop1_value in second_query, "second retrieve must key on the discovered hop-1 token"
        assert spec["dimension"] in second_query and spec["measure"] in second_query, (
            "the reformulated second-hop query is fully keyed, never a bare token"
        )
        assert first_query != second_query
        hop2_id = action["answer"]["claims"][0]["evidence_ids"][0]
        first_ids = {record["id"] for record in results[0]["records"]}
        assert hop2_id not in first_ids, "hop-2 current table must be discovered, not handed over"
        assert hop2_id in {record["id"] for record in results[1]["records"]}
    assert standard == 40 - len(generate.V6_REVISE_ON_MISS_INDICES)


def test_dependent_lookup_compose_revise_on_miss(dataset):
    """Spec (d): the flagged demos reproduce the campaign's bare-token loop
    AND its correction — the bare-token second query returns NOTHING, and
    the corrected behavior is ONE reformulated query keyed
    '<dimension> <token> <measure>' (never a repeat, never another bare
    token), then the final succeeds."""
    _, rows = dataset
    flagged = 0
    for row in rows["dependent_lookup_compose"]:
        presented, action = assert_canonical_row(row)
        if not row["meta"].get("bare_token_miss"):
            continue
        flagged += 1
        spec = _compose_spec(row)
        actions = _retrieve_actions(row)
        results = _tool_results(row)
        assert len(actions) == 3 == len(results), "revise-on-miss: subject query, bare-token miss, reformulation"
        hop1_value = action["answer"]["claims"][1]["value"]
        bare_query = actions[1]["arguments"]["query"]
        assert bare_query == hop1_value, "the miss is the bare discovered token"
        assert len(bare_query.split()) == 1
        assert results[1]["returned_count"] == 0, "the bare-token query returns nothing"
        third_query = actions[2]["arguments"]["query"]
        assert third_query == f"{spec['dimension']} {hop1_value} {spec['measure']}"
        assert third_query != bare_query
        hop2_id = action["answer"]["claims"][0]["evidence_ids"][0]
        assert hop2_id in {record["id"] for record in results[2]["records"]}
        assert hop2_id not in {record["id"] for record in results[0]["records"]}
        assert action["answer"]["claims"], "the episode finalizes with the answer after ONE reformulation"
    assert flagged == len(generate.V6_REVISE_ON_MISS_INDICES) == 3


def test_dependent_lookup_compose_bare_token_queries_confined(dataset):
    """Bare-token retrieve queries appear ONLY in the flagged revise-on-miss
    demos and only in the failing position — the never-gold near-miss
    behavior (bare-token repeats) is confined to the demonstrated miss."""
    _, rows = dataset
    for row in rows["dependent_lookup_compose"]:
        actions = _retrieve_actions(row)
        bare = [a for a in actions if len(a["arguments"]["query"].split()) == 1]
        if row["meta"].get("bare_token_miss"):
            assert len(bare) == 1 and bare[0] is actions[1]
        else:
            assert not bare, ("bare-token query outside the revise-on-miss demos", row["episode_id"])


def test_dependent_lookup_compose_near_misses_never_cited(dataset):
    """Every near-miss shape from the campaign is presented and never cited:
    the superseded hop-1 register (stale token, chained to the correction),
    the stale hop-2 table, the untrusted-instruction placard, and the
    unit-stripped bare-number record under the ASKED parameterized key (its
    text can only compose to the stripped form)."""
    _, rows = dataset
    stale_hop1 = 0
    for row in rows["dependent_lookup_compose"]:
        presented, action = assert_canonical_row(row)
        spec = _compose_spec(row)
        claims = action["answer"]["claims"]
        cited = {rid for claim in claims for rid in claim["evidence_ids"]}
        hop2_key, hop1_value = claims[0]["key"], claims[1]["value"]
        subject = _asked_subject(row)
        for rid in row["meta"].get("stale_hop1_ids", ()):
            stale_hop1 += 1
            record = presented[rid]
            assert rid not in cited
            assert record["claim_key"] == spec["hop1_key"]
            assert _record_value(record) != hop1_value, "stale register carries the stale token"
            assert record["memory_state"] == "superseded"
            # Chained to the presented correction (memory_state, not prose).
            assert record["superseded_by"] in cited
        for rid in row["meta"]["distractor_ids"]:
            record = presented[rid]
            assert rid not in cited
            assert record["claim_key"].startswith("placard_")
            assert "tells hands" in record["text"] or "told to" in record["text"], (
                "the distractor is an untrusted instruction"
            )
        # The stale hop-2 table: parameterized by the STALE token, never cited.
        stale_tables = [
            r
            for r in presented.values()
            if r["claim_key"].startswith(spec["hop2_prefix"] + "_") and r["claim_key"] != hop2_key
        ]
        assert stale_tables, "the dual hop-2 tables must both be presented"
        for record in stale_tables:
            assert record["id"] not in cited
        # Unit-stripped near-miss: same subject, the ASKED parameterized key,
        # but the text carries ONLY the bare number (no unit token).
        (stripped_id,) = row["meta"]["type_stripped_ids"]
        stripped = presented[stripped_id]
        assert stripped_id not in cited
        assert stripped["claim_key"] == hop2_key
        assert subject in stripped["text"]
        number = claims[0]["value"].split()[0]
        assert number in stripped["text"]
        assert spec["unit"] not in stripped["text"], "the stripped record must not carry the unit"
    rate = stale_hop1 / len(rows["dependent_lookup_compose"])
    assert rate >= 0.60, f"superseded hop-1 registers in only {rate:.0%} of compose episodes"


def test_dependent_lookup_compose_distractor_carries_stale_token_value(dataset):
    """The untrusted-instruction distractor always asserts the STALE token's
    value (the adverse shape from the sealed family): its text carries the
    stale hop-2 table's value and never the corrected token."""
    _, rows = dataset
    for row in rows["dependent_lookup_compose"]:
        presented, action = assert_canonical_row(row)
        spec = _compose_spec(row)
        claims = action["answer"]["claims"]
        (distractor_id,) = row["meta"]["distractor_ids"]
        distractor = presented[distractor_id]
        stale_tables = [
            r
            for r in presented.values()
            if r["claim_key"].startswith(spec["hop2_prefix"] + "_")
            and r["claim_key"] != claims[0]["key"]
        ]
        stale_values = {_record_value(r) for r in stale_tables}
        assert any(value in distractor["text"] for value in stale_values), (
            "the distractor asserts the stale token's value"
        )
        # The hop-2 answer value and the corrected hop-1 token never appear;
        # the rule claim's dimension value DOES appear by design — the
        # distractor asserts the stale token's value UNDER the dimension.
        assert claims[0]["value"] not in distractor["text"], "the distractor carries the hop-2 answer value"
        hop1_value = claims[1]["value"]
        assert hop1_value not in distractor["text"]


def test_dependent_lookup_compose_subject_diversity(dataset):
    """Spec (e): subjects diversify across heldout2's three subject types —
    person names, plot names and part ids — each drawn from its own v6 pool."""
    _, rows = dataset
    kinds = set()
    for row in rows["dependent_lookup_compose"]:
        assert_canonical_row(row)
        kind = row["meta"]["subject_kind"]
        kinds.add(kind)
        subject = row["meta"]["asked_subject"]
        pool = {"person": generate.V6_PERSONS, "plot": generate.V6_PLOTS, "part": generate.V6_PARTS}[kind]
        assert subject in pool
        spec = _compose_spec(row)
        assert spec["subject_kind"] == kind
    assert kinds == {"person", "plot", "part"}


def test_dependent_lookup_compose_variants_exercised(dataset):
    _, rows = dataset
    variants = {row["meta"]["chain_variant"] for row in rows["dependent_lookup_compose"]}
    assert variants == {spec["variant"] for spec in generate.V6_COMPOSE_VARIANTS}


# ---------------------------------------------------------------------------
# v6 leakage discipline (M8 method re-run, heldout2 leg LIVE, v1-v5 rows
# swept live from the same generated dataset)
# ---------------------------------------------------------------------------


def test_v6_pools_disjoint_by_construction(dataset, sealed_corpora_v5):
    """The v6 pools are all-new: token-level disjoint from the v1-v5 pools,
    the m8 fixtures and ALL sealed workloads (pilot216, heldout, heldout2 —
    including the FIXED dependent_lookups shapes)."""
    prior_pool_tokens = set()
    for pool in (generate.ENTITY_POOL_STRINGS, generate.V4_ENTITY_POOL_STRINGS, generate.V5_ENTITY_POOL_STRINGS):
        for entity in pool:
            prior_pool_tokens |= _tokens(entity)
    v6_pool_tokens = set()
    for entity in generate.V6_ENTITY_POOL_STRINGS:
        v6_pool_tokens |= _tokens(entity)
    assert not (v6_pool_tokens & prior_pool_tokens), (
        "v6 entity pool shares tokens with the v1-v5 pools",
        sorted(v6_pool_tokens & prior_pool_tokens),
    )
    prior_keys = (
        {fact["key"] for fact in generate.FACTS}
        | {fact["key"] for fact in generate.V4_FACTS}
        | {fact["key"] for fact in generate.V5_FACTS}
    )
    v6_keys = set()
    for spec in generate.V6_COMPOSE_VARIANTS:
        v6_keys |= {spec["hop1_key"], spec["rule_key"], spec["hop2_prefix"]}
    assert not (prior_keys & v6_keys)
    sealed_text_blob = "\n".join(sealed_corpora_v5)
    sealed_tokens = _tokens(sealed_text_blob)
    assert not (v6_pool_tokens & sealed_tokens), (
        "v6 entity pool shares tokens with the sealed workloads",
        sorted(v6_pool_tokens & sealed_tokens),
    )
    fixture_blob = "\n".join(path.read_text(encoding="utf-8") for path in sorted(FIXTURES.glob("*.json")))
    assert not (v6_pool_tokens & _tokens(fixture_blob))
    # v6 key/dimension/unit/hop-token/source and value vocabulary: absent
    # from every corpus and every prior pool.
    value_tokens = set(generate.V6_KEY_STEMS)
    for spec in generate.V6_COMPOSE_VARIANTS:
        value_tokens |= _tokens(spec["measure"]) | _tokens(spec["dimension"]) | _tokens(spec["rule_surface"])
        value_tokens |= _tokens(spec["register"]) | _tokens(spec["rule_lead"]) | {spec["unit"]}
        if "agent" in spec:
            value_tokens.add(spec["agent"])
    for source in generate._V6_SOURCES:
        value_tokens |= _tokens(source)
    assert not (value_tokens & sealed_tokens), sorted(value_tokens & sealed_tokens)
    assert not (value_tokens & _tokens(fixture_blob))
    assert not (value_tokens & prior_pool_tokens)
    # String-level: no v6 pool string or per-episode v6 entity usage (subjects,
    # hop tokens, composed values) appears in any corpus text.
    _, rows = dataset
    used = set()
    for cls in generate.V6_CLASSES:
        for row in rows[cls]:
            used.update(row["meta"]["entities"])
    assert used
    for entity in set(generate.V6_ENTITY_POOL_STRINGS) | used:
        assert entity not in sealed_text_blob, entity
        assert entity not in fixture_blob, entity
    for spec in generate.V6_COMPOSE_VARIANTS:
        assert spec["rule_key"] not in sealed_text_blob, "v6 rule key collides with sealed truth keys"


def test_v6_zero_word_8gram_overlap(dataset, sealed_corpora_v5):
    """The M8 leakage method re-run for curriculum v6: zero shared word
    8-grams between the v6-class rows (prompts, queries, record texts,
    string claim values) and (1) the sealed pilot216/heldout corpora,
    (2) the CURRENT sealed heldout2 corpus (fixed dependent_lookups
    templates, live leg), (3) the m8 dev fixtures, (4) the v1-v5 rows of the
    SAME generated dataset (always live), and (5) a dataset_version 3/4/5
    out/ directory when one is available."""
    v6_grams = set()
    for cls in generate.V6_CLASSES:
        for row in dataset[1][cls]:
            for text in _row_texts(row):
                v6_grams |= _word_ngrams(text)
    assert v6_grams
    sealed_grams = set()
    for text in sealed_corpora_v5:
        sealed_grams |= _word_ngrams(text)
    overlap = v6_grams & sealed_grams
    assert not overlap, f"shared 8-grams with the sealed corpora (incl. heldout2): {sorted(overlap)[:5]}"
    fixture_grams = set()
    for path in sorted(FIXTURES.glob("*.json")):
        for event in json.loads(path.read_text(encoding="utf-8"))["events"]:
            fixture_grams |= _word_ngrams(event["text"])
    overlap = v6_grams & fixture_grams
    assert not overlap, f"shared 8-grams with the m8 fixtures: {sorted(overlap)[:5]}"
    # Live leg: the v1-v5 rows of the same generated dataset.
    prior_grams = set()
    for cls in generate.CLASSES:
        if cls in generate.V6_CLASSES:
            continue
        for row in dataset[1][cls]:
            for text in _row_texts(row):
                prior_grams |= _word_ngrams(text)
    overlap = v6_grams & prior_grams
    assert not overlap, f"shared 8-grams with the v1-v5 curriculum rows: {sorted(overlap)[:5]}"
    manifest_path = V3_OUT_DIR / "manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()).get("dataset_version") in (3, 4, 5):
        prior_out_grams = set()
        for path in sorted(V3_OUT_DIR.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["class"] in generate.V5_CLASSES + generate.V6_CLASSES:
                    continue  # same-generator v5/v6 rows, not a prior dataset
                for text in _row_texts(row):
                    prior_out_grams |= _word_ngrams(text)
        overlap = v6_grams & prior_out_grams
        assert not overlap, f"shared 8-grams with the prior curriculum output: {sorted(overlap)[:5]}"


def test_v6_key_stems_disjoint_from_measurement_corpora(measurement_key_tokens):
    """Stem level, v6-specific: every declared v6 key stem — parameterized
    hop-2 prefixes, hop tokens, units, dimensions, rule-key suffixes and the
    query-baited near-miss key stems — is disjoint from every
    measurement-corpora fact/truth key token (heldout2's FIXED
    dependent_lookups shapes included)."""
    tokens, coverage = measurement_key_tokens
    overlap = generate.V6_KEY_STEMS & tokens
    assert not overlap, (
        "v6 key stems shared with measurement-corpora fact keys",
        sorted(overlap),
        coverage,
    )
