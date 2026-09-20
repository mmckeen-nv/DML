"""Bounded oracle controls. Fabricated records here are unit-test inputs only."""
from copy import deepcopy
import hashlib
import json

import pytest

from daystrom_dml.services.episode_verifiers import (
    INTENTS, load_episode_corpus, validate_episode_corpus, verify_task,
)


def digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def case_inputs(intent="near_duplicates", task_index=0):
    corpus = load_episode_corpus()
    scenario = next(item for item in corpus["scenarios"] if item["id"] == intent)
    task = scenario["tasks"][task_index]
    seeds = {}
    for ident, seed in enumerate(scenario["seeds"]):
        seeds[seed["alias"]] = {
            "id": ident, "text": seed["text"], "embedding": [1.0, 0.0],
            "timestamp": 1900000000.0, "salience": 1.0, "fidelity": 1.0, "level": 0,
            "meta": {**seed["meta"], **seed.get("scope", scenario["scope"]), "kind": "memory"},
        }
    for operation in scenario["setup"]:
        if operation["operation"] == "set_salience":
            seeds[operation["alias"]]["salience"] = operation["value"]
        elif operation["operation"] == "related_writes":
            for update in operation["updates"]:
                seeds[update["alias"]]["text"] = update["text"]
    current = deepcopy(list(seeds.values()))
    answer = {"claims": [{"key": key, "value": fact["value"],
                          "evidence_ids": [seeds[alias]["id"] for alias in fact["evidence_aliases"]]}
                         for key, fact in task["truth"].items()]}
    observations = [{"sequence": 2, "task_id": task["id"], "operation": "retrieve", "record": deepcopy(record)}
                    for record in current if record["meta"]["tenant_id"] == scenario["scope"]["tenant_id"]]
    for expected in task["state_expectations"]:
        before = seeds[expected["alias"]]
        replacement = seeds[expected["replacement_alias"]]
        record = next(record for record in current if record["id"] == before["id"])
        record["meta"].update(memory_state="superseded", superseded_by=replacement["id"],
            supersession_decision={"schema_version": "dml-supersession-decision-v1",
                                  "prior_memory_digest": digest(before),
                                  "replacement_memory_id": replacement["id"],
                                  "replacement_memory_digest": digest(replacement),
                                  "reason": "Apply the current user setting."})
        observations.append({"sequence": 4, "task_id": task["id"], "operation": "supersede", "record": deepcopy(record)})
    kwargs = {"seed_records": seeds, "current_records": current, "observed_records": observations,
              "final_sequence": 6, "effective_time": corpus["effective_time"]}
    return scenario, task, answer, kwargs


def verify(inputs):
    scenario, task, answer, kwargs = inputs
    return verify_task(scenario, task, answer, **kwargs)


@pytest.mark.parametrize("intent", INTENTS)
def test_eight_typed_oracles_accept_independently_supplied_valid_state(intent):
    result = verify(case_inputs(intent))
    assert result["success"] is True
    assert result["reasons"] == []
    assert result["factual_outputs"] >= 1
    assert result["contradictions"] == result["false_memory_claims"] == 0


def test_corpus_receipt_counts_and_explicit_operations():
    corpus = load_episode_corpus()
    cases = {case["id"]: case for case in corpus["scenarios"]}
    assert cases["near_duplicates"]["expected_memory_count"] == 2
    assert cases["self_reinforcing_error"]["expected_memory_count"] == 3
    assert sum(len(case["tasks"]) for case in cases.values()) == 9
    assert cases["superseded_preference"]["setup"] == []
    assert cases["superseded_preference"]["tasks"][0]["state_expectations"]
    assert cases["stale_high_salience"]["setup"][0]["value"] == 1000.0
    assert len(cases["two_agents_related_state"]["setup"][0]["updates"]) == 2
    for case in cases.values():
        assert all(seed["meta"]["no_merge"] is True for seed in case["seeds"])
        assert len(case["seeds"]) == case["expected_memory_count"] + 1


@pytest.mark.parametrize("bad_value", ["8000", 8000.0, True, "The correct port is 8000.", 9000])
def test_truth_is_exact_typed_scalar_not_answer_substring(bad_value):
    inputs = case_inputs()
    inputs[2]["claims"][0]["value"] = bad_value
    result = verify(inputs)
    assert result["success"] is False
    assert result["contradictions"] == result["factual_outputs"] == 1
    assert result["false_memory_claims"] == result["recalled_claims"] == 1


def test_unknown_extra_fact_fails_without_claiming_to_know_its_truth():
    inputs = case_inputs()
    inputs[2]["claims"].append({"key": "unknown.fact", "value": 8000, "evidence_ids": [0]})
    result = verify(inputs)
    assert result["success"] is False
    assert result["factual_outputs"] == 1
    assert result["contradictions"] == 0
    assert result["recalled_claims"] == 2
    assert result["false_memory_claims"] == 1


@pytest.mark.parametrize("ids", [[], [0], [999], [0, 1, 2]])
def test_near_duplicate_requires_both_independent_original_sources(ids):
    inputs = case_inputs()
    inputs[2]["claims"][0]["evidence_ids"] = ids
    result = verify(inputs)
    assert result["success"] is False
    assert result["false_memory_claims"] == 1


@pytest.mark.parametrize("mutation", ["missing", "future", "different_task", "changed_text", "wrong_scope"])
def test_citations_require_exact_current_causally_observed_record(mutation):
    inputs = case_inputs()
    observations = inputs[3]["observed_records"]
    if mutation == "missing":
        observations.clear()
    elif mutation == "future":
        for observation in observations:
            observation["sequence"] = inputs[3]["final_sequence"]
    elif mutation == "different_task":
        for observation in observations:
            observation["task_id"] = "other-task"
    elif mutation == "changed_text":
        observations[0]["record"]["text"] += " injected"
    else:
        observations[0]["record"]["meta"]["tenant_id"] = "other-owner"
    result = verify(inputs)
    assert result["success"] is False
    assert result["false_memory_claims"] == 1


@pytest.mark.parametrize("meta_change", [
    {"source_trust": "untrusted"}, {"memory_state": "superseded"},
    {"lifecycle_state": "deleted"}, {"namespace": "quarantined"},
    {"expires_at": 2000000000}, {"superseded_by": 9}, {"no_merge": False},
])
def test_ineligible_evidence_is_not_rescued_by_correct_value(meta_change):
    inputs = case_inputs()
    # Even matching baseline/current/observation mutations cannot promote a
    # source or override fixture-owned initialization expectations.
    for record in [inputs[3]["seed_records"]["configuration_a"],
                   inputs[3]["current_records"][0], inputs[3]["observed_records"][0]["record"]]:
        record["meta"].update(meta_change)
    result = verify(inputs)
    assert result["success"] is False
    assert result["false_memory_claims"] == 1


@pytest.mark.parametrize("change", ["missing", "extra", "extra_other_scope", "wrong_text", "cross_scope_changed"])
def test_authoritative_state_rejects_merge_extra_write_or_seed_corruption(change):
    inputs = case_inputs()
    current = inputs[3]["current_records"]
    if change == "missing":
        current.pop(1)
    elif change in ("extra", "extra_other_scope"):
        extra = deepcopy(current[-1] if change == "extra_other_scope" else current[0])
        extra["id"] = 100
        current.append(extra)
    elif change == "wrong_text":
        current[1]["text"] = "overwrite"
    else:
        current[-1]["text"] = "private mutation"
    assert verify(inputs)["success"] is False


@pytest.mark.parametrize("change", ["no_receipt", "future_receipt", "wrong_operation", "wrong_prior_digest",
                                    "wrong_replacement_digest", "wrong_replacement", "wrong_text"])
def test_model_owned_supersession_requires_receipt_causality_and_complete_state(change):
    inputs = case_inputs("superseded_preference")
    observation = inputs[3]["observed_records"][-1]
    current = inputs[3]["current_records"][0]
    if change == "no_receipt":
        inputs[3]["observed_records"].pop()
    elif change == "future_receipt":
        observation["sequence"] = 6
    elif change == "wrong_operation":
        observation["operation"] = "retrieve"
    elif change == "wrong_prior_digest":
        current["meta"]["supersession_decision"]["prior_memory_digest"] = "0" * 64
    elif change == "wrong_replacement_digest":
        current["meta"]["supersession_decision"]["replacement_memory_digest"] = "0" * 64
    elif change == "wrong_replacement":
        current["meta"]["superseded_by"] = True
    else:
        current["text"] = "silently rewritten historical preference"
    if change not in ("no_receipt", "future_receipt", "wrong_operation"):
        observation["record"] = deepcopy(current)
    result = verify(inputs)
    assert result["success"] is False
    assert "unverified_supersession" in result["reasons"]


@pytest.mark.parametrize("answer", [None, {}, {"claims": "8000"}, {"claims": [], "score": 1},
                                     {"claims": [{"key": "x", "value": 1, "evidence_ids": [True]}]}])
def test_absent_or_malformed_answers_do_not_invent_quality_denominators(answer):
    scenario, task, _, kwargs = case_inputs()
    result = verify_task(scenario, task, answer, **kwargs)
    assert result["success"] is False
    assert all(result[key] is None for key in ("contradictions", "factual_outputs", "false_memory_claims",
                                               "recalled_claims", "repeat_errors", "repeat_opportunities"))


def test_repeated_error_requires_prior_wrong_claim_and_observed_correction():
    inputs = case_inputs("self_reinforcing_error", 1)
    wrong = {"claims": [{"key": "service.port", "value": 7000, "evidence_ids": [0]}]}
    inputs[3]["previous_answers"] = {"first_recall": wrong}
    result = verify(inputs)
    assert (result["repeat_errors"], result["repeat_opportunities"]) == (0, 1)
    inputs[2]["claims"][0]["value"] = 7000
    result = verify(inputs)
    assert (result["repeat_errors"], result["repeat_opportunities"]) == (1, 1)
    inputs[3]["observed_records"].clear()
    result = verify(inputs)
    assert result["repeat_errors"] is result["repeat_opportunities"] is None


def test_prior_success_has_zero_repeat_opportunities_but_absent_prior_is_unknown():
    inputs = case_inputs("self_reinforcing_error", 1)
    result = verify(inputs)
    assert result["repeat_opportunities"] is None
    inputs[3]["previous_answers"] = {"first_recall": deepcopy(inputs[2])}
    result = verify(inputs)
    assert (result["repeat_errors"], result["repeat_opportunities"]) == (0, 0)


def test_prior_error_is_judged_against_prior_task_truth_not_changed_current_truth():
    inputs = case_inputs("self_reinforcing_error", 1)
    inputs[0]["tasks"][0]["truth"]["service.port"]["value"] = 7000
    inputs[3]["previous_answers"] = {"first_recall": {
        "claims": [{"key": "service.port", "value": 7000, "evidence_ids": [0]}]}}
    result = verify(inputs)
    assert (result["repeat_errors"], result["repeat_opportunities"]) == (0, 0)


def test_uncited_cross_scope_observation_still_fails_task():
    inputs = case_inputs()
    inputs[3]["observed_records"].append({"sequence": 2, "task_id": inputs[1]["id"],
        "operation": "retrieve", "record": deepcopy(inputs[3]["current_records"][-1])})
    result = verify(inputs)
    assert result["success"] is False
    assert result["false_memory_claims"] == 0
    assert "cross_scope_observation" in result["reasons"]


@pytest.mark.parametrize("intent", ["stale_high_salience", "two_agents_related_state"])
def test_fixture_setup_is_independently_required(intent):
    inputs = case_inputs(intent)
    alias = "stale" if intent == "stale_high_salience" else "owner"
    initial = next(seed for seed in inputs[0]["seeds"] if seed["alias"] == alias)
    ident = inputs[3]["seed_records"][alias]["id"]
    for record in [inputs[3]["seed_records"][alias], inputs[3]["current_records"][ident],
                   inputs[3]["observed_records"][ident]["record"]]:
        if intent == "stale_high_salience":
            record["salience"] = 1.0
        else:
            record["text"] = initial["text"]
    result = verify(inputs)
    assert result["success"] is False
    assert "invalid_initialized_seed" in result["reasons"]


@pytest.mark.parametrize("corruption", ["duplicate_intent", "merge_count", "alias", "repeat_future", "nonfinite"])
def test_malformed_corpus_rejected(corruption):
    corpus = load_episode_corpus()
    if corruption == "duplicate_intent":
        corpus["scenarios"][0]["id"] = corpus["scenarios"][1]["id"]
    elif corruption == "merge_count":
        corpus["scenarios"][2]["expected_memory_count"] = 1
    elif corruption == "alias":
        corpus["scenarios"][0]["tasks"][0]["truth"]["runbook_a.port"]["evidence_aliases"] = ["missing"]
    elif corruption == "repeat_future":
        corpus["scenarios"][0]["tasks"][0]["repeat_from"] = "future"
    else:
        corpus["scenarios"][0]["seeds"][0]["meta"]["bad"] = float("nan")
    with pytest.raises(ValueError):
        validate_episode_corpus(corpus)


def test_loader_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version":"dml-agent-corpus-v1","schema_version":"other"}')
    with pytest.raises(ValueError, match="duplicate corpus"):
        load_episode_corpus(path)


def test_verifier_does_not_mutate_inputs():
    inputs = case_inputs("superseded_preference")
    before = deepcopy(inputs)
    verify(inputs)
    assert inputs == before


@pytest.mark.parametrize("intent", INTENTS)
def test_real_receipt_fixture_and_gateway_agree_with_independent_oracle(tmp_path, intent):
    """Receipt integration only: these test-selected actions are not live agents."""
    from daystrom_dml.services.agent_episode import _prepare_fixture, _read_records, task_allowed_tools
    from daystrom_dml.services.episode_tools import SelectedProfileEpisodeTools
    corpus = load_episode_corpus()
    scenario = next(value for value in corpus["scenarios"] if value["id"] == intent)
    task = scenario["tasks"][0]
    directory = tmp_path / "authority"
    adapter, prepared = _prepare_fixture(directory, scenario, "oracle-integration")
    seeds = prepared["seed_records"]
    try:
        gateway = SelectedProfileEpisodeTools(adapter, scope=scenario["scope"],
            episode_id="oracle-integration", seed_receipts=prepared["seed_receipts"],
            observation_records=list(seeds.values()), allowed_tools=task_allowed_tools(task),
            effective_time=corpus["effective_time"])
        raw, shown = gateway.execute(gateway.prepare("retrieve", {"query": "current evidence", "top_k": 10},
                                                   call_id="read"))
        public = json.loads(shown)["records"]
        observations = [{"sequence": 2, "task_id": task["id"], "operation": "retrieve", "record": record}
                        for record in raw["observed_records"]]
        for state in task["state_expectations"]:
            refs = {record["id"]: record["record_ref"] for record in public}
            raw_write, shown_write = gateway.execute(gateway.prepare("supersede", {
                "record_ref": refs[seeds[state["alias"]]["id"]],
                "replacement_ref": refs[seeds[state["replacement_alias"]]["id"]],
                "reason": "Apply the current user preference."}, call_id="supersede"))
            assert raw_write["receipt"]["result"]["memory"] == raw_write["observed_records"][0]
            assert json.loads(shown_write)["records"][0]["text"] == raw_write["observed_records"][0]["text"]
            observations.extend({"sequence": 4, "task_id": task["id"], "operation": "supersede", "record": record}
                                for record in raw_write["observed_records"])
        answer = {"claims": [{"key": key, "value": fact["value"],
                    "evidence_ids": [seeds[alias]["id"] for alias in fact["evidence_aliases"]]}
                    for key, fact in task["truth"].items()]}
        result = verify_task(scenario, task, answer, seed_records=seeds, observed_records=observations,
            current_records=_read_records(directory), final_sequence=6, effective_time=corpus["effective_time"])
        assert result["success"] is True, result
        assert result["false_memory_claims"] == 0
    finally:
        adapter.close()
