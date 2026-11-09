"""M8 competence battery: schema boundaries, verifiers, runner with doubles.

Doubles only: no model, no tokenizer download, no DML adapter, no network.
"""

import json
from pathlib import Path
import re as _re
import types

import pytest

from daystrom_dml.services import m8_accounting
from daystrom_dml.services import m8_protocol as protocol
from daystrom_dml.services.m8_memory_baseline import PresentedEvidence
from scripts.m8_battery import runner, schema, verifiers

FIXTURES = Path(__file__).parent.parent / "scripts" / "m8_battery" / "fixtures"
SCOPE = {"tenant_id": "m8-battery", "client_id": None, "session_id": "project", "instance_id": None}


def event(record_id=10, timestamp=1000, **overrides):
    base = {
        "record_id": record_id,
        "text": "Project status is amber.",
        "scope": dict(SCOPE),
        "source_id": "bulletin",
        "fact_key": "status",
        "version": 1,
        "timestamp": timestamp,
        "corrects": None,
        "metadata": {"source_trust": "trusted"},
    }
    base.update(overrides)
    return base


def scenario_dict(**overrides):
    base = {
        "family": "changing_facts",
        "scenario_id": "changing_facts_minimal",
        "scope": dict(SCOPE),
        "events": [event()],
        "query": "What is the project status?",
        "query_clock": 2000,
        "truth": [{"key": "status", "value": "amber", "verifier": "status_value"}],
        "expects_retrieval": True,
    }
    base.update(overrides)
    return base


# --- schema boundaries -----------------------------------------------------


def test_shipped_fixtures_load_and_validate():
    scenarios = schema.load_scenarios(FIXTURES)
    assert len(scenarios) == len(list(FIXTURES.glob("*.json"))) >= 2
    assert all(s.expects_retrieval for s in scenarios)
    by_id = {s.scenario_id: s for s in scenarios}
    assert by_id["changing_facts_project_status"].family == "changing_facts"
    assert by_id["conflicting_independent_sources_status"].family == "conflicting_independent_sources"
    assert by_id["changing_facts_project_status"].events[1].corrects == 10
    envelopes = by_id["conflicting_independent_sources_status"].envelopes
    assert all(e["metadata"]["source_trust"] == "trusted" for e in envelopes)


def test_minimal_scenario_validates():
    scenario = schema.validate_scenario(scenario_dict())
    assert scenario.family == "changing_facts"
    assert len(scenario.events) == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.update(family="unknown_family"),
        lambda s: s.update(unknown_field=True),
        lambda s: s.update(scenario_id="CamelCase"),
        lambda s: s.update(scenario_id=""),
        lambda s: s.update(expects_retrieval="yes"),
        lambda s: s.update(query_clock=999),  # precedes the t=1000 event
        lambda s: s.update(query="  "),
        lambda s: s.update(truth=[]),
        lambda s: s["truth"][0].update(verifier="not_a_verifier"),
        lambda s: s["truth"][0].update(extra=True),
        lambda s: s["truth"][0].update(value=None),  # status_value needs string
        lambda s: s.update(events=[]),
        lambda s: s["events"][0].update(extra=True),
        lambda s: s["events"][0]["metadata"].update(source_trust="untrusted"),
        lambda s: s["events"][0]["metadata"].pop("source_trust"),
        lambda s: s["events"][0]["metadata"].update(distractor="yes"),
        lambda s: s["events"][0].update(
            scope={"tenant_id": "other", "client_id": None, "session_id": "project", "instance_id": None}
        ),
    ],
)
def test_invalid_scenarios_rejected(mutate):
    raw = scenario_dict()
    mutate(raw)
    with pytest.raises(schema.FixtureError):
        schema.validate_scenario(raw)


def test_event_flags_as_booleans_accepted():
    raw = scenario_dict()
    raw["events"][0]["metadata"].update(distractor=True, untrusted_instruction_text=False)
    scenario = schema.validate_scenario(raw)
    assert scenario.events[0].metadata["distractor"] is True


def test_duplicate_record_ids_rejected():
    raw = scenario_dict(events=[event(), event(record_id=10, timestamp=1001)])
    with pytest.raises(schema.FixtureError, match="duplicate"):
        schema.validate_scenario(raw)


def test_non_chronological_events_rejected():
    raw = scenario_dict(events=[event(10, 1001), event(11, 1000, source_id="other")])
    with pytest.raises(schema.FixtureError, match="non-chronological"):
        schema.validate_scenario(raw)


def test_valid_correction_chain_accepted():
    raw = scenario_dict(
        events=[
            event(10, 1000),
            event(11, 1001, text="Project status is green.", version=2, corrects=10),
        ],
        truth=[{"key": "status", "value": "green", "verifier": "status_value"}],
    )
    scenario = schema.validate_scenario(raw)
    assert [e.version for e in scenario.events] == [1, 2]


@pytest.mark.parametrize(
    "second",
    [
        event(11, 1001),  # implicit new version without corrects
        event(11, 1001, version=2, corrects=99),  # unknown predecessor
        event(11, 1001, version=1, corrects=10),  # nonmonotonic version
        event(11, 1001, version=2, corrects=10, source_id="other"),  # drifted source
        event(11, 1001, version=2, corrects=10, fact_key="other"),  # drifted fact
    ],
)
def test_invalid_correction_chains_rejected(second):
    raw = scenario_dict(events=[event(10, 1000), second])
    with pytest.raises(schema.FixtureError):
        schema.validate_scenario(raw)


def test_stale_correction_predecessor_rejected():
    raw = scenario_dict(
        events=[
            event(10, 1000),
            event(11, 1001, version=2, corrects=10),
            event(12, 1002, version=3, corrects=10),
        ]
    )
    with pytest.raises(schema.FixtureError, match="stale"):
        schema.validate_scenario(raw)


def test_multi_claim_truth_shape_boundaries():
    good = {
        "key": "pair",
        "verifier": "multi_claim",
        "claims": [{"key": "status", "value": "green", "verifier": "status_value"}],
    }
    assert schema.validate_scenario(scenario_dict(truth=[good])).truth[0]["key"] == "pair"
    for bad in (
        {**good, "value": None},  # multi claims forbid a value field
        {**good, "claims": []},
        {**good, "claims": [good]},  # nested multi_claim
    ):
        with pytest.raises(schema.FixtureError):
            schema.validate_scenario(scenario_dict(truth=[bad]))


def test_load_scenario_filename_and_size(tmp_path):
    path = tmp_path / "wrong_name.json"
    path.write_text(json.dumps(scenario_dict()))
    with pytest.raises(schema.FixtureError, match="filename"):
        schema.load_scenario(path)
    bad = tmp_path / "changing_facts_minimal.json"
    bad.write_text("{not json")
    with pytest.raises(schema.FixtureError, match="JSON"):
        schema.load_scenario(bad)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(schema.FixtureError, match="no scenario fixtures"):
        schema.load_scenarios(empty)


# --- verifiers -------------------------------------------------------------

GREEN = {"record_id": 11, "text": "Project status is green.", "fact_key": "status"}
RED = {"record_id": 21, "text": "Project status is red.", "fact_key": "status"}


def test_status_value_support_discipline():
    claim = {"key": "status", "value": "green"}
    assert verifiers.status_value(claim, GREEN) is True
    assert verifiers.status_value(claim, RED) is False
    assert verifiers.status_value(claim, {**GREEN, "text": "Project status is green"}) is False
    assert verifiers.status_value(claim, {"record_id": 1}) is False
    with pytest.raises(verifiers.VerifierError):
        verifiers.status_value({"key": "status", "value": 3}, GREEN)


def test_record_text_contains():
    claim = {"key": "note", "value": "status is green"}
    assert verifiers.record_text_contains(claim, GREEN) is True
    assert verifiers.record_text_contains(claim, RED) is False


def test_multi_claim_requires_one_record_supporting_all():
    joint = {"record_id": 1, "text": "Project status is green."}
    multi = {
        "key": "pair",
        "verifier": "multi_claim",
        "claims": [
            {"key": "status", "value": "green", "verifier": "status_value"},
            {"key": "note", "value": "green", "verifier": "record_text_contains"},
        ],
    }
    assert verifiers.multi_claim(multi, joint) is True
    assert verifiers.multi_claim(multi, RED) is False
    with pytest.raises(verifiers.VerifierError):
        verifiers.multi_claim({"key": "x", "claims": [multi]}, joint)
    with pytest.raises(verifiers.VerifierError):
        verifiers.verify("not_a_verifier", {}, {})


def test_supports_for_truth_key_admission_and_conflict():
    truth = (
        {"key": "status_bulletin", "value": "green", "verifier": "status_value"},
        {"key": "status_independent", "value": "red", "verifier": "status_value"},
    )
    supports = verifiers.supports_for_truth(truth)
    assert supports({"key": "status_bulletin", "value": "green"}, GREEN) is True
    assert supports({"key": "status_independent", "value": "red"}, RED) is True
    assert supports({"key": "status_bulletin", "value": "green"}, RED) is False
    assert supports({"key": "unknown", "value": "green"}, GREEN) is False
    assert supports({"key": "status_bulletin", "value": 1}, GREEN) is False
    assert supports("not a claim", GREEN) is False
    assert verifiers.expected_claims(truth) == {
        ("status_bulletin", "green"),
        ("status_independent", "red"),
    }


def test_status_value_splits_on_last_is():
    claim = {"key": "status", "value": "amber"}
    # rpartition: the value is everything after the LAST " is ".
    assert verifiers.status_value(claim, {"text": "The status is green is amber."}) is True
    assert verifiers.status_value({"key": "status", "value": "green"}, {"text": "The status is green is amber."}) is False
    # Empty head or empty value is not a statement.
    assert verifiers.status_value(claim, {"text": " is amber."}) is False
    assert verifiers.status_value({"key": "status", "value": ""}, {"text": "Status is ."}) is False
    with pytest.raises(verifiers.VerifierError):
        verifiers.status_value({"key": "  ", "value": "amber"}, GREEN)


def test_record_text_contains_word_bounded_semantics():
    # Word-bounded: a cited record may not "support" a value it never states as
    # a token (repaired from raw substring, which let "green" match "evergreen").
    assert verifiers.record_text_contains({"key": "n", "value": "green"}, {"text": "evergreen."}) is False
    assert verifiers.record_text_contains({"key": "n", "value": "green"}, {"text": "status is green."}) is True
    assert verifiers.record_text_contains({"key": "n", "value": "300"}, {"text": "cap is 3000 seconds"}) is False
    assert verifiers.record_text_contains({"key": "n", "value": "300"}, {"text": "cap is 300 seconds"}) is True
    assert verifiers.record_text_contains({"key": "n", "value": "300 seconds"}, {"text": "cap is 300 seconds"}) is True
    assert verifiers.record_text_contains({"key": "n", "value": "Green"}, GREEN) is False  # case-sensitive
    assert verifiers.record_text_contains({"key": "n", "value": ""}, GREEN) is True  # empty substring matches
    assert verifiers.record_text_contains({"key": "n", "value": "green"}, "not a record") is False
    assert verifiers.record_text_contains({"key": "n", "value": "green"}, {"text": 42}) is False
    with pytest.raises(verifiers.VerifierError):
        verifiers.record_text_contains({"key": "n", "value": None}, GREEN)


def test_multi_claim_fail_closed_edges():
    joint = {"text": "Project status is green."}
    with pytest.raises(verifiers.VerifierError):
        verifiers.multi_claim({"key": "pair", "claims": []}, joint)
    with pytest.raises(verifiers.VerifierError):
        verifiers.multi_claim({"key": "pair", "claims": "nope"}, joint)
    with pytest.raises(verifiers.VerifierError):
        verifiers.multi_claim({"key": "pair", "claims": ["nope"]}, joint)
    # A subclaim with an unknown or missing verifier propagates fail-closed, not False.
    bad_sub = {"key": "pair", "claims": [{"key": "s", "value": "green", "verifier": "invented"}]}
    with pytest.raises(verifiers.VerifierError):
        verifiers.multi_claim(bad_sub, joint)
    missing_verifier = {"key": "pair", "claims": [{"key": "s", "value": "green"}]}
    with pytest.raises(verifiers.VerifierError):
        verifiers.multi_claim(missing_verifier, joint)


def test_multi_claim_entry_support_ignores_model_value():
    truth = (
        {
            "key": "pair",
            "verifier": "multi_claim",
            "claims": [
                {"key": "pair", "value": "green", "verifier": "status_value"},
                {"key": "note", "value": "green", "verifier": "record_text_contains"},
            ],
        },
    )
    supports = verifiers.supports_for_truth(truth)
    joint = {"record_id": 1, "text": "Project status is green."}
    assert supports({"key": "pair", "value": "green"}, joint) is True
    # TODO: suspicious — a multi_claim truth entry never checks the model's stated
    # value; any value (or none) is supported once the record backs the subclaims.
    assert supports({"key": "pair", "value": "blue"}, joint) is True
    assert supports({"key": "pair"}, joint) is True
    assert supports({"key": "pair", "value": "green"}, RED) is False
    assert supports({"key": "other", "value": "green"}, joint) is False


def test_supports_for_truth_conflicting_same_key_entries():
    truth = (
        {"key": "status", "value": "green", "verifier": "status_value"},
        {"key": "status", "value": "red", "verifier": "status_value"},
    )
    supports = verifiers.supports_for_truth(truth)
    assert supports({"key": "status", "value": "green"}, GREEN) is True
    assert supports({"key": "status", "value": "red"}, RED) is True
    assert supports({"key": "status", "value": "red"}, GREEN) is False
    assert supports({"key": "status", "value": "green"}, RED) is False
    # expected_claims keeps both (key, value) pairs for the duplicated key.
    assert verifiers.expected_claims(truth) == {("status", "green"), ("status", "red")}


def test_supports_for_truth_empty_truth_and_missing_value():
    supports = verifiers.supports_for_truth(())
    assert supports({"key": "status", "value": "green"}, GREEN) is False
    assert verifiers.expected_claims(()) == set()

    supports = verifiers.supports_for_truth(({"key": "status", "value": "green", "verifier": "status_value"},))
    # Leaf entries require a string value; a claim without one is unsupported, never repaired.
    assert supports({"key": "status"}, GREEN) is False
    assert supports({"key": "status", "value": None}, GREEN) is False


def test_supports_for_truth_ignores_evidence_ids():
    # The supports oracle judges (key, value) against the presented record only;
    # evidence_ids are not consulted — citation validity is the caller's discipline.
    supports = verifiers.supports_for_truth(({"key": "status", "value": "green", "verifier": "status_value"},))
    assert supports({"key": "status", "value": "green"}, GREEN) is True
    assert supports({"key": "status", "value": "green", "evidence_ids": []}, GREEN) is True
    assert supports({"key": "status", "value": "green", "evidence_ids": [999]}, GREEN) is True


def test_expected_claims_multi_claim_primary_vs_subclaims():
    multi = {
        "key": "pair",
        "verifier": "multi_claim",
        "claims": [
            {"key": "pair", "value": "green", "verifier": "status_value"},
            {"key": "note", "value": "green", "verifier": "record_text_contains"},
        ],
    }
    # Only primary subclaims (key == entry key) are expected of the answer.
    assert verifiers.expected_claims([multi]) == {("pair", "green")}
    # No primary subclaim: all subclaims are kept so coverage is still checkable.
    orphaned = {
        "key": "pair",
        "verifier": "multi_claim",
        "claims": [
            {"key": "status", "value": "green", "verifier": "status_value"},
            {"key": "note", "value": "green", "verifier": "record_text_contains"},
        ],
    }
    assert verifiers.expected_claims([orphaned]) == {("status", "green"), ("note", "green")}
    mixed = (multi, {"key": "status", "value": "red", "verifier": "status_value"})
    assert verifiers.expected_claims(mixed) == {("pair", "green"), ("status", "red")}


# --- runner with doubles ---------------------------------------------------


def action(payload):
    return json.dumps({"schema_version": "dml-agent-action-v1", **payload})


RETRIEVE = action({"kind": "tool", "name": "retrieve", "arguments": {"query": "status", "top_k": 8}})


def final_for(scenario):
    claims = []
    for entry in scenario.truth:
        if entry["verifier"] == "multi_claim":
            primary = next(s for s in entry["claims"] if s["key"] == entry["key"])
            ids = [e["record_id"] for e in scenario.envelopes if verifiers.multi_claim(entry, e)]
            claims.append({"key": entry["key"], "value": primary["value"], "evidence_ids": ids})
        else:
            ids = [e["record_id"] for e in scenario.envelopes if verifiers.VERIFIER_FUNCS[entry["verifier"]](entry, e)]
            claims.append({"key": entry["key"], "value": entry["value"], "evidence_ids": ids})
    return action({"kind": "final", "answer": {"claims": claims}})


class TokenizerDouble:
    def encode(self, text, **kwargs):
        return list(text.encode())


class ConsumerDouble:
    def __init__(self, arm, script):
        self._arm = arm
        self._script = list(script)
        self._tokenizer = TokenizerDouble()
        self.identity = {"runtime_identity": f"double:{arm}"}
        self._calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def compile(self, messages, *, scope=None, **kwargs):
        self._calls += 1
        return types.SimpleNamespace(request_digest=f"{self._arm}-{self._calls}", arm=self._arm)

    def execute(self, compiled):
        raw = self._script.pop(0) if self._script else "not json"
        try:
            parsed, error = protocol.parse_action(raw, self._arm), None
        except ValueError as exc:
            parsed, error = None, str(exc)
        return {
            "input_token_count": 1,
            "output_token_count": 1,
            "finish_reason": "eos",
            "derived_action": parsed,
            "derived_action_text": raw,
            "action_error": error,
        }


class StoreDouble:
    def __init__(self, envelopes):
        self._envelopes = envelopes

    def retrieve(self, query, *, now, top_k, token_budget, count_tokens, render_records, scope=None):
        return PresentedEvidence(tuple(self._envelopes[:top_k]), "", 0)


class BridgeDouble:
    def __init__(self, envelopes):
        self._store = StoreDouble(envelopes)

    def retrieve(self, query, **kwargs):
        return {"evidence": self._store.retrieve(query, **kwargs)}


class EmbedderDouble:
    def embed(self, text):
        assert text.strip()
        return (1.0,) + (0.0,) * 383


def double_factories(scripts):
    state = {}

    def store_factory(scenario, workdir, *, embedder, snapshot):
        state["scenario"] = scenario
        envelopes = scenario.envelopes
        return (
            {
                "rag": StoreDouble(envelopes),
                "bridge": BridgeDouble(envelopes),
                "context_records": list(envelopes),
            },
            lambda: None,
        )

    def consumer_factory(snapshot, arm):
        return ConsumerDouble(arm, scripts(state["scenario"], arm))

    return store_factory, consumer_factory


def success_scripts(scenario, arm):
    if arm == "context_only":
        return [final_for(scenario)]
    return [RETRIEVE, final_for(scenario)]


def run_double_scenario(scenario, scripts=success_scripts, arms=runner.ARMS, tmp_path=None):
    store_factory, consumer_factory = double_factories(scripts)
    return runner.run_scenario(
        scenario,
        arms=arms,
        workdir=tmp_path,
        snapshot="double-snapshot",
        embedder=EmbedderDouble(),
        store_factory=store_factory,
        consumer_factory=consumer_factory,
    )


def test_arm_store_wiring_all_arms_valid(tmp_path):
    scenario = schema.load_scenario(FIXTURES / "changing_facts_project_status.json")
    episodes = run_double_scenario(scenario, tmp_path=tmp_path)
    assert [e["arm"] for e in episodes] == list(runner.ARMS)
    assert all(e["verdict"] == "valid_cited_final" for e in episodes)
    assert all(e["failure"] is None for e in episodes)
    assert all(e["truth_covered"] for e in episodes)
    by_arm = {e["arm"]: e for e in episodes}
    assert by_arm["context_only"]["retrieval_attempted"] is False
    assert by_arm["persistent_rag"]["retrieval_attempted"] is True
    assert by_arm["dml"]["retrieval_attempted"] is True
    assert 11 in by_arm["persistent_rag"]["presented_record_ids"]
    assert all(len(e["calls"]) == 2 for a, e in by_arm.items() if a != "context_only")


def test_conflicting_sources_final_cites_each_value(tmp_path):
    scenario = schema.load_scenario(FIXTURES / "conflicting_independent_sources_status.json")
    episodes = run_double_scenario(scenario, arms=("persistent_rag",), tmp_path=tmp_path)
    (episode,) = episodes
    assert episode["verdict"] == "valid_cited_final"
    claims = {(c["key"], c["value"]): c["evidence_ids"] for c in episode["answer"]["claims"]}
    assert claims == {("bulletin_status", "green"): [20], ("independent_witness_status", "red"): [21]}


def test_unparseable_action_failure_retained(tmp_path):
    scenario = schema.load_scenario(FIXTURES / "changing_facts_project_status.json")
    episodes = run_double_scenario(scenario, scripts=lambda s, a: ["garbage"], arms=("dml",), tmp_path=tmp_path)
    (episode,) = episodes
    assert episode["verdict"] is None
    assert episode["failure"].startswith("unparseable action retained:")
    assert episode["retrieval_attempted"] is False


def test_context_only_tool_action_rejected_by_protocol(tmp_path):
    scenario = schema.load_scenario(FIXTURES / "changing_facts_project_status.json")
    episodes = run_double_scenario(scenario, scripts=lambda s, a: [RETRIEVE], arms=("context_only",), tmp_path=tmp_path)
    assert "not advertised" in episodes[0]["failure"]


def test_dry_run_validates_shipped_fixtures(tmp_path, capsys):
    code = runner.main(
        [
            "--scenarios-dir",
            str(FIXTURES),
            "--workdir-root",
            str(tmp_path / "work"),
            "--output",
            str(tmp_path / "out.jsonl"),
            "--dry-run",
        ]
    )
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] == "fixtures_valid"
    assert len(report["scenarios"]) == len(list(FIXTURES.glob("*.json")))
    assert not (tmp_path / "out.jsonl").exists()


def test_dry_run_reports_invalid_fixture(tmp_path, capsys):
    bad = tmp_path / "changing_facts_minimal.json"
    bad.write_text(json.dumps(scenario_dict(family="nope")))
    code = runner.main(
        [
            "--scenarios-dir",
            str(tmp_path),
            "--workdir-root",
            str(tmp_path / "work"),
            "--output",
            str(tmp_path / "out.jsonl"),
            "--dry-run",
        ]
    )
    assert code == 1
    assert "fixture_error" in capsys.readouterr().err


def test_main_end_to_end_with_doubles(tmp_path):
    store_factory, consumer_factory = double_factories(success_scripts)
    output = tmp_path / "evidence.jsonl"
    code = runner.main(
        [
            "--scenarios-dir",
            str(FIXTURES),
            "--workdir-root",
            str(tmp_path / "work"),
            "--output",
            str(output),
            "--arms",
            "context_only,persistent_rag",
            "--snapshot",
            "double-snapshot",
        ],
        embedder_factory=EmbedderDouble,
        consumer_factory=consumer_factory,
        store_factory=store_factory,
    )
    assert code == 0
    episodes = [json.loads(line) for line in output.read_text().splitlines()]
    fixture_count = len(list(FIXTURES.glob("*.json")))
    assert len(episodes) == 2 * fixture_count
    assert all(e["verdict"] == "valid_cited_final" for e in episodes)
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["episodes"] == 2 * fixture_count
    for entry in summary["families"].values():
        assert entry["valid_cited_final_rate"] == 1.0
        assert entry["failures"] == []
    for family in ("changing_facts", "conflicting_independent_sources"):
        assert summary["families"][family]["retrieval_attempted_rate"] == 1.0


def test_summarize_retains_failures_and_rates():
    episodes = [
        {
            "family": "changing_facts",
            "scenario_id": "a",
            "arm": "context_only",
            "expects_retrieval": True,
            "retrieval_attempted": False,
            "verdict": "valid_cited_final",
            "failure": None,
        },
        {
            "family": "changing_facts",
            "scenario_id": "a",
            "arm": "dml",
            "expects_retrieval": True,
            "retrieval_attempted": False,
            "verdict": None,
            "failure": "boom",
        },
    ]
    entry = runner.summarize(episodes)["changing_facts"]
    assert entry["valid_cited_final_rate"] == 0.5
    assert entry["retrieval_attempted_rate"] == 0.0
    assert entry["failures"] == [{"scenario_id": "a", "arm": "dml", "failure": "boom"}]


def test_arms_subset_validation(tmp_path):
    with pytest.raises(SystemExit):
        runner.main(
            [
                "--scenarios-dir",
                str(FIXTURES),
                "--workdir-root",
                str(tmp_path / "work"),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--arms",
                "bogus",
                "--dry-run",
            ]
        )


def test_seeds_replicate_and_label_episodes(tmp_path):
    store_factory, consumer_factory = double_factories(success_scripts)
    output = tmp_path / "evidence.jsonl"
    code = runner.main(
        [
            "--scenarios-dir",
            str(FIXTURES),
            "--workdir-root",
            str(tmp_path / "work"),
            "--output",
            str(output),
            "--arms",
            "context_only",
            "--snapshot",
            "double-snapshot",
            "--seeds",
            "0,3",
        ],
        embedder_factory=EmbedderDouble,
        consumer_factory=consumer_factory,
        store_factory=store_factory,
    )
    assert code == 0
    episodes = [json.loads(line) for line in output.read_text().splitlines()]
    fixture_count = len(list(FIXTURES.glob("*.json")))
    assert len(episodes) == 2 * fixture_count
    assert {e["sampling_seed"] for e in episodes} == {0, 3}
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["seeds"] == [0, 3]


def test_seeds_validation_rejects_duplicates_and_negatives(tmp_path):
    for bad in ("0,0", "-1", ""):
        with pytest.raises(SystemExit):
            runner.main(
                [
                    "--scenarios-dir",
                    str(FIXTURES),
                    "--workdir-root",
                    str(tmp_path / "work"),
                    "--output",
                    str(tmp_path / "out.jsonl"),
                    "--seeds",
                    bad,
                    "--dry-run",
                ]
            )


# --- cost accounting with doubles ------------------------------------------

METERING_FIELDS = (
    "model_calls",
    "input_tokens",
    "output_tokens",
    "embedding_calls",
    "embedding_wall_seconds",
    "retrievals",
    "wall_seconds",
    "gpu_wall_seconds",
)


class CountingEmbedder(EmbedderDouble):
    def __init__(self):
        self.calls = 0

    def embed(self, text):
        self.calls += 1
        return super().embed(text)


class UnknownUsageConsumerDouble(ConsumerDouble):
    """Consumer double whose backend cannot report usage (None counts)."""

    def execute(self, compiled):
        result = super().execute(compiled)
        result["input_token_count"] = None
        result["output_token_count"] = None
        return result


def test_episode_metering_fields_exist_and_are_consistent(tmp_path):
    scenario = schema.load_scenario(FIXTURES / "changing_facts_project_status.json")
    store_factory, consumer_factory = double_factories(success_scripts)
    embedder = CountingEmbedder()
    episodes = runner.run_scenario(
        scenario,
        arms=runner.ARMS,
        workdir=tmp_path,
        snapshot="double-snapshot",
        embedder=embedder,
        store_factory=store_factory,
        consumer_factory=consumer_factory,
    )
    by_arm = {e["arm"]: e for e in episodes}
    for episode in episodes:
        for field in METERING_FIELDS:
            assert field in episode, field
        assert episode["model_calls"] == len(episode["calls"])
        assert episode["input_tokens"] == sum(c["input_token_count"] for c in episode["calls"])
        assert episode["output_tokens"] == sum(c["output_token_count"] for c in episode["calls"])
        assert 0.0 <= episode["embedding_wall_seconds"] <= episode["wall_seconds"]
        assert episode["gpu_wall_seconds"] is None  # caller-reported by the station wrapper
    assert by_arm["context_only"]["retrievals"] == 0
    assert by_arm["context_only"]["embedding_calls"] == 0
    assert by_arm["persistent_rag"]["retrievals"] == 1
    assert by_arm["persistent_rag"]["embedding_calls"] == 1
    assert by_arm["dml"]["retrievals"] == 1
    # The dml arm embeds inside its bridge, not in the runner's retrieval path.
    assert by_arm["dml"]["embedding_calls"] == 0
    assert embedder.calls == 1


def test_episode_metering_uses_injected_clock(tmp_path):
    scenario = schema.load_scenario(FIXTURES / "changing_facts_project_status.json")
    store_factory, consumer_factory = double_factories(success_scripts)
    ticks = iter([0.0, 1.0, 1.25, 3.5])  # start, embed start, embed end, episode end
    (episode,) = runner.run_scenario(
        scenario,
        arms=("persistent_rag",),
        workdir=tmp_path,
        snapshot="double-snapshot",
        embedder=EmbedderDouble(),
        store_factory=store_factory,
        consumer_factory=consumer_factory,
        clock=lambda: next(ticks),
    )
    assert episode["embedding_wall_seconds"] == 0.25
    assert episode["wall_seconds"] == 3.5


def test_unknown_usage_makes_token_totals_unknown(tmp_path):
    scenario = schema.load_scenario(FIXTURES / "changing_facts_project_status.json")
    store_factory, _ = double_factories(success_scripts)
    (episode,) = runner.run_scenario(
        scenario,
        arms=("context_only",),
        workdir=tmp_path,
        snapshot="double-snapshot",
        embedder=EmbedderDouble(),
        store_factory=store_factory,
        consumer_factory=lambda snapshot, arm: UnknownUsageConsumerDouble(arm, success_scripts(scenario, arm)),
    )
    assert episode["model_calls"] == len(episode["calls"]) == 1
    assert episode["input_tokens"] is None
    assert episode["output_tokens"] is None


def _accounting_episode(arm, **overrides):
    episode = {
        "arm": arm,
        "verdict": "valid_cited_final",
        "model_calls": 1,
        "input_tokens": 10,
        "output_tokens": 5,
        "embedding_calls": 0,
        "embedding_wall_seconds": 0.0,
        "retrievals": 1,
        "wall_seconds": 1.0,
        "gpu_wall_seconds": None,
    }
    episode.update(overrides)
    return episode


def test_accounting_summary_per_arm_totals_and_cost():
    episodes = [
        _accounting_episode("dml"),
        _accounting_episode("dml", verdict=None, failure="boom", input_tokens=20, output_tokens=6),
        _accounting_episode("context_only", input_tokens=4, output_tokens=2, retrievals=0),
    ]
    per_arm = runner.accounting_summary(episodes)
    assert "persistent_rag" not in per_arm  # arms without episodes are omitted
    dml = per_arm["dml"]
    assert dml["episodes"] == 2
    assert dml["model_calls"] == 2
    assert dml["input_tokens"] == 30
    assert dml["output_tokens"] == 11
    assert dml["retrievals"] == 2
    assert dml["valid_cited_final"] == 1
    # cost_per_correct_success semantics: the failed attempt is in the numerator.
    assert dml["cost_per_valid_final"] == 41.0
    assert dml["gpu_wall_seconds"] is None
    assert per_arm["context_only"]["cost_per_valid_final"] == 6.0
    reported = [
        _accounting_episode("dml", gpu_wall_seconds=2.0),
        _accounting_episode("dml", gpu_wall_seconds=3.5),
    ]
    assert runner.accounting_summary(reported)["dml"]["gpu_wall_seconds"] == 5.5


def test_accounting_summary_fails_loud_on_unknown_denominator():
    with pytest.raises(ValueError, match="zero correct successes"):
        runner.accounting_summary([_accounting_episode("dml", verdict=None, failure="boom")])
    with pytest.raises(ValueError, match="unknown attempt cost"):
        runner.accounting_summary([_accounting_episode("dml", input_tokens=None)])


def test_main_writes_durable_ledger_and_accounting_block(tmp_path):
    def scripts(scenario, arm):
        if arm == "dml" and scenario.scenario_id == "changing_facts_project_status":
            return ["garbage"]
        return success_scripts(scenario, arm)

    store_factory, consumer_factory = double_factories(scripts)
    output = tmp_path / "evidence.jsonl"
    code = runner.main(
        [
            "--scenarios-dir",
            str(FIXTURES),
            "--workdir-root",
            str(tmp_path / "work"),
            "--output",
            str(output),
            "--arms",
            "context_only,dml",
            "--snapshot",
            "double-snapshot",
        ],
        embedder_factory=EmbedderDouble,
        consumer_factory=consumer_factory,
        store_factory=store_factory,
    )
    assert code == 0
    episodes = [json.loads(line) for line in output.read_text().splitlines()]
    for episode in episodes:
        for field in METERING_FIELDS:
            assert field in episode, field
    # Reloading the ledger from disk validates the hash chain; failures included.
    ledger_path = output.with_suffix(".ledger.jsonl")
    ledger = m8_accounting.DurableLedger(ledger_path)
    records = ledger.records()
    assert len(records) == len(episodes) > 0
    assert all(record["kind"] == "step" for record in records)
    assert [record["payload"]["episode"] for record in records] == episodes
    assert all(record["payload"]["name"].startswith("episode:") for record in records)
    retained = [r for r in records if r["payload"]["episode"]["failure"]]
    assert retained and all(r["payload"]["episode"]["arm"] == "dml" for r in retained)
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    accounting = summary["accounting"]
    assert accounting["cost_unit"] == "tokens"
    assert accounting["ledger"] == str(ledger_path)
    per_arm = accounting["per_arm"]
    assert set(per_arm) == {"context_only", "dml"}
    for arm, entry in per_arm.items():
        arm_episodes = [e for e in episodes if e["arm"] == arm]
        assert entry["episodes"] == len(arm_episodes)
        assert entry["model_calls"] == sum(e["model_calls"] for e in arm_episodes)
        assert entry["input_tokens"] == sum(e["input_tokens"] for e in arm_episodes)
        assert entry["output_tokens"] == sum(e["output_tokens"] for e in arm_episodes)
        assert entry["embedding_calls"] == sum(e["embedding_calls"] for e in arm_episodes)
        assert entry["retrievals"] == sum(e["retrievals"] for e in arm_episodes)
        assert entry["gpu_wall_seconds"] is None
        expected = (entry["input_tokens"] + entry["output_tokens"]) / entry["valid_cited_final"]
        assert entry["cost_per_valid_final"] == pytest.approx(expected)
    assert per_arm["dml"]["valid_cited_final"] < per_arm["dml"]["episodes"]


def test_main_fails_loud_when_an_arm_has_no_valid_final(tmp_path):
    store_factory, consumer_factory = double_factories(lambda s, a: ["garbage"])
    output = tmp_path / "evidence.jsonl"
    with pytest.raises(ValueError, match="zero correct successes"):
        runner.main(
            [
                "--scenarios-dir",
                str(FIXTURES),
                "--workdir-root",
                str(tmp_path / "work"),
                "--output",
                str(output),
                "--snapshot",
                "double-snapshot",
            ],
            embedder_factory=EmbedderDouble,
            consumer_factory=consumer_factory,
            store_factory=store_factory,
        )
    # The unknown denominator never passes, but episode evidence is retained.
    assert output.exists()
    episodes = [json.loads(line) for line in output.read_text().splitlines()]
    assert episodes and all(e["verdict"] is None for e in episodes)
    ledger = m8_accounting.DurableLedger(output.with_suffix(".ledger.jsonl"))
    assert len(ledger.records()) == len(episodes)


# ---------------------------------------------------------------------------
# Fixture-authoring hygiene rules (WS-A).
#
# RED by design: these tests encode the hygiene rules the fixture repair must
# satisfy and are expected to FAIL on several shipped fixtures. The failing
# output is the repair spec. Rules judge the raw fixture JSON exactly as
# authored: the readable surface is the query text plus event ``text`` fields
# (fact_key/metadata are scaffolding, not language a model could read).
# ---------------------------------------------------------------------------

_HYGIENE_FIXTURES = sorted(FIXTURES.glob("*.json"))
_HYGIENE_IDS = [p.stem for p in _HYGIENE_FIXTURES]

_TOKEN_RE = _re.compile(r"[a-z0-9]+")
_SNAKE_RE = _re.compile(r"[a-z0-9]+(_[a-z0-9]+)*")

_UNIT_WORDS = frozenset({
    "second", "seconds", "minute", "minutes", "hour", "hours",
    "day", "days", "week", "weeks", "month", "months", "year", "years",
    "ms", "millisecond", "milliseconds",
    "percent", "percentage",
    "byte", "bytes", "kb", "mb", "gb", "tb",
})


def _hygiene_tokens(text):
    return set(_TOKEN_RE.findall(text.lower()))


def _hygiene_corpus(raw):
    """(token set, lowered text) of everything readable: query + event texts."""
    texts = [raw["query"], *(e["text"] for e in raw["events"])]
    return set().union(*(_hygiene_tokens(t) for t in texts)), "\n".join(texts).lower()


def _hygiene_truth_leaves(raw):
    """Every truth entry whose key a model could claim.

    Leaf entries plus multi_claim entries themselves and their subclaims:
    a final answer cites the entry key, and expected_claims flattens to the
    subclaim (key, value) pairs.
    """
    for entry in raw["truth"]:
        if entry["verifier"] == "multi_claim":
            yield entry
            yield from entry["claims"]
        else:
            yield entry


@pytest.mark.parametrize("path", _HYGIENE_FIXTURES, ids=_HYGIENE_IDS)
def test_fixture_hygiene_truth_keys_snake_case(path):
    """Rule 1a: every truth key (incl. multi_claim subclaims) is snake_case."""
    raw = json.loads(path.read_text())
    bad = sorted({
        leaf["key"]
        for leaf in _hygiene_truth_leaves(raw)
        if not _SNAKE_RE.fullmatch(leaf["key"])
    })
    assert not bad, f"{path.name}: non-snake_case truth keys: {bad}"


@pytest.mark.parametrize("path", _HYGIENE_FIXTURES, ids=_HYGIENE_IDS)
def test_fixture_hygiene_truth_keys_natural(path):
    """Rule 1b: every truth key is NATURAL — each of its words appears
    (case-insensitive token match) in the query text or an event text, so a
    model could plausibly produce the key from what it read."""
    raw = json.loads(path.read_text())
    corpus_tokens, _ = _hygiene_corpus(raw)
    problems = {}
    for leaf in _hygiene_truth_leaves(raw):
        missing = [w for w in leaf["key"].split("_") if w not in corpus_tokens]
        if missing:
            problems.setdefault(leaf["key"], missing)
    assert not problems, (
        f"{path.name}: truth keys a reader could not plausibly produce: "
        + "; ".join(f"{key} (missing words: {words})" for key, words in sorted(problems.items()))
    )


@pytest.mark.parametrize("path", _HYGIENE_FIXTURES, ids=_HYGIENE_IDS)
def test_fixture_hygiene_truth_value_units_grounded(path):
    """Rule 2: a truth value carrying a unit (seconds/percent/...) must have
    that unit present in the query or event text."""
    raw = json.loads(path.read_text())
    corpus_tokens, corpus_text = _hygiene_corpus(raw)
    problems = []
    for leaf in _hygiene_truth_leaves(raw):
        value = leaf.get("value")
        if type(value) is not str:
            continue
        units = {t for t in _hygiene_tokens(value) if t in _UNIT_WORDS}
        if "%" in value:
            units.add("%")
        missing = [
            u for u in sorted(units)
            if (u == "%" and "%" not in corpus_text) or (u != "%" and u not in corpus_tokens)
        ]
        if missing:
            problems.append({"key": leaf["key"], "value": value, "units_absent_from_text": missing})
    assert not problems, f"{path.name}: unit-bearing truth values with ungrounded units: {problems}"


@pytest.mark.parametrize("path", _HYGIENE_FIXTURES, ids=_HYGIENE_IDS)
def test_fixture_hygiene_conflicting_query_names_both_sources(path):
    """Rule 3: a conflicting-family query must explicitly ask about BOTH
    sources — every event source_id contributes at least one token to the
    query text."""
    raw = json.loads(path.read_text())
    if raw["family"] != "conflicting_independent_sources":
        pytest.skip("not a conflicting-family fixture")
    query_tokens = _hygiene_tokens(raw["query"])
    sources = sorted({e["source_id"] for e in raw["events"]})
    missing = [s for s in sources if not (_hygiene_tokens(s) & query_tokens)]
    assert not missing, (
        f"{path.name}: conflicting-family query does not explicitly ask about "
        f"source(s) {missing}; query: {raw['query']!r}"
    )


@pytest.mark.parametrize("path", _HYGIENE_FIXTURES, ids=_HYGIENE_IDS)
def test_fixture_hygiene_truth_keys_echoed_by_event_fact_keys(path):
    """Rule 4: every truth key (incl. multi_claim entries and subclaims) is
    echoed by at least one event fact_key in the same fixture, so the
    canonical claim key is always echoable from a presented record
    (control.py maps fact_key to the model-visible claim_key). For
    conflicting-family fixtures every echoing event's source must be the
    source the key names (each side carries its source-qualified key)."""
    raw = json.loads(path.read_text())
    by_key = {}
    for event in raw["events"]:
        by_key.setdefault(event["fact_key"], []).append(event)
    truth_keys = {leaf["key"] for leaf in _hygiene_truth_leaves(raw)}
    unechoed = sorted(truth_keys - set(by_key))
    assert not unechoed, f"{path.name}: truth keys no event fact_key echoes: {unechoed}"
    if raw["family"] != "conflicting_independent_sources":
        return
    mismatched = {}
    for key in sorted(truth_keys):
        key_tokens = set(key.split("_"))
        for event in by_key[key]:
            if not _hygiene_tokens(event["source_id"]) <= key_tokens:
                mismatched.setdefault(key, set()).add(event["source_id"])
    assert not mismatched, (
        f"{path.name}: conflicting truth keys echoed by events from sources the key does not name: "
        + "; ".join(f"{key} <- {sorted(sources)}" for key, sources in sorted(mismatched.items()))
    )
