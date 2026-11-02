"""M8 competence battery: schema boundaries, verifiers, runner with doubles.

Doubles only: no model, no tokenizer download, no DML adapter, no network.
"""

import json
from pathlib import Path
import types

import pytest

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
    assert claims == {("status_bulletin", "green"): [20], ("status_independent", "red"): [21]}


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
