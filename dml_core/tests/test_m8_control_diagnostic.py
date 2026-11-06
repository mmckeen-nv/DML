"""M8 DML-policy control diagnostic: fixture mapping, cells, controls, evidence.

Doubles only: no model, no tokenizer, no network. The scripted consumer is
bound to the llama3-sft-v3 event identity so the episode contract admits it;
every model decision is explicitly supplied by the test.
"""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from daystrom_dml.services.agent_episode import (
    EpisodeLimits,
    _fixture_adapter,
    _read_records,
)
from daystrom_dml.services.episode_tools import canonical
from scripts.m8_battery import control, schema
from test_agent_episode_runtime import ScriptedConsumer, final, tool

FIXTURES = Path(__file__).parent.parent / "scripts" / "m8_battery" / "fixtures"
SCOPE = {"tenant_id": "m8-battery", "client_id": "ctl", "session_id": None, "instance_id": None}
LIMITS = EpisodeLimits(max_steps=6, output_tokens=128)
CALL_KEYS = {
    "step",
    "request_digest",
    "input_token_count",
    "output_token_count",
    "finish_reason",
    "action",
    "action_error",
}


class Llama3ControlConsumer(ScriptedConsumer):
    """Scripted double under the llama3-sft-v3 runtime identity; no model."""

    def compile(self, messages, tools, *, output_reserved_tokens):
        artifact = super().compile(messages, tools, output_reserved_tokens=output_reserved_tokens)
        return replace(
            artifact,
            identity=replace(artifact.identity, runtime_identity="dml-llama3-sft-action-runtime-v2:" + "d" * 64),
        )

    def execute(self, artifact):
        self.dispatched.append(artifact)
        action = next(self.actions)
        if callable(action):
            action = action(self.requests[-1])
        return SimpleNamespace(
            artifact_digest=artifact.artifact_digest,
            input_token_count=artifact.input_tokens,
            output_token_count=2,
            output_ids=(5, 6),
            text=action if type(action) is str else canonical(action),
        )


def event(record_id=10, timestamp=1000, **overrides):
    base = {
        "record_id": record_id,
        "text": "The service port is 8000.",
        "scope": dict(SCOPE),
        "source_id": "runbook",
        "fact_key": "port",
        "version": 1,
        "timestamp": timestamp,
        "corrects": None,
        "metadata": {"source_trust": "trusted"},
    }
    base.update(overrides)
    return base


def make_scenario(events, **overrides):
    base = {
        "family": "changing_facts",
        "scenario_id": "ctl_minimal",
        "scope": dict(SCOPE),
        "events": events,
        "query": "What is the service port?",
        "query_clock": 5000,
        "truth": [{"key": "port", "value": "8000", "verifier": "status_value"}],
        "expects_retrieval": True,
    }
    base.update(overrides)
    return schema.validate_scenario(base)


def chain_scenario():
    """Three-event correction chain; the current record states 9000."""
    return make_scenario(
        [
            event(),
            event(11, 1001, text="The service port is 9000.", version=2, corrects=10),
            event(
                12,
                1002,
                text="Team chat anecdote about port 7000.",
                source_id="chat",
                fact_key="anecdote",
                metadata={"source_trust": "trusted", "distractor": True},
            ),
        ],
        scenario_id="ctl_chain",
        truth=[{"key": "port", "value": "9000", "verifier": "status_value"}],
    )


def conflicting_scenario():
    return make_scenario(
        [
            event(source_id="alpha"),
            event(11, 1001, text="The service port is 9000.", source_id="beta"),
        ],
        family="conflicting_independent_sources",
        scenario_id="ctl_conflicting",
        truth=[
            {"key": "port_alpha", "value": "8000", "verifier": "status_value"},
            {"key": "port_beta", "value": "9000", "verifier": "status_value"},
        ],
    )


def cite_containing(key, value):
    """Final claiming key=value, citing every presented record containing value."""

    def callback(request):
        records = json.loads(request["messages"][-1]["content"])["records"]
        ids = [record["id"] for record in records if value in record["text"]]
        return final({"claims": [{"key": key, "value": value, "evidence_ids": ids}]})

    return callback


def run_episode(scenario, ctl, actions, tmp_path, **kwargs):
    return control.run_control_episode(
        scenario,
        ctl,
        workdir=tmp_path,
        snapshot=None,
        consumer_factory=lambda snapshot: Llama3ControlConsumer(actions),
        limits=LIMITS,
        **kwargs,
    )


def test_label_cell_maps_fixture_content():
    assert control.label_cell(make_scenario([event()])) == "answer_present"
    assert control.label_cell(chain_scenario()) == "correction_chain"
    assert control.label_cell(conflicting_scenario()) == "conflicting"
    # The family declares conflict intent and wins over any correction edge;
    # empty_store is never labeled from content.
    conflict_with_chain = make_scenario(
        [event(), event(11, 1001, text="The service port is 9000.", version=2, corrects=10)],
        family="conflicting_independent_sources",
        scenario_id="ctl_mixed",
    )
    assert control.label_cell(conflict_with_chain) == "conflicting"
    assert "empty_store" not in {control.label_cell(scenario) for scenario in schema.load_scenarios(FIXTURES)}


def test_cell_for_empty_store_construction_overrides_content():
    scenario = chain_scenario()
    assert control.cell_for(scenario, "empty_store") == "empty_store"
    assert control.cell_for(scenario, "inline_evidence") == "correction_chain"
    assert control.cell_for(scenario, "none") == "correction_chain"


def test_shipped_fixtures_label_into_declared_cells():
    labels = {scenario.scenario_id: control.label_cell(scenario) for scenario in schema.load_scenarios(FIXTURES)}
    assert set(labels.values()) <= set(control.CELLS) - {"empty_store"}
    assert {"answer_present", "correction_chain", "conflicting"} <= set(labels.values())


def test_seed_meta_maps_event_fields_without_inventing_claim_value():
    meta = control.seed_meta(event(7, source_id="deploy-ledger", fact_key="backoff"))
    assert meta == {"source": "deploy-ledger", "claim_key": "backoff", "source_trust": "trusted"}


def test_build_prompt_controls():
    scenario = chain_scenario()
    assert control.build_prompt(scenario, "none") == scenario.query
    # The unanswerable construction must not leak inline evidence either.
    assert control.build_prompt(scenario, "empty_store") == scenario.query
    inline = control.build_prompt(scenario, "inline_evidence")
    assert inline.startswith(control.INLINE_HEADER)
    for item in scenario.events:
        assert f"- record {item.record_id}: {item.text}" in inline
    assert inline.endswith("Task: " + scenario.query)


def test_unknown_control_fails_closed(tmp_path):
    with pytest.raises(schema.FixtureError, match="unknown control"):
        run_episode(chain_scenario(), "bogus", [], tmp_path)


def test_ingest_scenario_receipts_corrections_as_supersession(tmp_path):
    scenario = chain_scenario()
    adapter = _fixture_adapter(tmp_path / "authority")
    try:
        ingested = control.ingest_scenario(adapter, scenario, key_prefix="test:ingest")
        assert len(ingested["receipts"]) == 3
        assert len(ingested["supersessions"]) == 1
        assert set(ingested["records"]) == {10, 11, 12}
        records = {record["id"]: record for record in _read_records(tmp_path / "authority")}
        source = records[ingested["records"][10]["id"]]
        replacement = records[ingested["records"][11]["id"]]
        assert source["meta"]["memory_state"] == "superseded"
        assert source["meta"]["superseded_by"] == replacement["id"]
        assert replacement["meta"].get("memory_state") != "superseded"
        report = adapter.retrieve_context("service port", top_k=10, as_of=control.EPISODE_EFFECTIVE_TIME, **SCOPE)
        texts = [item["text"] for item in report["items"]]
        assert "The service port is 9000." in texts
        assert "The service port is 8000." not in texts
    finally:
        adapter.close()


def test_correction_chain_episode_valid_cited_final(tmp_path):
    record = run_episode(
        chain_scenario(), "none", [tool(query="service port", top_k=10), cite_containing("port", "9000")], tmp_path
    )
    assert set(record) == set(control.RECORD_KEYS)
    assert record["schema"] == control.EVIDENCE_SCHEMA
    assert record["arm"] == "dml_policy" and record["cell"] == "correction_chain"
    assert record["control"] == "none" and record["retrieval_needless"] is False
    assert record["consumer_profile"] == control.CONSUMER_PROFILE
    assert record["retrieval_attempted"] is True and record["retrieve_first"] is True
    assert record["terminal_status"] == "completed"
    assert record["verdict"] == "valid_cited_final" and record["truth_covered"] is True
    assert record["failure"] is None
    assert record["ingestion"] == {"events": 3, "supersessions": 1}
    assert all(set(call) == CALL_KEYS for call in record["calls"])
    assert record["model_calls"] == 2
    assert record["input_tokens"] == 8 and record["output_tokens"] == 4
    cited = record["answer"]["claims"][0]["evidence_ids"][0]
    assert cited in record["presented_record_ids"]


def test_inline_evidence_control_presents_records_and_populates_store(tmp_path):
    scenario = chain_scenario()
    record = run_episode(
        scenario, "inline_evidence", [tool(query="service port", top_k=10), cite_containing("port", "9000")], tmp_path
    )
    assert record["cell"] == "correction_chain" and record["retrieval_needless"] is True
    assert record["ingestion"]["events"] == 3
    # Retrieval here is needless (evidence inline): surface-cued per H2b.
    assert record["retrieval_attempted"] is True
    assert record["verdict"] == "valid_cited_final"


def test_empty_store_construction_ingests_nothing(tmp_path):
    record = run_episode(chain_scenario(), "empty_store", [final({"claims": []})], tmp_path)
    assert record["cell"] == "empty_store" and record["retrieval_needless"] is True
    assert record["ingestion"] == {"events": 0, "supersessions": 0}
    assert record["retrieval_attempted"] is False and record["retrieve_first"] is False
    assert record["verdict"] == "empty_claims_final" and record["truth_covered"] is False
    assert record["presented_record_ids"] == []


def test_surface_cued_retrieval_in_empty_store_is_measured(tmp_path):
    record = run_episode(
        chain_scenario(), "empty_store", [tool(query="service port", top_k=10), final({"claims": []})], tmp_path
    )
    assert record["cell"] == "empty_store"
    assert record["retrieval_attempted"] is True and record["retrieve_first"] is True
    assert record["verdict"] == "empty_claims_final"


def test_immediate_final_without_retrieval_is_retained(tmp_path):
    record = run_episode(chain_scenario(), "none", [final({"claims": []})], tmp_path)
    assert record["retrieval_attempted"] is False and record["retrieve_first"] is False
    assert record["terminal_status"] == "completed"
    assert record["verdict"] == "empty_claims_final"


def test_unsupported_claim_is_not_credited(tmp_path):
    record = run_episode(
        chain_scenario(), "none", [tool(query="service port", top_k=10), cite_containing("port", "7000")], tmp_path
    )
    assert record["terminal_status"] == "completed"
    assert record["verdict"] == "unsupported_final" and record["truth_covered"] is False


def test_unparseable_action_is_retained_never_repaired(tmp_path):
    record = run_episode(chain_scenario(), "none", ["not JSON"], tmp_path)
    assert record["terminal_status"] == "invalid_action"
    assert record["verdict"] is None and record["retrieval_attempted"] is False
    assert record["retrieve_first"] is None
    assert record["calls"][0]["action"] is None
    assert record["calls"][0]["action_error"]
    assert record["output_tokens"] == 2


def test_runner_exception_is_retained_as_failure(tmp_path):
    def broken_factory(snapshot):
        raise RuntimeError("station absent")

    record = control.run_control_episode(
        chain_scenario(), "none", workdir=tmp_path, snapshot=None, consumer_factory=broken_factory, limits=LIMITS
    )
    assert record["failure"] == "RuntimeError: station absent"
    assert record["terminal_status"] is None and record["verdict"] is None
    assert record["ingestion"] == {"events": 3, "supersessions": 1}


def test_control_verifier_evidence_shape():
    scenario = chain_scenario()
    verifier = control.control_verifier(scenario)
    observed = [
        {
            "sequence": 1,
            "task_id": control.TASK_ID,
            "operation": "retrieve",
            "record": {"id": 5, "text": "The service port is 9000."},
        }
    ]
    kwargs = dict(
        seed_records={},
        current_records=[],
        final_sequence=3,
        effective_time=control.EPISODE_EFFECTIVE_TIME,
        previous_answers=None,
    )
    verdict = verifier(
        None,
        None,
        {"claims": [{"key": "port", "value": "9000", "evidence_ids": [5]}]},
        observed_records=observed,
        **kwargs,
    )
    assert verdict["success"] is True and verdict["reasons"] == []
    verdict = verifier(
        None,
        None,
        {"claims": [{"key": "port", "value": "9000", "evidence_ids": [6]}]},
        observed_records=observed,
        **kwargs,
    )
    assert verdict["success"] is False
    assert any("unsupported" in reason for reason in verdict["reasons"])
    verdict = verifier(None, None, {"claims": []}, observed_records=observed, **kwargs)
    assert verdict["success"] is False
    assert "empty claims list" in verdict["reasons"]
    assert any("truth not covered" in reason for reason in verdict["reasons"])


def test_summarize_rates_and_needless_split():
    episodes = [
        {
            "scenario_id": "a",
            "cell": "answer_present",
            "control": "none",
            "retrieval_needless": False,
            "retrieval_attempted": True,
            "retrieve_first": True,
            "verdict": "valid_cited_final",
            "terminal_status": "completed",
            "failure": None,
        },
        {
            "scenario_id": "a",
            "cell": "answer_present",
            "control": "inline_evidence",
            "retrieval_needless": True,
            "retrieval_attempted": True,
            "retrieve_first": True,
            "verdict": "valid_cited_final",
            "terminal_status": "completed",
            "failure": None,
        },
        {
            "scenario_id": "b",
            "cell": "empty_store",
            "control": "empty_store",
            "retrieval_needless": True,
            "retrieval_attempted": True,
            "retrieve_first": True,
            "verdict": "empty_claims_final",
            "terminal_status": "completed",
            "failure": None,
        },
        {
            "scenario_id": "b",
            "cell": "empty_store",
            "control": "empty_store",
            "retrieval_needless": True,
            "retrieval_attempted": False,
            "retrieve_first": False,
            "verdict": "empty_claims_final",
            "terminal_status": "completed",
            "failure": None,
        },
        {
            "scenario_id": "c",
            "cell": "conflicting",
            "control": "none",
            "retrieval_needless": False,
            "retrieval_attempted": False,
            "retrieve_first": None,
            "verdict": None,
            "terminal_status": "model_error",
            "failure": None,
        },
    ]
    summary = control.summarize(episodes)
    assert set(summary) == {"answer_present", "empty_store", "conflicting"}
    assert summary["answer_present"]["retrieval_attempted_rate"] == 1.0
    assert summary["answer_present"]["needless_episodes"] == 1
    assert summary["answer_present"]["needless_retrieval_rate"] == 1.0
    assert summary["empty_store"]["needless_retrieval_rate"] == 0.5
    assert summary["empty_store"]["empty_claims_final"] == 2
    assert summary["conflicting"]["needless_retrieval_rate"] is None
    assert summary["conflicting"]["failures"] == [
        {"scenario_id": "c", "control": "none", "failure": "terminal_status:model_error"}
    ]


def test_select_scenarios_fail_closed():
    scenarios = [make_scenario([event()]), chain_scenario(), conflicting_scenario()]
    by_id = [scenario.scenario_id for scenario in scenarios]
    assert [s.scenario_id for s in control.select_scenarios(scenarios, None)] == by_id
    subset = control.select_scenarios(scenarios, "ctl_chain,ctl_minimal")
    assert [s.scenario_id for s in subset] == ["ctl_chain", "ctl_minimal"]
    for bad in ("missing_id", "ctl_chain,ctl_chain", "ctl_chain,,ctl_minimal", ""):
        with pytest.raises(schema.FixtureError):
            control.select_scenarios(scenarios, bad)


def test_main_dry_run_and_snapshot_requirement(tmp_path, capsys):
    output = tmp_path / "episodes.jsonl"
    argv = ["--scenarios-dir", str(FIXTURES), "--workdir-root", str(tmp_path / "work"), "--output", str(output)]
    assert control.main(argv + ["--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] == "fixtures_valid"
    assert len(report["scenarios"]) == len(schema.load_scenarios(FIXTURES))
    assert control.main(argv) == 2
    assert not output.exists()
    assert control.main(argv + ["--dry-run", "--fixtures-subset", "expcorr_ingest_backoff_chain"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [s["scenario_id"] for s in report["scenarios"]] == ["expcorr_ingest_backoff_chain"]
    assert report["scenarios"][0]["cell"] == "correction_chain"


def test_main_end_to_end_with_doubles(tmp_path):
    output = tmp_path / "episodes.jsonl"

    def factory(snapshot):
        return Llama3ControlConsumer(
            [
                tool(query="ingest pipeline retry backoff cap", top_k=10),
                cite_containing("ingest_retry_backoff_cap", "300 seconds"),
            ]
        )

    code = control.main(
        [
            "--scenarios-dir",
            str(FIXTURES),
            "--workdir-root",
            str(tmp_path / "work"),
            "--output",
            str(output),
            "--snapshot",
            str(tmp_path),
            "--fixtures-subset",
            "expcorr_ingest_backoff_chain",
            "--controls",
            "none,empty_store",
        ],
        consumer_factory=factory,
    )
    assert code == 0
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    records = {json.loads(line)["control"]: json.loads(line) for line in lines}
    assert set(records) == {"none", "empty_store"}
    for record in records.values():
        assert set(record) == set(control.RECORD_KEYS)
    assert records["none"]["verdict"] == "valid_cited_final"
    empty = records["empty_store"]
    assert empty["cell"] == "empty_store" and empty["ingestion"]["events"] == 0
    # The scripted policy retrieves first even with an empty store: surface-cued.
    assert empty["retrieval_attempted"] is True and empty["retrieve_first"] is True
    summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == control.SUMMARY_SCHEMA
    assert summary["episodes"] == 2
    assert summary["cells"]["correction_chain"]["retrieval_attempted_rate"] == 1.0
    assert summary["cells"]["empty_store"]["needless_retrieval_rate"] == 1.0


def test_development_x86_factory_requires_declared_env(tmp_path, monkeypatch):
    monkeypatch.delenv("DML_DEV_RUNTIME", raising=False)
    with pytest.raises(RuntimeError, match="DML_DEV_RUNTIME=development-x86-20261102"):
        control.dev_x86_consumer_factory(tmp_path)
    monkeypatch.setenv("DML_DEV_RUNTIME", "development-x86-20261102")
    with pytest.raises(Exception) as excinfo:  # gate passed; fails on the absent snapshot instead
        control.dev_x86_consumer_factory(tmp_path)
    assert "DML_DEV_RUNTIME" not in str(excinfo.value)


def test_record_carries_declared_development_identity(tmp_path):
    class DevConsumer(Llama3ControlConsumer):
        @property
        def development_runtime_identity(self):
            return dict(control.DEVELOPMENT_X86_RUNTIME)

    scenario = chain_scenario()
    record = control.run_control_episode(
        scenario,
        "none",
        workdir=tmp_path,
        snapshot=None,
        consumer_factory=lambda snapshot: DevConsumer([final({"claims": []})]),
        limits=LIMITS,
    )
    assert record["development_runtime"] == dict(control.DEVELOPMENT_X86_RUNTIME)
    json.dumps(record)  # still JSONL-serializable with the identity present


def test_record_development_identity_none_on_qualified_path(tmp_path):
    record = run_episode(chain_scenario(), "none", [final({"claims": []})], tmp_path)
    assert record["development_runtime"] is None


def test_development_flag_uses_dev_factory(monkeypatch, tmp_path):
    monkeypatch.setenv("DML_DEV_RUNTIME", "development-x86-20261102")
    selected = {}
    original = control.dev_x86_consumer_factory

    def wrapped(snapshot):
        selected["used"] = True
        pytest.raises(Exception, original, snapshot)
        raise RuntimeError("dev factory selected")

    monkeypatch.setattr(control, "dev_x86_consumer_factory", wrapped)
    code = control.main(
        [
            "--scenarios-dir",
            str(FIXTURES),
            "--workdir-root",
            str(tmp_path / "work"),
            "--output",
            str(tmp_path / "out.jsonl"),
            "--snapshot",
            str(tmp_path),
            "--fixtures-subset",
            "changing_facts_project_status",
            "--controls",
            "none",
            "--development-runtime-x86",
        ]
    )
    assert code == 0 and selected["used"] is True


def test_inline_only_citation_is_not_supported(tmp_path):
    # Pin: in an inline_evidence episode, a final that cites ONLY the fixture
    # record id from the inline-embedded text (record 11 states "The service
    # port is 9000." in the prompt) is scored unsupported_final, even though
    # the cited text states the truth. control_verifier credits a claim only
    # via observed_records, which the episode contract builds exclusively from
    # tool_completed (retrieve) presentations; the inline prompt text never
    # enters observed_records, and its fixture record_ids are not DML ids
    # anyway (module docstring: presented_record_ids are "the DML-assigned
    # integer record ids the model could cite, not fixture record_ids").
    # Intended contract of the verifier: yes — INLINE_HEADER declares the
    # inline records untrusted source data, so support must be grounded in a
    # retrieved (verified-memory) record.
    # TODO(observer question "should inline-only citations be creditable in
    # inline_evidence?"): this contract is in tension with the control's own
    # design note that retrieval is "needless" in inline_evidence and with
    # amendments v2+ inviting the model to cite "the provided text" — an
    # immediate inline-citing final can never score valid_cited_final.
    record = run_episode(
        chain_scenario(),
        "inline_evidence",
        [final({"claims": [{"key": "port", "value": "9000", "evidence_ids": [11]}]})],
        tmp_path,
    )
    assert record["terminal_status"] == "completed"
    assert record["verdict"] == "unsupported_final" and record["truth_covered"] is False
    assert record["presented_record_ids"] == []


def test_unretrieved_dml_record_citation_is_not_supported(tmp_path):
    # Pin (stronger form of the inline-only rule): even a genuine DML record
    # id whose stored text states the truth (id 1 = "The service port is
    # 9000.") is unsupported when that record was never presented by a
    # retrieve call in the episode. Support requires tool presentation, not
    # store membership — "the episode oracle is never consulted beyond
    # presented records".
    record = run_episode(
        chain_scenario(),
        "inline_evidence",
        [final({"claims": [{"key": "port", "value": "9000", "evidence_ids": [1]}]})],
        tmp_path,
    )
    assert record["verdict"] == "unsupported_final" and record["truth_covered"] is False


def test_retrieved_supporting_citation_is_supported_inline(tmp_path):
    # Pin: a claim citing a retrieved DML id whose text supports it is
    # credited — valid_cited_final with truth covered, and the cited id is
    # exactly a member of presented_record_ids.
    record = run_episode(
        chain_scenario(),
        "inline_evidence",
        [tool(query="service port", top_k=10), cite_containing("port", "9000")],
        tmp_path,
    )
    assert record["verdict"] == "valid_cited_final" and record["truth_covered"] is True
    cited = record["answer"]["claims"][0]["evidence_ids"]
    assert cited and set(cited) <= set(record["presented_record_ids"])


def test_inline_evidence_empty_claims_final_classification(tmp_path):
    # Pin: an empty-claims final in an inline_evidence episode classifies as
    # empty_claims_final (the declared insufficiency signal), not a failure.
    record = run_episode(chain_scenario(), "inline_evidence", [final({"claims": []})], tmp_path)
    assert record["terminal_status"] == "completed"
    assert record["verdict"] == "empty_claims_final" and record["truth_covered"] is False
    assert record["retrieval_needless"] is True


def test_inline_presented_record_ids_are_exactly_retrieved_dml_ids(tmp_path):
    # Pin: presented_record_ids in an inline_evidence episode is exactly the
    # deduplicated set of DML-assigned ids the retrieve tool showed the model,
    # in first-shown order. For the chain fixture ingested as DML ids 0, 1, 2
    # (in event order), retrieve top_k=10 returns only the two non-superseded
    # records: id 1 ("The service port is 9000.") and id 2 (the 7000
    # anecdote). The superseded id 0 ("The service port is 8000.") is stored
    # but never presented, and the inline fixture record ids 10/11/12 never
    # appear — presented_record_ids is DML ids only.
    record = run_episode(
        chain_scenario(),
        "inline_evidence",
        [tool(query="service port", top_k=10), final({"claims": []})],
        tmp_path,
    )
    assert record["presented_record_ids"] == [1, 2]
    assert not (set(record["presented_record_ids"]) & {10, 11, 12})


def test_prompt_amendment_appended_and_recorded(tmp_path):
    record = run_episode(chain_scenario(), "none", [final({"claims": []})], tmp_path,
                         amendment="terminate-on-empty-v1")
    assert record["prompt_amendment"] == "terminate-on-empty-v1"
    task = control.build_task(chain_scenario(), "none", "terminate-on-empty-v1")
    assert "Never emit the same tool call twice in a row" in task["prompt"]
    assert task["prompt"].startswith(chain_scenario().query)


def test_unknown_prompt_amendment_fails_closed(tmp_path):
    with pytest.raises(Exception, match="unknown prompt amendment"):
        control.build_task(chain_scenario(), "none", "invented")
    record = run_episode(chain_scenario(), "none", [final({"claims": []})], tmp_path)
    assert record["prompt_amendment"] == "none"


# --- _fill_from_report token-total edges ------------------------------------


def test_token_total_unknown_when_any_count_unknown():
    assert control._token_total([], "input_token_count") == 0
    calls = [{"input_token_count": 3}, {"input_token_count": 4}]
    assert control._token_total(calls, "input_token_count") == 7
    # Any unknown (None) count makes the whole total unknown, never a partial sum.
    assert control._token_total(calls + [{"input_token_count": None}], "input_token_count") is None
    # A missing key reads as unknown (None) and poisons the total the same way.
    assert control._token_total([{"output_token_count": 1}], "input_token_count") is None


def _report_record():
    return {"retrieve_first": None, "retrieval_attempted": False, "failure": None}


def test_fill_from_report_requested_without_completion_leaves_unknown_tokens():
    record = _report_record()
    report = {
        "events": [
            {"kind": "model_requested", "payload": {"step": 0, "artifact_digest": "d0"}},
            # A completion for a step with no request is dropped, not invented.
            {"kind": "model_completed", "payload": {"step": 1, "input_token_count": 5, "output_token_count": 2, "text": "garbage"}},
        ],
        "terminal": {"status": "completed", "answer": {"claims": []}, "success": False},
    }
    control._fill_from_report(record, report)
    assert [call["step"] for call in record["calls"]] == [0]
    assert record["calls"][0]["input_token_count"] is None
    assert record["calls"][0]["output_token_count"] is None
    # The runner.py-shaped totals then come out unknown, matching the accounting rule.
    assert control._token_total(record["calls"], "input_token_count") is None
    assert control._token_total(record["calls"], "output_token_count") is None
    assert record["verdict"] == "empty_claims_final"


def test_fill_from_report_explicit_none_counts_poison_totals():
    record = _report_record()
    report = {
        "events": [
            {"kind": "model_requested", "payload": {"step": 0, "artifact_digest": "d0"}},
            {"kind": "model_completed", "payload": {"step": 0, "input_token_count": None, "output_token_count": 2, "text": "garbage"}},
            {"kind": "model_requested", "payload": {"step": 1, "artifact_digest": "d1"}},
            {"kind": "model_completed", "payload": {"step": 1, "input_token_count": 5, "output_token_count": 3, "text": "garbage"}},
        ],
        "terminal": {"status": "completed", "answer": {"claims": []}, "success": False},
    }
    control._fill_from_report(record, report)
    assert len(record["calls"]) == 2
    # One unknown input count makes the input total unknown; outputs stay summable.
    assert control._token_total(record["calls"], "input_token_count") is None
    assert control._token_total(record["calls"], "output_token_count") == 5


def test_shared_consumer_reused_and_not_closed_per_episode(tmp_path):
    closes = []

    class SharedConsumer(Llama3ControlConsumer):
        def close(self):
            closes.append(True)

    shared = SharedConsumer([final({"claims": []}), final({"claims": []})])
    scenario = chain_scenario()
    for ctl in ("none", "empty_store"):
        record = control.run_control_episode(
            scenario,
            ctl,
            workdir=tmp_path / ctl,
            snapshot=None,
            consumer_factory=lambda snapshot: pytest.fail("factory called despite shared consumer"),
            limits=LIMITS,
            shared_consumer=shared,
        )
        assert record["failure"] is None
        assert record["terminal_status"] == "completed"
    assert closes == []  # caller owns the shared consumer's lifecycle


def test_main_shares_one_consumer_across_episodes(monkeypatch, tmp_path):
    constructions = []

    def factory(snapshot):
        constructions.append(snapshot)
        return Llama3ControlConsumer([final({"claims": []})] * 10)

    monkeypatch.setattr(control, "default_consumer_factory", factory)
    code = control.main(
        [
            "--scenarios-dir", str(FIXTURES),
            "--workdir-root", str(tmp_path / "work"),
            "--output", str(tmp_path / "out.jsonl"),
            "--snapshot", str(tmp_path),
            "--fixtures-subset", "changing_facts_project_status",
            "--controls", "none,empty_store",
        ]
    )
    assert code == 0
    assert len(constructions) == 1  # one admission for the whole run
    records = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert len(records) == 2 and all(r["failure"] is None for r in records)


def test_main_shared_construction_failure_retained_per_episode(monkeypatch, tmp_path):
    def broken(snapshot):
        raise RuntimeError("station absent")

    monkeypatch.setattr(control, "default_consumer_factory", broken)
    code = control.main(
        [
            "--scenarios-dir", str(FIXTURES),
            "--workdir-root", str(tmp_path / "work"),
            "--output", str(tmp_path / "out.jsonl"),
            "--snapshot", str(tmp_path),
            "--fixtures-subset", "changing_facts_project_status",
            "--controls", "none,empty_store",
        ]
    )
    assert code == 0
    records = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert all(r["failure"] == "RuntimeError: station absent" for r in records)
