"""Transport-native M8 battery: per-arm retrieve backends, parity, evidence.

Doubles only: no model, no tokenizer, no network. The scripted consumer is
bound to the llama3-sft-v3 event identity so the episode contract admits it;
every model decision is explicitly supplied by the test. The RAG arm runs the
real PersistentRAGBaseline (real SQLite, deterministic embedded double) so
"the baseline receives the same event stream" is exercised, not mocked.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from daystrom_dml.contracts.agent_episode import LLAMA3_SFT_V3_CONSUMER_PROFILE, QWEN3_SFT_V1_CONSUMER_PROFILE
from daystrom_dml.services.agent_episode import EpisodeLimits
from daystrom_dml.services.episode_tools import canonical
from daystrom_dml.services.m8_memory_baseline import PersistentRAGBaseline, validate_event
from scripts.m8_battery import runner_transport_native as battery
from scripts.m8_battery import schema
from test_agent_episode_runtime import ScriptedConsumer, final, tool
from test_m8_control_diagnostic import chain_scenario, cite_containing

FIXTURES = Path(__file__).parent.parent / "scripts" / "m8_battery" / "fixtures"
LIMITS = EpisodeLimits(max_steps=6, output_tokens=128)
EMBEDDING_IDENTITY = "test-lexical-double-v1"
EMBEDDING_DIMS = 16


class Llama3BatteryConsumer(ScriptedConsumer):
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


class LexicalEmbedder:
    """Deterministic hashing double; no learned-quality claim, no model load."""

    model_name = EMBEDDING_IDENTITY

    def embed(self, text):
        import hashlib as _hashlib
        import re

        vector = np.zeros(EMBEDDING_DIMS, dtype=np.float32)
        for word in re.findall(r"\w+", text.casefold()):
            digest = _hashlib.sha256(word.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:2], "big") % EMBEDDING_DIMS] += 1
        if not vector.any():
            vector[0] = 1
        return vector / np.linalg.norm(vector)


class SpyRAGBaseline(PersistentRAGBaseline):
    """Captures every ingested event/vector before delegating; real store."""

    def __init__(self, path, **kwargs):
        self.seen = []
        super().__init__(path, **kwargs)

    def ingest(self, event, vector):
        self.seen.append(validate_event(event))
        super().ingest(event, vector)


def rag_store_factory(path, holder=None):
    store = SpyRAGBaseline(path, embedding_identity=EMBEDDING_IDENTITY, dimensions=EMBEDDING_DIMS)
    if holder is not None:
        holder.append(store)
    return store


def run_episode(scenario, arm, actions, tmp_path, stores=None, **kwargs):
    return battery.run_battery_episode(
        scenario,
        arm,
        workdir=tmp_path,
        snapshot=None,
        consumer_factory=lambda snapshot: Llama3BatteryConsumer(actions),
        limits=LIMITS,
        embedder=LexicalEmbedder(),
        rag_store_factory=lambda path: rag_store_factory(path, stores),
        **kwargs,
    )


# --- event-clock parity -----------------------------------------------------


def test_event_stream_digest_identical_across_arms(tmp_path):
    scenario = chain_scenario()
    records = {arm: run_episode(scenario, arm, [final({"claims": []})], tmp_path / arm) for arm in battery.ARMS}
    digests = {arm: record["event_stream_digest"] for arm, record in records.items()}
    assert len(set(digests.values())) == 1
    expected = hashlib.sha256(canonical([validate_event(e) for e in scenario.events]).encode("utf-8")).hexdigest()
    assert digests["dml_memory"] == expected
    for record in records.values():
        assert record["failure"] is None and record["terminal_status"] == "completed"


def test_rag_baseline_receives_the_same_events(tmp_path):
    scenario = chain_scenario()
    stores = []
    record = run_episode(scenario, "rag_memory", [final({"claims": []})], tmp_path, stores=stores)
    assert record["failure"] is None
    (store,) = stores
    # The baseline admitted exactly the fixture stream, in event order.
    assert [envelope["record_id"] for envelope in store.seen] == [10, 11, 12]
    assert store.seen == [validate_event(event) for event in scenario.events]
    assert record["ingestion"] == {"events": 3, "supersessions": 1}


def test_no_memory_receives_stream_and_presents_nothing(tmp_path):
    scenario = chain_scenario()
    consumer_actions = [tool(query="service port", top_k=8), final({"claims": []})]
    holder = {}

    def factory(snapshot):
        consumer = Llama3BatteryConsumer(consumer_actions)
        holder["consumer"] = consumer
        return consumer

    record = battery.run_battery_episode(
        scenario,
        "no_memory",
        workdir=tmp_path,
        snapshot=None,
        consumer_factory=factory,
        limits=LIMITS,
        embedder=LexicalEmbedder(),
        rag_store_factory=rag_store_factory,
    )
    assert record["failure"] is None
    assert record["ingestion"] == {"events": 3, "supersessions": 1}  # received, discarded
    assert record["backend"] == {"kind": "empty"}
    assert record["retrieval_attempted"] is True and record["retrieve_first"] is True
    assert record["presented_record_ids"] == []
    # The model-facing retrieve result carried zero records.
    feedback = json.loads(holder["consumer"].requests[-1]["messages"][-1]["content"])
    assert feedback["records"] == [] and feedback["returned_count"] == 0


# --- record schema and per-arm verdicts --------------------------------------


def test_record_schema_stable_across_arms(tmp_path):
    scenario = chain_scenario()
    for arm in battery.ARMS:
        record = run_episode(scenario, arm, [final({"claims": []})], tmp_path / arm)
        assert set(record) == set(battery.RECORD_KEYS)
        assert record["schema"] == "dml.m8.transport-native-battery.v1"
        assert record["arm"] == arm
        # The inline arm carries the context-floor cell tag (C2); the other
        # arms keep the content-derived cell.
        assert record["cell"] == ("context_inline" if arm == "context_inline" else "correction_chain")
        assert record["retrieve_top_k_cap"] == battery.TOP_K_CAP
        assert record["consumer_profile"] == LLAMA3_SFT_V3_CONSUMER_PROFILE
        json.dumps(record)  # JSONL-serializable


def test_dml_memory_arm_valid_cited_final(tmp_path):
    record = run_episode(
        chain_scenario(),
        "dml_memory",
        [tool(query="service port", top_k=8), cite_containing("port", "9000")],
        tmp_path,
    )
    assert record["backend"] == {"kind": "dml_receipt_store", "profile_id": battery.PROFILE_ID}
    assert record["ingestion"] == {"events": 3, "supersessions": 1}
    assert record["terminal_status"] == "completed"
    assert record["verdict"] == "valid_cited_final" and record["truth_covered"] is True
    assert record["retrieval_attempted"] is True and record["retrieve_first"] is True
    assert record["input_tokens"] == 8 and record["output_tokens"] == 4


def test_rag_memory_arm_valid_cited_final_from_current_view(tmp_path):
    scenario = chain_scenario()
    record = run_episode(
        scenario,
        "rag_memory",
        [tool(query="service port", top_k=8), cite_containing("port", "9000")],
        tmp_path,
    )
    assert record["failure"] is None
    assert record["backend"]["kind"] == "persistent_rag_baseline"
    assert record["backend"]["embedding_identity"] == EMBEDDING_IDENTITY
    assert record["backend"]["dimensions"] == EMBEDDING_DIMS
    assert record["backend"]["top_k_cap"] == battery.RAG_TOP_K_CAP
    assert record["verdict"] == "valid_cited_final" and record["truth_covered"] is True
    # The correction-aware current view presents the 9000 record (assigned id
    # 1) and the 7000 anecdote (id 2); the superseded 8000 record (id 0) is
    # stored but never presented. Ids are backend-assigned, not fixture ids.
    assert record["presented_record_ids"] == [1, 2]
    assert not (set(record["presented_record_ids"]) & {10, 11, 12})
    cited = record["answer"]["claims"][0]["evidence_ids"]
    assert cited == [1] and set(cited) <= set(record["presented_record_ids"])


def test_retrieve_top_k_cap_presented_and_enforced_without_tool_error(tmp_path):
    # C4: both persistent arms are PRESENTED the same top_k cap (TOP_K_CAP, the
    # manifest fixed-candidate limit) at the tool-schema layer, and the bridge
    # enforces it as a proven no-dispatch validation rejection with model
    # feedback — never a silent clamp, never a tool_error terminal.
    holder = {}

    def factory(snapshot):
        consumer = Llama3BatteryConsumer(
            [
                tool(query="service port", top_k=10),  # exceeds the declared cap
                tool(query="service port", top_k=battery.TOP_K_CAP),
                cite_containing("port", "9000"),
            ]
        )
        holder["consumer"] = consumer
        return consumer

    record = battery.run_battery_episode(
        chain_scenario(),
        "rag_memory",
        workdir=tmp_path,
        snapshot=None,
        consumer_factory=factory,
        limits=LIMITS,
        embedder=LexicalEmbedder(),
        rag_store_factory=rag_store_factory,
    )
    assert record["failure"] is None and record["terminal_status"] == "completed"
    assert record["verdict"] == "valid_cited_final"
    assert record["retrieve_top_k_cap"] == battery.TOP_K_CAP
    # The model-facing retrieve schema presents the declared cap.
    (retrieve_def,) = (
        tool_def for tool_def in holder["consumer"].requests[0]["tools"] if tool_def["function"]["name"] == "retrieve"
    )
    assert retrieve_def["function"]["parameters"]["properties"]["top_k"]["maximum"] == battery.TOP_K_CAP
    # The over-cap call got the declared rejection feedback, not records.
    rejected = json.loads(holder["consumer"].requests[1]["messages"][-1]["content"])
    assert rejected["error"]["code"] == "top_k_cap_exceeded" and rejected["effects"] == "none"
    # The compliant retry executed and presented the current view.
    assert record["presented_record_ids"] == [1, 2]
    # The episode_started envelope declares the cap (fixed-protocol validated).
    # (The request tools above are checked against that declaration by the
    # episode event validator on every step.)


def test_no_memory_unsupported_final_is_measured_not_hidden(tmp_path):
    # A policy that fabricates against a dead retrieve is recorded as
    # unsupported_final: evidence about the policy, not a harness defect.
    record = run_episode(
        chain_scenario(),
        "no_memory",
        [
            tool(query="service port", top_k=5),
            final({"claims": [{"key": "port", "value": "9000", "evidence_ids": [0]}]}),
        ],
        tmp_path,
    )
    assert record["terminal_status"] == "completed"
    assert record["verdict"] == "unsupported_final" and record["truth_covered"] is False


def test_unparseable_action_is_retained_never_repaired(tmp_path):
    record = run_episode(chain_scenario(), "dml_memory", ["not JSON"], tmp_path)
    assert record["terminal_status"] == "invalid_action"
    assert record["verdict"] is None and record["retrieval_attempted"] is False
    assert record["calls"][0]["action"] is None and record["calls"][0]["action_error"]


def test_unknown_arm_fails_closed(tmp_path):
    with pytest.raises(schema.FixtureError, match="unknown arm"):
        run_episode(chain_scenario(), "bogus", [], tmp_path)


def test_runner_exception_is_retained_as_failure(tmp_path):
    def broken_factory(snapshot):
        raise RuntimeError("station absent")

    record = battery.run_battery_episode(
        chain_scenario(),
        "dml_memory",
        workdir=tmp_path,
        snapshot=None,
        consumer_factory=broken_factory,
        limits=LIMITS,
    )
    assert record["failure"] == "RuntimeError: station absent"
    assert record["terminal_status"] is None and record["verdict"] is None
    assert record["ingestion"] == {"events": 3, "supersessions": 1}
    assert record["event_stream_digest"] is not None


def test_rag_arm_requires_embedder(tmp_path):
    record = battery.run_battery_episode(
        chain_scenario(),
        "rag_memory",
        workdir=tmp_path,
        snapshot=None,
        consumer_factory=lambda snapshot: Llama3BatteryConsumer([final({"claims": []})]),
        limits=LIMITS,
        embedder=None,
        rag_store_factory=rag_store_factory,
    )
    assert record["failure"] == "FixtureError: rag_memory arm requires an embedder"


def test_shared_consumer_reused_across_arms_and_not_closed(tmp_path):
    closes = []

    class SharedConsumer(Llama3BatteryConsumer):
        def close(self):
            closes.append(True)

    shared = SharedConsumer([final({"claims": []})] * len(battery.ARMS))
    scenario = chain_scenario()
    for arm in battery.ARMS:
        record = battery.run_battery_episode(
            scenario,
            arm,
            workdir=tmp_path / arm,
            snapshot=None,
            consumer_factory=lambda snapshot: pytest.fail("factory called despite shared consumer"),
            limits=LIMITS,
            embedder=LexicalEmbedder(),
            rag_store_factory=rag_store_factory,
            shared_consumer=shared,
        )
        assert record["failure"] is None and record["terminal_status"] == "completed"
    assert closes == []  # caller owns the shared consumer's lifecycle


# --- summary math -------------------------------------------------------------


def _episode(arm, **overrides):
    base = {
        "scenario_id": "s",
        "arm": arm,
        "retrieval_attempted": True,
        "retrieve_first": True,
        "verdict": "valid_cited_final",
        "terminal_status": "completed",
        "failure": None,
        "model_calls": 2,
        "input_tokens": 10,
        "output_tokens": 4,
    }
    base.update(overrides)
    return base


def test_summarize_per_arm_rates_and_costs():
    episodes = [
        _episode("dml_memory"),
        _episode("dml_memory", verdict="unsupported_final"),
        _episode("rag_memory", verdict="empty_claims_final", abstention_cause="records_returned_no_support"),
        _episode("rag_memory", terminal_status="step_limit", verdict=None, retrieval_attempted=True),
        _episode("context_inline", verdict="empty_claims_final", abstention_cause="no_records_returned"),
        _episode("no_memory", verdict="unsupported_final", retrieval_attempted=True),
        _episode("no_memory", terminal_status="model_error", verdict=None, input_tokens=None),
    ]
    summary = battery.summarize(episodes)
    assert set(summary) == set(battery.ARMS)
    dml = summary["dml_memory"]
    assert dml["role"] == "persistent"
    assert dml["episodes"] == 2
    assert dml["retrieval_attempted_rate"] == 1.0
    assert dml["valid_cited_final_rate"] == 0.5
    assert dml["unsupported_final_rate"] == 0.5
    assert dml["abstention_rate"] == 0.0
    assert dml["abstention_causes"] == {cause: 0 for cause in battery.ABSTENTION_CAUSES}
    assert dml["step_limit_rate"] == 0.0
    assert dml["input_tokens"] == 20 and dml["output_tokens"] == 8
    # cost_per_valid_final: all attempted costs in the numerator (2 x 14).
    assert dml["cost_per_valid_final"] == 28.0
    rag = summary["rag_memory"]
    assert rag["abstention_rate"] == 0.5
    assert rag["abstention_causes"]["records_returned_no_support"] == 1
    assert rag["step_limit_rate"] == 0.5
    # The step-limit episode is retained as a failure, never collapsed.
    assert rag["failures"] == [{"scenario_id": "s", "failure": "terminal_status:step_limit"}]
    inline = summary["context_inline"]
    # The context floor is labeled by role and excluded from the persistent
    # grouping; its abstention cause breakdown is pre-registered (C6).
    assert inline["role"] == "context_floor"
    assert inline["abstention_rate"] == 1.0
    assert inline["abstention_causes"]["no_records_returned"] == 1
    empty = summary["no_memory"]
    assert empty["role"] == "policy_floor"
    # Zero valid finals: no cost rate is issued (never a free pass), and the
    # unknown token count makes the input aggregate unknown.
    assert empty["valid_cited_final_rate"] == 0.0
    assert empty["cost_per_valid_final"] is None
    assert empty["input_tokens"] is None and empty["output_tokens"] == 8
    assert empty["failures"] == [{"scenario_id": "s", "failure": "terminal_status:model_error"}]


def test_summarize_absent_arms_omitted():
    assert battery.summarize([_episode("dml_memory")]) == {
        "dml_memory": battery.summarize([_episode("dml_memory")])["dml_memory"]
    }


# --- CLI ----------------------------------------------------------------------


def test_main_dry_run_and_snapshot_requirement(tmp_path, capsys):
    output = tmp_path / "episodes.jsonl"
    argv = ["--scenarios-dir", str(FIXTURES), "--workdir-root", str(tmp_path / "work"), "--output", str(output)]
    assert battery.main(argv + ["--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] == "fixtures_valid" and report["arms"] == list(battery.ARMS)
    assert len(report["scenarios"]) == len(schema.load_scenarios(FIXTURES))
    assert all(len(s["event_stream_digest"]) == 64 for s in report["scenarios"])
    assert battery.main(argv) == 2
    assert not output.exists()


def test_main_end_to_end_all_arms_with_doubles(tmp_path):
    output = tmp_path / "episodes.jsonl"

    def factory(snapshot):
        return Llama3BatteryConsumer(
            [
                tool(query="ingest pipeline retry backoff cap", top_k=8),
                cite_containing("ingest_retry_backoff_cap", "300 seconds"),
            ]
        )

    code = battery.main(
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
        ],
        consumer_factory=factory,
        embedder_factory=LexicalEmbedder,
        rag_store_factory=rag_store_factory,
    )
    assert code == 0
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["arm"] for record in records] == list(battery.ARMS)
    for record in records:
        assert set(record) == set(battery.RECORD_KEYS)
        assert record["failure"] is None
    # Event-clock parity: one stream digest across the four arms.
    assert len({record["event_stream_digest"] for record in records}) == 1
    by_arm = {record["arm"]: record for record in records}
    assert by_arm["dml_memory"]["verdict"] == "valid_cited_final"
    assert by_arm["rag_memory"]["verdict"] == "valid_cited_final"
    assert by_arm["no_memory"]["presented_record_ids"] == []
    # C2: the inline arm carries its cell tag and an empty store.
    assert by_arm["context_inline"]["cell"] == "context_inline"
    assert by_arm["context_inline"]["ingestion"] == {"events": 0, "supersessions": 0}
    assert by_arm["context_inline"]["presented_record_ids"] == []
    # C4: structural return parity between the persistent arms.
    assert by_arm["dml_memory"]["return_parity_digest"] == by_arm["rag_memory"]["return_parity_digest"]
    # C5: the first retrieve query is recorded and identical across the
    # memory-bearing arms of this scenario.
    queries = {arm: by_arm[arm]["first_retrieve_query"] for arm in battery.MEMORY_BEARING_ARMS}
    assert len(set(queries.values())) == 1 and None not in queries.values()
    summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == battery.SUMMARY_SCHEMA
    assert summary["episodes"] == 4 and summary["consumer_profile"] == LLAMA3_SFT_V3_CONSUMER_PROFILE
    assert summary["retrieve_top_k_cap"] == battery.TOP_K_CAP
    assert summary["persistent_arms"] == ["dml_memory", "rag_memory"]
    assert summary["arm_roles"]["no_memory"] == "policy_floor"
    assert summary["arm_roles"]["context_inline"] == "context_floor"
    assert set(summary["per_arm"]) == set(battery.ARMS)
    assert summary["per_arm"]["dml_memory"]["valid_cited_final_rate"] == 1.0
    assert summary["per_arm"]["rag_memory"]["valid_cited_final_rate"] == 1.0
    assert summary["per_arm"]["no_memory"]["retrieval_attempted_rate"] == 1.0
    (check,) = summary["first_retrieve_query_checks"]
    assert check["scenario_id"] == "expcorr_ingest_backoff_chain" and check["identical"] is True
    assert check["note"] is None


def test_main_shares_one_consumer_across_arms(monkeypatch, tmp_path):
    constructions = []

    def factory(snapshot):
        constructions.append(snapshot)
        return Llama3BatteryConsumer([final({"claims": []})] * 10)

    monkeypatch.setattr(battery.control, "default_consumer_factory", factory)
    code = battery.main(
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
        ],
        embedder_factory=LexicalEmbedder,
        rag_store_factory=rag_store_factory,
    )
    assert code == 0
    assert len(constructions) == 1  # one admission for the whole run, all arms
    records = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert len(records) == len(battery.ARMS) and all(r["failure"] is None for r in records)


def test_consumer_profile_qwen3_requires_development_runtime_x86(tmp_path, capsys):
    code = battery.main(
        [
            "--scenarios-dir",
            str(FIXTURES),
            "--workdir-root",
            str(tmp_path / "work"),
            "--output",
            str(tmp_path / "out.jsonl"),
            "--snapshot",
            str(tmp_path),
            "--consumer-profile",
            "qwen3",
        ]
    )
    assert code == 2
    assert "requires --development-runtime-x86" in capsys.readouterr().err
    assert not (tmp_path / "out.jsonl").exists()


def test_consumer_profile_default_remains_llama3(tmp_path):
    output = tmp_path / "episodes.jsonl"

    def factory(snapshot):
        return Llama3BatteryConsumer([final({"claims": []})])

    code = battery.main(
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
            "changing_facts_project_status",
            "--arms",
            "no_memory",
        ],
        consumer_factory=factory,
    )
    assert code == 0
    record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert record["consumer_profile"] == LLAMA3_SFT_V3_CONSUMER_PROFILE


def test_qwen3_profile_episode_carries_qwen_profile_id(tmp_path):
    class Qwen3BatteryConsumer(ScriptedConsumer):
        def compile(self, messages, tools, *, output_reserved_tokens):
            artifact = super().compile(messages, tools, output_reserved_tokens=output_reserved_tokens)
            return replace(
                artifact,
                identity=replace(artifact.identity, runtime_identity="dml-qwen3-sft-action-runtime-v1:" + "e" * 64),
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

    record = battery.run_battery_episode(
        chain_scenario(),
        "rag_memory",
        workdir=tmp_path,
        snapshot=None,
        consumer_factory=lambda snapshot: Qwen3BatteryConsumer(
            [tool(query="service port", top_k=8), cite_containing("port", "9000")]
        ),
        limits=LIMITS,
        consumer_profile=QWEN3_SFT_V1_CONSUMER_PROFILE,
        embedder=LexicalEmbedder(),
        rag_store_factory=rag_store_factory,
    )
    assert set(record) == set(battery.RECORD_KEYS)
    assert record["consumer_profile"] == "qwen3_5-9b-sft-v1-bf16-action-json-v1"
    assert record["failure"] is None and record["verdict"] == "valid_cited_final"


def test_development_checkpoint_flag_validation(tmp_path, monkeypatch, capsys):
    base_args = [
        "--scenarios-dir",
        str(FIXTURES),
        "--workdir-root",
        str(tmp_path / "work"),
        "--output",
        str(tmp_path / "out.jsonl"),
        "--snapshot",
        str(tmp_path),
        "--arms",
        "no_memory",
    ]
    decl = tmp_path / "decl.json"
    decl.write_text(
        json.dumps(
            {
                "label": "development-v4-gate",
                "manifest_sha256": "a" * 64,
                "adapter_sha256": "b" * 64,
            }
        )
    )
    # requires --development-runtime-x86
    assert battery.main(base_args + ["--development-checkpoint", str(decl)]) == 2
    # malformed declaration rejected fail-closed
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"label": "qualified-x"}))
    assert battery.main(base_args + ["--development-runtime-x86", "--development-checkpoint", str(bad)]) == 2
    # valid declaration reaches the factory with the declared identity
    seen = {}

    def spy(snapshot, development_checkpoint=None):
        seen["identity"] = development_checkpoint
        raise RuntimeError("stop before model load")

    monkeypatch.setattr(battery.control, "dev_x86_consumer_factory", spy)
    monkeypatch.setenv("DML_DEV_RUNTIME", "development-x86-20261102")
    code = battery.main(
        base_args
        + [
            "--development-runtime-x86",
            "--development-checkpoint",
            str(decl),
            "--fixtures-subset",
            "changing_facts_project_status",
        ]
    )
    assert code == 0 and seen["identity"]["label"] == "development-v4-gate"
    records = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert records and all(r["failure"] == "RuntimeError: stop before model load" for r in records)


# --- review conditions C2-C7 (2026-11-09 review) ----------------------------


def test_context_inline_arm_embeds_stream_with_functional_empty_retrieve(tmp_path):
    # C2/C3: the inline_evidence construction is a full arm — events embedded
    # in the prompt as untrusted inline source data, retrieve present and
    # functional on an empty store, cell tagged context_inline.
    scenario = chain_scenario()
    holder = {}

    def factory(snapshot):
        consumer = Llama3BatteryConsumer([tool(query="service port", top_k=8), final({"claims": []})])
        holder["consumer"] = consumer
        return consumer

    record = battery.run_battery_episode(
        scenario,
        "context_inline",
        workdir=tmp_path,
        snapshot=None,
        consumer_factory=factory,
        limits=LIMITS,
    )
    assert record["failure"] is None and record["terminal_status"] == "completed"
    assert record["cell"] == "context_inline"
    assert record["backend"] == {"kind": "empty_store", "evidence": "inline_prompt"}
    assert record["ingestion"] == {"events": 0, "supersessions": 0}
    # The prompt embeds every fixture event behind the untrusted-data header.
    prompt = holder["consumer"].requests[0]["messages"][-1]["content"]
    assert battery.control.INLINE_HEADER in prompt
    for event in scenario.events:
        assert f"record {event.record_id}: {event.text}" in prompt
    # Retrieve is present and functional; the store is empty.
    assert record["retrieval_attempted"] is True and record["retrieve_first"] is True
    feedback = json.loads(holder["consumer"].requests[1]["messages"][-1]["content"])
    assert feedback["records"] == [] and feedback["returned_count"] == 0
    # Event-clock parity: the inline arm shares the scenario stream digest.
    dml = run_episode(scenario, "dml_memory", [final({"claims": []})], tmp_path / "dml")
    assert record["event_stream_digest"] == dml["event_stream_digest"]
    # The inline arm is not part of the memory-bearing pairing check.
    assert battery.first_retrieve_query_checks([record]) == []


def test_fourth_arm_stream_parity(tmp_path):
    # C2: the fourth arm sees the identical validated event stream digest.
    scenario = chain_scenario()
    digests = {
        arm: run_episode(scenario, arm, [final({"claims": []})], tmp_path / arm)["event_stream_digest"]
        for arm in battery.ARMS
    }
    assert len(set(digests.values())) == 1


def test_return_parity_digest_structural_parity(tmp_path):
    # C4: with the same first query, the persistent arms' presented-record
    # list shapes (presentation order + field sets + requested top_k) are
    # structurally identical; an empty-return arm digests differently.
    scenario = chain_scenario()
    records = {}
    presented = {}
    for arm in battery.PERSISTENT_ARMS:
        holder = {}

        def factory(snapshot, holder=holder):
            consumer = Llama3BatteryConsumer([tool(query="service port", top_k=8), cite_containing("port", "9000")])
            holder["consumer"] = consumer
            return consumer

        records[arm] = battery.run_battery_episode(
            scenario,
            arm,
            workdir=tmp_path / arm,
            snapshot=None,
            consumer_factory=factory,
            limits=LIMITS,
            embedder=LexicalEmbedder(),
            rag_store_factory=rag_store_factory,
        )
        assert records[arm]["failure"] is None
        feedback = json.loads(holder["consumer"].requests[1]["messages"][-1]["content"])
        assert feedback["requested_top_k"] == battery.TOP_K_CAP
        presented[arm] = feedback["records"]
    dml_digest = records["dml_memory"]["return_parity_digest"]
    assert dml_digest is not None and dml_digest == records["rag_memory"]["return_parity_digest"]
    # Identical presentation field sets, position by position (retrieval rank).
    assert [sorted(record) for record in presented["dml_memory"]] == [
        sorted(record) for record in presented["rag_memory"]
    ]
    assert presented["dml_memory"]  # parity is over nonempty presentations here
    empty = run_episode(
        scenario, "no_memory", [tool(query="service port", top_k=8), final({"claims": []})], tmp_path / "empty"
    )
    assert empty["return_parity_digest"] != dml_digest


def test_first_retrieve_query_recorded_and_mismatch_flagged(tmp_path):
    # C5: the first retrieve query is recorded per episode; the per-scenario
    # identity check flags divergence with a retained note, never silent.
    scenario = chain_scenario()
    dml = run_episode(
        scenario, "dml_memory", [tool(query="service port", top_k=8), final({"claims": []})], tmp_path / "dml"
    )
    rag = run_episode(
        scenario, "rag_memory", [tool(query="service port", top_k=8), final({"claims": []})], tmp_path / "rag"
    )
    assert dml["first_retrieve_query"] == "service port" == rag["first_retrieve_query"]
    (check,) = battery.first_retrieve_query_checks([dml, rag])
    assert check["identical"] is True and check["note"] is None
    assert check["queries"] == {"dml_memory": "service port", "rag_memory": "service port"}
    # A divergence is retained with an explicit difference note.
    diverged = dict(rag, first_retrieve_query="a different query")
    (check,) = battery.first_retrieve_query_checks([dml, diverged])
    assert check["identical"] is False and "DIFFERENCE RETAINED" in check["note"]
    assert check["queries"]["rag_memory"] == "a different query"
    # An arm that never retrieves is reported, not silently dropped.
    no_call = run_episode(scenario, "dml_memory", [final({"claims": []})], tmp_path / "no_call")
    assert no_call["first_retrieve_query"] is None
    (check,) = battery.first_retrieve_query_checks([no_call, rag])
    assert check["identical"] is False and check["note"] is not None


def test_abstention_cause_all_three_values(tmp_path):
    # C6: every empty_claims_final carries a pre-registered abstention cause.
    scenario = chain_scenario()
    direct = run_episode(scenario, "dml_memory", [final({"claims": []})], tmp_path / "direct")
    assert direct["verdict"] == "empty_claims_final" and direct["abstention_cause"] == "no_retrieve_call"
    empty = run_episode(
        scenario, "no_memory", [tool(query="service port", top_k=8), final({"claims": []})], tmp_path / "empty"
    )
    assert empty["verdict"] == "empty_claims_final" and empty["abstention_cause"] == "no_records_returned"
    supported = run_episode(
        scenario, "dml_memory", [tool(query="service port", top_k=8), final({"claims": []})], tmp_path / "supported"
    )
    assert supported["presented_record_ids"] and supported["verdict"] == "empty_claims_final"
    assert supported["abstention_cause"] == "records_returned_no_support"
    answered = run_episode(
        scenario,
        "dml_memory",
        [tool(query="service port", top_k=8), cite_containing("port", "9000")],
        tmp_path / "answered",
    )
    assert answered["verdict"] == "valid_cited_final" and answered["abstention_cause"] is None
    summary = battery.summarize([direct, empty, supported, answered])
    assert summary["dml_memory"]["abstention_causes"] == {
        "no_records_returned": 0,
        "records_returned_no_support": 1,
        "no_retrieve_call": 1,
    }
    assert summary["no_memory"]["abstention_causes"]["no_records_returned"] == 1
