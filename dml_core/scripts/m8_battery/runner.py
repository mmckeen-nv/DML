"""Development-only M8 competence battery runner.

Generalizes m8_neutral_smoke: load strict scenario fixtures from a directory,
build the three neutral arms in a fresh workdir per scenario, run the same
compile/execute/dispatch/validate consumer loop per arm, and emit one JSONL
evidence record per arm-episode plus a compact per-family competence summary
(retrieval-attempted rate, valid-cited-final rate, retained failures).

This is a development protocol-competence battery, not a benchmark and not
held-out data; no answer keys gate the loop beyond fixture support
verification, and every failure is retained as evidence.

Station use plugs the pinned MiniLM in through the embedder factory and the
authenticated snapshot consumer through the consumer factory; tests inject
doubles and never load a model.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

from daystrom_dml.services import m8_protocol as protocol
from scripts.m8_battery import schema as fixture_schema
from scripts.m8_battery import verifiers

ARMS = protocol.ARMS
EVIDENCE_SCHEMA = "dml.m8.competence-battery.v1"
SUMMARY_SCHEMA = "dml.m8.competence-battery-summary.v1"
EMBEDDING_DIMENSIONS = 384
TOKEN_BUDGET = 2048
MAX_MODEL_CALLS = 6


def default_embedder_factory():
    """Pinned station MiniLM (CPU fp32 mean-pool L2, 384 dims)."""
    from scripts.m8_neutral_smoke import MiniLMEmbedder

    return MiniLMEmbedder()


def default_consumer_factory(snapshot, arm, *, seed=0):
    from daystrom_dml.services.m8_neutral_runtime import M8NeutralConsumer

    return M8NeutralConsumer(snapshot, arm=arm, offline=False, seed=seed)


def _build_dml_bridge(embedder, workdir, scope):
    from daystrom_dml.dml_adapter import DMLAdapter
    from daystrom_dml.settings import DMLSettings
    from daystrom_dml.services.m8_dml_memory import DMLDevelopmentEventMemory
    from scripts.m8_neutral_smoke import EMBEDDING_IDENTITY, EMBEDDING_MODEL

    config = {
        "production_profile": "dml-receipted-local-v1",
        "storage_dir": str(workdir / "dml-authority"),
        "model_name": "dummy",
        "llm_backend": "dummy",
        "embedding_model": EMBEDDING_MODEL,
        "strict_embedding_required": True,
        "persistence": {
            "enable": False,
            "interval_sec": 0,
            "journal": True,
            "receipts": True,
            "outbox": False,
            "receipt_embedding_identity": EMBEDDING_IDENTITY,
            "snapshot_interval": 1,
        },
        "rag_store": {"enable": False},
        "dpm": {
            "enable": False,
            "mode": "disabled",
            "include_in_context": False,
            "include_in_preamble": False,
        },
        "ann_min_items": 0,
        "checkpoint_interval_seconds": 0,
        "skip_rag_state_import": True,
        "survival_ledger_enabled": False,
        "background_processing_enabled": False,
        "enable_stm_controller": False,
        "enable_workflow_cache": False,
        "enable_quality_on_retrieval": False,
        "mirror_agentic_memory_to_rag": False,
        "gpu_acceleration": False,
        "metrics_enabled": False,
    }
    adapter = DMLAdapter(_validated_settings=DMLSettings(**config), embedder=embedder, start_aging_loop=False)
    bridge = DMLDevelopmentEventMemory(adapter, scope=scope, evidence_path=workdir / "dml-evidence.jsonl")
    return adapter, bridge


def default_store_factory(scenario, workdir, *, embedder, snapshot):
    """Build the three arm stores, ingest all events, prove RAG durability."""
    from daystrom_dml.services.m8_memory_baseline import ContextBuffer, PersistentRAGBaseline
    from scripts.m8_neutral_smoke import EMBEDDING_IDENTITY

    rag_path = workdir / "rag.sqlite"
    rag = PersistentRAGBaseline(rag_path, embedding_identity=EMBEDDING_IDENTITY, dimensions=EMBEDDING_DIMENSIONS)
    adapter, bridge = _build_dml_bridge(embedder, workdir, scenario.scope)
    holder = {}

    def count(text):
        if "tok" not in holder:
            from transformers import AutoTokenizer

            holder["tok"] = AutoTokenizer.from_pretrained(
                str(Path(snapshot) / "model"), local_files_only=True, trust_remote_code=False
            )
        return protocol.count_tokens(holder["tok"], text)

    buffer = ContextBuffer(
        scope=scenario.scope,
        token_budget=TOKEN_BUDGET,
        count_tokens=count,
        render_records=protocol.render_records,
    )
    for event in scenario.events:
        rag.ingest(event, embedder.embed(event.text))
        bridge.ingest(event)
        buffer.ingest(event)
    rag.close()
    # Reopen to prove durable ingestion before the query path.
    rag = PersistentRAGBaseline(rag_path, embedding_identity=EMBEDDING_IDENTITY, dimensions=EMBEDDING_DIMENSIONS)
    stores = {"rag": rag, "bridge": bridge, "context_records": buffer.present().records}

    def close():
        adapter.close()
        rag.close()

    return stores, close


def run_arm_episode(
    arm,
    scenario,
    stores,
    *,
    embedder,
    consumer_factory,
    snapshot,
    max_model_calls=MAX_MODEL_CALLS,
    seed=0,
):
    """One neutral consumer lifecycle for one scenario arm; retain everything."""
    record = {
        "schema": EVIDENCE_SCHEMA,
        "scenario_id": scenario.scenario_id,
        "family": scenario.family,
        "arm": arm,
        "sampling_seed": seed,
        "expects_retrieval": scenario.expects_retrieval,
        "retrieval_attempted": False,
        "calls": [],
        "presented_record_ids": [],
        "verdict": None,
        "failure": None,
    }
    supports = verifiers.supports_for_truth(scenario.truth)
    scope = scenario.scope
    context_records = list(stores["context_records"]) if arm == "context_only" else []
    messages = protocol.initial_messages(arm, scenario.query, records=context_records, scope=scope)
    presented = list(context_records)
    try:
        with consumer_factory(snapshot, arm) as consumer:
            record["runtime_identity"] = consumer.identity["runtime_identity"]

            def count(text):
                # Exact pinned tokenizer already admitted by this consumer.
                return protocol.count_tokens(consumer._tokenizer, text)

            for step in range(1, max_model_calls + 1):
                compiled = consumer.compile(messages, scope=scope)
                result = consumer.execute(compiled)
                record["calls"].append(
                    {
                        "step": step,
                        "request_digest": compiled.request_digest,
                        "input_token_count": result["input_token_count"],
                        "output_token_count": result["output_token_count"],
                        "finish_reason": result.get("finish_reason"),
                        "action": result.get("derived_action"),
                        "action_error": result.get("action_error"),
                    }
                )
                action = result.get("derived_action")
                if action is None:
                    record["failure"] = "unparseable action retained: " + str(result.get("action_error"))
                    break
                if action["kind"] == "final":
                    answer = protocol.validate_final(action, presented, scope=scope, supports=supports)
                    record["verdict"] = "valid_cited_final"
                    record["answer"] = answer
                    actual = {(c["key"], c["value"]) for c in answer["claims"]}
                    record["truth_covered"] = verifiers.expected_claims(scenario.truth) <= actual
                    break
                # Tool action: retrieve through THIS arm's store and continue.
                record["retrieval_attempted"] = True
                args = action["arguments"]
                if arm == "persistent_rag":
                    ev = stores["rag"].retrieve(
                        embedder.embed(args["query"]),
                        scope=scope,
                        now=scenario.query_clock,
                        top_k=args["top_k"],
                        token_budget=TOKEN_BUDGET,
                        count_tokens=count,
                        render_records=protocol.render_records,
                    )
                elif arm == "dml":
                    ev = stores["bridge"].retrieve(
                        args["query"],
                        now=scenario.query_clock,
                        top_k=args["top_k"],
                        token_budget=TOKEN_BUDGET,
                        count_tokens=count,
                        render_records=protocol.render_records,
                    )["evidence"]
                else:
                    record["failure"] = "context_only issued a tool action"
                    break
                presented.extend(deepcopy(list(ev.records)))
                messages += protocol.tool_feedback(
                    result["derived_action_text"],
                    arm=arm,
                    call_id=f"retrieve-{step}",
                    records=list(ev.records),
                    scope=scope,
                )
            else:
                record["failure"] = "model call ceiling reached without final"
    except Exception as exc:  # noqa: BLE001 — every failure is retained evidence
        record["failure"] = f"{type(exc).__name__}: {exc}"
    record["presented_record_ids"] = [r["record_id"] for r in presented]
    return record


def run_scenario(
    scenario,
    *,
    arms,
    workdir,
    snapshot,
    embedder,
    store_factory,
    consumer_factory,
    max_model_calls=MAX_MODEL_CALLS,
    seed=0,
):
    """Run every requested arm for one scenario in its own fresh workdir."""
    stores, close = store_factory(scenario, workdir, embedder=embedder, snapshot=snapshot)
    try:
        return [
            run_arm_episode(
                arm,
                scenario,
                stores,
                embedder=embedder,
                consumer_factory=consumer_factory,
                snapshot=snapshot,
                max_model_calls=max_model_calls,
                seed=seed,
            )
            for arm in arms
        ]
    finally:
        close()


FAMILY_ORDER = fixture_schema.FAMILIES


def summarize(episodes):
    """Compact per-family competence summary over arm-episode records.

    retrieval_attempted_rate is computed over episodes where the scenario
    expects retrieval and the arm advertises the retrieve tool (context_only
    is never eligible); it is null when a family has no eligible episodes.
    Failures are retained, never collapsed.
    """
    families = {}
    for episode in episodes:
        entry = families.setdefault(
            episode["family"],
            {
                "scenarios": set(),
                "episodes": 0,
                "valid_cited_final": 0,
                "retrieval_eligible": 0,
                "retrieval_attempted": 0,
                "failures": [],
            },
        )
        entry["scenarios"].add(episode["scenario_id"])
        entry["episodes"] += 1
        if episode.get("verdict") == "valid_cited_final":
            entry["valid_cited_final"] += 1
        if episode.get("expects_retrieval") and episode["arm"] != "context_only":
            entry["retrieval_eligible"] += 1
            if episode.get("retrieval_attempted"):
                entry["retrieval_attempted"] += 1
        if episode.get("failure"):
            entry["failures"].append(
                {
                    "scenario_id": episode["scenario_id"],
                    "arm": episode["arm"],
                    "failure": episode["failure"],
                }
            )
    summary = {}
    for family in FAMILY_ORDER:
        if family not in families:
            continue
        entry = families[family]
        eligible = entry["retrieval_eligible"]
        summary[family] = {
            "scenarios": len(entry["scenarios"]),
            "episodes": entry["episodes"],
            "valid_cited_final_rate": round(entry["valid_cited_final"] / entry["episodes"], 4),
            "retrieval_attempted_rate": (round(entry["retrieval_attempted"] / eligible, 4) if eligible else None),
            "failures": entry["failures"],
        }
    return summary


def _parse_arms(value):
    arms = tuple(part.strip() for part in value.split(","))
    if not arms or any(arm not in ARMS for arm in arms) or len(set(arms)) != len(arms):
        raise argparse.ArgumentTypeError(f"--arms must be a comma subset of {','.join(ARMS)}")
    return arms


def _parse_seeds(value):
    try:
        seeds = tuple(int(part.strip()) for part in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("--seeds must be comma-separated nonnegative integers") from None
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("--seeds must be distinct nonnegative integers")
    return seeds


def _fresh_workdir(root, scenario_id):
    target = root / scenario_id
    if target.exists() and any(target.iterdir()):
        raise fixture_schema.FixtureError(f"workdir is not fresh: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def main(argv=None, *, embedder_factory=None, consumer_factory=None, store_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios-dir", required=True)
    parser.add_argument("--workdir-root", required=True)
    parser.add_argument("--output", required=True, help="episode evidence JSONL path")
    parser.add_argument("--arms", type=_parse_arms, default=ARMS)
    parser.add_argument("--snapshot", help="authenticated snapshot directory; required unless --dry-run")
    parser.add_argument("--max-model-calls", type=int, default=MAX_MODEL_CALLS)
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=(0,),
        help="comma-separated sampler seeds; seed 0 is the pinned qualified default, "
        "declared alternatives label development diagnostics only",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate fixtures only; no stores or models")
    args = parser.parse_args(argv)

    started = time.time()
    try:
        scenarios = fixture_schema.load_scenarios(Path(args.scenarios_dir))
    except fixture_schema.FixtureError as exc:
        print(json.dumps({"fixture_error": str(exc)}), file=sys.stderr)
        return 1

    if args.dry_run:
        report = [
            {
                "scenario_id": s.scenario_id,
                "family": s.family,
                "events": len(s.events),
                "truth_claims": len(s.truth),
                "expects_retrieval": s.expects_retrieval,
            }
            for s in scenarios
        ]
        print(json.dumps({"dry_run": "fixtures_valid", "scenarios": report}, indent=1))
        return 0

    if not args.snapshot:
        print(json.dumps({"error": "--snapshot is required unless --dry-run"}), file=sys.stderr)
        return 2
    if args.max_model_calls < 1:
        print(json.dumps({"error": "--max-model-calls must be positive"}), file=sys.stderr)
        return 2

    embedder = (embedder_factory or default_embedder_factory)()
    consumer_injected = consumer_factory is not None
    consumer_factory = consumer_factory or default_consumer_factory
    store_factory = store_factory or default_store_factory
    workdir_root = Path(args.workdir_root)
    workdir_root.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    episodes = []
    try:
        with output.open("w", encoding="utf-8") as stream:
            for seed in args.seeds:
                seed_consumer_factory = consumer_factory
                if not consumer_injected:

                    def seed_consumer_factory(snapshot, arm, _seed=seed):
                        return default_consumer_factory(snapshot, arm, seed=_seed)

                for scenario in scenarios:
                    workdir = _fresh_workdir(workdir_root, f"{scenario.scenario_id}__seed{seed}")
                    for episode in run_scenario(
                        scenario,
                        arms=args.arms,
                        workdir=workdir,
                        snapshot=args.snapshot,
                        embedder=embedder,
                        store_factory=store_factory,
                        consumer_factory=seed_consumer_factory,
                        max_model_calls=args.max_model_calls,
                        seed=seed,
                    ):
                        episodes.append(episode)
                        stream.write(json.dumps(episode, sort_keys=True) + "\n")
                        stream.flush()
    except fixture_schema.FixtureError as exc:
        print(json.dumps({"fixture_error": str(exc)}), file=sys.stderr)
        return 1

    summary = {
        "schema": SUMMARY_SCHEMA,
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - started, 1),
        "arms": list(args.arms),
        "seeds": sorted(args.seeds),
        "scenarios": len(scenarios),
        "episodes": len(episodes),
        "families": summarize(episodes),
        "limitations": (
            "Development protocol-competence battery only: fixture workloads, "
            "scripted retrieval dispatch, no benchmark, no held-out data, no "
            "quality claim; every failure retained."
        ),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps(summary["families"], indent=1))
    print("episodes:", len(episodes), "evidence:", output, "summary:", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
