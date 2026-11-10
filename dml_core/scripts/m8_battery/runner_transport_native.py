"""Development-only transport-native M8 battery runner (amendment executable form).

Implements the proposed arm binding of docs/m8-transport-native-fairness-2026-11-05.md:
every arm runs through the DML surface — same policy, same system prompt
composition, same action schema, same sampler, same step limits, same episode
machinery (``run_episode_with_test_dependencies`` + the receipted
``SelectedProfileEpisodeTools`` bridge, retrieve-only allowlist). The arms
differ in exactly one component: what the ``retrieve`` tool is backed by.

- ``dml_memory``: the DML receipt store, fed through control.py's existing
  receipted ingestion path (``ingest_scenario``; corrections as receipted
  supersessions).
- ``rag_memory``: the independent RAG baseline
  (``daystrom_dml.services.m8_memory_baseline.PersistentRAGBaseline``) fed the
  SAME event stream, receipt-free; retrieval ranking/selection is the
  baseline's own cosine+recency current view.
- ``no_memory``: the retrieve tool is present but its backend discards the
  stream and always returns zero records (the policy with a dead retrieve).

Fairness contract: all three backends receive the identical fixture event
stream; ``event_stream_digest`` (a canonical digest of the validated event
envelopes) is recorded per episode and MUST match across arms of one
scenario/seed — that equality is the event-clock parity evidence. The episode
contract requires the exact ``SelectedProfileEpisodeTools`` type under
execution protocol v2, so the per-arm backend enters behind the bridge's
adapter seam: the bridge still owns scope, presentation, CAS references and
the model-facing result shape; the backend only answers
``retrieve_context``. ``presented_record_ids`` are backend-assigned integer
ids (DML-assigned for dml_memory; ingestion-order ids for rag_memory), never
fixture record_ids.

Verifier, fixtures, evidence discipline (retain everything, repair nothing),
token accounting (any unknown per-call count makes the total unknown) and the
consumer/development-runtime flags are control.py's, unchanged.

KNOWN GAPS (inherited from control.py unless noted; nothing here is faked):

1. NOT live_local evidence: episodes use the explicitly labeled in-process
   injection path (``execution_path == "test_injected"``); no supervisor
   deadline/kill, no worker isolation, no seed_receipts_digest binding.
2. Pinned sampler: ``--seeds`` replicates identical runs (interface parity).
3. Authority time: fixture timestamps are not bound into DML authority time;
   retrieval ``as_of`` uses the episode convention effective time. The RAG
   baseline ranks by its own envelope timestamps (event-clock parity contract
   `dml-append-request-v2` remains pending independent API review).
4. The rag_memory backend's top_k is clamped to the baseline's frozen limit
   (8); the cap is declared in the backend identity of every rag episode.
5. rag_memory presentation records are synthesized in the DML record shape
   from the baseline's envelopes so the unchanged truth-based verifier can
   ground citations in presented records; they carry no DML receipts and are
   labeled by the backend identity.
6. A station run requires the authenticated snapshot (and, for rag_memory, the
   pinned MiniLM embedder); anywhere else only injected doubles execute. Tests
   never load a model.

This is a development diagnostic only: no benchmark, no held-out data, no
quality or competence claim; every failure is retained as evidence.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

from daystrom_dml.contracts.agent_episode import (
    EXECUTION_PROTOCOL_V2,
    LLAMA3_SFT_V3_CONSUMER_PROFILE,
    QWEN3_SFT_V1_CONSUMER_PROFILE,
)
from daystrom_dml.contracts.profile import PROFILE_ID
from daystrom_dml.services import m8_accounting
from daystrom_dml.services.agent_episode import (
    EpisodeLimits,
    _fixture_adapter,
    _read_records,
    run_episode_with_test_dependencies,
)
from daystrom_dml.services.episode_tools import SelectedProfileEpisodeTools, canonical
from daystrom_dml.services.m8_memory_baseline import PersistentRAGBaseline, validate_event
from scripts.m8_battery import control
from scripts.m8_battery import schema as fixture_schema

ARMS = ("dml_memory", "rag_memory", "no_memory")
EVIDENCE_SCHEMA = "dml.m8.transport-native-battery.v1"
SUMMARY_SCHEMA = "dml.m8.transport-native-battery-summary.v1"
TASK_ID = "m8-transport-native-battery"
EPISODE_EFFECTIVE_TIME = control.EPISODE_EFFECTIVE_TIME
ALLOWED_TOOLS = control.ALLOWED_TOOLS
PROFILES = control.PROFILES
EMBEDDING_DIMENSIONS = 384
TOKEN_BUDGET = 2048
RAG_TOP_K_CAP = 8  # PersistentRAGBaseline's frozen retrieval limit

RECORD_KEYS = (
    "schema",
    "scenario_id",
    "family",
    "arm",
    "backend",
    "cell",
    "consumer_profile",
    "sampling_seed",
    "expects_retrieval",
    "runtime_identity",
    "development_runtime",
    "development_checkpoint",
    "retrieval_attempted",
    "retrieve_first",
    "calls",
    "presented_record_ids",
    "ingestion",
    "event_stream_digest",
    "terminal_status",
    "verdict",
    "truth_covered",
    "answer",
    "failure",
    "model_calls",
    "input_tokens",
    "output_tokens",
    "wall_seconds",
)


def event_stream_envelopes(scenario):
    """Validated canonical envelope snapshot of the fixture event stream.

    Every arm's parity digest is computed over exactly these envelopes; the
    RAG baseline independently re-validates the same events at ingest.
    """
    return [validate_event(event) for event in scenario.events]


def event_stream_digest(scenario):
    """sha256 over the canonical validated event stream; identical across arms."""
    return hashlib.sha256(canonical(event_stream_envelopes(scenario)).encode("utf-8")).hexdigest()


def _rag_record(assigned_id, envelope, vector):
    """One baseline envelope in the DML record shape; presentation identity only.

    The id is backend-assigned (ingestion order), never the fixture record_id.
    The embedding is the exact vector committed to the baseline for this event.
    """
    return {
        "id": assigned_id,
        "text": envelope["text"],
        "timestamp": float(envelope["timestamp"]),
        "level": 0,
        "fidelity": 1.0,
        "salience": 1.0,
        "meta": {
            **envelope["scope"],
            "source": envelope["source_id"],
            "claim_key": envelope["fact_key"],
            "source_trust": envelope["metadata"].get("source_trust", "trusted"),
        },
        "embedding": [float(value) for value in vector],
    }


def _rag_item(record):
    """retrieve_context item in the DML report's shape (string id, full meta)."""
    return {
        "id": str(record["id"]),
        "text": record["text"].strip(),
        "meta": deepcopy(record["meta"]),
        "timestamp": float(record["timestamp"]),
        "level": record["level"],
        "fidelity": float(record["fidelity"]),
        "salience": float(record["salience"]),
    }


def _budget_count_tokens(text):
    """Deterministic whitespace counter; drives ONLY the baseline's budget cut."""
    return len(text.split())


def _budget_render_records(records):
    return "\n".join("- " + record["text"] for record in records)


class EmptyRetrieveBackend:
    """no_memory arm: the retrieve tool is present and always returns zero records."""

    production_profile_id = PROFILE_ID
    identity = {"kind": "empty"}

    def retrieve_context(self, query, *, top_k, as_of, tenant_id, client_id, session_id, instance_id):
        return {
            "items": [],
            "top_k": top_k,
            "as_of": as_of,
            "backend": dict(self.identity),
            "token_count_kind": "none",
            "raw_context": "",
            "context_tokens": 0,
        }


class RagRetrieveBackend:
    """rag_memory arm: retrieve served by the independent PersistentRAGBaseline.

    Ranking, cutoff and the correction-aware current view are the baseline's
    own; this facade only maps returned envelopes into the report shape the
    receipted bridge presents. top_k is clamped to the baseline's frozen
    limit; the clamp is declared in the report and the backend identity.
    """

    production_profile_id = PROFILE_ID

    def __init__(
        self,
        store,
        *,
        embedder,
        records,
        token_budget=TOKEN_BUDGET,
        count_tokens=_budget_count_tokens,
        render_records=_budget_render_records,
    ):
        self._store = store
        self._embedder = embedder
        self._records = records  # fixture record_id -> presentation record
        self._token_budget = token_budget
        self._count_tokens = count_tokens
        self._render_records = render_records
        self.identity = {
            "kind": "persistent_rag_baseline",
            "embedding_identity": store.embedding_identity,
            "dimensions": store.dimensions,
            "top_k_cap": RAG_TOP_K_CAP,
        }

    def retrieve_context(self, query, *, top_k, as_of, tenant_id, client_id, session_id, instance_id):
        scope = {"tenant_id": tenant_id, "client_id": client_id, "session_id": session_id, "instance_id": instance_id}
        effective_top_k = min(top_k, RAG_TOP_K_CAP)
        evidence = self._store.retrieve(
            self._embedder.embed(query),
            scope=scope,
            now=as_of,
            top_k=effective_top_k,
            token_budget=self._token_budget,
            count_tokens=self._count_tokens,
            render_records=self._render_records,
        )
        return {
            "items": [_rag_item(self._records[envelope["record_id"]]) for envelope in evidence.records],
            "top_k": effective_top_k,
            "as_of": as_of,
            "backend": dict(self.identity),
            "token_count_kind": "whitespace-words",
            "raw_context": evidence.context,
            "context_tokens": evidence.token_count,
        }


def default_embedder_factory():
    """Pinned station MiniLM (CPU fp32 mean-pool L2, 384 dims)."""
    from scripts.m8_neutral_smoke import MiniLMEmbedder

    return MiniLMEmbedder()


def default_rag_store_factory(path):
    from scripts.m8_neutral_smoke import EMBEDDING_IDENTITY

    return PersistentRAGBaseline(path, embedding_identity=EMBEDDING_IDENTITY, dimensions=EMBEDDING_DIMENSIONS)


def ingest_rag(store, scenario, *, embedder):
    """Feed the baseline the same event stream; corrections via its own admission.

    Ingestion is chronological per the baseline's event clock; ``corrects``
    edges are validated by the baseline itself (same-source monotonic
    correction), so the retrieved current view excludes superseded records
    without any runner-side repair.
    """
    records = {}
    for index, event in enumerate(scenario.events):
        vector = embedder.embed(event.text)
        store.ingest(event, vector)
        records[event.record_id] = _rag_record(index, validate_event(event), vector)
    return records


def run_battery_episode(
    scenario,
    arm,
    *,
    workdir,
    snapshot,
    consumer_factory,
    limits,
    seed=0,
    consumer_profile=LLAMA3_SFT_V3_CONSUMER_PROFILE,
    adapter_factory=_fixture_adapter,
    embedder=None,
    rag_store_factory=default_rag_store_factory,
    clock=time.monotonic,
    shared_consumer=None,
):
    """One transport-native arm episode; every failure is retained in the record.

    With shared_consumer the episode reuses a single admitted consumer (one
    snapshot verification + model load per RUN, not per episode); such a
    consumer is never closed here — the caller owns its lifecycle.
    """
    if arm not in ARMS:
        raise fixture_schema.FixtureError(f"unknown arm: {arm}")
    workdir.mkdir(parents=True, exist_ok=True)
    started = clock()
    record = {
        "schema": EVIDENCE_SCHEMA,
        "scenario_id": scenario.scenario_id,
        "family": scenario.family,
        "arm": arm,
        "backend": None,
        "cell": control.label_cell(scenario),
        "consumer_profile": consumer_profile,
        "sampling_seed": seed,
        "expects_retrieval": scenario.expects_retrieval,
        "runtime_identity": None,
        "development_runtime": None,
        "development_checkpoint": None,
        "retrieval_attempted": False,
        "retrieve_first": None,
        "calls": [],
        "presented_record_ids": [],
        "ingestion": {"events": 0, "supersessions": 0},
        "event_stream_digest": None,
        "terminal_status": None,
        "verdict": None,
        "truth_covered": False,
        "answer": None,
        "failure": None,
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "wall_seconds": 0.0,
    }
    episode_id = f"m8tn-{scenario.scenario_id}-{arm}-s{seed}"[:128]
    adapter = None
    store = None
    consumer = None
    try:
        record["event_stream_digest"] = event_stream_digest(scenario)
        corrections = sum(1 for event in scenario.events if event.corrects is not None)
        if arm == "dml_memory":
            directory = workdir / "authority"
            adapter = adapter_factory(directory)
            ingested = control.ingest_scenario(
                adapter, scenario, key_prefix=f"m8tn:{scenario.scenario_id}:{arm}:{seed}"
            )
            record["ingestion"] = {
                "events": len(ingested["receipts"]),
                "supersessions": len(ingested["supersessions"]),
            }
            backend = adapter
            record["backend"] = {"kind": "dml_receipt_store", "profile_id": PROFILE_ID}
            seed_receipts = ingested["receipts"]
            observations = list(ingested["records"].values())
            seed_records = {str(record_id): rec for record_id, rec in ingested["records"].items()}

            def current_records():
                return _read_records(directory)
        elif arm == "rag_memory":
            if embedder is None:
                raise fixture_schema.FixtureError("rag_memory arm requires an embedder")
            store = rag_store_factory(workdir / "rag.sqlite")
            rag_records = ingest_rag(store, scenario, embedder=embedder)
            record["ingestion"] = {"events": len(scenario.events), "supersessions": corrections}
            backend = RagRetrieveBackend(store, embedder=embedder, records=rag_records)
            record["backend"] = dict(backend.identity)
            seed_receipts = ()
            observations = list(rag_records.values())
            seed_records = {str(record_id): rec for record_id, rec in rag_records.items()}

            def current_records():
                return list(rag_records.values())
        else:
            # no_memory: the same stream is validated for the parity digest and
            # discarded; nothing is retained or presented.
            record["ingestion"] = {"events": len(scenario.events), "supersessions": corrections}
            backend = EmptyRetrieveBackend()
            record["backend"] = dict(backend.identity)
            seed_receipts = ()
            observations = []
            seed_records = {}

            def current_records():
                return []

        toolbox = SelectedProfileEpisodeTools(
            backend,
            scope=scenario.scope,
            episode_id=episode_id,
            seed_receipts=seed_receipts,
            observation_records=observations,
            allowed_tools=ALLOWED_TOOLS,
            effective_time=EPISODE_EFFECTIVE_TIME,
            execution_protocol=EXECUTION_PROTOCOL_V2,
        )
        consumer = shared_consumer if shared_consumer is not None else consumer_factory(snapshot)
        identity = getattr(consumer, "identity", None)
        record["runtime_identity"] = getattr(identity, "runtime_identity", None)
        record["development_runtime"] = getattr(consumer, "development_runtime_identity", None)
        record["development_checkpoint"] = getattr(consumer, "development_checkpoint_identity", None)
        report = run_episode_with_test_dependencies(
            consumer=consumer,
            toolbox=toolbox,
            task={"id": TASK_ID, "prompt": scenario.query},
            scenario={"id": scenario.scenario_id, "scope": deepcopy(scenario.scope)},
            seed_records=seed_records,
            current_records=current_records,
            limits=limits,
            verifier=control.control_verifier(scenario),
            effective_time=EPISODE_EFFECTIVE_TIME,
            episode_id=episode_id,
            consumer_profile=consumer_profile,
        )
        control._fill_from_report(record, report)
    except Exception as exc:  # noqa: BLE001 — every failure is retained evidence
        record["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        if shared_consumer is None:
            close = getattr(consumer, "close", None)
            if callable(close):
                close()
        if store is not None:
            store.close()
        if adapter is not None:
            adapter.close()
    record["model_calls"] = len(record["calls"])
    record["input_tokens"] = control._token_total(record["calls"], "input_token_count")
    record["output_tokens"] = control._token_total(record["calls"], "output_token_count")
    record["wall_seconds"] = clock() - started
    if set(record) != set(RECORD_KEYS):
        raise fixture_schema.FixtureError("transport-native evidence record drifted from its declared keys")
    return record


def _optional_total(values):
    """Sum measurements; any unknown (None) value makes the aggregate unknown."""
    total = 0
    for value in values:
        if value is None:
            return None
        total += value
    return total


def summarize(episodes):
    """Per-arm summary; failures are retained, never collapsed.

    Rates cover every episode of the arm. abstention is the declared
    insufficiency signal (empty_claims_final); step_limit counts episodes the
    step ceiling stopped without a final. cost_per_valid_final applies
    m8_accounting.cost_per_correct_success semantics where the runtime
    provides complete token counts and the arm has at least one valid final;
    otherwise it is None — never a silent zero and never a free pass.
    """
    per_arm = {}
    for arm in ARMS:
        arm_episodes = [episode for episode in episodes if episode["arm"] == arm]
        if not arm_episodes:
            continue
        total = len(arm_episodes)
        attempts = []
        for episode in arm_episodes:
            cost = _optional_total((episode["input_tokens"], episode["output_tokens"]))
            attempts.append({"correct": episode.get("verdict") == "valid_cited_final", "cost": cost})
        try:
            cost_per_valid = m8_accounting.cost_per_correct_success(attempts)
        except ValueError:
            cost_per_valid = None
        failures = []
        for episode in arm_episodes:
            failure = episode.get("failure")
            status = episode.get("terminal_status")
            if failure or status not in (None, "completed"):
                failures.append(
                    {
                        "scenario_id": episode["scenario_id"],
                        "failure": failure or f"terminal_status:{status}",
                    }
                )
        per_arm[arm] = {
            "episodes": total,
            "retrieval_attempted_rate": round(sum(bool(e.get("retrieval_attempted")) for e in arm_episodes) / total, 4),
            "retrieve_first_rate": round(sum(bool(e.get("retrieve_first")) for e in arm_episodes) / total, 4),
            "valid_cited_final_rate": round(
                sum(e.get("verdict") == "valid_cited_final" for e in arm_episodes) / total, 4
            ),
            "unsupported_final_rate": round(
                sum(e.get("verdict") == "unsupported_final" for e in arm_episodes) / total, 4
            ),
            "abstention_rate": round(sum(e.get("verdict") == "empty_claims_final" for e in arm_episodes) / total, 4),
            "step_limit_rate": round(sum(e.get("terminal_status") == "step_limit" for e in arm_episodes) / total, 4),
            "model_calls": sum(e["model_calls"] for e in arm_episodes),
            "input_tokens": _optional_total(e["input_tokens"] for e in arm_episodes),
            "output_tokens": _optional_total(e["output_tokens"] for e in arm_episodes),
            "cost_per_valid_final": cost_per_valid,
            "failures": failures,
        }
    return per_arm


def _parse_arms(value):
    arms = tuple(part.strip() for part in value.split(","))
    if not arms or any(arm not in ARMS for arm in arms) or len(set(arms)) != len(arms):
        raise argparse.ArgumentTypeError(f"--arms must be a comma subset of {','.join(ARMS)}")
    return arms


def main(argv=None, *, consumer_factory=None, adapter_factory=None, embedder_factory=None, rag_store_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios-dir", default=str(Path(__file__).parent / "fixtures"))
    parser.add_argument("--workdir-root", required=True)
    parser.add_argument("--output", required=True, help="episode evidence JSONL path")
    parser.add_argument("--snapshot", help="authenticated snapshot; required unless --dry-run")
    parser.add_argument(
        "--fixtures-subset",
        help="comma-separated scenario ids to run (a single id runs one scenario)",
    )
    parser.add_argument("--arms", type=_parse_arms, default=ARMS)
    parser.add_argument(
        "--seeds",
        type=control._parse_seeds,
        default=(0,),
        help="comma-separated run replications; the qualified runtime pins sampler "
        "seed 0 per request, so seeds replicate identical runs (development only)",
    )
    parser.add_argument(
        "--consumer-profile",
        default="llama3",
        choices=sorted(PROFILES),
        help="policy consumer profile: llama3 (default; qualified SFTv3) or qwen3 "
        "(M9 Track-B Qwen3.5 SFTv1; requires --development-runtime-x86)",
    )
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true", help="validate fixtures and stream digests only")
    parser.add_argument(
        "--development-runtime-x86",
        action="store_true",
        help="development-only: use the declared x86 runtime identity "
        "(requires DML_DEV_RUNTIME=development-x86-20261102 in the environment)",
    )
    parser.add_argument(
        "--development-checkpoint",
        metavar="PATH",
        help="JSON file declaring a candidate checkpoint identity "
        "(label, manifest_sha256, adapter_sha256); requires --development-runtime-x86",
    )
    args = parser.parse_args(argv)

    started = time.time()
    try:
        scenarios = control.select_scenarios(
            fixture_schema.load_scenarios(Path(args.scenarios_dir)), args.fixtures_subset
        )
    except fixture_schema.FixtureError as exc:
        print(json.dumps({"fixture_error": str(exc)}), file=sys.stderr)
        return 1

    if args.dry_run:
        report = [
            {
                "scenario_id": scenario.scenario_id,
                "family": scenario.family,
                "cell": control.label_cell(scenario),
                "events": len(scenario.events),
                "corrections": sum(1 for event in scenario.events if event.corrects is not None),
                "expects_retrieval": scenario.expects_retrieval,
                "event_stream_digest": event_stream_digest(scenario),
            }
            for scenario in scenarios
        ]
        print(json.dumps({"dry_run": "fixtures_valid", "arms": list(args.arms), "scenarios": report}, indent=1))
        return 0

    if not args.snapshot:
        print(json.dumps({"error": "--snapshot is required unless --dry-run"}), file=sys.stderr)
        return 2
    try:
        limits = EpisodeLimits(max_steps=args.max_steps, output_tokens=args.output_tokens)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2

    profile = PROFILES[args.consumer_profile]
    injected_factory = consumer_factory is not None
    if profile == QWEN3_SFT_V1_CONSUMER_PROFILE and not args.development_runtime_x86 and not injected_factory:
        # The qwen3 SFTv1 qualified pins are fail-closed placeholders (no
        # trained adapter admitted yet); only the declared x86 development
        # runtime can construct its consumer.
        print(json.dumps({"error": "--consumer-profile qwen3 requires --development-runtime-x86"}), file=sys.stderr)
        return 2
    if consumer_factory is None:
        if args.development_runtime_x86:
            if profile == QWEN3_SFT_V1_CONSUMER_PROFILE:

                def consumer_factory(snapshot, _profile=profile):
                    return control.dev_x86_consumer_factory(snapshot, profile=_profile)

            else:
                consumer_factory = control.dev_x86_consumer_factory
        else:
            consumer_factory = control.default_consumer_factory
    if args.development_checkpoint:
        if not args.development_runtime_x86:
            print(json.dumps({"error": "--development-checkpoint requires --development-runtime-x86"}), file=sys.stderr)
            return 2
        from daystrom_dml.contracts.model_input import ModelInputError

        if profile == QWEN3_SFT_V1_CONSUMER_PROFILE:
            from daystrom_dml.services.qwen3_sft_v1_action_input import _validate_development_checkpoint
        else:
            from daystrom_dml.services.llama3_sft_v3_action_input import _validate_development_checkpoint

        try:
            checkpoint_identity = json.loads(Path(args.development_checkpoint).read_text(encoding="utf-8"))
            _validate_development_checkpoint(checkpoint_identity)
        except (OSError, json.JSONDecodeError, ModelInputError) as exc:
            print(json.dumps({"error": f"invalid development checkpoint declaration: {exc}"}), file=sys.stderr)
            return 2

        if profile == QWEN3_SFT_V1_CONSUMER_PROFILE:

            def consumer_factory(snapshot, _identity=checkpoint_identity, _profile=profile):
                return control.dev_x86_consumer_factory(snapshot, development_checkpoint=_identity, profile=_profile)

        else:

            def consumer_factory(snapshot, _identity=checkpoint_identity):
                return control.dev_x86_consumer_factory(snapshot, development_checkpoint=_identity)

    adapter_factory = adapter_factory or _fixture_adapter
    rag_store_factory = rag_store_factory or default_rag_store_factory
    # The embedder is constructed only when a rag_memory arm is planned.
    embedder = None
    if "rag_memory" in args.arms:
        embedder = (embedder_factory or default_embedder_factory)()
    workdir_root = Path(args.workdir_root)
    workdir_root.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # One admission + model load per run (not per episode), shared across arms.
    # Injected factories (tests) keep per-episode construction. A
    # shared-construction failure is retained as the failure of every planned
    # episode, exactly as per-episode construction would have produced.
    shared_consumer = None
    if not injected_factory:
        try:
            shared_consumer = consumer_factory(args.snapshot)
        except Exception as shared_error:  # noqa: BLE001 — retained per episode below

            def consumer_factory(_snapshot, _exc=shared_error):
                raise _exc

    episodes = []
    try:
        with output.open("w", encoding="utf-8") as stream:
            for seed in args.seeds:
                for scenario in scenarios:
                    for arm in args.arms:
                        workdir = control._fresh_workdir(workdir_root, f"{scenario.scenario_id}__{arm}__seed{seed}")
                        episode = run_battery_episode(
                            scenario,
                            arm,
                            workdir=workdir,
                            snapshot=args.snapshot,
                            consumer_factory=consumer_factory,
                            adapter_factory=adapter_factory,
                            embedder=embedder,
                            rag_store_factory=rag_store_factory,
                            limits=limits,
                            seed=seed,
                            consumer_profile=profile,
                            shared_consumer=shared_consumer,
                        )
                        episodes.append(episode)
                        stream.write(json.dumps(episode, sort_keys=True) + "\n")
                        stream.flush()
    except fixture_schema.FixtureError as exc:
        print(json.dumps({"fixture_error": str(exc)}), file=sys.stderr)
        return 1
    finally:
        if shared_consumer is not None:
            close = getattr(shared_consumer, "close", None)
            if callable(close):
                close()

    summary = {
        "schema": SUMMARY_SCHEMA,
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - started, 1),
        "arms": list(args.arms),
        "consumer_profile": profile,
        "seeds": sorted(args.seeds),
        "scenarios": len(scenarios),
        "episodes": len(episodes),
        "per_arm": summarize(episodes),
        "limitations": (
            "Development transport-native battery only: test_injected episode "
            "envelope (not live_local), fixture workloads, pinned sampler seed, "
            "no benchmark, no held-out data, no competence claim; arms differ "
            "only in the retrieve backend and every failure is retained."
        ),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps(summary["per_arm"], indent=1))
    print("episodes:", len(episodes), "evidence:", output, "summary:", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
