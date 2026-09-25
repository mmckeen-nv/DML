"""Replay a frozen live campaign without executing a model or granting authenticity.

Passing this checker establishes the declared evidence/coverage conditions only.
The independent review must separately establish trained-weight provenance, real
execution, the pre-generation freeze, and exact-source CI qualification.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path

from daystrom_dml.contracts.agent_episode import (
    canonical_json, decode_json, validate_episode_events, presented_record_identities,
    execution_protocol_for_profile, EXECUTION_PROTOCOL_V2, VALIDATION_CONSUMER_PROFILE, RECOVERY_CONSUMER_PROFILE, EVENT_VERSION_V2,
)
from daystrom_dml.services.agent_episode import task_allowed_tools, validate_consumer_profile
from daystrom_dml.services.episode_outcomes import build_terminal, summarize_episode_outcomes
from daystrom_dml.services.episode_verifiers import INTENTS, load_episode_corpus, verify_task
from daystrom_dml.services.model_input_snapshot import REQUIRED_FILES, verify_local_snapshot
from scripts.agent_episodes import MAX_CAMPAIGN_BYTES, _atomic_json, _source_digests

SPEC_VERSION = "dml-agent-campaign-spec-v1"
EVIDENCE_VERSION = "dml-agent-campaign-evidence-v1"
SPEC_VERSION_V2 = "dml-agent-campaign-spec-v2"
EVIDENCE_VERSION_V2 = "dml-agent-campaign-evidence-v2"
GATES = {
    "attempted_tasks": 9,
    "intent_count": 8,
    "model_generation_for_every_task": True,
    "retrieval_and_measurable_final_for_every_intent": True,
    "verified_model_owned_supersession": True,
    "actual_predecessor_context": True,
    "complete_failure_inclusive_evidence": True,
    "all_task_successes_required": False,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same(left, right):
    return canonical_json(left, limit=MAX_CAMPAIGN_BYTES) == canonical_json(right, limit=MAX_CAMPAIGN_BYTES)


def _digest(value):
    return hashlib.sha256(canonical_json(value, limit=MAX_CAMPAIGN_BYTES)).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _observations(events):
    """Reconstruct only record bytes actually shown to the model by a tool."""
    observations = []
    for event in events:
        if event["kind"] != "tool_completed":
            continue
        payload = event["payload"]
        displayed = decode_json(payload["model_result"]).get("records", [])
        for record in payload["result"].get("observed_records", []):
            if payload["name"] != "retrieve":
                _require(_same(payload["result"].get("receipt", {}).get("result", {}).get("memory"), record),
                         "Mutation record differs from receipt")
            if any(type(item) is dict and type(item.get("id")) is int and item["id"] == record["id"]
                   and item.get("text") == record["text"]
                   and all(_same(item[key], record["meta"].get(key)) for key in (
                       "source", "claim_key", "claim_value", "source_trust", "memory_state") if key in item)
                   for item in displayed):
                observations.append({"sequence": event["sequence"], "task_id": event["task_id"],
                                     "operation": payload["name"], "record": deepcopy(record)})
    return observations


def _replay_presented_authority(events, prepared):
    """Bind the causal v2 ledger to scoped prepared records and acknowledged receipts."""
    from daystrom_dml.journal import _validated_receipt
    from daystrom_dml.persistence import validate_record
    from daystrom_dml.services.episode_tools import _digest as record_digest
    from daystrom_dml.services.receipt_supersession import canonical_supersession_request, _supersession_receipt
    scope = events[0]["payload"]["scope"]
    owned, stores, references = set(), set(), {}
    for receipt in prepared["seed_receipts"]:
        _validated_receipt(receipt)
        record = receipt["result"]["memory"]
        validate_record(record)
        _require(_same(receipt["scope"], {key: record["meta"].get(key) for key in scope}),
                 "Seed receipt record and scope differ")
        if _same(receipt["scope"], scope):
            stores.add(receipt["store_id"])
            owned.add(canonical_json(record))
            references.setdefault(canonical_json(record), "r" + str(len(references)))
    for record in prepared["seed_records"].values():
        validate_record(record)
        if _same({key: record["meta"].get(key) for key in scope}, scope):
            owned.add(canonical_json(record))
    pending, ledger = None, {}
    for event in events:
        payload = event["payload"]
        if event["kind"] == "tool_requested":
            pending = event
        elif event["kind"] == "tool_completed":
            if payload["name"] != "retrieve":
                receipt = payload["result"]["receipt"]
                _validated_receipt(receipt)
                _require(_same(receipt["scope"], scope) and receipt["store_id"] in stores
                         and pending is not None and receipt["key"] == pending["payload"]["idempotency_key"]
                         and receipt["key"] == "episode:" + event["episode_id"] + ":" + event["call_id"],
                         "Acknowledged mutation receipt differs from scoped prepared authority")
                _require(payload["name"] == "supersede", "Mutation is outside the fixed campaign authority")
                arguments = pending["payload"]["arguments"]
                _require(all(arguments[field] in ledger for field in ("record_ref", "replacement_ref")),
                         "Mutation did not use previously presented immutable records")
                source = decode_json(ledger[arguments["record_ref"]][1])
                replacement = decode_json(ledger[arguments["replacement_ref"]][1])
                request, digest = canonical_supersession_request(source["id"],
                    replacement_memory_id=replacement["id"], expected_memory_digest=record_digest(source),
                    expected_replacement_digest=record_digest(replacement), reason=arguments["reason"], **scope)
                _require(receipt["request_digest"] == digest, "Mutation receipt request digest differs from proposal")
                _supersession_receipt(receipt, request)
                record = receipt["result"]["memory"]
                validate_record(record)
                owned.add(canonical_json(record))
                references.setdefault(canonical_json(record), "r" + str(len(references)))
            identities = presented_record_identities(payload, scope)
            _require(all(raw in owned and references.get(raw) == reference
                         for reference, (_, raw) in identities.items()),
                     "Presented immutable identity was not owned by prepared or receipted authority")
            ledger.update(identities)
            pending = None
        elif event["kind"] == "tool_failed":
            pending = None


def _replay_model(events, identity, tokenizer, consumer_profile):
    generated = 0
    grammar_matcher = None
    for event in events:
        payload = event["payload"]
        if event["kind"] == "model_requested":
            compiled, request = payload["compiled"], payload["request"]
            _require(_same(compiled["identity"], identity), "Model identity differs from frozen snapshot")
            encoded = tokenizer.apply_chat_template(request["messages"], tools=request["tools"],
                chat_template=tokenizer.chat_template, tokenize=True, add_generation_prompt=False,
                continue_final_message=False, truncation=False, padding=False, return_dict=True,
                tokenizer_kwargs={"return_attention_mask": True})
            _require(_same(encoded["input_ids"], compiled["input_ids"])
                     and _same(encoded["attention_mask"], compiled["attention_mask"]),
                     "Recorded input tokens differ from independent tokenization")
            if consumer_profile in ("qwen2-action-json-v1", VALIDATION_CONSUMER_PROFILE, RECOVERY_CONSUMER_PROFILE):
                from daystrom_dml.services.agent_action_grammar import compile_action_grammar
                import xgrammar as xgr
                # Membership below separately rejects unused model rows. A
                # syntax replay needs only the tokenizer's actual ID space.
                grammar = compile_action_grammar(tokenizer, max(tokenizer.get_vocab().values()) + 1,
                                                 request["tools"])
                grammar_matcher = xgr.GrammarMatcher(
                    grammar, override_stop_tokens=[tokenizer.eos_token_id],
                    terminate_without_stop_token=False, max_rollback_tokens=-1, default_temperature=None,
                )
        elif event["kind"] == "model_completed":
            output_ids = payload["output_ids"]
            if consumer_profile in {"qwen2-instruct-v1", "qwen2-action-json-v1", VALIDATION_CONSUMER_PROFILE, RECOVERY_CONSUMER_PROFILE}:
                vocabulary = tokenizer.get_vocab()
                allowed_ids = frozenset(vocabulary.values())
                _require(all(type(token) is int and token in allowed_ids for token in output_ids),
                         "Qwen output contains an invalid or unused vocabulary ID")
                eos = vocabulary.get("<|im_end|>")
                _require(type(eos) is int and tokenizer.eos_token == "<|im_end|>"
                         and eos == tokenizer.eos_token_id and eos in tokenizer.all_special_ids,
                         "Qwen terminal EOS identity differs")
                if consumer_profile in ("qwen2-action-json-v1", VALIDATION_CONSUMER_PROFILE, RECOVERY_CONSUMER_PROFILE):
                    _require(grammar_matcher is not None, "Constrained output lacks its request grammar")
                    special = {index for index, token in tokenizer.added_tokens_decoder.items() if token.special}
                    _require(all(token not in special - {eos} and grammar_matcher.accept_token(token)
                                 for token in output_ids), "Recorded output violates its request-owned grammar")
                    grammar_matcher = None
                if output_ids and output_ids[-1] == eos:
                    output_ids = output_ids[:-1]
            text = tokenizer.decode(output_ids, skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
            _require(text == payload["text"], "Recorded output text differs from its token IDs")
            generated += 1
    return generated


def replay_campaign(campaign, spec, *, identity, tokenizer):
    """Replay evidence and return measured gate results; never infer authenticity."""
    corpus = load_episode_corpus()
    selected = [(scenario, task) for scenario in corpus["scenarios"] for task in scenario["tasks"]]
    selection = [{"scenario_id": scenario["id"], "task_id": task["id"]} for scenario, task in selected]
    consumer_profile = validate_consumer_profile(spec["consumer_profile"])
    protocol = execution_protocol_for_profile(consumer_profile)
    v2 = protocol == EXECUTION_PROTOCOL_V2
    _require(spec.get("schema_version") == (SPEC_VERSION_V2 if v2 else SPEC_VERSION)
             and _same(spec.get("acceptance"), GATES), "Campaign specification or fixed acceptance gates differ")
    for value in (spec, campaign):
        _require(value.get("execution_protocol") == protocol if v2 else "execution_protocol" not in value,
                 "Campaign execution protocol differs")
    _require(campaign.get("consumer_profile") == consumer_profile, "Consumer profile differs")
    _require(campaign.get("schema_version") == ("dml-agent-campaign-v2" if v2 else "dml-agent-campaign-v1"),
             "Campaign schema differs")
    _require(_same(spec["selection"], selection) and _same(campaign["selection"], selection),
             "Campaign selection differs from the complete ordered corpus")
    _require(campaign["corpus_digest"] == spec["corpus_digest"] == _digest(corpus), "Corpus digest differs")
    _require(_same(campaign["limits"], spec["limits"]), "Campaign limits differ")
    _require(_same(campaign["source_sha256"], spec["producer_source_sha256"]), "Producer source differs")
    _require(campaign["ranking_scope"] == "synthetic_fixture" and campaign["live_qualified"] is False
             and campaign["source_ci_qualified"] is False, "Producer overclaims qualification")
    _require(type(campaign["episodes"]) is list and len(campaign["episodes"]) == len(selected),
             "Every declared task requires exactly one retained attempt")
    seen, prior, rows, terminals = set(), {}, [], []
    for episode, (scenario, task) in zip(campaign["episodes"], selected):
        events, terminal, prepared = episode["events"], episode["terminal"], episode["prepared"]
        _require(episode.get("consumer_profile") == consumer_profile, "Episode consumer profile differs")
        validate_episode_events(events)
        _require((events[0]["schema_version"] == EVENT_VERSION_V2) is v2, "Episode protocol differs from campaign")
        _require(_same(events[-1]["payload"], terminal), "Retained terminal differs from events")
        _require(episode["scenario_id"] == scenario["id"] and terminal["task_id"] == task["id"],
                 "Retained attempt order or task identity differs")
        _require(terminal["episode_id"] not in seen, "Duplicate episode identity")
        seen.add(terminal["episode_id"])
        start = events[0]["payload"]
        if v2:
            _require(start["consumer_profile"] == terminal["consumer_profile"] == consumer_profile,
                     "Episode boundary consumer profile differs from campaign")
        _require(start["execution_path"] == terminal["execution_path"] == "live_local", "Injected execution")
        for key, value in {"limits": spec["limits"], "scope": scenario["scope"], "prompt": task["prompt"],
                           "effective_time": corpus["effective_time"],
                           "allowed_tools": list(task_allowed_tools(task))}.items():
            _require(_same(start[key], value), "Episode start differs: " + key)
        previous = prior.get((scenario["id"], task.get("repeat_from")))
        expected_context = None if previous is None else {key: previous[key]
            for key in ("episode_id", "task_id", "evidence_digest", "answer")}
        _require(_same(start["prior_context"], expected_context), "Actual predecessor context differs")
        complete = (not episode.get("raw_evidence_incomplete") and type(prepared) is dict
                    and bool(prepared.get("seed_records")) and not terminal["usage_unknown"]
                    and not terminal["effects_unknown"] and terminal["unknown_input_calls"] == 0
                    and terminal["unknown_output_calls"] == 0)
        if prepared and start["seed_receipts_digest"] is not None:
            _require(start["seed_receipts_digest"] == _digest(prepared), "Seed evidence digest differs")
            if v2:
                _replay_presented_authority(events, prepared)
        else:
            complete = False
        generated = _replay_model(events, identity, tokenizer, consumer_profile)
        observations = _observations(events)
        verdict = terminal["verifier"]
        if terminal["status"] == "completed":
            replay = verify_task(scenario, task, terminal["answer"], seed_records=prepared["seed_records"],
                observed_records=observations, current_records=episode["current_records"],
                final_sequence=events[-1]["sequence"], effective_time=corpus["effective_time"],
                previous_answers={} if previous is None else {previous["task_id"]: previous["answer"]})
            _require(_same(replay, verdict), "Independent task verifier differs")
        else:
            _require(all(verdict[key] is None for key in (
                "contradictions", "factual_outputs", "false_memory_claims", "recalled_claims",
                "repeat_errors", "repeat_opportunities")), "Unfinished task cannot invent measured quality")
        rebuilt = build_terminal(events[:-1], verdict, status=terminal["status"],
            latency_ms=terminal["latency_ms"], retrieval_ms=terminal["retrieval_ms"],
            answer=terminal["answer"], usage_unknown=terminal["usage_unknown"])
        _require(_same(rebuilt, terminal), "Failure-inclusive terminal accounting differs")
        retrieved = any(event["kind"] == "tool_completed" and event["payload"]["name"] == "retrieve"
                        for event in events)
        measurable = (terminal["status"] == "completed" and all(type(verdict[key]) is int
            for key in ("contradictions", "factual_outputs", "false_memory_claims", "recalled_claims"))
            and verdict["factual_outputs"] > 0 and verdict["recalled_claims"] > 0)
        supersession = (scenario["id"] == "superseded_preference" and terminal["success"]
                        and any(item["operation"] == "supersede" for item in observations))
        rows.append({"scenario_id": scenario["id"], "task_id": task["id"], "status": terminal["status"],
                     "success": terminal["success"], "generated_calls": generated, "retrieved": retrieved,
                     "measurable_final": measurable, "complete": complete,
                     "prior_answer_available": previous is not None and previous["answer"] is not None,
                     "repeat_quality_measured": all(type(verdict[key]) is int
                         for key in ("repeat_errors", "repeat_opportunities")),
                     "verified_model_owned_supersession": supersession})
        terminals.append(terminal)
        prior[(scenario["id"], task["id"])] = terminal
    summary = summarize_episode_outcomes(terminals)
    _require(_same(campaign["summary"], summary), "Campaign summary differs from all retained attempts")
    complete = all(row["complete"] for row in rows)
    _require(campaign["raw_evidence_complete"] is complete, "Producer completeness flag differs")
    coverage = {intent: any(row["scenario_id"] == intent and row["retrieved"] and row["measurable_final"]
                           for row in rows) for intent in INTENTS}
    feedback = [row for row in rows if row["scenario_id"] == "self_reinforcing_error"]
    feedback_complete = (len(feedback) == 2 and all(row["retrieved"] and row["measurable_final"] for row in feedback)
                         and feedback[1]["prior_answer_available"] and feedback[1]["repeat_quality_measured"])
    gates = {"all_attempts_generated": all(row["generated_calls"] > 0 for row in rows),
             "all_intents_reached_retrieval_and_final": all(coverage.values()),
             "verified_model_owned_supersession": any(row["verified_model_owned_supersession"] for row in rows),
             "complete_failure_inclusive_evidence": complete, "actual_predecessor_context": feedback_complete}
    return {"schema_version": EVIDENCE_VERSION_V2 if v2 else EVIDENCE_VERSION,
            **({"execution_protocol": protocol} if v2 else {}), "validation_scope": "independent_data_replay_only",
            "execution_authenticity_verified": False, "source_ci_qualified": False,
            "predeclared_gates_passed": all(gates.values()), "gates": gates, "intent_coverage": coverage,
            "attempts": rows, "summary": summary}


def verify_files(*, spec_path, spec_sha256, campaign_path, snapshot_directory, source_root):
    _require(file_digest(spec_path) == spec_sha256, "Frozen specification digest differs")
    spec = decode_json(Path(spec_path).read_bytes(), limit=MAX_CAMPAIGN_BYTES)
    root = Path(source_root).resolve()
    for relative, digest in spec["source_sha256"].items():
        path = (root / relative).resolve()
        _require(path.is_relative_to(root) and file_digest(path) == digest, "Frozen source differs: " + relative)
    _require(_same(_source_digests(), spec["producer_source_sha256"]), "Executing producer sources differ")
    bundle = Path(snapshot_directory)
    _require(set(spec["snapshot_sha256"]) == REQUIRED_FILES | {"snapshot.json"}, "Snapshot inventory differs")
    for name, digest in spec["snapshot_sha256"].items():
        _require(file_digest(bundle / name) == digest, "Frozen snapshot bytes differ: " + name)
    campaign_bytes = Path(campaign_path).read_bytes()
    campaign = decode_json(campaign_bytes, limit=MAX_CAMPAIGN_BYTES)
    consumer_profile = validate_consumer_profile(spec["consumer_profile"])
    if consumer_profile in {"qwen2-instruct-v1", "qwen2-action-json-v1", VALIDATION_CONSUMER_PROFILE, RECOVERY_CONSUMER_PROFILE}:
        from daystrom_dml.services.qwen_model_snapshot import verify_qwen_snapshot
        verify_snapshot = verify_qwen_snapshot
    else:
        verify_snapshot = verify_local_snapshot
    with verify_snapshot(bundle) as snapshot:
        from transformers import PreTrainedTokenizerFast
        if consumer_profile in ("qwen2-action-json-v1", VALIDATION_CONSUMER_PROFILE, RECOVERY_CONSUMER_PROFILE):
            from daystrom_dml.services.qwen_action_input import constrained_identity
            identity = constrained_identity(snapshot.identity, consumer_profile=consumer_profile).to_payload()
        else:
            identity = snapshot.identity.to_payload()
        _require(_same(identity, spec["model_identity"]), "Frozen model identity differs")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(snapshot.path / "tokenizer.json"),
                                            **snapshot.special_tokens)
        tokenizer.chat_template = snapshot.chat_template
        tokenizer.model_max_length = snapshot.identity.model_window_tokens
        tokenizer.backend_tokenizer.no_truncation()
        tokenizer.backend_tokenizer.no_padding()
        evidence = replay_campaign(campaign, spec, identity=identity, tokenizer=tokenizer)
    return {**evidence, "specification_sha256": spec_sha256,
            "campaign_sha256": hashlib.sha256(campaign_bytes).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("spec", "spec-sha256", "campaign", "snapshot-directory", "source-root", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    evidence = verify_files(spec_path=args.spec, spec_sha256=args.spec_sha256, campaign_path=args.campaign,
        snapshot_directory=args.snapshot_directory, source_root=args.source_root)
    _atomic_json(args.output, evidence)
    return 0 if evidence["predeclared_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
