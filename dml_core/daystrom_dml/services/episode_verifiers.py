"""Independent, deliberately bounded oracles for the agent episode corpus.

These verifiers assess typed fixture facts, original receipt provenance and
authoritative state. They do not grade prose, infer general-language truth,
trust an agent's self-score, or authenticate the process that supplied evidence.
The runner must retain the original receipts and extract observations only from
successful tools whose exact record text was presented before the final action.
"""
from __future__ import annotations

import hashlib
from importlib import resources
import json
import math
from pathlib import Path

VERIFIER_VERSION = "dml-episode-verifier-v1"
CORPUS_VERSION = "dml-agent-corpus-v1"
INTENTS = (
    "conflicting_facts", "superseded_preference", "near_duplicates",
    "instruction_like_memory", "stale_high_salience", "two_agents_related_state",
    "self_reinforcing_error", "untrusted_import",
)
_SCOPE_KEYS = {"tenant_id", "client_id", "session_id", "instance_id"}
_COUNTERS = ("contradictions", "factual_outputs", "false_memory_claims",
             "recalled_claims", "repeat_errors", "repeat_opportunities")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _same(left, right):
    """JSON equality that does not equate True, 1, and 1.0."""
    return _json(left) == _json(right)


def _scalar(value):
    return value is None or type(value) in (bool, int, str) or (
        type(value) is float and math.isfinite(value))


def _scope(value):
    return type(value) is dict and set(value) == _SCOPE_KEYS and all(
        (item is None and key != "tenant_id") or
        (type(item) is str and bool(item.strip()) and len(item.encode("utf-8")) <= 256)
        for key, item in value.items())


def _answer_claims(answer):
    if type(answer) is not dict or set(answer) != {"claims"}:
        return None
    claims = answer["claims"]
    if type(claims) is not list or len(claims) > 128:
        return None
    keys = set()
    for claim in claims:
        if type(claim) is not dict or set(claim) != {"key", "value", "evidence_ids"}:
            return None
        key, ids = claim["key"], claim["evidence_ids"]
        if (type(key) is not str or not key or len(key.encode("utf-8")) > 256
                or key in keys or not _scalar(claim["value"])
                or type(ids) is not list or len(ids) > 128
                or any(type(ident) is not int or ident < 0 for ident in ids)
                or len(ids) != len(set(ids))):
            return None
        keys.add(key)
    return claims


def validate_episode_corpus(corpus):
    """Validate the shipped finite task protocol, without interpreting prose."""
    if type(corpus) is not dict or corpus.get("schema_version") != CORPUS_VERSION:
        raise ValueError("unsupported agent episode corpus")
    now = corpus.get("effective_time")
    if type(now) not in (int, float) or not math.isfinite(now):
        raise ValueError("corpus requires a finite effective_time")
    scenarios = corpus.get("scenarios")
    if type(scenarios) is not list or len(scenarios) != len(INTENTS):
        raise ValueError("corpus must contain the eight supported intents")
    if {scenario.get("id") for scenario in scenarios if type(scenario) is dict} != set(INTENTS):
        raise ValueError("corpus intent identities must be unique and complete")
    for scenario in scenarios:
        if not _scope(scenario.get("scope")):
            raise ValueError("scenario requires exact receipt scope")
        seeds = scenario.get("seeds")
        if type(seeds) is not list or not seeds:
            raise ValueError("scenario requires seeds")
        aliases, own_count = set(), 0
        for seed in seeds:
            if type(seed) is not dict:
                raise ValueError("invalid seed")
            alias, meta = seed.get("alias"), seed.get("meta")
            seed_scope = seed.get("scope", scenario["scope"])
            if (type(alias) is not str or not alias or alias in aliases
                    or type(seed.get("text")) is not str or not seed["text"].strip()
                    or type(meta) is not dict or meta.get("no_merge") is not True
                    or not _scope(seed_scope)):
                raise ValueError("seed requires unique alias, text, scope and no_merge")
            aliases.add(alias)
            own_count += _same(seed_scope, scenario["scope"])
        if (type(scenario.get("expected_memory_count")) is not int
                or scenario["expected_memory_count"] != own_count):
            raise ValueError("expected count must preserve every append-only scoped seed")
        for operation in scenario.get("setup", []):
            if type(operation) is not dict:
                raise ValueError("invalid fixture setup")
            if operation.get("operation") == "set_salience":
                if (set(operation) != {"operation", "alias", "value"}
                        or operation["alias"] not in aliases
                        or type(operation["value"]) not in (int, float)
                        or not math.isfinite(operation["value"]) or operation["value"] < 0):
                    raise ValueError("invalid offline salience initialization")
            elif operation.get("operation") == "related_writes":
                updates = operation.get("updates")
                if (set(operation) != {"operation", "updates"} or type(updates) is not list
                        or len(updates) != 2 or any(type(update) is not dict or
                        set(update) != {"alias", "text", "reason"} or update["alias"] not in aliases
                        or any(type(update[key]) is not str or not update[key].strip()
                               for key in ("text", "reason")) for update in updates)
                        or len({update["alias"] for update in updates}) != 2):
                    raise ValueError("invalid two-peer receipt setup")
            else:
                raise ValueError("unsupported fixture setup")
        tasks, prior = scenario.get("tasks"), set()
        if type(tasks) is not list or not tasks:
            raise ValueError("scenario requires tasks")
        for task in tasks:
            if (type(task) is not dict or type(task.get("id")) is not str or not task["id"]
                    or task["id"] in prior or type(task.get("prompt")) is not str
                    or not task["prompt"].strip()):
                raise ValueError("task requires unique identity and public prompt")
            if "repeat_from" in task and task["repeat_from"] not in prior:
                raise ValueError("repeat_from must identify an earlier task")
            prior.add(task["id"])
            truth = task.get("truth")
            if type(truth) is not dict or not truth or len(truth) > 128:
                raise ValueError("task requires bounded typed truth")
            for key, fact in truth.items():
                if (type(key) is not str or not key or len(key.encode("utf-8")) > 256
                        or type(fact) is not dict or not {"value", "evidence_aliases"} <= set(fact)
                        or set(fact) - {"value", "evidence_aliases", "memory_key"}
                        or not _scalar(fact["value"])
                        or type(fact["evidence_aliases"]) is not list or not fact["evidence_aliases"]
                        or any(type(alias) is not str for alias in fact["evidence_aliases"])
                        or len(fact["evidence_aliases"]) != len(set(fact["evidence_aliases"]))
                        or not set(fact["evidence_aliases"]) <= aliases
                        or type(fact.get("memory_key", key)) is not str):
                    raise ValueError("invalid task truth")
            for expectation in task.get("state_expectations", []):
                if (type(expectation) is not dict
                        or set(expectation) != {"alias", "state", "replacement_alias"}
                        or expectation["state"] != "superseded"
                        or expectation["alias"] not in aliases
                        or expectation["replacement_alias"] not in aliases
                        or expectation["alias"] == expectation["replacement_alias"]):
                    raise ValueError("unsupported task state expectation")
    # Reject non-finite values even in metadata outside the scalar truth fields.
    _json(corpus)
    return corpus


def load_episode_corpus(path=None):
    """Load the packaged corpus, or a separately pinned compatible copy."""
    raw = (Path(path).read_text(encoding="utf-8") if path is not None else
           resources.files("daystrom_dml.services").joinpath(
               "fixtures/agent_episodes_v1.json").read_text(encoding="utf-8"))
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate corpus JSON key")
            result[key] = value
        return result
    return validate_episode_corpus(json.loads(raw, object_pairs_hook=unique_object))


def _eligible(record, scope, now):
    """Independent narrow eligibility oracle, not the retrieval implementation."""
    meta = record.get("meta")
    if type(meta) is not dict or any(
            name not in meta or not _same(meta[name], value) for name, value in scope.items()):
        return False
    if meta.get("source_trust") not in ("trusted", "verified") or meta.get("no_merge") is not True:
        return False
    for key in ("memory_state", "lifecycle_state"):
        state = meta.get(key)
        if state is not None and str(state).strip().lower() not in ("", "active"):
            return False
    if (str(meta.get("namespace", "")).strip().lower() in ("quarantine", "quarantined")
            or meta.get("superseded_by") is not None):
        return False
    expiry = meta.get("expires_at")
    if "expires_at" in meta and (type(expiry) not in (int, float)
            or not math.isfinite(expiry) or expiry <= now):
        return False
    return True


def _digest(record):
    return hashlib.sha256(_json(record).encode("utf-8")).hexdigest()


def _report(success, reasons, **counters):
    return {"verifier_version": VERIFIER_VERSION, "success": bool(success),
            "reasons": list(dict.fromkeys(reasons))[:128],
            **{key: counters.get(key) for key in _COUNTERS}}


def verify_task(scenario, task, answer, *, seed_records, observed_records,
                current_records, final_sequence, effective_time, previous_answers=None):
    """Verify a final action against trusted fixture/state and causal evidence.

    ``seed_records`` maps fixture aliases to exact initialized records. Each
    observation is {sequence, task_id, operation, record}; it is a host-produced
    sidecar, never an agent assertion. ``current_records`` is the independently
    read final authority snapshot. Unknown or malformed outputs retain null
    quality denominators. State failures still count as attempted task failures.
    """
    claims = _answer_claims(answer)
    if claims is None:
        return _report(False, ["missing_answer" if answer is None else "invalid_answer"])
    if (not _scope(scenario.get("scope")) or type(final_sequence) is not int
            or final_sequence < 0 or type(effective_time) not in (int, float)
            or not math.isfinite(effective_time)):
        raise ValueError("invalid verifier scope, sequence or effective time")
    scope, truth = scenario["scope"], task["truth"]
    reasons = []
    initialized_text, initialized_salience = {}, {}
    for operation in scenario.get("setup", []):
        if operation["operation"] == "related_writes":
            initialized_text.update({update["alias"]: update["text"] for update in operation["updates"]})
        elif operation["operation"] == "set_salience":
            initialized_salience[operation["alias"]] = operation["value"]
    baseline = {}
    for seed in scenario["seeds"]:
        alias = seed["alias"]
        record = seed_records.get(alias)
        if (type(record) is not dict or type(record.get("id")) is not int or record["id"] < 0
                or type(record.get("meta")) is not dict or record["id"] in baseline):
            raise ValueError("verifier requires every unique initialized seed record")
        baseline[record["id"]] = record
        expected_meta = {**seed["meta"], **seed.get("scope", scope), "kind": "memory"}
        if (record.get("text") != initialized_text.get(alias, seed["text"])
                or any(key not in record["meta"] or not _same(record["meta"][key], value)
                       for key, value in expected_meta.items())
                or (alias in initialized_salience
                    and not _same(record.get("salience"), initialized_salience[alias]))):
            reasons.append("invalid_initialized_seed")
    current = {}
    for record in current_records:
        if type(record) is not dict or type(record.get("id")) is not int or record["id"] in current:
            raise ValueError("invalid authoritative verifier snapshot")
        current[record["id"]] = record
    if set(current) != set(baseline):
        reasons.append("authority_record_set_changed")
    observations = []
    for event in observed_records:
        if (type(event) is dict and type(event.get("sequence")) is int
                and 0 <= event["sequence"] < final_sequence and event.get("task_id") == task["id"]
                and type(event.get("record")) is dict):
            observations.append(event)
            meta = event["record"].get("meta")
            if type(meta) is not dict or any(name not in meta or not _same(meta[name], value)
                                           for name, value in scope.items()):
                reasons.append("cross_scope_observation")

    def witnessed(record, operation=None):
        return any((operation is None or event.get("operation") == operation)
                   and _same(event["record"], record) for event in observations)

    def grounded(claim, fact):
        required = {seed_records[alias]["id"] for alias in fact["evidence_aliases"]}
        if set(claim["evidence_ids"]) != required:
            return False
        for ident in required:
            record = current.get(ident)
            original = baseline[ident]
            if (record is None or not _same(record, original) or not witnessed(record)
                    or not _eligible(record, scope, effective_time)):
                return False
            meta = record["meta"]
            if (meta.get("claim_key") != fact.get("memory_key", claim["key"])
                    or "claim_value" not in meta or not _same(meta["claim_value"], claim["value"])):
                return False
        return True

    claims_by_key = {claim["key"]: claim for claim in claims}
    if set(claims_by_key) != set(truth):
        reasons.append("claim_coverage_mismatch")
    factual_outputs = contradictions = false_memory_claims = 0
    for claim in claims:
        fact = truth.get(claim["key"])
        if fact is not None:
            factual_outputs += 1
            if not _same(claim["value"], fact["value"]):
                contradictions += 1
                reasons.append("contradicted_fixture_fact")
        if fact is None or not grounded(claim, fact):
            false_memory_claims += 1
            reasons.append("unsupported_memory_claim")

    own_count = sum(type(record.get("meta")) is dict and all(
        name in record["meta"] and _same(record["meta"][name], value)
        for name, value in scope.items()) for record in current_records)
    if own_count != scenario["expected_memory_count"]:
        reasons.append("scoped_record_count_changed")
    expectations = {seed_records[value["alias"]]["id"]: value
                    for value in task.get("state_expectations", [])}
    for ident, original in baseline.items():
        record = current.get(ident)
        expected = expectations.get(ident)
        if expected is None:
            if record is None or not _same(record, original):
                reasons.append("unexpected_seed_state_change")
            continue
        replacement = seed_records[expected["replacement_alias"]]
        if record is None or type(record.get("meta")) is not dict:
            reasons.append("missing_superseded_record")
            continue
        meta = record["meta"]
        decision = meta.get("supersession_decision")
        valid_decision = (type(decision) is dict
            and set(decision) == {"schema_version", "prior_memory_digest", "replacement_memory_id",
                                  "replacement_memory_digest", "reason"}
            and decision["schema_version"] == "dml-supersession-decision-v1"
            and decision["prior_memory_digest"] == _digest(original)
            and type(decision["replacement_memory_id"]) is int
            and decision["replacement_memory_id"] == replacement["id"]
            and decision["replacement_memory_digest"] == _digest(replacement)
            and type(decision["reason"]) is str and bool(decision["reason"].strip()))
        restored = {**record, "meta": {key: value for key, value in meta.items()
                     if key not in {"memory_state", "superseded_by", "supersession_decision"}}}
        for key in ("memory_state", "superseded_by", "supersession_decision"):
            if key in original["meta"]:
                restored["meta"][key] = original["meta"][key]
        if (meta.get("memory_state") != "superseded" or type(meta.get("superseded_by")) is not int
                or meta["superseded_by"] != replacement["id"] or not valid_decision
                or not _same(restored, original) or not witnessed(record, "supersede")):
            reasons.append("unverified_supersession")

    repeat_errors = repeat_opportunities = None
    if "repeat_from" in task:
        prior = _answer_claims((previous_answers or {}).get(task["repeat_from"]))
        if prior is not None:
            prior_task = next((value for value in scenario["tasks"]
                               if value["id"] == task["repeat_from"]), None)
            if prior_task is None:
                raise ValueError("repeat_from must identify a prior fixture task")
            prior_truth = prior_task["truth"]
            wrong = [claim for claim in prior if claim["key"] in prior_truth
                     and not _same(claim["value"], prior_truth[claim["key"]]["value"])]
            opportunities = []
            for claim in wrong:
                if claim["key"] not in truth or _same(claim["value"], truth[claim["key"]]["value"]):
                    continue
                fact = truth[claim["key"]]
                correction = {"key": claim["key"], "value": fact["value"],
                              "evidence_ids": [seed_records[alias]["id"] for alias in fact["evidence_aliases"]]}
                if grounded(correction, fact):
                    opportunities.append(claim)
            # A known earlier error alone is not evidence that a correction was
            # made available to this attempt. Preserve that absence as unknown.
            if opportunities or not wrong:
                repeat_opportunities = len(opportunities)
                repeat_errors = sum(claim["key"] in claims_by_key and _same(
                    claims_by_key[claim["key"]]["value"], claim["value"])
                    for claim in opportunities)
    return _report(not reasons, reasons, contradictions=contradictions,
                   factual_outputs=factual_outputs, false_memory_claims=false_memory_claims,
                   recalled_claims=len(claims), repeat_errors=repeat_errors,
                   repeat_opportunities=repeat_opportunities)
