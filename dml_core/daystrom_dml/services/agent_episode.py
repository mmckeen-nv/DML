"""Bounded actual-model episode production with a supervised local worker.

The concrete entry point constructs the admitted local consumer itself.  An
explicitly labeled in-process injection helper supports deterministic tests; it
does not establish live execution, trained-model provenance or qualification.
"""
from __future__ import annotations

from contextlib import closing
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import math
import multiprocessing
from pathlib import Path
from queue import Empty, Full, Queue
import re
import sqlite3
from threading import Event, Thread
import time
import uuid

from ..contracts.agent_episode import (
    AGENT_POLICY, DIAGNOSTIC_RESERVE_BYTES, MAX_EPISODE_BYTES,
    AgentEpisodeError, canonical_json, decode_json, make_event,
    initial_messages, parse_agent_action, validate_episode_events,
    validate_prior_context, validate_verifier,
)
from ..contracts.model_input import ModelInputBudgetError
from .episode_tools import SelectedProfileEpisodeTools, episode_tool_definitions
from .episode_outcomes import build_terminal


@dataclass(frozen=True, slots=True)
class EpisodeLimits:
    max_steps: int = 6
    output_tokens: int = 128
    max_input_tokens: int = 32768
    max_output_tokens: int = 1024
    max_transcript_bytes: int = 262144
    max_event_bytes: int = 4 * 1024 * 1024
    max_episode_bytes: int = 16 * 1024 * 1024
    wall_time_seconds: float = 60.0

    def __post_init__(self):
        ceilings = {"max_steps": 64, "output_tokens": 4096,
                    "max_input_tokens": 1024 * 1024, "max_output_tokens": 1024 * 1024,
                    "max_transcript_bytes": 1024 * 1024,
                    "max_event_bytes": 16 * 1024 * 1024,
                    "max_episode_bytes": MAX_EPISODE_BYTES - DIAGNOSTIC_RESERVE_BYTES}
        for name, maximum in ceilings.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("Invalid bounded episode limit: " + name)
        if self.max_event_bytes > self.max_episode_bytes or self.output_tokens > self.max_output_tokens:
            raise ValueError("Per-operation limits exceed episode limits")
        if (type(self.wall_time_seconds) not in (int, float)
                or not math.isfinite(self.wall_time_seconds) or not 0 < self.wall_time_seconds <= 300):
            raise ValueError("Episode deadline must be greater than zero and at most 300 seconds")


_POLICY = AGENT_POLICY


def _failure_verdict(reason: str) -> dict:
    return {"verifier_version": "dml-episode-verifier-v1", "success": False, "reasons": [reason],
            "contradictions": None, "factual_outputs": None, "false_memory_claims": None,
            "recalled_claims": None, "repeat_errors": None, "repeat_opportunities": None}


def _elapsed(start: float) -> float:
    return max(0.0, (time.monotonic() - start) * 1000)


def _started(episode_id, task, scope, limits, execution_path, seed_receipts_digest=None, allowed_tools=None,
             effective_time=2000000000, prior_context=None):
    return make_event(episode_id=episode_id, task_id=task["id"], sequence=0,
        kind="episode_started", payload={"execution_path": execution_path,
        "limits": asdict(limits), "prompt": task["prompt"], "scope": scope,
        "seed_receipts_digest": seed_receipts_digest, "ranking_scope": "synthetic_fixture",
        "allowed_tools": list(task_allowed_tools(task) if allowed_tools is None else allowed_tools),
        "effective_time": effective_time, "prior_context": prior_context})


def _tool_effects(exc, name):
    if name == "retrieve":
        return "none"
    from ..journal import IdempotencyConflict
    from .receipt_ingestion import ReceiptCapacityError, ReceiptCommitRejected, ReceiptEmbeddingError
    from .receipt_lifecycle import ReceiptLifecycleConflict, ReceiptMemoryNotFound
    return "none" if isinstance(exc, (IdempotencyConflict, ReceiptCapacityError,
        ReceiptCommitRejected, ReceiptEmbeddingError, ReceiptLifecycleConflict, ReceiptMemoryNotFound)) else "unknown"


def task_allowed_tools(task):
    return ("retrieve", "supersede") if any(
        value.get("state") == "superseded" for value in task.get("state_expectations", [])) else ("retrieve",)


def build_episode_request(task, *, messages=None, limits=EpisodeLimits(), allowed_tools=None, prior_context=None):
    """Build the complete next request without truncation; useful for preflight."""
    allowed = task_allowed_tools(task) if allowed_tools is None else tuple(allowed_tools)
    return {"messages": deepcopy(messages) if messages is not None else initial_messages(task["prompt"], prior_context),
        "tools": [tool for tool in episode_tool_definitions() if tool["function"]["name"] in allowed],
        "output_reserved_tokens": limits.output_tokens}


def _run_loop(consumer, toolbox, *, task, limits, emit, prior_context=None):
    """Emit actual request/results; only admitted parsed model text selects work."""
    request = build_episode_request(task, limits=limits, allowed_tools=toolbox.allowed_tools, prior_context=prior_context)
    messages, tools = request["messages"], request["tools"]
    input_tokens = output_tokens = 0
    retrieval_ms = 0.0
    for step in range(limits.max_steps):
        request = {"messages": deepcopy(messages), "tools": deepcopy(tools),
                   "output_reserved_tokens": limits.output_tokens}
        if len(canonical_json(request)) > limits.max_transcript_bytes:
            emit("admission_rejected", "admission-" + str(step), {"step": step, "limit": "transcript_bytes",
                "observed": len(canonical_json(request)), "maximum": limits.max_transcript_bytes,
                "request": request, "compiled": None, "artifact_digest": None})
            return {"status": "transcript_limit", "answer": None, "retrieval_ms": retrieval_ms}
        if output_tokens + limits.output_tokens > limits.max_output_tokens:
            emit("admission_rejected", "admission-" + str(step), {"step": step, "limit": "output_tokens",
                "observed": output_tokens + limits.output_tokens, "maximum": limits.max_output_tokens,
                "request": request, "compiled": None, "artifact_digest": None})
            return {"status": "token_limit", "answer": None, "retrieval_ms": retrieval_ms}
        call_id = "model-" + str(step)
        before = time.monotonic()
        try:
            artifact = consumer.compile(request["messages"], request["tools"],
                                        output_reserved_tokens=limits.output_tokens)
        except Exception as exc:
            emit("model_failed", call_id, {"step": step, "phase": "compile", "request": request,
                "error_code": type(exc).__name__, "input_token_count": 0, "output_token_count": 0,
                "latency_ms": _elapsed(before), "ttft_ms": None})
            return {"status": "input_limit" if isinstance(exc, ModelInputBudgetError) else "model_error",
                    "answer": None, "retrieval_ms": retrieval_ms}
        if input_tokens + artifact.input_tokens > limits.max_input_tokens:
            emit("admission_rejected", "admission-" + str(step), {"step": step, "limit": "input_tokens",
                "observed": input_tokens + artifact.input_tokens, "maximum": limits.max_input_tokens,
                "request": request, "compiled": artifact.signing_payload(), "artifact_digest": artifact.artifact_digest})
            return {"status": "token_limit", "answer": None, "retrieval_ms": retrieval_ms}
        emit("model_requested", call_id, {"step": step, "request": request,
            "compiled": artifact.signing_payload(), "artifact_digest": artifact.artifact_digest})
        before = time.monotonic()
        try:
            result = consumer.execute(artifact)
            if (result.artifact_digest != artifact.artifact_digest
                    or result.input_token_count != artifact.input_tokens
                    or type(result.output_token_count) is not int
                    or result.output_token_count != len(result.output_ids)
                    or not 0 <= result.output_token_count <= artifact.output_reserved_tokens):
                raise ValueError("Model result differs from the dispatched artifact")
        except Exception as exc:
            emit("model_failed", call_id, {"step": step, "phase": "execute", "error_code": type(exc).__name__,
                "input_token_count": artifact.input_tokens, "output_token_count": None,
                "latency_ms": _elapsed(before), "ttft_ms": None})
            return {"status": "model_error", "answer": None, "retrieval_ms": retrieval_ms}
        emit("model_completed", call_id, {"step": step, "artifact_digest": result.artifact_digest,
            "input_token_count": result.input_token_count, "output_ids": list(result.output_ids),
            "output_token_count": result.output_token_count, "text": result.text,
            "latency_ms": _elapsed(before), "ttft_ms": None})
        input_tokens += result.input_token_count
        output_tokens += result.output_token_count
        try:
            action = parse_agent_action(result.text)
            if action["kind"] == "final":
                return {"status": "completed", "answer": action["answer"], "retrieval_ms": retrieval_ms}
            tool_id = "tool-" + str(step)
            prepared = toolbox.prepare(action["name"], action["arguments"], call_id=tool_id)
        except Exception as exc:
            emit("action_rejected", call_id, {"step": step, "error_code": type(exc).__name__})
            return {"status": "invalid_action", "answer": None, "retrieval_ms": retrieval_ms}
        emit("tool_requested", tool_id, prepared.request_payload())
        before = time.monotonic()
        try:
            raw_result, model_result = toolbox.execute(prepared)
        except Exception as exc:
            duration = _elapsed(before)
            if prepared.name == "retrieve":
                retrieval_ms += duration
            emit("tool_failed", tool_id, {"name": prepared.name, "error_code": type(exc).__name__,
                "effects": _tool_effects(exc, prepared.name), "latency_ms": duration})
            return {"status": "tool_error", "answer": None, "retrieval_ms": retrieval_ms}
        duration = _elapsed(before)
        if prepared.name == "retrieve":
            retrieval_ms += duration
        emit("tool_completed", tool_id, {"name": prepared.name, "result": raw_result,
            "model_result": model_result, "latency_ms": duration})
        messages.extend([
            {"role": "assistant", "content": result.text, "tool_calls": [{"id": tool_id,
             "type": "function", "function": {"name": prepared.name, "arguments": prepared.arguments_json}}]},
            {"role": "tool", "tool_call_id": tool_id, "name": prepared.name, "content": model_result},
        ])
    return {"status": "step_limit", "answer": None, "retrieval_ms": retrieval_ms}


def _observed(events):
    observed = []
    for event in events:
        if event["kind"] != "tool_completed":
            continue
        payload = event["payload"]
        shown = decode_json(payload["model_result"])
        records = shown.get("records", [])
        for record in payload["result"].get("observed_records", []):
            if payload["name"] != "retrieve" and payload["result"].get("receipt", {}).get("result", {}).get("memory") != record:
                raise AgentEpisodeError("Mutation evidence differs from its acknowledged receipt")
            # A hidden receipt sidecar alone cannot create a model citation.
            if any(type(item) is dict and type(item.get("id")) is int
                   and item["id"] == record["id"] and item.get("text") == record["text"]
                   and all(item.get(key) == record["meta"].get(key) for key in (
                       "source", "claim_key", "claim_value", "source_trust", "memory_state") if key in item)
                   for item in records):
                observed.append({"sequence": event["sequence"], "task_id": event["task_id"],
                                 "operation": payload["name"], "record": deepcopy(record)})
    return observed


def _finish(events, outcome, *, scenario, task, seed_records, current_records,
            effective_time, previous_answers, elapsed_ms, usage_unknown=False, verifier=None):
    status = outcome["status"]
    verdict = _failure_verdict(status)
    if status == "completed":
        try:
            if verifier is None:
                from .episode_verifiers import verify_task
                verifier = verify_task
            verdict = verifier(scenario, task, outcome["answer"], seed_records=seed_records,
                observed_records=_observed(events), current_records=current_records,
                final_sequence=len(events), effective_time=effective_time, previous_answers=previous_answers)
            validate_verifier(verdict)
        except Exception:
            status = "verifier_error"
            verdict = _failure_verdict(status)
    terminal = build_terminal(events, verdict, status=status, latency_ms=elapsed_ms,
        retrieval_ms=outcome.get("retrieval_ms"), answer=outcome.get("answer"), usage_unknown=usage_unknown)
    events.append(make_event(episode_id=events[0]["episode_id"], task_id=task["id"],
        sequence=len(events), kind="terminal", payload=terminal))
    validate_episode_events(events)
    return terminal


def _prior_feedback(task, prior_context, previous_answers):
    """One source supplies both explicitly untrusted model context and scoring."""
    validate_prior_context(prior_context)
    if previous_answers is not None and type(previous_answers) is not dict:
        raise ValueError("Previous model answers require a task-keyed object")
    repeat_from = task.get("repeat_from")
    if prior_context is None:
        if repeat_from is not None and (previous_answers or {}).get(repeat_from) is not None:
            raise ValueError("Previous verifier answer has no matching model context")
        return None, {}
    if prior_context["task_id"] != repeat_from:
        raise ValueError("Prior model context must identify the task's repeat_from source")
    answer = prior_context["answer"]
    if (previous_answers is not None and repeat_from in previous_answers
            and canonical_json(previous_answers[repeat_from]) != canonical_json(answer)):
        raise ValueError("Prior model context and verifier answer differ")
    return deepcopy(prior_context), {repeat_from: deepcopy(answer)}


def run_episode_with_test_dependencies(*, consumer, toolbox, task: dict, scenario: dict,
        seed_records: dict, current_records, limits=EpisodeLimits(), verifier=None,
        effective_time=2000000000, previous_answers=None, prior_context=None, episode_id=None) -> dict:
    """Explicit in-process test plumbing.  It does not promise a hard deadline."""
    ident = episode_id or "test-" + uuid.uuid4().hex
    start = time.monotonic()
    prior_context, previous_answers = _prior_feedback(task, prior_context, previous_answers)
    events = [_started(ident, task, toolbox.scope, limits, "test_injected", allowed_tools=toolbox.allowed_tools,
                       effective_time=effective_time, prior_context=prior_context)]
    if toolbox.effective_time != effective_time:
        raise ValueError("Tool and verifier effective times differ")
    def emit(kind, call_id, payload):
        event = make_event(episode_id=ident, task_id=task["id"], sequence=len(events),
                           kind=kind, call_id=call_id, payload=payload)
        proposed = [*events, event]
        validate_episode_events(proposed, require_terminal=False)
        events.append(event)
    usage_unknown = False
    try:
        outcome = _run_loop(consumer, toolbox, task=task, limits=limits, emit=emit, prior_context=prior_context)
    except Exception:
        outcome = {"status": "runner_error", "answer": None, "retrieval_ms": None}
        usage_unknown = True
        _interrupt_pending(events, status="runner_error")
    try:
        records = current_records() if callable(current_records) else current_records
    except Exception:
        records = []
        outcome = {"status": "runner_error", "answer": None, "retrieval_ms": None}
        usage_unknown = True
    terminal = _finish(events, outcome, scenario=scenario, task=task, seed_records=seed_records,
        current_records=records, effective_time=effective_time, previous_answers=previous_answers,
        elapsed_ms=_elapsed(start), verifier=verifier, usage_unknown=usage_unknown)
    return {"events": events, "terminal": terminal}


class _EpisodeLexicalEmbedder:
    """Versioned synthetic lexical ranking fixture; no learned-quality claim."""
    model_name = "dml-episode-lexical-fixture-v1"

    def embed(self, text):
        import numpy as np
        vector = np.zeros(16, dtype=np.float32)
        for word in re.findall(r"\w+", text.casefold()):
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:2], "big") % 16] += 1
        if not vector.any():
            vector[0] = 1
        return vector / np.linalg.norm(vector)


def _fixture_adapter(directory):
    from ..dml_adapter import DMLAdapter
    from ..settings import DMLSettings
    config = {"production_profile": "dml-receipted-local-v1", "storage_dir": str(directory),
        "model_name": "dummy", "llm_backend": "dummy", "embedding_model": _EpisodeLexicalEmbedder.model_name,
        "strict_embedding_required": True,
        "persistence": {"enable": False, "interval_sec": 0, "journal": True, "receipts": True,
            "outbox": False, "receipt_embedding_identity": "dml-episode-lexical-fixture-v1", "snapshot_interval": 1},
        "rag_store": {"enable": False},
        "dpm": {"enable": False, "mode": "disabled", "include_in_context": False, "include_in_preamble": False},
        "ann_min_items": 0, "checkpoint_interval_seconds": 0, "skip_rag_state_import": True,
        "survival_ledger_enabled": False, "background_processing_enabled": False,
        "enable_stm_controller": False, "enable_workflow_cache": False,
        "enable_quality_on_retrieval": False, "mirror_agentic_memory_to_rag": False,
        "gpu_acceleration": False, "metrics_enabled": False}
    return DMLAdapter(_validated_settings=DMLSettings(**config),
                      embedder=_EpisodeLexicalEmbedder(), start_aging_loop=False)


def _read_records(directory, *, timeout=0.25):
    """Bound final observation independently of child survival or database locks."""
    path = Path(directory) / "dml_state.sqlite3"
    deadline = time.monotonic() + timeout
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=min(timeout, 0.1))) as connection:
        connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 100)
        connection.execute("BEGIN")
        records, size = [], 0
        for row in connection.execute("SELECT payload FROM records WHERE bucket='items' ORDER BY position LIMIT 257"):
            size += len(row[0].encode("utf-8"))
            if len(records) >= 256 or size > 4 * 1024 * 1024 or time.monotonic() >= deadline:
                raise ValueError("Final authority observation exceeds fixture bounds")
            records.append(decode_json(row[0], limit=1024 * 1024))
        return records


def _prepare_fixture(directory, scenario, episode_id):
    """Create fresh receipt-backed fixtures; preserve every original acknowledgement."""
    adapter = _fixture_adapter(directory)
    receipts, aliases, setup_records = [], {}, []
    try:
        for index, seed in enumerate(scenario["seeds"]):
            receipt = adapter.ingest_memory_receipted(seed["text"],
                idempotency_key=episode_id + ":seed:" + str(index),
                meta=deepcopy(seed["meta"]), **seed.get("scope", scenario["scope"]))
            receipts.append(receipt)
            aliases[seed["alias"]] = receipt["result"]["memory"]["id"]
        for index, setup in enumerate(scenario.get("setup", [])):
            if setup["operation"] == "set_salience":
                # Explicit fresh-fixture initialization, never an agent tool or
                # public production API.  Existing authority CAS owns the write.
                revision, state = adapter._journal.read_snapshot()
                updated = deepcopy(state)
                record = next(item for item in updated["items"] if item["id"] == aliases[setup["alias"]])
                before = deepcopy(record)
                record["salience"] = setup["value"]
                adapter._journal.save(updated, expected_revision=revision)
                setup_records.append({"operation": "set_salience", "before": before,
                    "after": deepcopy(record), "source": "offline_fresh_fixture_initialization"})
            elif setup["operation"] == "related_writes":
                peer = _fixture_adapter(directory)
                try:
                    for offset, write in enumerate(setup["updates"]):
                        current = next(record for record in _read_records(directory)
                                       if record["id"] == aliases[write["alias"]])
                        from .episode_tools import _digest
                        receipt = (adapter if offset % 2 == 0 else peer).update_memory_receipted(
                            current["id"], text=write["text"], reason=write["reason"],
                            expected_memory_digest=_digest(current),
                            idempotency_key=episode_id + ":setup:" + str(index) + ":" + str(offset),
                            **scenario["scope"])
                        receipts.append(receipt)
                        setup_records.append({"operation": "related_write", "receipt": receipt,
                                              "source": "scheduled_receipted_peer"})
                finally:
                    peer.close()
            else:
                raise ValueError("Unsupported episode fixture setup")
        records = _read_records(directory)
        indexed = {record["id"]: record for record in records}
        return adapter, {"seed_receipts": receipts, "seed_records": {
            alias: indexed[ident] for alias, ident in aliases.items()}, "setup_records": setup_records}
    except BaseException:
        adapter.close()
        raise


def _worker(connection, config):
    """Spawn target. Every operation request waits for the supervisor's ACK."""
    sequence = 1
    limits = EpisodeLimits(**config["limits"])
    adapter = consumer = None
    def send(value):
        raw = canonical_json(value, limit=limits.max_event_bytes)
        connection.send_bytes(raw)
        acknowledgement = connection.recv_bytes(128)
        if acknowledgement != b"ack":
            raise RuntimeError("Supervisor did not admit worker progress")
    def emit(kind, call_id, payload):
        nonlocal sequence
        event = make_event(episode_id=config["episode_id"], task_id=config["task"]["id"],
            sequence=sequence, kind=kind, call_id=call_id, payload=payload)
        send({"kind": "event", "event": event})
        sequence += 1
    try:
        adapter, prepared = _prepare_fixture(config["authority_directory"], config["scenario"], config["episode_id"])
        send({"kind": "prepared", "prepared": prepared})
        from .model_input import LocalTransformersInputConsumer
        consumer = LocalTransformersInputConsumer(config["snapshot_directory"])
        toolbox = SelectedProfileEpisodeTools(adapter, scope=config["scenario"]["scope"],
            episode_id=config["episode_id"], seed_receipts=prepared["seed_receipts"],
            observation_records=list(prepared["seed_records"].values()),
            allowed_tools=task_allowed_tools(config["task"]), effective_time=config["effective_time"])
        outcome = _run_loop(consumer, toolbox, task=config["task"], limits=limits, emit=emit,
                            prior_context=config.get("prior_context"))
        consumer.close()
        consumer = None
        adapter.close()
        adapter = None
        send({"kind": "finished", "outcome": outcome})
    except BaseException as exc:
        try:
            send({"kind": "failed", "error_code": type(exc).__name__})
        except BaseException:
            pass
    finally:
        if consumer is not None:
            consumer.close()
        if adapter is not None:
            adapter.close()
        connection.close()


def _read_frames(connection, queue, limit, stop):
    """Only this daemon may block on a partial child frame; supervisor never does."""
    def publish(value):
        while not stop.is_set():
            try:
                queue.put(value, timeout=0.05)
                return
            except Full:
                continue
    try:
        while not stop.is_set():
            publish(("frame", connection.recv_bytes(limit)))
    except (EOFError, OSError, ValueError):
        publish(("closed", None))


def _interrupt_pending(events, *, status):
    requested = next((event for event in reversed(events)
                      if event["kind"] in ("model_requested", "tool_requested")), None)
    if requested is None or any(event["call_id"] == requested["call_id"] and event["kind"] in (
            "model_completed", "model_failed", "tool_completed", "tool_failed") for event in events):
        return False
    payload = requested["payload"]
    if requested["kind"] == "model_requested":
        kind = "model_failed"
        result = {"step": payload["step"], "phase": "execute", "error_code": "worker_" + status,
            "input_token_count": None, "output_token_count": None, "latency_ms": None, "ttft_ms": None}
    else:
        kind = "tool_failed"
        result = {"name": payload["name"], "error_code": "worker_" + status,
            "effects": "none" if payload["name"] == "retrieve" else "unknown", "latency_ms": None}
    events.append(make_event(episode_id=requested["episode_id"], task_id=requested["task_id"],
        sequence=len(events), kind=kind, call_id=requested["call_id"], payload=result))
    return True


def _supervise(config, *, worker_target=_worker):
    """Parent owns the deadline and ACKs only fully validated bounded frames."""
    limits = EpisodeLimits(**config["limits"])
    start = time.monotonic()
    deadline = start + limits.wall_time_seconds
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    process = context.Process(target=worker_target, args=(child, config), daemon=True)
    events, prepared = [], {"seed_receipts": [], "seed_records": {}, "setup_records": []}
    outcome = {"status": "runner_error", "answer": None, "retrieval_ms": None}
    completed = False
    queue = Queue(maxsize=1)
    stop = Event()
    reader = None
    started = False
    try:
        process.start()
        started = True
        child.close()
        reader = Thread(target=_read_frames, args=(parent, queue, limits.max_event_bytes, stop),
                        name="dml-episode-frame-reader", daemon=True)
        reader.start()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                outcome["status"] = "timeout"
                break
            try:
                kind, raw = queue.get(timeout=min(remaining, 0.05))
            except Empty:
                if not process.is_alive():
                    # One final bounded opportunity lets the reader publish EOF.
                    try:
                        kind, raw = queue.get(timeout=min(max(deadline - time.monotonic(), 0.0), 0.05))
                    except Empty:
                        outcome["status"] = "killed"
                        break
                else:
                    continue
            if kind == "closed":
                outcome["status"] = "killed"
                break
            value = decode_json(raw, limit=limits.max_event_bytes)
            if type(value) is not dict or type(value.get("kind")) is not str:
                raise AgentEpisodeError("Invalid worker frame")
            if value["kind"] == "prepared":
                if events or set(value) != {"kind", "prepared"}:
                    raise AgentEpisodeError("Duplicate or malformed fixture preparation")
                prepared = value["prepared"]
                if type(prepared) is not dict or set(prepared) != {"seed_receipts", "seed_records", "setup_records"}:
                    raise AgentEpisodeError("Invalid prepared receipt sidecar")
                digest = hashlib.sha256(canonical_json(prepared, limit=limits.max_event_bytes)).hexdigest()
                events.append(_started(config["episode_id"], config["task"], config["scenario"]["scope"],
                    limits, config["execution_path"], digest, effective_time=config["effective_time"],
                    prior_context=config.get("prior_context")))
            elif value["kind"] == "event":
                if not events or set(value) != {"kind", "event"}:
                    raise AgentEpisodeError("Event preceded prepared authority")
                proposed = [*events, value["event"]]
                validate_episode_events(proposed, require_terminal=False)
                events.append(value["event"])
            elif value["kind"] == "finished":
                if not events or set(value) != {"kind", "outcome"}:
                    raise AgentEpisodeError("Invalid worker completion")
                outcome = value["outcome"]
                if (type(outcome) is not dict or set(outcome) != {"status", "answer", "retrieval_ms"}
                        or outcome["status"] not in {"completed", "invalid_action", "model_error", "tool_error",
                                                    "step_limit", "input_limit", "transcript_limit", "token_limit"}):
                    raise AgentEpisodeError("Invalid worker outcome")
                completed = True
                parent.send_bytes(b"ack")
                break
            elif value["kind"] == "failed":
                if set(value) != {"kind", "error_code"}:
                    raise AgentEpisodeError("Invalid worker failure")
                parent.send_bytes(b"ack")
                break
            else:
                raise AgentEpisodeError("Unknown worker frame")
            # Parent observation and validation precede acknowledgement; a child
            # may not begin generation or a tool effect until this byte arrives.
            parent.send_bytes(b"ack")
    except (ValueError, TypeError, KeyError, OSError, EOFError, RuntimeError, AgentEpisodeError):
        outcome = {"status": "runner_error", "answer": None, "retrieval_ms": None}
    finally:
        if started and process.is_alive():
            if completed:
                process.join(timeout=min(0.1, max(0, deadline - time.monotonic())))
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.2)
            if process.is_alive():
                process.kill()
                process.join(timeout=0.2)
        elif started:
            process.join(timeout=0)
        child.close()
        stop.set()
        parent.close()
        if reader is not None:
            reader.join(timeout=0.1)
    if not events:
        events.append(_started(config["episode_id"], config["task"], config["scenario"]["scope"],
                               limits, config["execution_path"], effective_time=config["effective_time"],
                               prior_context=config.get("prior_context")))
    if not completed:
        _interrupt_pending(events, status=outcome["status"])
    try:
        records = _read_records(config["authority_directory"])
    except (OSError, sqlite3.Error, ValueError):
        records = []
        if completed:
            outcome = {"status": "runner_error", "answer": None, "retrieval_ms": None}
            completed = False
    terminal = _finish(events, outcome, scenario=config["scenario"], task=config["task"],
        seed_records=prepared["seed_records"], current_records=records,
        effective_time=config["effective_time"], previous_answers=config.get("previous_answers"),
        elapsed_ms=_elapsed(start), usage_unknown=not completed)
    return {"events": events, "terminal": terminal, "prepared": prepared,
            "current_records": records, "live_qualified": False}


def run_local_episode(*, snapshot_directory, work_directory, scenario: dict, task: dict,
                      limits=EpisodeLimits(), effective_time=2000000000, previous_answers=None,
                      prior_context=None) -> dict:
    """Run a fresh fixture through the actual local exact-input consumer.

    Startup, fixture writes, compile, generation and agent tools are inside the
    supervised deadline. No model is downloaded and no supplied flag establishes
    trained-model provenance, semantic success or live qualification.
    """
    if type(limits) is not EpisodeLimits:
        raise ValueError("Exact EpisodeLimits are required")
    from .episode_verifiers import load_episode_corpus
    corpus = load_episode_corpus()
    if (not any(canonical_json(scenario) == canonical_json(value) for value in corpus["scenarios"])
            or not any(canonical_json(task) == canonical_json(value) for value in scenario["tasks"])
            or type(effective_time) is not int or effective_time != corpus["effective_time"]):
        raise ValueError("Concrete execution requires a task in the pinned packaged corpus")
    canonical_json(previous_answers, limit=1024 * 1024)
    prior_context, previous_answers = _prior_feedback(task, prior_context, previous_answers)
    directory = Path(work_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    ident = "episode-" + uuid.uuid4().hex
    config = {"episode_id": ident, "execution_path": "live_local", "scenario": deepcopy(scenario),
        "task": deepcopy(task), "limits": asdict(limits), "effective_time": effective_time,
        "prior_context": prior_context,
        "previous_answers": deepcopy(previous_answers), "snapshot_directory": str(Path(snapshot_directory).resolve()),
        "authority_directory": str(directory / "authority")}
    return _supervise(config)
