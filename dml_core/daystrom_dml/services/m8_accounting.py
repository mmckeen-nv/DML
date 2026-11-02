"""Failure-inclusive development accounting, not complete M8 cost qualification.

Durable dispatch supervision, GPU/CPU metering and model attestation are separate
execution gates. Unknown dispatch usage blocks continuation and any cost claim.

The durable ledger, deadline supervisor and episode metering below are
development-only instrumentation: they record observed costs faithfully but do
not qualify execution costs, and GPU wall seconds are caller-reported only.
"""

from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

from ..contracts.agent_episode import decode_json


class BudgetExhausted(ValueError):
    pass


def _integer(value):
    if type(value) is not int or value < 0:
        raise ValueError("nonnegative integer required")
    return value


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("finite nonnegative measurement required")
    return value


def _ids(values):
    if type(values) not in (list, tuple):
        raise ValueError("exact token IDs required")
    return tuple(_integer(value) for value in values)


class DevelopmentAccounting:
    """Original ceilings; every admitted dispatch consumes its input and step."""

    def __init__(self, *, clock=time.monotonic):
        self.clock = clock
        self.started = _number(clock())
        self._turns, self._stages = [], []
        self._pending = None
        self.input_tokens = self.output_tokens = 0
        self.usage_unknown = False
        self.terminal = False

    def _elapsed(self):
        elapsed = self.clock() - self.started
        return _number(elapsed)

    def admit(self, call_id, input_ids, *, reserved_output=256):
        ids = _ids(input_ids)
        if type(call_id) is not str or not call_id or any(t["call_id"] == call_id for t in self._turns):
            raise ValueError("distinct owned call identity required")
        if type(reserved_output) is not int or not 1 <= reserved_output <= 256:
            raise ValueError("invalid output reservation")
        if self._pending is not None or self.usage_unknown or self.terminal:
            raise BudgetExhausted("pending, unknown or terminal episode")
        if (
            len(self._turns) >= 6
            or self.input_tokens + len(ids) > 32768
            or self.output_tokens + reserved_output > 1536
            or len(ids) + reserved_output > 8192
            or self._elapsed() >= 300
        ):
            raise BudgetExhausted("original budget exhausted")
        row = {
            "call_id": call_id,
            "input_ids": list(ids),
            "output_ids": None,
            "reserved_output": reserved_output,
            "status": "pending",
            "error": None,
        }
        self._turns.append(row)
        self._pending = row
        self.input_tokens += len(ids)
        return deepcopy(row)

    def complete(self, call_id, output_ids, *, error=None, usage_known=True):
        if self._pending is None or self._pending["call_id"] != call_id:
            raise ValueError("completion has no matching dispatch")
        if type(usage_known) is not bool or (error is not None and type(error) is not str):
            raise ValueError("invalid completion metadata")
        ids = _ids(output_ids)
        row = self._pending
        row.update(output_ids=list(ids), error=error, status="error" if error is not None else "completed")
        self.output_tokens += len(ids)
        self._pending = None
        if not usage_known:
            self.usage_unknown = True
            row["status"] = "unknown_usage"
        if len(ids) > row["reserved_output"]:
            row["status"] = "output_overrun"
        if self._elapsed() >= 300:
            row["status"] = "timeout"
        if row["status"] != "completed":
            self.terminal = True
        return deepcopy(row)

    def record_stage(
        self,
        name,
        *,
        wall_seconds,
        cpu_seconds=None,
        gpu_seconds=None,
        embedding_tokens=0,
        embedding_calls=0,
        bytes_written=0,
    ):
        if name not in {
            "startup",
            "warmup",
            "ingestion",
            "embedding",
            "retrieval",
            "maintenance",
            "checkpoint",
            "rebuild",
            "teardown",
        }:
            raise ValueError("unknown cost stage")
        row = {
            "name": name,
            "wall_seconds": _number(wall_seconds),
            "cpu_seconds": None if cpu_seconds is None else _number(cpu_seconds),
            "gpu_seconds": None if gpu_seconds is None else _number(gpu_seconds),
            "embedding_tokens": _integer(embedding_tokens),
            "embedding_calls": _integer(embedding_calls),
            "bytes_written": _integer(bytes_written),
        }
        self._stages.append(row)

    def snapshot(self):
        return deepcopy(
            {
                "schema": "dml-m8-development-accounting-v1",
                "development_only": True,
                "execution_costs_qualified": False,
                "turns": self._turns,
                "stages": self._stages,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "usage_unknown": self.usage_unknown or self._pending is not None,
                "terminal": self.terminal,
                "elapsed_seconds": self._elapsed(),
            }
        )


def tokens_per_correct_success(episodes):
    """Unknown or zero-success aggregate cannot establish token efficiency."""
    total, successes = 0, 0
    for episode in episodes:
        if type(episode["correct"]) is not bool:
            raise ValueError("independently graded correctness required")
        accounting = episode["accounting"]
        if accounting["usage_unknown"]:
            return None
        total += _integer(accounting["input_tokens"]) + _integer(accounting["output_tokens"])
        successes += episode["correct"]
    return total / successes if successes else None


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class DurableLedger:
    """Fsync'd append-only JSONL evidence ledger with hash-chained replay.

    Every admitted input, metering step and output is recorded with a sequence
    number, including failed and timed-out attempts; nothing is rewritten or
    dropped. The chain detects accidental alteration, not an adversary
    rewriting the whole file; external freeze hashes remain required. After
    any append I/O failure this object is failed and refuses further records.
    """

    version = "dml-m8-durable-ledger-v1"
    KINDS = frozenset({"input", "step", "output"})
    OUTPUT_STATUSES = frozenset({"completed", "error", "timeout", "unknown_usage", "output_overrun"})

    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists() and not self.path.is_file():
            raise ValueError("ledger path must be a regular file")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._head = "0" * 64
        self._records = []
        self.failed = False
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        for line in self.path.read_bytes().splitlines():
            row = decode_json(line, limit=16 * 1024 * 1024)
            if type(row) is not dict:
                self.failed = True
                raise ValueError("invalid durable ledger chain")
            digest = row.pop("sha256", None)
            if (
                row.get("previous") != self._head
                or row.get("sequence") != len(self._records)
                or row.get("version") != self.version
                or digest != hashlib.sha256(_canonical(row)).hexdigest()
            ):
                self.failed = True
                raise ValueError("invalid durable ledger chain")
            self._validate_payload(row.get("kind"), row.get("payload"))
            row["sha256"] = digest
            self._records.append(row)
            self._head = digest

    def _validate_payload(self, kind, payload):
        if kind not in self.KINDS:
            raise ValueError("unknown ledger record kind")
        if type(payload) is not dict:
            raise ValueError("ledger payload must be an object")
        if kind == "input":
            if type(payload.get("call_id")) is not str or not payload["call_id"]:
                raise ValueError("owned call identity required")
            _ids(payload.get("input_ids"))
        elif kind == "step":
            if type(payload.get("name")) is not str or not payload["name"]:
                raise ValueError("named metering step required")
            _number(payload.get("wall_seconds"))
            for key in ("cpu_seconds", "gpu_seconds"):
                if payload.get(key) is not None:
                    _number(payload[key])
            for key in ("embedding_tokens", "embedding_calls", "bytes_written"):
                if key in payload:
                    _integer(payload[key])
        else:
            if type(payload.get("call_id")) is not str or not payload["call_id"]:
                raise ValueError("owned call identity required")
            if payload.get("status") not in self.OUTPUT_STATUSES:
                raise ValueError("explicit failure-inclusive output status required")
            _ids(payload.get("output_ids"))
            if payload.get("error") is not None and type(payload["error"]) is not str:
                raise ValueError("invalid output error record")

    def __len__(self):
        return len(self._records)

    def records(self):
        return deepcopy(self._records)

    def record(self, kind, payload):
        if self.failed:
            raise ValueError("ledger failed; no further records permitted")
        payload = deepcopy(payload)
        self._validate_payload(kind, payload)
        row = {
            "version": self.version,
            "sequence": len(self._records),
            "previous": self._head,
            "kind": kind,
            "payload": payload,
        }
        digest = hashlib.sha256(_canonical(row)).hexdigest()
        row["sha256"] = digest
        # Fail-closed before the first durable write: any exception, including
        # evidence I/O failure, poisons this object against further records.
        self.failed = True
        with self.path.open("ab") as stream:
            stream.write(_canonical(row) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self._head = digest
        self._records.append(row)
        self.failed = False
        return deepcopy(row)

    def record_episode(self, snapshot):
        """Durably record every admitted input, output and stage of an episode.

        Failure-inclusive: error, timeout, unknown-usage and overrun outputs
        are recorded, never dropped. A snapshot with a pending dispatch is
        rejected before any write; per-turn input/output pairs precede steps.
        """
        if type(snapshot) is not dict or type(snapshot.get("turns")) is not list or type(snapshot.get("stages")) is not list:
            raise ValueError("development accounting snapshot required")
        for turn in snapshot["turns"]:
            if type(turn) is not dict or turn.get("status") == "pending":
                raise ValueError("pending dispatch has no output to record")
        for turn in snapshot["turns"]:
            self.record("input", {"call_id": turn["call_id"], "input_ids": turn["input_ids"]})
            self.record(
                "output",
                {
                    "call_id": turn["call_id"],
                    "status": turn["status"],
                    "output_ids": turn["output_ids"],
                    "error": turn["error"],
                },
            )
        for stage in snapshot["stages"]:
            self.record("step", stage)


class DeadlineExceeded(TimeoutError):
    """Declared wall deadline reached; the attempt is charged its full deadline."""


class DeadlineSupervisor:
    """Single-use monotonic wall-clock deadline context for one attempt.

    A timed-out attempt is charged the FULL declared deadline — never the
    partial elapsed time and never zero — while a completed attempt is charged
    its observed wall time. This is dispatch supervision only; it does not
    qualify execution costs.
    """

    def __init__(self, budget_seconds, *, clock=time.monotonic):
        self.budget = _number(budget_seconds)
        if self.budget <= 0:
            raise ValueError("positive deadline budget required")
        if not callable(clock):
            raise ValueError("monotonic clock required")
        self.clock = clock
        self.started = _number(clock())
        self._attempts = []

    def _elapsed(self):
        return _number(self.clock() - self.started)

    def check(self):
        remaining = self.budget - self._elapsed()
        if remaining <= 0:
            raise DeadlineExceeded("declared wall deadline reached")
        return remaining

    def __enter__(self):
        if self._attempts:
            raise ValueError("deadline supervisor is single-use")
        self.check()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = self._elapsed()
        timed_out = (exc_type is not None and issubclass(exc_type, DeadlineExceeded)) or elapsed >= self.budget
        self._attempts.append(
            {
                "deadline_seconds": self.budget,
                "observed_seconds": elapsed,
                "timed_out": timed_out,
                "charged_seconds": self.budget if timed_out else elapsed,
            }
        )
        if timed_out and exc_type is None:
            raise DeadlineExceeded("declared wall deadline reached")
        return False

    def records(self):
        return deepcopy(self._attempts)


def _rusage_sample():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is bytes on Darwin and kibibytes on Linux; normalize to bytes.
    peak = usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
    return usage.ru_utime + usage.ru_stime, int(peak)


class EpisodeMetering:
    """Full manifest metering vector for one attempted episode.

    CPU core-seconds and peak RSS are sampled from resource.getrusage. GPU
    wall seconds are explicitly caller-reported; this module never queries
    NVML or any device. Unmeasured costs remain None (unknown) and are never
    reported as zero; only an actual measurement may be zero.
    """

    version = "dml-m8-episode-metering-v1"
    STAGES = frozenset({"warmup", "ingestion", "retrieval", "maintenance", "compaction", "checkpoint", "rebuild"})

    def __init__(self, *, sample=None):
        if sample is not None and not callable(sample):
            raise ValueError("rusage sampler must be callable")
        self._sample = sample or _rusage_sample
        self._start_core, self._peak_rss = self._sample_pair()
        self._embedding = None
        self._stages = {}
        self._disk_bytes = None
        self._gpu_wall = None

    def _sample_pair(self):
        pair = self._sample()
        if type(pair) not in (tuple, list) or len(pair) != 2:
            raise ValueError("sampler must return (cpu_core_seconds, peak_rss_bytes)")
        return _number(pair[0]), _integer(pair[1])

    def record_embedding(self, *, input_tokens, calls, cpu_seconds):
        input_tokens, calls, cpu_seconds = _integer(input_tokens), _integer(calls), _number(cpu_seconds)
        if self._embedding is None:
            self._embedding = {"input_tokens": 0, "calls": 0, "cpu_seconds": 0.0}
        self._embedding["input_tokens"] += input_tokens
        self._embedding["calls"] += calls
        self._embedding["cpu_seconds"] += cpu_seconds

    def record_stage(self, name, *, wall_seconds, cpu_seconds=None):
        if name not in self.STAGES:
            raise ValueError("unknown manifest cost stage")
        row = self._stages.setdefault(name, {"wall_seconds": 0.0, "cpu_seconds": None})
        row["wall_seconds"] += _number(wall_seconds)
        if cpu_seconds is not None:
            row["cpu_seconds"] = _number(cpu_seconds) + (0.0 if row["cpu_seconds"] is None else row["cpu_seconds"])

    def record_disk(self, *, bytes_written):
        self._disk_bytes = _integer(bytes_written) + (self._disk_bytes or 0)

    def record_gpu(self, *, wall_seconds):
        """Caller-reported GPU wall seconds only; never device-queried here."""
        self._gpu_wall = _number(wall_seconds) + (self._gpu_wall or 0.0)

    def snapshot(self):
        core, peak = self._sample_pair()
        self._peak_rss = max(self._peak_rss, peak)
        return deepcopy(
            {
                "schema": self.version,
                "execution_costs_qualified": False,
                "embedding": self._embedding,
                "stages": self._stages,
                "cpu_core_seconds": core - self._start_core,
                "peak_rss_bytes": self._peak_rss,
                "disk_bytes_written": self._disk_bytes,
                "gpu_wall_seconds": self._gpu_wall,
                "gpu_wall_seconds_source": None if self._gpu_wall is None else "caller_reported",
            }
        )


def cost_per_correct_success(episodes):
    """Cost per correct success over every attempted episode.

    The numerator sums the cost of ALL attempts, failures included; the
    denominator counts only independently graded correct successes. A
    zero-success aggregate or any unknown (None) attempt cost is rejected —
    unknown cost is never treated as zero and a free pass is never issued.
    """
    if type(episodes) not in (list, tuple) or not episodes:
        raise ValueError("attempted episodes required")
    total, successes = 0.0, 0
    for episode in episodes:
        if type(episode) is not dict or type(episode.get("correct")) is not bool:
            raise ValueError("independently graded correctness required")
        if episode.get("cost") is None:
            raise ValueError("unknown attempt cost cannot establish cost per success")
        total += _number(episode["cost"])
        successes += episode["correct"]
    if not successes:
        raise ValueError("zero correct successes cannot establish cost per success")
    return total / successes
