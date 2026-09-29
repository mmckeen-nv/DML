"""Failure-inclusive development accounting, not complete M8 cost qualification.

Durable dispatch supervision, GPU/CPU metering and model attestation are separate
execution gates. Unknown dispatch usage blocks continuation and any cost claim.
"""

from __future__ import annotations
from copy import deepcopy
import math
import time


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
