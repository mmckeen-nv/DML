#!/usr/bin/env python3
"""Compile two exact ContextPacket generations into payload-free native work."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Sequence

_DML_CORE = Path(__file__).resolve().parents[1]
if str(_DML_CORE) not in sys.path:
    sys.path.insert(0, str(_DML_CORE))

from daystrom_dml.api_contracts import ContractError  # noqa: E402
from daystrom_dml.context.manifest import ContextPacket  # noqa: E402
from daystrom_dml.context.native_transition import (  # noqa: E402
    NativeContextCheckpointBinding,
    NativeContextTransitionCompiler,
)
from daystrom_dml.context.probe import atomic_write_json  # noqa: E402

_ALLOWED_FIELDS = frozenset(
    {
        "parent_packet",
        "current_packet",
        "parent_checkpoint",
        "model_native_limit",
        "served_limit",
        "observed_at",
    }
)
_REQUIRED_FIELDS = frozenset(
    {
        "parent_packet",
        "current_packet",
        "model_native_limit",
        "served_limit",
    }
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load(path: Path) -> tuple[ContextPacket, ContextPacket, NativeContextCheckpointBinding | None, int, int, float]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("input must be readable UTF-8 JSON") from exc
    if not isinstance(raw, dict):
        raise ContractError("input must be an object")
    unknown = sorted(set(raw) - _ALLOWED_FIELDS)
    missing = sorted(_REQUIRED_FIELDS - set(raw))
    if unknown:
        raise ContractError(f"unknown top-level fields: {','.join(unknown)}")
    if missing:
        raise ContractError(f"missing top-level fields: {','.join(missing)}")
    if not isinstance(raw["parent_packet"], dict) or not isinstance(raw["current_packet"], dict):
        raise ContractError("packet fields must be objects")
    parent = ContextPacket.from_dict(raw["parent_packet"])
    current = ContextPacket.from_dict(raw["current_packet"])
    checkpoint_raw = raw.get("parent_checkpoint")
    if checkpoint_raw is None:
        checkpoint = None
    elif isinstance(checkpoint_raw, dict):
        checkpoint = NativeContextCheckpointBinding.from_dict(checkpoint_raw)
    else:
        raise ContractError("parent_checkpoint must be an object or null")
    observed = raw.get("observed_at", time.time())
    if (
        isinstance(observed, bool)
        or not isinstance(observed, (int, float))
        or not math.isfinite(float(observed))
    ):
        raise ContractError("observed_at must be finite numeric epoch seconds")
    return (
        parent,
        current,
        checkpoint,
        raw["model_native_limit"],
        raw["served_limit"],
        float(observed),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Transition input JSON")
    parser.add_argument("--artifact", required=True, type=Path, help="Payload-free plan JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        parent, current, checkpoint, native_limit, served_limit, observed = _load(args.input)
        plan = NativeContextTransitionCompiler(clock=lambda: observed).compile(
            parent_packet=parent,
            current_packet=current,
            parent_checkpoint=checkpoint,
            model_native_limit=native_limit,
            served_limit=served_limit,
        )
        report = {**plan.to_dict(), "pass": plan.feasible}
        status = 0 if plan.feasible else 2
    except Exception as exc:
        report = {
            "schema_version": "daystrom-native-context-transition-failure-v1",
            "pass": False,
            "error_type": type(exc).__name__ if isinstance(exc, ContractError) else "TransitionError",
            "reason_digest": _digest(str(exc)),
        }
        status = 1
    atomic_write_json(args.artifact, report)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return status


if __name__ == "__main__":
    sys.exit(main())
