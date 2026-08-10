#!/usr/bin/env python3
"""Produce a deterministic, payload-free native-context lifecycle profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

# Prefer this checkout over any older globally installed daystrom_dml package.
_DML_CORE = Path(__file__).resolve().parents[1]
if str(_DML_CORE) not in sys.path:
    sys.path.insert(0, str(_DML_CORE))

from daystrom_dml.api_contracts import ContractError
from daystrom_dml.context.native_profile import (
    NativeContextProfileConfig,
    NativeContextProfiler,
    NativeContextSpan,
)
from daystrom_dml.context.probe import atomic_write_json

_ALLOWED_TOP_LEVEL = frozenset({"config", "spans", "requested_span_ids"})
_ALLOWED_CONFIG = frozenset(
    {
        "model_id",
        "runtime_id",
        "model_native_limit",
        "served_limit",
        "target_hot_tokens",
        "stale_after_turns",
        "freeze_after_turns",
        "runtime_state_bytes_per_token",
    }
)
_ALLOWED_SPAN = frozenset(
    {
        "span_id",
        "content_digest",
        "start_token",
        "token_count",
        "authority",
        "priority",
        "exact_required",
        "resident",
        "age_turns",
        "reference_count",
        "summary_digest",
        "summary_tokens",
    }
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    return value


def _reject_unknown(value: dict[str, Any], allowed: frozenset[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ContractError(f"unknown {label} fields: {','.join(unknown)}")


def _load(path: Path) -> tuple[NativeContextProfileConfig, list[NativeContextSpan], list[str]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("input must be readable UTF-8 JSON") from exc

    root = _mapping(raw, "input")
    _reject_unknown(root, _ALLOWED_TOP_LEVEL, "top-level")

    config_raw = _mapping(root.get("config"), "config")
    _reject_unknown(config_raw, _ALLOWED_CONFIG, "config")
    try:
        config = NativeContextProfileConfig(**config_raw)
    except TypeError as exc:
        raise ContractError("config fields do not match the lifecycle contract") from exc

    spans_raw = root.get("spans")
    if not isinstance(spans_raw, list):
        raise ContractError("spans must be an array")
    spans: list[NativeContextSpan] = []
    for index, raw_span in enumerate(spans_raw):
        span = _mapping(raw_span, f"spans[{index}]")
        _reject_unknown(span, _ALLOWED_SPAN, f"spans[{index}]")
        try:
            spans.append(NativeContextSpan(**span))
        except TypeError as exc:
            raise ContractError(f"spans[{index}] fields do not match the manifest contract") from exc

    requested = root.get("requested_span_ids", [])
    if not isinstance(requested, list) or not all(isinstance(item, str) for item in requested):
        raise ContractError("requested_span_ids must be an array of strings")
    return config, spans, requested


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Lifecycle manifest JSON")
    parser.add_argument("--artifact", required=True, type=Path, help="Digest-only output JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config, spans, requested = _load(args.input)
        profile = NativeContextProfiler(config).profile(spans, requested_span_ids=requested)
        report = {**profile.to_dict(), "pass": profile.feasible}
        status = 0
    except Exception as exc:  # Failure artifacts intentionally disclose only class and digest.
        error_type = type(exc).__name__ if isinstance(exc, ContractError) else "ProfileError"
        report = {
            "schema_version": "dcm.native-context-profile.failure.v1",
            "pass": False,
            "error_type": error_type,
            "reason_digest": _digest(str(exc)),
        }
        status = 1

    atomic_write_json(args.artifact, report)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return status


if __name__ == "__main__":
    sys.exit(main())
