#!/usr/bin/env python3
"""Report and optionally quarantine noisy Daystrom DML runtime memories.

Dry-run is the default. With --apply, matching records in dml_state.jsonl are
marked with retrieval-suppression metadata rather than deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any


DEFAULT_STORE = Path(
    "/Users/markmckeen/.hermes/hermes-agent/integrations/daystrom-dml/stores/hermes-runtime-store"
)
STATE_TYPE = "daystrom_dml.memory"
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("smoke_or_self_test", re.compile(r"\b(?:smoke[- ]?test|self[- ]?test|test record|pre[- ]?fix)\b", re.IGNORECASE)),
    ("completed_turn_artifact", re.compile(r"\bcompleted (?:snips_?2|citizen snips) turn\b", re.IGNORECASE)),
    ("injected_memory_context", re.compile(r"<\s*/?\s*memory-context\s*>", re.IGNORECASE)),
    ("daystrom_injected_block", re.compile(r"=== Daystrom (?:Personality Matrix Overlay|DML Active Continuity|DML Retrieved Memory) ===", re.IGNORECASE)),
    ("repeated_role_prefix", re.compile(r"\b(?:user|assistant):\s*\|?\s*(?:user|assistant):", re.IGNORECASE)),
    ("tool_output_boilerplate", re.compile(r"\b(?:chunk id:|wall time:|process exited|original token count|functions\.exec_command|apply_patch)\b", re.IGNORECASE)),
    ("assistant_scaffolding", re.compile(r"^(?:i(?:'|’)ll|let me|checking|reading|inspecting)\b", re.IGNORECASE | re.MULTILINE)),
    ("system_memory_wrapper", re.compile(r"\b(?:System note: The following is recalled memory context|Internal memory context:|authoritative reference data|persistent memory and should inform all responses)\b", re.IGNORECASE)),
    ("recent_channel_prelude", re.compile(r"\[(?:Recent channel messages|New message)\]", re.IGNORECASE)),
    ("summary_instruction_preface", re.compile(r"\bHere is a summary\b|\b(?:tokens|characters) or less\b", re.IGNORECASE)),
)
SENSITIVE_RE = re.compile(r"(?i)\b(api[_-]?key|token|secret|password|authorization|bearer)\b\s*[:=]\s*\S+")


def redact(text: str) -> str:
    return SENSITIVE_RE.sub(lambda m: f"{m.group(1)}=[REDACTED]", text or "")


def compact(text: str, limit: int = 220) -> str:
    text = " ".join(redact(text).split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rstrip()
    word = cut.rfind(" ")
    if word > 80:
        cut = cut[:word]
    return cut + "..."


def load_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise SystemExit(f"empty state file: {path}")
    header = json.loads(lines[0])
    records = [json.loads(line) for line in lines[1:] if line.strip()]
    return header, records


def classify(record: dict[str, Any]) -> list[str]:
    text = str(record.get("text") or "")
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    haystack = text + "\n" + json.dumps(meta, sort_keys=True, default=str)
    reasons = [name for name, pattern in PATTERNS if pattern.search(haystack)]
    if meta.get("source") == "hermes-memory-provider" and meta.get("summary_source") != "hermes_daystrom_hygiene_v1":
        reasons.append("legacy_unhygiened_provider_turn")
    return reasons


def write_state(path: Path, records: list[dict[str, Any]]) -> Path:
    payload_lines = [json.dumps(record, separators=(",", ":"), sort_keys=True) for record in records]
    checksum = hashlib.sha256("\n".join(payload_lines).encode("utf-8")).hexdigest()
    header = {
        "type": STATE_TYPE,
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "count": len(records),
        "checksum": checksum,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(header, separators=(",", ":"), sort_keys=True)
        + ("\n" + "\n".join(payload_lines) if payload_lines else ""),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--state-file", default="dml_state.jsonl")
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--apply", action="store_true", help="mark matching records quarantined; default is dry-run")
    args = parser.parse_args()

    state_path = args.store_dir / args.state_file
    header, records = load_jsonl(state_path)
    counts: dict[str, int] = {}
    examples: dict[str, list[dict[str, Any]]] = {}
    matched: list[tuple[dict[str, Any], list[str]]] = []

    for record in records:
        reasons = classify(record)
        if not reasons:
            continue
        matched.append((record, reasons))
        for reason in reasons:
            counts[reason] = counts.get(reason, 0) + 1
            bucket = examples.setdefault(reason, [])
            if len(bucket) < max(0, args.examples):
                bucket.append(
                    {
                        "id": record.get("id"),
                        "text": compact(str(record.get("text") or "")),
                        "meta": {
                            key: value
                            for key, value in (record.get("meta") or {}).items()
                            if key in {"source", "phase", "provider", "summary_source", "memory_state", "namespace"}
                        },
                    }
                )

    result: dict[str, Any] = {
        "state_path": str(state_path),
        "dry_run": not args.apply,
        "total_records": len(records),
        "matched_records": len(matched),
        "counts": dict(sorted(counts.items())),
        "examples": examples,
    }

    if args.apply and matched:
        backup = state_path.with_suffix(state_path.suffix + f".bak.{int(time.time())}")
        shutil.copy2(state_path, backup)
        for record, reasons in matched:
            meta = record.setdefault("meta", {})
            if isinstance(meta, dict):
                meta["memory_state"] = "quarantined"
                meta["namespace"] = "quarantine"
                meta["quarantine_reason"] = "daystrom_hygiene_scan"
                meta["quarantine_matches"] = sorted(set(reasons))
                meta["quarantined_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_state(state_path, records)
        result["backup_path"] = str(backup)
        result["applied"] = True
    else:
        result["applied"] = False

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
