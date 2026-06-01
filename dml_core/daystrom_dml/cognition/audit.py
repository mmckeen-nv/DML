"""DCN audit trail storage and redaction helpers."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SENSITIVE_KEY_PARTS = ("secret", "token", "api_key", "apikey", "password", "credential", "authorization")
REDACTED_VALUE_KEYS = {"raw_transcript", "tool_log", "prompt_scaffold", "raw_context", "assembled_context"}


class DCNAuditStore:
    """Append-only JSONL audit store for DCN decisions and outcomes.

    Audit records intentionally store decisions/outcomes, not payload content.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        record = {
            "event_type": event_type,
            "timestamp": time.time(),
            "payload": sanitize_audit_payload(payload),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        return record

    def tail(self, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        safe_limit = max(0, min(int(limit), 500))
        lines = self.path.read_text(encoding="utf-8").splitlines()[-safe_limit:]
        records = []
        for line in lines:
            if not line.strip():
                continue
            records.append(json.loads(line))
        return records


def sanitize_audit_payload(payload: Any) -> Any:
    """Redact secrets and raw context-like payloads before persistence."""

    if isinstance(payload, dict):
        clean: Dict[str, Any] = {}
        for key, value in payload.items():
            key_str = str(key)
            lowered = key_str.lower()
            if key_str in REDACTED_VALUE_KEYS or any(part in lowered for part in SENSITIVE_KEY_PARTS):
                clean[key_str] = "[REDACTED]"
            else:
                clean[key_str] = sanitize_audit_payload(value)
        return clean
    if isinstance(payload, list):
        return [sanitize_audit_payload(item) for item in payload]
    if isinstance(payload, tuple):
        return [sanitize_audit_payload(item) for item in payload]
    return payload


def has_forbidden_audit_content(records: Iterable[Dict[str, Any]], forbidden: Iterable[str]) -> bool:
    text = json.dumps(list(records), sort_keys=True, default=str).lower()
    return any(item.lower() in text for item in forbidden)
