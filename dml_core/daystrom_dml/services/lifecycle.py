"""Explicit lifecycle decisions. Retrieval never promotes source authority."""
import math
from typing import Mapping

POLICY_VERSION = "explicit-memory-lifecycle-v1"


def suppression_reason(meta: Mapping, *, now: float, include_quarantined: bool = False) -> str | None:
    if include_quarantined:
        return None  # trusted operator inspection; not a promotion
    state = str(meta.get("memory_state") or meta.get("lifecycle_state") or "").strip().lower()
    namespace = str(meta.get("namespace") or "").strip().lower()
    if state in {"quarantine", "quarantined", "suppressed", "deleted", "superseded", "expired"}:
        return "state_" + state
    if namespace in {"quarantine", "quarantined"}:
        return "quarantine_namespace"
    if str(meta.get("source_trust") or "").strip().lower() == "untrusted":
        return "untrusted_source"
    if meta.get("superseded_by") is not None:
        return "explicitly_superseded"
    if "expires_at" in meta:
        expires = meta["expires_at"]
        if type(expires) not in (int, float) or not math.isfinite(expires):
            return "invalid_expiry"
        if now >= expires:
            return "expired"
    return None
