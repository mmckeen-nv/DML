"""Context manifest and packet contracts."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from daystrom_dml.api_contracts import AuditInfo, ContractError, DaystromScope, SerializableDataclass, _serialize
from daystrom_dml.context.budget import ContextBudget
from daystrom_dml.context.capabilities import RuntimeCapabilities
from daystrom_dml.context.schema import ContextSegment

CONTEXT_MANIFEST_V1 = "daystrom-context-manifest-v1"
CONTEXT_PACKET_V1 = "daystrom-context-packet-v1"


@dataclass
class ContextManifest(SerializableDataclass):
    """Deterministic manifest for an ordered set of admitted context segments."""

    scope: DaystromScope = field(default_factory=DaystromScope)
    model_id: str = ""
    runtime_id: str = ""
    segment_ids: List[str] = field(default_factory=list)
    estimated_input_tokens: int = 0
    exact_input_tokens: Optional[int] = None
    parent_checkpoint_id: Optional[str] = None
    parent_manifest_id: Optional[str] = None
    decisions: Dict[str, Any] = field(default_factory=dict)
    audit: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    manifest_version: str = CONTEXT_MANIFEST_V1
    content_digest: str = ""

    def __post_init__(self) -> None:
        if self.manifest_version != CONTEXT_MANIFEST_V1:
            raise ContractError(f"manifest_version must be {CONTEXT_MANIFEST_V1!r}")
        if isinstance(self.scope, dict):
            self.scope = DaystromScope.from_dict(self.scope)
        if not self.model_id:
            raise ContractError("model_id must be non-empty")
        if not self.runtime_id:
            raise ContractError("runtime_id must be non-empty")
        if any(not segment_id for segment_id in self.segment_ids):
            raise ContractError("segment_ids must not contain empty values")
        if not isinstance(self.estimated_input_tokens, int) or isinstance(self.estimated_input_tokens, bool):
            raise ContractError("estimated_input_tokens must be an integer")
        if self.estimated_input_tokens < 0:
            raise ContractError("estimated_input_tokens must be non-negative")
        if self.exact_input_tokens is not None:
            if not isinstance(self.exact_input_tokens, int) or isinstance(self.exact_input_tokens, bool):
                raise ContractError("exact_input_tokens must be an integer")
            if self.exact_input_tokens < 0:
                raise ContractError("exact_input_tokens must be non-negative")
        for name in ("decisions", "audit"):
            if not isinstance(getattr(self, name), dict):
                raise ContractError(f"{name} must be a dictionary")
        computed_digest = self.compute_content_digest()
        if self.content_digest and self.content_digest != computed_digest:
            raise ContractError("content_digest does not match manifest content")
        self.content_digest = computed_digest

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]):
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ContractError(f"{cls.__name__}.from_dict expected dict, got {type(data).__name__}")
        if not data.get("content_digest"):
            raise ContractError("content_digest is required when deserializing a context manifest")
        payload = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        if "scope" in payload:
            payload["scope"] = DaystromScope.from_dict(payload["scope"])
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        if self.content_digest != self.compute_content_digest():
            raise ContractError("content_digest no longer matches manifest content")
        return super().to_dict()

    def compute_content_digest(self) -> str:
        stable = {
            "scope": _serialize(self.scope),
            "model_id": self.model_id,
            "runtime_id": self.runtime_id,
            "segment_ids": list(self.segment_ids),
            "estimated_input_tokens": self.estimated_input_tokens,
            "exact_input_tokens": self.exact_input_tokens,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "parent_manifest_id": self.parent_manifest_id,
            "decisions": _serialize(self.decisions),
            "audit": _serialize(self.audit),
            "manifest_version": self.manifest_version,
        }
        payload = json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class ContextPacket(SerializableDataclass):
    """Context packet handed from DCM toward inference/prompt integration."""

    scope: DaystromScope = field(default_factory=DaystromScope)
    capabilities: RuntimeCapabilities = field(
        default_factory=lambda: RuntimeCapabilities.api_only(model_id="unknown", backend_id="unknown")
    )
    budget: ContextBudget = field(default_factory=ContextBudget)
    segments: List[ContextSegment] = field(default_factory=list)
    manifest: ContextManifest = field(
        default_factory=lambda: ContextManifest(model_id="unknown", runtime_id="unknown")
    )
    rendered_messages: List[Dict[str, Any]] = field(default_factory=list)
    decisions: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    audit: AuditInfo = field(default_factory=AuditInfo)
    packet_version: str = CONTEXT_PACKET_V1
    packet_content_digest: str = ""

    def __post_init__(self) -> None:
        if self.packet_version != CONTEXT_PACKET_V1:
            raise ContractError(f"packet_version must be {CONTEXT_PACKET_V1!r}")
        if isinstance(self.scope, dict):
            self.scope = DaystromScope.from_dict(self.scope)
        if isinstance(self.capabilities, dict):
            self.capabilities = RuntimeCapabilities.from_dict(self.capabilities)
        if isinstance(self.budget, dict):
            self.budget = ContextBudget.from_dict(self.budget)
        self.segments = [
            ContextSegment.from_dict(segment) if isinstance(segment, dict) else segment for segment in self.segments
        ]
        if isinstance(self.manifest, dict):
            self.manifest = ContextManifest.from_dict(self.manifest)
        if isinstance(self.audit, dict):
            self.audit = AuditInfo.from_dict(self.audit)
        for segment in self.segments:
            if not isinstance(segment, ContextSegment):
                raise ContractError("segments must contain ContextSegment values")
        total = sum(segment.effective_tokens for segment in self.segments)
        if self.manifest.scope != self.scope:
            raise ContractError("manifest.scope must match packet scope")
        if self.manifest.model_id != self.capabilities.model_id:
            raise ContractError("manifest.model_id must match packet capabilities")
        if self.manifest.runtime_id != self.capabilities.backend_id:
            raise ContractError("manifest.runtime_id must match packet capabilities")
        if any(segment.scope != self.scope for segment in self.segments):
            raise ContractError("segment scopes must match packet scope")
        if total > self.budget.available_input_tokens:
            raise ContractError("segment token total cannot exceed available input budget")
        if self.budget.admitted_input_tokens != total:
            raise ContractError("budget admitted_input_tokens must match segment token total")
        if self.manifest.segment_ids and self.manifest.segment_ids != [segment.segment_id for segment in self.segments]:
            raise ContractError("manifest.segment_ids must match ordered packet segments")
        if self.manifest.estimated_input_tokens != total:
            raise ContractError("manifest estimated_input_tokens must match segment token total")
        for name in ("rendered_messages", "warnings"):
            if not isinstance(getattr(self, name), list):
                raise ContractError(f"{name} must be a list")
        if not isinstance(self.decisions, dict):
            raise ContractError("decisions must be a dictionary")
        computed_digest = self.compute_content_digest()
        if self.packet_content_digest and self.packet_content_digest != computed_digest:
            raise ContractError("packet_content_digest does not match packet content")
        self.packet_content_digest = computed_digest

    def to_dict(self) -> Dict[str, Any]:
        # Re-run contract invariants so post-construction mutation cannot produce
        # a packet whose segments, rendered messages, budget, and manifest disagree.
        self.__post_init__()
        return super().to_dict()

    def compute_content_digest(self) -> str:
        stable = {
            "scope": _serialize(self.scope),
            "capabilities": _serialize(self.capabilities),
            "budget": _serialize(self.budget),
            "segments": _serialize(self.segments),
            "manifest": _serialize(self.manifest),
            "rendered_messages": _serialize(self.rendered_messages),
            "decisions": _serialize(self.decisions),
            "warnings": _serialize(self.warnings),
            "audit": _serialize(self.audit),
            "packet_version": self.packet_version,
        }
        payload = json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]):
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ContractError(f"{cls.__name__}.from_dict expected dict, got {type(data).__name__}")
        if not data.get("packet_content_digest"):
            raise ContractError("packet_content_digest is required when deserializing a context packet")
        payload = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        return cls(**payload)
