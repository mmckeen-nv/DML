"""Provider-neutral contracts for bounded semantic context-page catalogs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol

from daystrom_dml.api_contracts import ContractError, DaystromScope, SerializableDataclass
from daystrom_dml.context.schema import ContextSegment


PAGE_CATALOG_V1 = "daystrom-page-catalog-v1"


@dataclass
class PageCatalogQuery(SerializableDataclass):
    """Read-only scoped lookup request for durable context-page candidates."""

    scope: DaystromScope = field(default_factory=DaystromScope)
    query: str = ""
    exact_handles: List[str] = field(default_factory=list)
    causal_anchor_ids: List[str] = field(default_factory=list)
    max_candidates: int = 8
    max_payload_bytes: int = 262_144
    max_payload_tokens: int = 4_096
    schema_version: str = PAGE_CATALOG_V1

    def __post_init__(self) -> None:
        if self.schema_version != PAGE_CATALOG_V1:
            raise ContractError("page catalog schema_version is not supported")
        if isinstance(self.scope, dict):
            self.scope = DaystromScope.from_dict(self.scope)
        if not isinstance(self.scope, DaystromScope):
            raise ContractError("scope must be a DaystromScope")
        if not isinstance(self.query, str):
            raise ContractError("query must be a string")
        if len(self.query) > 32_768:
            raise ContractError("query exceeds 32768 characters")
        _positive_int("max_candidates", self.max_candidates)
        _positive_int("max_payload_bytes", self.max_payload_bytes)
        _positive_int("max_payload_tokens", self.max_payload_tokens)
        self.exact_handles = _bounded_string_list("exact_handles", self.exact_handles, self.max_candidates)
        self.causal_anchor_ids = _bounded_string_list(
            "causal_anchor_ids", self.causal_anchor_ids, max(32, self.max_candidates * 8)
        )
        if len(self.exact_handles) > self.max_candidates:
            raise ContractError("exact_handles exceed max_candidates")
        if not self.query.strip() and not self.exact_handles:
            raise ContractError("page catalog query or exact_handles must be provided")

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None):
        payload = dict(data or {})
        if "scope" in payload:
            payload["scope"] = DaystromScope.from_dict(payload["scope"])
        return cls(**{key: value for key, value in payload.items() if key in cls.__dataclass_fields__})


@dataclass
class PageCatalogResult(SerializableDataclass):
    """Integrity-ready context segments selected from one page catalog."""

    scope: DaystromScope
    segments: List[ContextSegment] = field(default_factory=list)
    telemetry: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = PAGE_CATALOG_V1

    def __post_init__(self) -> None:
        if self.schema_version != PAGE_CATALOG_V1:
            raise ContractError("page catalog schema_version is not supported")
        if isinstance(self.scope, dict):
            self.scope = DaystromScope.from_dict(self.scope)
        if not isinstance(self.segments, list):
            raise ContractError("page catalog segments must be a list")
        self.segments = [
            item if isinstance(item, ContextSegment) else ContextSegment.from_dict(item) for item in self.segments
        ]
        if any(segment.scope != self.scope for segment in self.segments):
            raise ContractError("page catalog result contains a scope mismatch")
        if not isinstance(self.telemetry, dict):
            raise ContractError("page catalog telemetry must be a dictionary")
        try:
            self.telemetry = json.loads(json.dumps(self.telemetry, sort_keys=True))
        except (TypeError, ValueError) as exc:
            raise ContractError("page catalog telemetry must be JSON serializable") from exc

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None):
        payload = dict(data or {})
        if "scope" in payload:
            payload["scope"] = DaystromScope.from_dict(payload["scope"])
        if "segments" in payload:
            payload["segments"] = [ContextSegment.from_dict(item) for item in payload["segments"]]
        return cls(**{key: value for key, value in payload.items() if key in cls.__dataclass_fields__})


class SemanticPageCatalog(Protocol):
    def lookup(self, query: PageCatalogQuery) -> PageCatalogResult:
        """Return at most ``query.max_candidates`` strictly scoped pages."""
        ...


def _positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ContractError(f"{name} must be a positive integer")


def _bounded_string_list(name: str, values: List[str], limit: int) -> List[str]:
    if not isinstance(values, list):
        raise ContractError(f"{name} must be a list")
    result: List[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ContractError(f"{name} must contain non-empty strings")
        clean = value.strip()
        if len(clean) > 1024:
            raise ContractError(f"{name} entries exceed 1024 characters")
        if clean not in result:
            result.append(clean)
        if len(result) > max(1, limit):
            raise ContractError(f"{name} exceeds its bound")
    return result
