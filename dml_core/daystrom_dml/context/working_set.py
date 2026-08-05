"""Deterministic provider-neutral working-set replacement for virtual context."""
from __future__ import annotations

import json
import time
from itertools import islice
from typing import Any, Dict, Iterable, Optional

from daystrom_dml.api_contracts import AuditInfo, ContractError, DaystromScope
from daystrom_dml.context.adapters.api_messages import APIMessageAdapter
from daystrom_dml.context.adapters.base import RuntimeContextAdapter
from daystrom_dml.context.admission import PageOutCallback, admit_context_segments
from daystrom_dml.context.budget import ContextBudget
from daystrom_dml.context.capabilities import RuntimeCapabilities
from daystrom_dml.context.manifest import ContextManifest, ContextPacket
from daystrom_dml.context.schema import ContextSegment, context_segment_digest

WORKING_SET_TRANSITION_V1 = "daystrom-working-set-transition-v1"


class WorkingSetManager:
    """Reconcile bounded candidates into an integrity-bound resident set.

    Semantic selection remains provider neutral. Runtime-specific prefix/KV
    reuse may consume the resulting manifest lineage as an optional accelerator.
    """

    def __init__(
        self,
        runtime_adapter: Optional[RuntimeContextAdapter] = None,
        *,
        max_candidates: int = 4096,
        clock: Any = None,
    ) -> None:
        if not isinstance(max_candidates, int) or isinstance(max_candidates, bool) or max_candidates < 1:
            raise ContractError("max_candidates must be a positive integer")
        self.runtime_adapter = runtime_adapter or APIMessageAdapter()
        self.max_candidates = max_candidates
        self.clock = clock or time.time

    def reconcile(
        self,
        *,
        scope: DaystromScope | Dict[str, Any] | None = None,
        segments: Iterable[ContextSegment | Dict[str, Any]],
        model_id: str,
        runtime_id: str,
        model_limit_tokens: int,
        output_reserved_tokens: int = 0,
        runtime_reserved_tokens: int = 0,
        endpoint_url: Optional[str] = None,
        parent_manifest: ContextManifest | Dict[str, Any] | None = None,
        page_out: Optional[PageOutCallback] = None,
    ) -> ContextPacket:
        scope_obj = scope if isinstance(scope, DaystromScope) else DaystromScope.from_dict(scope)
        if not isinstance(model_id, str) or not model_id:
            raise ContractError("model_id must be non-empty")
        if not isinstance(runtime_id, str) or not runtime_id:
            raise ContractError("runtime_id must be non-empty")
        parent = self._copy_parent(parent_manifest)
        self._validate_parent(parent, scope=scope_obj, model_id=model_id, runtime_id=runtime_id)
        candidates = list(islice(iter(segments), self.max_candidates + 1))
        if len(candidates) > self.max_candidates:
            raise ContractError("working-set candidates exceed max_candidates")

        admitted = admit_context_segments(
            scope=scope_obj,
            segments=candidates,
            model_id=model_id,
            runtime_id=runtime_id,
            endpoint_url=endpoint_url,
            model_limit_tokens=model_limit_tokens,
            output_reserved_tokens=output_reserved_tokens,
            runtime_reserved_tokens=runtime_reserved_tokens,
            runtime_adapter=self.runtime_adapter,
            page_out=page_out,
        )
        current_digests = {segment.segment_id: context_segment_digest(segment) for segment in admitted.segments}
        previous_digests = dict(parent.segment_digests) if parent is not None else {}
        previous_ids = list(parent.segment_ids) if parent is not None else []
        current_ids = [segment.segment_id for segment in admitted.segments]
        stable_prefix = _stable_prefix(previous_ids, previous_digests, current_ids, current_digests)
        transition = {
            "version": WORKING_SET_TRANSITION_V1,
            "parent_manifest_id": parent.content_digest if parent is not None else None,
            "retained": [item for item in current_ids if previous_digests.get(item) == current_digests[item]],
            "replaced": [
                item for item in current_ids if item in previous_digests and previous_digests[item] != current_digests[item]
            ],
            "added": [item for item in current_ids if item not in previous_ids],
            "evicted": [item for item in previous_ids if item not in current_ids],
            "omitted": list(admitted.decisions.get("omitted", [])),
            "stable_prefix": stable_prefix,
            "prefill_from_index": len(stable_prefix),
            "prefill_segment_ids": current_ids[len(stable_prefix) :],
        }
        decisions = _json_copy(admitted.decisions)
        decisions["working_set"] = transition
        manifest = ContextManifest(
            scope=scope_obj,
            model_id=admitted.capabilities.model_id,
            runtime_id=admitted.capabilities.backend_id,
            segment_ids=current_ids,
            segment_digests=current_digests,
            estimated_input_tokens=admitted.manifest.estimated_input_tokens,
            exact_input_tokens=admitted.manifest.exact_input_tokens,
            parent_manifest_id=parent.content_digest if parent is not None else None,
            decisions=decisions,
            audit={
                "reason": "deterministic working-set reconciliation",
                "reason_codes": ["working_set_reconciled"],
            },
            created_at=float(self.clock()),
        )
        return ContextPacket(
            scope=scope_obj,
            capabilities=RuntimeCapabilities.from_dict(admitted.capabilities.to_dict()),
            budget=ContextBudget.from_dict(admitted.budget.to_dict()),
            segments=[ContextSegment.from_dict(segment.to_dict()) for segment in admitted.segments],
            manifest=manifest,
            rendered_messages=_json_copy(admitted.rendered_messages),
            decisions=decisions,
            warnings=list(admitted.warnings),
            audit=AuditInfo(
                reason="deterministic working-set reconciliation",
                policy="working_set_v1",
                reason_codes=["working_set_reconciled"],
            ),
        )

    @staticmethod
    def _copy_parent(parent: ContextManifest | Dict[str, Any] | None) -> Optional[ContextManifest]:
        if parent is None:
            return None
        if isinstance(parent, ContextManifest):
            return ContextManifest.from_dict(_json_copy(parent.to_dict()))
        if isinstance(parent, dict):
            return ContextManifest.from_dict(_json_copy(parent))
        raise ContractError("parent_manifest must be a ContextManifest or dictionary")

    @staticmethod
    def _validate_parent(
        parent: Optional[ContextManifest],
        *,
        scope: DaystromScope,
        model_id: str,
        runtime_id: str,
    ) -> None:
        if parent is None:
            return
        if parent.scope != scope:
            raise ContractError("parent manifest scope does not match working-set scope")
        if parent.model_id != model_id:
            raise ContractError("parent manifest model does not match working-set model")
        if parent.runtime_id != runtime_id:
            raise ContractError("parent manifest runtime does not match working-set runtime")
        if not parent.segment_digests:
            raise ContractError("parent manifest lacks segment digest bindings")


def _stable_prefix(
    previous_ids: list[str],
    previous_digests: Dict[str, str],
    current_ids: list[str],
    current_digests: Dict[str, str],
) -> list[str]:
    prefix: list[str] = []
    for previous_id, current_id in zip(previous_ids, current_ids):
        if previous_id != current_id or previous_digests.get(previous_id) != current_digests.get(current_id):
            break
        prefix.append(current_id)
    return prefix


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))
