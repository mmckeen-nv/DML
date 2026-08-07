"""Experimental controller-gated vLLM 0.20 SimpleCPUOffload KV connector.

This module is **dependency-gated**: it imports vLLM and torch at module load.
Importing the ``daystrom_dml.context.vllm_bridge`` package does NOT import this
module, so the package remains usable on platforms without vLLM.  Tests that
exercise the connector stub the minimal vLLM modules explicitly before
importing this module.

The connector subclasses vLLM 0.20's ``SimpleCPUOffloadConnector`` (which
itself inherits ``KVConnectorBase_V1`` and ``SupportsHMA``) and gates the
scheduler-side save/restore methods by request-level ``kv_transfer_params``
and checkpoint identity via :class:`DaystromKVPolicy`.  Unapproved requests
neither store nor load KV.  Actual GPU<->CPU transfer is delegated to the
parent ``SimpleCPUOffloadConnector``.

Design notes
------------
* ``checkpoint_digest`` is an independent controller identity digest.  A
  controller cannot know vLLM block hashes in advance, so the digest is NOT
  required to equal ``SHA256(block_hashes)``.  On authorized save the digest is
  bound to the exact ordered ``request.block_hashes`` in the in-memory record.
* ``get_num_new_matched_tokens``: returns ``(0, False)`` unless a restore is
  authorized.  When authorized, delegates to the parent scheduler manager and
  uses the parent's native matched token count directly — does NOT treat the
  count of block hashes as a token count.
* ``update_state_after_alloc``: delegates to the parent scheduler manager for
  authorized **restore** requests (when ``num_external_tokens > 0``) AND for
  authorized **save** requests (with ``num_external_tokens`` normally 0),
  because eager mode creates ``_reqs_to_store`` entries in
  ``update_state_after_alloc``.
* ``request_finished`` / ``request_finished_all_groups``: gate the save path
  and return a payload-free ``kv_transfer_params`` result under a ``daystrom``
  key, merging safely with any parent output.  The result includes schema,
  operation, checkpoint digest, reason/status, and native matched/scheduled
  counts only — no request_id, nonce, signatures, block hashes, prompts, or
  token IDs.
* ``reset_cache``: fails closed by raising ``NotImplementedError`` (same as
  the parent).  We do NOT fake physical deletion.
* Telemetry exposes only payload-free fields: checkpoint digest, operation,
  matched/saved token counts.  No prompt text, token ids, block-hash bytes, or
  secrets.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch  # noqa: F401  (required by vLLM connector base at import time)

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

from daystrom_dml.context.vllm_bridge.policy import (
    DaystromKVDecision,
    DaystromKVPolicy,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = logging.getLogger("daystrom_dml.context.vllm_bridge.connector")

__all__ = ["DaystromCooperativeKVConnector"]


class DaystromCooperativeKVConnector(SimpleCPUOffloadConnector, SupportsHMA):
    """Controller-gated SimpleCPUOffloadConnector for Daystrom cooperative KV.

    The connector reads the HMAC secret file path from
    ``kv_connector_extra_config["daystrom_secret_path"]`` and optional limits
    ``daystrom_max_ttl_seconds`` / ``daystrom_max_records``.  All save/restore
    decisions are delegated to :class:`DaystromKVPolicy`.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        extra_config = self._kv_transfer_config.kv_connector_extra_config or {}
        secret_path_str = extra_config.get("daystrom_secret_path")
        if not secret_path_str:
            raise ValueError(
                "daystrom_secret_path must be configured in "
                "kv_connector_extra_config"
            )
        max_ttl = int(extra_config.get("daystrom_max_ttl_seconds", 3600))
        max_records = int(extra_config.get("daystrom_max_records", 4096))
        self._policy = DaystromKVPolicy(
            Path(secret_path_str),
            max_ttl_seconds=max_ttl,
            max_records=max_records,
        )
        self._decision_telemetry: dict[str, dict[str, Any]] = {}

    # -- policy access ------------------------------------------------------ #

    @property
    def policy(self) -> DaystromKVPolicy:
        return self._policy

    # -- helpers ------------------------------------------------------------ #

    def _request_block_hashes(self, request: "Request") -> tuple[bytes, ...]:
        """Extract the request's ordered block hashes as raw bytes."""

        return tuple(bytes(bh) for bh in getattr(request, "block_hashes", []) or [])

    def _evaluate(
        self, request: "Request", *, tokens: int = 0
    ) -> DaystromKVDecision:
        kv_transfer_params = getattr(request, "kv_transfer_params", None)
        block_hashes = self._request_block_hashes(request)
        decision = self._policy.evaluate(
            kv_transfer_params, block_hashes, tokens=tokens
        )
        tele = decision.telemetry()
        # Merge with existing telemetry so native matched_tokens learned from
        # the parent scheduler manager in an earlier call are preserved when
        # _evaluate is called again (e.g. update_state_after_alloc after
        # get_num_new_matched_tokens).
        existing = self._decision_telemetry.get(request.request_id)
        if existing is not None:
            if existing.get("matched_tokens", 0) > tele.get("matched_tokens", 0):
                tele["matched_tokens"] = existing["matched_tokens"]
        self._decision_telemetry[request.request_id] = tele
        return decision

    # -- Scheduler-side gated methods --------------------------------------- #

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        decision = self._evaluate(request)
        if not decision.authorized or decision.operation != "restore":
            return 0, False
        # Delegate to the parent scheduler manager for the native matched
        # token count.  Do NOT treat the count of block hashes as a token
        # count — the parent's count is based on actual CPU cache hits.
        if self.scheduler_manager is not None:
            parent_matched, parent_async = (
                self.scheduler_manager.get_num_new_matched_tokens(
                    request, num_computed_tokens
                )
            )
            if parent_matched is not None and parent_matched > 0:
                # Update decision telemetry with the actual native count.
                self._decision_telemetry[request.request_id][
                    "matched_tokens"
                ] = parent_matched
                return parent_matched, parent_async
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        decision = self._evaluate(request)
        if not decision.authorized:
            return
        if decision.operation == "restore":
            # Delegate to parent for the load path when there are external
            # tokens to restore from CPU cache.
            if self.scheduler_manager is not None and num_external_tokens > 0:
                self.scheduler_manager.update_state_after_alloc(
                    request, blocks, num_external_tokens
                )
        elif decision.operation == "save":
            # Delegate to parent for save registration too.  In eager mode the
            # parent's update_state_after_alloc creates _reqs_to_store entries
            # before the num_external_tokens==0 early return.  Without this
            # delegation, approved save requests would never be registered and
            # _prepare_eager_store_specs would skip them.  num_external_tokens
            # is normally 0 for save (no external load).
            if self.scheduler_manager is not None:
                self.scheduler_manager.update_state_after_alloc(
                    request, blocks, num_external_tokens
                )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        # Delegate metadata building to the parent; gating happens in the
        # per-request methods above.
        if self.scheduler_manager is not None:
            return self.scheduler_manager.build_connector_meta(scheduler_output)
        from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata

        return SimpleCPUOffloadMetadata()

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        if self.scheduler_manager is not None:
            self.scheduler_manager.update_connector_output(connector_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        decision = self._evaluate(
            request, tokens=getattr(request, "num_computed_tokens", 0)
        )
        if not decision.authorized or decision.operation not in {"save", "restore"}:
            return False, None
        # Both paths must delegate: save finalizes store tracking, while restore
        # releases CPU/GPU touch refs and load-event state in the parent manager.
        parent_retain, parent_meta = False, None
        if self.scheduler_manager is not None:
            parent_retain, parent_meta = self.scheduler_manager.request_finished(
                request, block_ids
            )
        # Preserve the native restore count captured during prefix matching.
        telemetry = self._decision_telemetry.get(request.request_id, decision.telemetry())
        daystrom_tele = self._build_daystrom_response_meta(telemetry)
        merged = self._merge_daystrom_meta(parent_meta, daystrom_tele)
        return parent_retain, merged

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        decision = self._evaluate(
            request, tokens=getattr(request, "num_computed_tokens", 0)
        )
        if not decision.authorized or decision.operation not in {"save", "restore"}:
            return False, None
        parent_retain, parent_meta = False, None
        if self.scheduler_manager is not None:
            parent_retain, parent_meta = (
                self.scheduler_manager.request_finished_all_groups(request, block_ids)
            )
        telemetry = self._decision_telemetry.get(request.request_id, decision.telemetry())
        daystrom_tele = self._build_daystrom_response_meta(telemetry)
        merged = self._merge_daystrom_meta(parent_meta, daystrom_tele)
        return parent_retain, merged

    # -- response telemetry helpers ------------------------------------------ #

    @staticmethod
    def _build_daystrom_response_meta(
        telemetry: DaystromKVDecision | dict[str, Any],
    ) -> dict[str, Any]:
        """Build a payload-free daystrom telemetry dict for the response.

        Includes only: schema, operation, checkpoint digest, reason/status,
        and native matched/scheduled counts.  Does NOT include request_id,
        nonce, signatures, block hashes, prompts, or token IDs.
        """

        values = telemetry.telemetry() if isinstance(telemetry, DaystromKVDecision) else telemetry
        return {
            "schema_version": values.get("schema_version", ""),
            "operation": values.get("operation", "unknown"),
            "checkpoint_digest": values.get("checkpoint_digest", ""),
            "reason_code": values.get("reason_code", "unknown"),
            "matched_tokens": values.get("matched_tokens", 0),
            "saved_tokens": values.get("saved_tokens", 0),
        }

    @staticmethod
    def _merge_daystrom_meta(
        parent_meta: dict[str, Any] | None,
        daystrom_tele: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge the daystrom telemetry under a ``daystrom`` key into the
        parent's connector metadata dict, safely preserving any existing
        parent keys.
        """

        if parent_meta is None:
            return {"daystrom": daystrom_tele}
        merged = dict(parent_meta)
        merged["daystrom"] = daystrom_tele
        return merged

    # -- reset_cache: fail closed, do not fake deletion --------------------- #

    def reset_cache(self) -> bool | None:
        # The parent raises NotImplementedError; we keep that fail-closed
        # behavior and additionally clear our authorization index so that
        # subsequent restores fail closed.  We do NOT claim physical purge.
        self._policy.reset_cache()
        raise NotImplementedError(
            "DaystromCooperativeKVConnector does not support reset_cache(). "
            "Physical purge of offloaded CPU KV bytes is not safely "
            "implemented; clearing the authorization index only."
        )

    # -- telemetry ---------------------------------------------------------- #

    def take_telemetry(self) -> dict[str, dict[str, Any]]:
        """Return and clear per-request payload-free decision telemetry."""

        snapshot = self._decision_telemetry
        self._decision_telemetry = {}
        return snapshot
