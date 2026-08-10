"""Experimental GPU-first vLLM 0.20 hybrid KV connector.

This module is **dependency-gated**: it imports vLLM and torch at module load.
Importing the ``daystrom_dml.context.vllm_bridge`` package does NOT import this
module, so the package remains usable on platforms without vLLM.  Tests that
exercise the connector stub the minimal vLLM modules explicitly before
importing this module.

The connector subclasses vLLM 0.20's ``SimpleCPUOffloadConnector`` (which
itself inherits ``KVConnectorBase_V1`` and ``SupportsHMA``) and gates the
scheduler-side save/restore/purge methods by request-level ``kv_transfer_params``
and checkpoint identity via :class:`DaystromKVPolicy`.  Unapproved requests
neither store nor load KV.  Actual GPU<->CPU transfer is delegated to the
parent ``SimpleCPUOffloadConnector``; selective purge adds a version-pinned
scheduler-to-worker command that zeroes protected CPU rows after DMA flush.

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
* Selective ``purge`` removes one checkpoint's unshared resident rows: the
  scheduler pins and evicts them logically, every worker flushes DMA and zeros
  its CPU tensors, and the scheduler frees rows only after all-worker evidence.
  Shared, missing, busy, or incomplete cases are retained or rejected explicitly.
* ``reset_cache`` still fails closed with ``NotImplementedError`` because an
  all-cache physical deletion contract is not implemented.
* Telemetry exposes only payload-free fields: checkpoint digest, operation,
  matched/saved token counts, and physical purge counters.  No prompt text,
  token ids, block-hash bytes, or secrets.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch  # noqa: F401  (required by vLLM connector base at import time)

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)

from daystrom_dml.context.vllm_bridge.policy import (
    DaystromKVAuthorizationError,
    DaystromKVDecision,
    DaystromKVPolicy,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = logging.getLogger("daystrom_dml.context.vllm_bridge.connector")

__all__ = ["DaystromCooperativeKVConnector"]


@dataclass
class DaystromPurgeMetadata(SimpleCPUOffloadMetadata):
    """Scheduler-to-worker command for one protected CPU KV zeroization."""

    purge_event: int = -1
    purge_cpu_blocks: list[int] = field(default_factory=list)


@dataclass
class DaystromPurgeWorkerMetadata(SimpleCPUOffloadWorkerMetadata):
    """Worker acknowledgements for physical purge events.

    Values are ``(worker_count, block_rows_zeroed, bytes_zeroed)`` and are
    summed across all runtime workers before the scheduler commits deletion.
    """

    completed_purge_events: dict[int, tuple[int, int, int]] = field(
        default_factory=dict
    )

    def aggregate(
        self, other: KVConnectorWorkerMetadata
    ) -> KVConnectorWorkerMetadata:
        if not isinstance(other, DaystromPurgeWorkerMetadata):
            return super().aggregate(other)
        completed_stores = dict(self.completed_store_events)
        for event, count in other.completed_store_events.items():
            completed_stores[event] = completed_stores.get(event, 0) + count
        completed_purges = dict(self.completed_purge_events)
        for event, values in other.completed_purge_events.items():
            previous = completed_purges.get(event, (0, 0, 0))
            completed_purges[event] = (
                previous[0] + values[0],
                previous[1] + values[1],
                previous[2] + values[2],
            )
        return DaystromPurgeWorkerMetadata(
            completed_store_events=completed_stores,
            completed_purge_events=completed_purges,
        )


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
        if not bool(vllm_config.cache_config.enable_prefix_caching):
            raise ValueError(
                "Daystrom GPU-first hybrid mode requires vLLM prefix caching"
            )
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
        self._save_request_to_checkpoint: dict[str, str] = {}
        self._checkpoint_stored_hashes: dict[str, set[bytes]] = {}
        self._checkpoint_store_completed: set[str] = set()
        self._purge_event_counter = 0
        self._purge_event_to_checkpoint: dict[int, str] = {}
        self._purge_event_to_request: dict[int, str] = {}
        self._purge_event_to_blocks: dict[int, list[Any]] = {}
        self._purge_event_pending_counts: dict[int, tuple[int, int, int]] = {}
        self._purge_unsent_events: list[int] = []
        self._purge_schedule_errors: dict[str, str] = {}
        self._purge_completion_errors: dict[str, str] = {}
        self._expected_worker_count = int(vllm_config.parallel_config.world_size)
        self._bound_purge_event = -1
        self._bound_purge_blocks: list[int] = []
        self._completed_purge_events: dict[int, tuple[int, int, int]] = {}

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
            for key in (
                "gpu_apc_matched_tokens",
                "cpu_offload_matched_tokens",
                "cache_route",
            ):
                if key in existing:
                    tele[key] = existing[key]
        self._decision_telemetry[request.request_id] = tele
        return decision

    def _record_purge_error(self, request: "Request", reason_code: str) -> None:
        self._purge_schedule_errors[request.request_id] = reason_code
        telemetry = self._decision_telemetry.get(request.request_id, {})
        telemetry.update(
            {
                "authorized": False,
                "operation": "purge",
                "reason_code": reason_code,
                "purged_blocks": 0,
                "purged_bytes": 0,
                "shared_blocks": 0,
            }
        )
        self._decision_telemetry[request.request_id] = telemetry

    def _capture_resident_checkpoint_inventory(self, checkpoint: str) -> None:
        """Bind already-resident CPU rows reused by an authorized save.

        The parent eager store skips allocation when an exact row is already in
        the CPU cache. Those rows produce no new store event, but they are still
        confirmed physical state owned by the new checkpoint and must remain in
        its purge inventory.
        """

        manager = self.scheduler_manager
        record = self._policy.record_for(checkpoint)
        if manager is None or record is None or getattr(manager, "_lazy_mode", False):
            return
        try:
            from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

            groups = manager.cpu_kv_cache_config.kv_cache_groups
            hash_index = manager.cpu_block_pool.cached_block_hash_to_block
            inventory = self._checkpoint_stored_hashes.setdefault(checkpoint, set())
            for block_hash in record.block_hashes:
                for group_id in range(len(groups)):
                    exact_hash = make_block_hash_with_group_id(block_hash, group_id)
                    block = hash_index.get_one_block(exact_hash)
                    if block is not None and block.block_hash is not None:
                        inventory.add(bytes(block.block_hash))
            expected = len(record.block_hashes) * len(groups)
            if expected > 0 and len(inventory) == expected:
                self._checkpoint_store_completed.add(checkpoint)
        except Exception:
            logger.exception(
                "Failed to capture resident CPU KV inventory for checkpoint %s",
                checkpoint,
            )

    def _checkpoint_inventory_counts(self, checkpoint: str) -> tuple[int, int]:
        """Return currently resident and expected CPU rows for a checkpoint.

        Inventory entries are revalidated against the live CPU block hash map so
        status cannot report stale readiness after ordinary capacity eviction.
        """

        manager = self.scheduler_manager
        record = self._policy.record_for(checkpoint)
        if manager is None or record is None:
            return 0, 0
        try:
            groups = manager.cpu_kv_cache_config.kv_cache_groups
            expected = len(record.block_hashes) * len(groups)
            inventory = self._checkpoint_stored_hashes.get(checkpoint, set())
            hash_index = manager.cpu_block_pool.cached_block_hash_to_block
            resident = {
                exact_hash
                for exact_hash in inventory
                if hash_index.get_one_block(exact_hash) is not None
            }
            if checkpoint in self._checkpoint_stored_hashes:
                self._checkpoint_stored_hashes[checkpoint] = resident
            return len(resident), expected
        except Exception:
            logger.exception(
                "Failed to inspect CPU KV readiness for checkpoint %s", checkpoint
            )
            return 0, 0

    def _checkpoint_status_telemetry(
        self, decision: DaystromKVDecision
    ) -> dict[str, Any]:
        """Add physical readiness to a valid signed status decision."""

        telemetry = decision.telemetry()
        telemetry.update(
            {
                "checkpoint_ready": False,
                "stored_blocks": 0,
                "expected_blocks": 0,
            }
        )
        completion_error = self._purge_completion_errors.get(
            decision.checkpoint_digest
        )
        if completion_error is not None:
            telemetry["reason_code"] = completion_error
            return telemetry
        if decision.reason_code in {
            "record_not_found",
            "record_expired",
            "purge_pending",
            "purge_complete",
        }:
            return telemetry
        stored, expected = self._checkpoint_inventory_counts(
            decision.checkpoint_digest
        )
        completed = decision.checkpoint_digest in self._checkpoint_store_completed
        if expected == 0:
            reason = "checkpoint_below_granularity"
        elif not completed:
            reason = "checkpoint_pending"
        elif stored == expected:
            reason = "checkpoint_ready"
        elif stored == 0:
            reason = "checkpoint_evicted"
        else:
            reason = "checkpoint_partial"
        telemetry.update(
            {
                "reason_code": reason,
                "checkpoint_ready": reason == "checkpoint_ready",
                "stored_blocks": stored,
                "expected_blocks": expected,
            }
        )
        return telemetry

    def _schedule_purge(self, request: "Request", decision: DaystromKVDecision) -> None:
        """Protect, logically evict, and queue one selective CPU KV purge."""

        if request.request_id in self._purge_schedule_errors:
            return
        if request.request_id in self._purge_event_to_request.values():
            return
        manager = self.scheduler_manager
        if manager is None:
            self._record_purge_error(request, "purge_scheduler_unavailable")
            return

        try:
            from vllm.v1.core.kv_cache_utils import get_block_hash

            if manager.has_pending_stores():
                self._record_purge_error(request, "purge_transfers_busy")
                return
            if getattr(manager, "_lazy_mode", False):
                self._record_purge_error(request, "purge_lazy_mode_unsupported")
                return
            self._capture_resident_checkpoint_inventory(decision.checkpoint_digest)
            unique_hashes, shared_hashes = self._policy.partition_purge_hashes(
                decision.checkpoint_digest
            )
            inventory = self._checkpoint_stored_hashes.get(
                decision.checkpoint_digest
            )
            if inventory is None:
                self._record_purge_error(request, "purge_inventory_missing")
                return
            record = self._policy.record_for(decision.checkpoint_digest)
            if not inventory and record is not None and record.block_hashes:
                self._record_purge_error(request, "purge_inventory_empty")
                return
            unique_set = set(unique_hashes)
            shared_set = set(shared_hashes)
            target_hashes = {
                exact_hash
                for exact_hash in inventory
                if bytes(get_block_hash(exact_hash)) in unique_set
            }
            shared_exact_hashes = {
                exact_hash
                for exact_hash in inventory
                if bytes(get_block_hash(exact_hash)) in shared_set
            }
            pool = manager.cpu_block_pool
            blocks_by_id: dict[int, Any] = {}
            for exact_hash in target_hashes:
                block = pool.cached_block_hash_to_block.get_one_block(exact_hash)
                if block is not None:
                    blocks_by_id[block.block_id] = block
            if len(blocks_by_id) != len(target_hashes):
                self._record_purge_error(request, "purge_blocks_missing")
                return
            blocks = [blocks_by_id[block_id] for block_id in sorted(blocks_by_id)]
            if any(block.ref_cnt != 0 for block in blocks):
                self._record_purge_error(request, "purge_blocks_busy")
                return

            event = self._purge_event_counter
            self._purge_event_counter += 1
            if blocks:
                # Pin the rows out of the free queue before removing their hash
                # keys so no concurrent store can reuse a row before zeroization.
                pool.touch(blocks)
                pool.evict_blocks(set(blocks_by_id))
            self._policy.begin_purge(
                decision.checkpoint_digest,
                purge_event=event,
                blocks_scheduled=len(blocks),
                shared_blocks=len(shared_exact_hashes),
                shared_hashes=shared_hashes,
            )
            self._purge_event_to_checkpoint[event] = decision.checkpoint_digest
            self._purge_event_to_request[event] = request.request_id
            self._purge_event_to_blocks[event] = blocks
            if not blocks:
                completed = self._policy.complete_purge(
                    event, blocks_zeroed=0, bytes_zeroed=0
                )
                self._decision_telemetry[request.request_id] = completed.telemetry()
                self._checkpoint_stored_hashes.pop(
                    decision.checkpoint_digest, None
                )
                self._checkpoint_store_completed.discard(
                    decision.checkpoint_digest
                )
                self._purge_event_to_checkpoint.pop(event, None)
                self._purge_event_to_request.pop(event, None)
                self._purge_event_to_blocks.pop(event, None)
            else:
                self._purge_unsent_events.append(event)
        except DaystromKVAuthorizationError as exc:
            self._record_purge_error(request, exc.reason_code)
        except Exception:
            logger.exception(
                "Failed to schedule selective CPU KV purge for checkpoint %s",
                decision.checkpoint_digest,
            )
            self._record_purge_error(request, "purge_schedule_failed")

    def _capture_completed_store_inventory(
        self, connector_output: KVConnectorOutput
    ) -> set[str]:
        """Bind confirmed CPU rows to checkpoint identities before parent cleanup."""

        manager = self.scheduler_manager
        worker_meta = getattr(connector_output, "kv_connector_worker_meta", None)
        if manager is None or not isinstance(
            worker_meta, SimpleCPUOffloadWorkerMetadata
        ):
            return set()
        try:
            from vllm.v1.core.kv_cache_utils import get_block_hash

            meta = cast(SimpleCPUOffloadWorkerMetadata, worker_meta)
            completed_reqs: set[str] = set()
            for event, count in meta.completed_store_events.items():
                total = manager._store_event_pending_counts.get(event, 0) + count
                if total < manager._expected_worker_count:
                    continue
                transfer = manager._store_event_to_blocks.get(event)
                request_ids = manager._store_event_to_reqs.get(event, [])
                if transfer is None:
                    continue
                exact_hashes = []
                for block_id in transfer.cpu_block_ids:
                    block_hash = manager.cpu_block_pool.blocks[block_id].block_hash
                    if block_hash is not None:
                        exact_hashes.append(bytes(block_hash))
                for request_id in request_ids:
                    checkpoint = self._save_request_to_checkpoint.get(request_id)
                    if checkpoint is None:
                        continue
                    record = self._policy.record_for(checkpoint)
                    if record is None:
                        continue
                    record_hashes = set(record.block_hashes)
                    inventory = self._checkpoint_stored_hashes.setdefault(
                        checkpoint, set()
                    )
                    inventory.update(
                        exact_hash
                        for exact_hash in exact_hashes
                        if bytes(get_block_hash(exact_hash)) in record_hashes
                    )
                    # Worker completion is distinct from full residency: a
                    # completed save may still be partial after capacity loss.
                    self._checkpoint_store_completed.add(checkpoint)
                    completed_reqs.add(request_id)
            return completed_reqs
        except Exception:
            logger.exception("Failed to capture completed CPU KV store inventory")
            return set()

    # -- Worker-side selective purge ---------------------------------------- #

    def bind_connector_metadata(
        self, connector_metadata: KVConnectorMetadata
    ) -> None:
        super().bind_connector_metadata(connector_metadata)
        self._bound_purge_event = int(
            getattr(connector_metadata, "purge_event", -1)
        )
        self._bound_purge_blocks = list(
            getattr(connector_metadata, "purge_cpu_blocks", [])
        )

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()
        self._bound_purge_event = -1
        self._bound_purge_blocks = []

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        finished_sending, finished_recving = super().get_finished(finished_req_ids)
        event = self._bound_purge_event
        block_ids = self._bound_purge_blocks
        if event < 0 or not block_ids or self.worker_handler is None:
            return finished_sending, finished_recving
        try:
            # Wait for every pending GPU<->CPU DMA before touching CPU rows.
            # The scheduler pinned these rows out of the free allocator first.
            self.worker_handler._flush_and_sync_all()
            cpu_caches = self.worker_handler.cpu_kv_caches
            if not cpu_caches:
                raise RuntimeError("worker CPU KV caches are not registered")
            if min(block_ids) < 0 or max(block_ids) >= self.worker_handler.num_cpu_blocks:
                raise RuntimeError("purge CPU block id out of range")
            indices = torch.tensor(block_ids, dtype=torch.long, device="cpu")
            bytes_zeroed = 0
            for tensor in cpu_caches.values():
                bytes_per_row = tensor[0].numel() * tensor.element_size()
                tensor.index_fill_(0, indices, 0)
                bytes_zeroed += bytes_per_row * len(block_ids)
            self._completed_purge_events[event] = (
                1,
                len(block_ids),
                bytes_zeroed,
            )
        except Exception:
            # No acknowledgement means scheduler rows remain protected and the
            # authorization record stays purge-pending rather than failing open.
            logger.exception("Worker failed to zero CPU KV rows for purge %d", event)
        return finished_sending, finished_recving

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        parent_meta = super().build_connector_worker_meta()
        completed_stores = (
            dict(parent_meta.completed_store_events)
            if isinstance(parent_meta, SimpleCPUOffloadWorkerMetadata)
            else {}
        )
        if not completed_stores and not self._completed_purge_events:
            return None
        metadata = DaystromPurgeWorkerMetadata(
            completed_store_events=completed_stores,
            completed_purge_events=self._completed_purge_events,
        )
        self._completed_purge_events = {}
        return metadata

    # -- Scheduler-side gated methods --------------------------------------- #

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        decision = self._evaluate(request)
        if not decision.authorized or decision.operation not in {"restore", "transition"}:
            return 0, False
        # vLLM 0.20 computes local GPU prefix-cache matches before invoking a
        # connector and passes that native count as num_computed_tokens. Keep
        # the managed CPU tier second and report both sources independently.
        # GPU APC remains opportunistic and is not an authorization boundary.
        telemetry = self._decision_telemetry[request.request_id]
        gpu_tokens = max(0, int(num_computed_tokens))
        telemetry["gpu_apc_matched_tokens"] = gpu_tokens
        telemetry["cpu_offload_matched_tokens"] = 0
        telemetry["cache_route"] = "gpu_apc" if gpu_tokens else "miss"
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
                telemetry["matched_tokens"] = parent_matched
                telemetry["cpu_offload_matched_tokens"] = parent_matched
                telemetry["cache_route"] = (
                    "gpu_apc_and_cpu" if gpu_tokens else "cpu_fallback"
                )
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
        if decision.operation in {"restore", "transition"}:
            # The parent manager's single call can both load an external prefix
            # and register this request for eager stores. A plain restore avoids
            # store registration when there is no external work; transition must
            # always delegate so its completed child can be published.
            if decision.operation == "transition":
                child = decision.child_checkpoint_digest
                previous_checkpoint = self._save_request_to_checkpoint.get(
                    request.request_id
                )
                if previous_checkpoint != child:
                    self._checkpoint_stored_hashes[child] = set()
                    self._checkpoint_store_completed.discard(child)
                self._save_request_to_checkpoint[request.request_id] = child
            if self.scheduler_manager is not None and (
                num_external_tokens > 0 or decision.operation == "transition"
            ):
                self.scheduler_manager.update_state_after_alloc(
                    request, blocks, num_external_tokens
                )
            if decision.operation == "transition":
                self._capture_resident_checkpoint_inventory(
                    decision.child_checkpoint_digest
                )
        elif decision.operation == "save":
            # Delegate to parent for save registration too.  In eager mode the
            # parent's update_state_after_alloc creates _reqs_to_store entries
            # before the num_external_tokens==0 early return.  Without this
            # delegation, approved save requests would never be registered and
            # _prepare_eager_store_specs would skip them.  num_external_tokens
            # is normally 0 for save (no external load).
            previous_checkpoint = self._save_request_to_checkpoint.get(
                request.request_id
            )
            if previous_checkpoint != decision.checkpoint_digest:
                self._checkpoint_stored_hashes[decision.checkpoint_digest] = set()
                self._checkpoint_store_completed.discard(
                    decision.checkpoint_digest
                )
            self._save_request_to_checkpoint[request.request_id] = (
                decision.checkpoint_digest
            )
            if self.scheduler_manager is not None:
                self.scheduler_manager.update_state_after_alloc(
                    request, blocks, num_external_tokens
                )
            self._capture_resident_checkpoint_inventory(decision.checkpoint_digest)
        elif (
            decision.operation == "purge"
            and decision.reason_code == "purge_authorized"
        ):
            self._schedule_purge(request, decision)
        elif (
            decision.operation == "purge"
            and decision.reason_code == "purge_pending"
            and decision.checkpoint_digest not in self._purge_completion_errors
        ):
            # A prior worker attempt may have failed without acknowledgement.
            # A separately signed status request safely retries the idempotent
            # zero command while the scheduler still protects the same rows.
            for event, checkpoint in self._purge_event_to_checkpoint.items():
                if (
                    checkpoint == decision.checkpoint_digest
                    and event not in self._purge_unsent_events
                    and event not in self._purge_event_pending_counts
                ):
                    self._purge_unsent_events.append(event)
                    break

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        if self.scheduler_manager is not None:
            parent = self.scheduler_manager.build_connector_meta(scheduler_output)
        else:
            parent = SimpleCPUOffloadMetadata()
        if not self._purge_unsent_events:
            return parent
        event = self._purge_unsent_events.pop(0)
        blocks = self._purge_event_to_blocks[event]
        return DaystromPurgeMetadata(
            load_event=getattr(parent, "load_event", -1),
            load_gpu_blocks=list(getattr(parent, "load_gpu_blocks", [])),
            load_cpu_blocks=list(getattr(parent, "load_cpu_blocks", [])),
            load_event_to_reqs=dict(getattr(parent, "load_event_to_reqs", {})),
            store_event=getattr(parent, "store_event", -1),
            store_gpu_blocks=list(getattr(parent, "store_gpu_blocks", [])),
            store_cpu_blocks=list(getattr(parent, "store_cpu_blocks", [])),
            need_flush=bool(getattr(parent, "need_flush", False)),
            purge_event=event,
            purge_cpu_blocks=[block.block_id for block in blocks],
        )

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        completed_save_reqs = self._capture_completed_store_inventory(
            connector_output
        )
        if self.scheduler_manager is not None:
            self.scheduler_manager.update_connector_output(connector_output)
            active_stores = getattr(
                self.scheduler_manager, "_reqs_to_store", {}
            )
            for request_id in completed_save_reqs:
                checkpoint = self._save_request_to_checkpoint.get(request_id)
                if checkpoint is not None:
                    # The parent publishes completed rows into the live hash
                    # index during update_connector_output(). Revalidate only
                    # after that publication; checking before the parent call
                    # incorrectly prunes newly completed inventory to empty.
                    self._capture_resident_checkpoint_inventory(checkpoint)
                    self._checkpoint_inventory_counts(checkpoint)
                if request_id not in active_stores:
                    self._save_request_to_checkpoint.pop(request_id, None)
        worker_meta = getattr(connector_output, "kv_connector_worker_meta", None)
        if not isinstance(worker_meta, DaystromPurgeWorkerMetadata):
            return
        for event, current in list(worker_meta.completed_purge_events.items()):
            blocks = self._purge_event_to_blocks.get(event)
            if blocks is None:
                continue
            previous = self._purge_event_pending_counts.get(event, (0, 0, 0))
            worker_count = previous[0] + current[0]
            rows_zeroed = previous[1] + current[1]
            bytes_zeroed = previous[2] + current[2]
            if worker_count < self._expected_worker_count:
                self._purge_event_pending_counts[event] = (
                    worker_count,
                    rows_zeroed,
                    bytes_zeroed,
                )
                continue
            self._purge_event_pending_counts.pop(event, None)
            expected_rows = len(blocks) * self._expected_worker_count
            if rows_zeroed != expected_rows:
                logger.error(
                    "Purge %d worker row evidence mismatch: got=%d expected=%d",
                    event,
                    rows_zeroed,
                    expected_rows,
                )
                continue
            checkpoint = self._purge_event_to_checkpoint.get(event)
            if checkpoint is None:
                logger.error("Purge %d checkpoint identity is missing", event)
                continue
            if not self._policy.purge_shared_owners_valid(checkpoint):
                reason = "purge_shared_ownership_changed"
                logger.error(
                    "Purge %d cannot commit because shared ownership changed",
                    event,
                )
                self._purge_completion_errors[checkpoint] = reason
                orphaned_request_id = self._purge_event_to_request.get(event)
                if orphaned_request_id is not None:
                    telemetry = self._decision_telemetry.get(orphaned_request_id, {})
                    telemetry.update(
                        {
                            "authorized": False,
                            "operation": "purge",
                            "checkpoint_digest": checkpoint,
                            "reason_code": reason,
                            "purged_blocks": 0,
                            "purged_bytes": 0,
                        }
                    )
                    self._decision_telemetry[orphaned_request_id] = telemetry
                # Keep the purge nonterminal and its rows protected. A runtime
                # restart or future explicit remediation must resolve the now-
                # orphaned shared rows; never claim or retry physical cleanup.
                continue
            completed = self._policy.complete_purge(
                event,
                blocks_zeroed=len(blocks),
                bytes_zeroed=bytes_zeroed,
            )
            # Worker zeroization is complete on every rank. Only now may these
            # protected CPU rows return to the allocator's free queue.
            if self.scheduler_manager is not None and blocks:
                self.scheduler_manager.cpu_block_pool.free_blocks(blocks)
            completed_request_id = (
                self._purge_event_to_request.pop(event)
                if event in self._purge_event_to_request
                else None
            )
            if completed_request_id is not None:
                self._decision_telemetry[completed_request_id] = completed.telemetry()
            completed_checkpoint = (
                self._purge_event_to_checkpoint.pop(event)
                if event in self._purge_event_to_checkpoint
                else None
            )
            if completed_checkpoint is not None:
                self._checkpoint_stored_hashes.pop(completed_checkpoint, None)
                self._checkpoint_store_completed.discard(completed_checkpoint)
            self._purge_event_to_blocks.pop(event, None)

    def _status_response(
        self, request_id: str, decision: DaystromKVDecision
    ) -> tuple[bool, dict[str, Any] | None]:
        """Return signed payload-free lifecycle state without transfer side effects."""

        if decision.reason_code in {"record_not_found", "record_expired"}:
            self._checkpoint_stored_hashes.pop(decision.checkpoint_digest, None)
            self._checkpoint_store_completed.discard(decision.checkpoint_digest)
        telemetry = self._checkpoint_status_telemetry(decision)
        self._decision_telemetry.pop(request_id, None)
        return False, self._merge_daystrom_meta(
            None, self._build_daystrom_response_meta(telemetry)
        )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        decision = self._evaluate(
            request, tokens=getattr(request, "num_computed_tokens", 0)
        )
        if decision.operation == "status":
            return self._status_response(request.request_id, decision)
        if decision.operation == "purge":
            telemetry = self._decision_telemetry.get(
                request.request_id, decision.telemetry()
            )
            error = self._purge_schedule_errors.pop(request.request_id, None)
            if error is None:
                error = self._purge_completion_errors.get(
                    decision.checkpoint_digest
                )
            if error is not None:
                telemetry = dict(telemetry)
                telemetry.update({"authorized": False, "reason_code": error})
            return False, self._merge_daystrom_meta(
                None, self._build_daystrom_response_meta(telemetry)
            )
        if not decision.authorized:
            # Preserve payload-free lifecycle evidence for a correctly signed
            # restore denied because this checkpoint is being, or has been,
            # physically purged. Other failures stay silent so this does not
            # become a checkpoint-existence oracle.
            if decision.operation == "restore" and decision.reason_code in {
                "purge_pending",
                "purge_complete",
            }:
                return False, self._merge_daystrom_meta(
                    None, self._build_daystrom_response_meta(decision)
                )
            return False, None
        if decision.operation not in {"save", "restore", "transition"}:
            return False, None
        # Both paths must delegate: save finalizes store tracking, while restore
        # releases CPU/GPU touch refs and load-event state in the parent manager.
        parent_retain, parent_meta = False, None
        if self.scheduler_manager is not None:
            parent_retain, parent_meta = self.scheduler_manager.request_finished(
                request, block_ids
            )
            if (
                decision.operation in {"save", "transition"}
                and request.request_id
                not in getattr(self.scheduler_manager, "_reqs_to_store", {})
            ):
                self._save_request_to_checkpoint.pop(request.request_id, None)
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
        if decision.operation == "status":
            return self._status_response(request.request_id, decision)
        if decision.operation == "purge":
            telemetry = self._decision_telemetry.get(
                request.request_id, decision.telemetry()
            )
            error = self._purge_schedule_errors.pop(request.request_id, None)
            if error is None:
                error = self._purge_completion_errors.get(
                    decision.checkpoint_digest
                )
            if error is not None:
                telemetry = dict(telemetry)
                telemetry.update({"authorized": False, "reason_code": error})
            return False, self._merge_daystrom_meta(
                None, self._build_daystrom_response_meta(telemetry)
            )
        if not decision.authorized:
            # Preserve payload-free lifecycle evidence for a correctly signed
            # restore denied because this checkpoint is being, or has been,
            # physically purged. Other failures stay silent so this does not
            # become a checkpoint-existence oracle.
            if decision.operation == "restore" and decision.reason_code in {
                "purge_pending",
                "purge_complete",
            }:
                return False, self._merge_daystrom_meta(
                    None, self._build_daystrom_response_meta(decision)
                )
            return False, None
        if decision.operation not in {"save", "restore", "transition"}:
            return False, None
        parent_retain, parent_meta = False, None
        if self.scheduler_manager is not None:
            parent_retain, parent_meta = (
                self.scheduler_manager.request_finished_all_groups(request, block_ids)
            )
            if (
                decision.operation in {"save", "transition"}
                and request.request_id
                not in getattr(self.scheduler_manager, "_reqs_to_store", {})
            ):
                self._save_request_to_checkpoint.pop(request.request_id, None)
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
        result = {
            "schema_version": values.get("schema_version", ""),
            "operation": values.get("operation", "unknown"),
            "checkpoint_digest": values.get("checkpoint_digest", ""),
            "reason_code": values.get("reason_code", "unknown"),
            "matched_tokens": values.get("matched_tokens", 0),
            "gpu_apc_matched_tokens": values.get("gpu_apc_matched_tokens", 0),
            "cpu_offload_matched_tokens": values.get(
                "cpu_offload_matched_tokens", 0
            ),
            "cache_route": values.get(
                "cache_route",
                "miss" if values.get("operation") in {"restore", "transition"} else "not_applicable",
            ),
            "saved_tokens": values.get("saved_tokens", 0),
            "purged_blocks": values.get("purged_blocks", 0),
            "purged_bytes": values.get("purged_bytes", 0),
            "shared_blocks": values.get("shared_blocks", 0),
        }
        child_checkpoint = values.get("child_checkpoint_digest")
        if child_checkpoint:
            result["child_checkpoint_digest"] = child_checkpoint
        for key in ("checkpoint_ready", "stored_blocks", "expected_blocks"):
            if key in values:
                result[key] = values[key]
        return result

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
        pending_checkpoints = set(self._purge_event_to_checkpoint.values())
        self._checkpoint_store_completed.intersection_update(pending_checkpoints)
        self._checkpoint_stored_hashes = {
            checkpoint: hashes
            for checkpoint, hashes in self._checkpoint_stored_hashes.items()
            if checkpoint in pending_checkpoints
        }
        self._save_request_to_checkpoint.clear()
        raise NotImplementedError(
            "DaystromCooperativeKVConnector does not support reset_cache(). "
            "All-cache physical purge of offloaded CPU KV bytes is not safely "
            "implemented; use signed request-bound selective purge."
        )

    # -- telemetry ---------------------------------------------------------- #

    def take_telemetry(self) -> dict[str, dict[str, Any]]:
        """Return and clear per-request payload-free decision telemetry."""

        snapshot = self._decision_telemetry
        self._decision_telemetry = {}
        return snapshot
