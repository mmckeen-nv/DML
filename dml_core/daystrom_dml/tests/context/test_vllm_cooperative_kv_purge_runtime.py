from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from daystrom_dml.context.vllm_bridge.policy import build_kv_transfer_params


@pytest.fixture()
def secret_file(tmp_path: Path) -> Path:
    path = tmp_path / "daystrom.key"
    path.write_text("purge-runtime-test-secret\n")
    return path


@pytest.fixture()
def fixed_time() -> float:
    return 1_000_000.0


_HELPER_PATH = Path(__file__).with_name("test_vllm_cooperative_kv_connector.py")
_HELPER_SPEC = importlib.util.spec_from_file_location("_daystrom_vllm_test_helpers", _HELPER_PATH)
assert _HELPER_SPEC is not None and _HELPER_SPEC.loader is not None
_HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPERS)


def _install_purge_stubs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _HELPERS._install_vllm_stubs(monkeypatch, tmp_path)

    torch_mod = sys.modules["torch"]
    torch_mod.long = "long"  # type: ignore[attr-defined]
    torch_mod.tensor = lambda values, **kwargs: list(values)  # type: ignore[attr-defined]

    base_mod = sys.modules[
        "vllm.distributed.kv_transfer.kv_connector.v1.base"
    ]
    metadata_mod = sys.modules["vllm.v1.simple_kv_offload.metadata"]

    @dataclass
    class SimpleCPUOffloadMetadata(base_mod.KVConnectorMetadata):
        load_event: int = -1
        load_gpu_blocks: list[int] = field(default_factory=list)
        load_cpu_blocks: list[int] = field(default_factory=list)
        load_event_to_reqs: dict[int, list[str]] = field(default_factory=dict)
        store_event: int = -1
        store_gpu_blocks: list[int] = field(default_factory=list)
        store_cpu_blocks: list[int] = field(default_factory=list)
        need_flush: bool = False

    @dataclass
    class SimpleCPUOffloadWorkerMetadata(base_mod.KVConnectorWorkerMetadata):
        completed_store_events: dict[int, int] = field(default_factory=dict)

        def aggregate(self, other):
            merged = dict(self.completed_store_events)
            for event, count in other.completed_store_events.items():
                merged[event] = merged.get(event, 0) + count
            return SimpleCPUOffloadWorkerMetadata(merged)

    metadata_mod.SimpleCPUOffloadMetadata = SimpleCPUOffloadMetadata  # type: ignore[attr-defined]
    metadata_mod.SimpleCPUOffloadWorkerMetadata = SimpleCPUOffloadWorkerMetadata  # type: ignore[attr-defined]

    kv_utils_mod = types.ModuleType("vllm.v1.core.kv_cache_utils")
    kv_utils_mod.make_block_hash_with_group_id = (  # type: ignore[attr-defined]
        lambda block_hash, group_id: block_hash + group_id.to_bytes(4, "big")
    )
    kv_utils_mod.get_block_hash = lambda exact_hash: exact_hash[:-4]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.v1.core.kv_cache_utils", kv_utils_mod)

    class FakeWorkerHandler:
        def __init__(self):
            self.cpu_kv_caches = None
            self.num_cpu_blocks = 0
            self.flushed = False
            self.metadata = None
            self.completed_store_events = {}

        def bind_connector_metadata(self, metadata):
            self.metadata = metadata

        def clear_connector_metadata(self):
            self.metadata = None

        def get_finished(self, finished_req_ids):
            return None, None

        def build_connector_worker_meta(self):
            if not self.completed_store_events:
                return None
            result = SimpleCPUOffloadWorkerMetadata(
                completed_store_events=self.completed_store_events
            )
            self.completed_store_events = {}
            return result

        def _flush_and_sync_all(self):
            self.flushed = True

    parent_mod = sys.modules[
        "vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector"
    ]
    Parent = parent_mod.SimpleCPUOffloadConnector

    def bind_connector_metadata(self, metadata):
        if self.worker_handler is not None:
            self.worker_handler.bind_connector_metadata(metadata)

    def clear_connector_metadata(self):
        if self.worker_handler is not None:
            self.worker_handler.clear_connector_metadata()

    def get_finished(self, finished_req_ids):
        if self.worker_handler is not None:
            return self.worker_handler.get_finished(finished_req_ids)
        return None, None

    def build_connector_worker_meta(self):
        if self.worker_handler is not None:
            return self.worker_handler.build_connector_worker_meta()
        return None

    Parent.bind_connector_metadata = bind_connector_metadata
    Parent.clear_connector_metadata = clear_connector_metadata
    Parent.get_finished = get_finished
    Parent.build_connector_worker_meta = build_connector_worker_meta

    sys.modules.pop("daystrom_dml.context.vllm_bridge.connector", None)
    import daystrom_dml.context.vllm_bridge.connector as connector_mod

    return connector_mod, FakeWorkerHandler


@pytest.fixture()
def purge_env(monkeypatch, tmp_path, secret_file):
    connector_mod, worker_handler_cls = _install_purge_stubs(monkeypatch, tmp_path)
    VllmConfig = sys.modules["vllm.config"].VllmConfig
    KVConnectorRole = sys.modules[
        "vllm.distributed.kv_transfer.kv_connector.v1.base"
    ].KVConnectorRole
    extra = {"daystrom_secret_path": str(secret_file)}
    return connector_mod, worker_handler_cls, VllmConfig, KVConnectorRole, extra


class _Row:
    def __init__(self, width: int):
        self.width = width

    def numel(self) -> int:
        return self.width


class _Tensor:
    def __init__(self, rows: int, width: int, value: int):
        self.data = [[value] * width for _ in range(rows)]
        self.width = width

    def __getitem__(self, index: int) -> _Row:
        return _Row(self.width)

    def element_size(self) -> int:
        return 2

    def index_fill_(self, dim: int, indices: list[int], value: int):
        assert dim == 0
        for index in indices:
            self.data[index] = [value] * self.width
        return self


def test_worker_zeroes_only_commanded_rows_and_reports_bytes(
    purge_env, secret_file
) -> None:
    connector_mod, worker_handler_cls, VllmConfig, KVConnectorRole, extra = purge_env
    worker = connector_mod.DaystromCooperativeKVConnector(
        VllmConfig(extra_config=extra), KVConnectorRole.WORKER
    )
    worker.worker_handler = worker_handler_cls()
    tensor = _Tensor(rows=5, width=4, value=9)
    worker.worker_handler.cpu_kv_caches = {"kv": tensor}
    worker.worker_handler.num_cpu_blocks = 5

    worker.bind_connector_metadata(
        connector_mod.DaystromPurgeMetadata(
            purge_event=3,
            purge_cpu_blocks=[1, 4],
        )
    )
    worker.get_finished(set())

    assert worker.worker_handler.flushed is True
    assert tensor.data[0] == [9, 9, 9, 9]
    assert tensor.data[1] == [0, 0, 0, 0]
    assert tensor.data[3] == [9, 9, 9, 9]
    assert tensor.data[4] == [0, 0, 0, 0]
    ack = worker.build_connector_worker_meta()
    assert ack.completed_purge_events == {3: (1, 2, 16)}


def test_worker_metadata_aggregates_all_rank_evidence(purge_env) -> None:
    connector_mod, *_ = purge_env
    first = connector_mod.DaystromPurgeWorkerMetadata(
        completed_store_events={4: 1},
        completed_purge_events={7: (1, 2, 16)},
    )
    second = connector_mod.DaystromPurgeWorkerMetadata(
        completed_store_events={4: 1},
        completed_purge_events={7: (1, 2, 16)},
    )

    combined = first.aggregate(second)

    assert combined.completed_store_events == {4: 2}
    assert combined.completed_purge_events == {7: (2, 4, 32)}


def test_completed_store_inventory_tracks_only_confirmed_cpu_rows(
    purge_env, secret_file, fixed_time
) -> None:
    connector_mod, _, VllmConfig, KVConnectorRole, extra = purge_env
    connector = connector_mod.DaystromCooperativeKVConnector(
        VllmConfig(extra_config=extra), KVConnectorRole.SCHEDULER
    )
    connector.policy._time_fn = lambda: fixed_time  # type: ignore[attr-defined]
    block_hashes = _HELPERS._block_hashes(3)
    checkpoint = _HELPERS._digest("inventory")
    save_params = build_kv_transfer_params(
        operation="save",
        checkpoint_digest=checkpoint,
        expires_at=fixed_time + 500,
        nonce="inventory-save",
        secret_path=secret_file,
    )
    assert connector.policy.evaluate(save_params, block_hashes).authorized
    connector._save_request_to_checkpoint["save-request"] = checkpoint  # type: ignore[attr-defined]
    connector._checkpoint_stored_hashes[checkpoint] = set()  # type: ignore[attr-defined]

    exact_hash = block_hashes[1] + (0).to_bytes(4, "big")
    cpu_blocks = [types.SimpleNamespace(block_hash=None) for _ in range(3)]
    cpu_blocks[2] = types.SimpleNamespace(block_hash=exact_hash)

    published = {"value": False}

    class HashIndex:
        def get_one_block(self, key):
            if published["value"] and key == exact_hash:
                return cpu_blocks[2]
            return None

    manager = connector.scheduler_manager
    manager.cpu_block_pool = types.SimpleNamespace(
        blocks=cpu_blocks,
        cached_block_hash_to_block=HashIndex(),
    )
    manager.cpu_kv_cache_config = types.SimpleNamespace(
        kv_cache_groups=[object()]
    )
    manager._store_event_pending_counts = {}
    manager._expected_worker_count = 1
    manager._store_event_to_blocks = {
        5: types.SimpleNamespace(cpu_block_ids=[2])
    }
    manager._store_event_to_reqs = {5: ["save-request"]}
    manager._reqs_to_store = {}

    def publish_completed_store(output):
        published["value"] = True

    manager.update_connector_output = publish_completed_store

    connector.update_connector_output(
        types.SimpleNamespace(
            kv_connector_worker_meta=connector_mod.DaystromPurgeWorkerMetadata(
                completed_store_events={5: 1},
                completed_purge_events={},
            )
        )
    )

    assert connector._checkpoint_stored_hashes[checkpoint] == {exact_hash}  # type: ignore[attr-defined]
    assert checkpoint in connector._checkpoint_store_completed  # type: ignore[attr-defined]
    status = connector._checkpoint_status_telemetry(  # type: ignore[attr-defined]
        connector.policy.evaluate(
            build_kv_transfer_params(
                operation="status",
                checkpoint_digest=checkpoint,
                expires_at=fixed_time + 500,
                nonce="inventory-status",
                secret_path=secret_file,
            ),
            [],
        )
    )
    assert status["reason_code"] == "checkpoint_partial"
    assert status["stored_blocks"] == 1
    assert status["expected_blocks"] == 3


def test_save_reuses_confirmed_resident_rows_in_checkpoint_inventory(
    purge_env, secret_file, fixed_time
) -> None:
    connector_mod, _, VllmConfig, KVConnectorRole, extra = purge_env
    connector = connector_mod.DaystromCooperativeKVConnector(
        VllmConfig(extra_config=extra), KVConnectorRole.SCHEDULER
    )
    connector.policy._time_fn = lambda: fixed_time  # type: ignore[attr-defined]
    block_hash = _HELPERS._block_hashes(1)[0]
    exact_hash = block_hash + (0).to_bytes(4, "big")

    class HashMap:
        def get_one_block(self, key):
            if key == exact_hash:
                return types.SimpleNamespace(block_id=2, block_hash=exact_hash, ref_cnt=0)
            return None

    manager = connector.scheduler_manager
    manager.cpu_block_pool = types.SimpleNamespace(
        cached_block_hash_to_block=HashMap()
    )
    manager.cpu_kv_cache_config = types.SimpleNamespace(kv_cache_groups=[object()])

    checkpoint = _HELPERS._digest("resident-reuse")
    params = build_kv_transfer_params(
        operation="save",
        checkpoint_digest=checkpoint,
        expires_at=fixed_time + 500,
        nonce="resident-reuse-save",
        secret_path=secret_file,
    )
    request = _HELPERS._FakeRequest(
        "resident-reuse-request",
        kv_transfer_params=params,
        block_hashes=[block_hash],
    )

    connector.update_state_after_alloc(request, blocks=[], num_external_tokens=0)

    assert connector._checkpoint_stored_hashes[checkpoint] == {exact_hash}  # type: ignore[attr-defined]


def test_nonempty_checkpoint_with_empty_inventory_cannot_report_purge_complete(
    purge_env, secret_file, fixed_time
) -> None:
    connector_mod, _, VllmConfig, KVConnectorRole, extra = purge_env
    connector = connector_mod.DaystromCooperativeKVConnector(
        VllmConfig(extra_config=extra), KVConnectorRole.SCHEDULER
    )
    connector.policy._time_fn = lambda: fixed_time  # type: ignore[attr-defined]
    checkpoint = _HELPERS._digest("empty-inventory")
    block_hash = _HELPERS._block_hashes(1)[0]
    save_params = build_kv_transfer_params(
        operation="save",
        checkpoint_digest=checkpoint,
        expires_at=fixed_time + 500,
        nonce="empty-inventory-save",
        secret_path=secret_file,
    )
    assert connector.policy.evaluate(save_params, [block_hash]).authorized
    connector._checkpoint_stored_hashes[checkpoint] = set()  # type: ignore[attr-defined]
    manager = connector.scheduler_manager
    manager.cpu_kv_cache_config = types.SimpleNamespace(kv_cache_groups=[object()])
    manager.cpu_block_pool = types.SimpleNamespace(
        cached_block_hash_to_block=types.SimpleNamespace(
            get_one_block=lambda key: None
        )
    )
    manager.has_pending_stores = lambda: False

    purge_params = build_kv_transfer_params(
        operation="purge",
        checkpoint_digest=checkpoint,
        expires_at=fixed_time + 500,
        nonce="empty-inventory-purge",
        secret_path=secret_file,
    )
    request = _HELPERS._FakeRequest(
        "empty-inventory-purge-request",
        kv_transfer_params=purge_params,
        block_hashes=[],
    )

    connector.update_state_after_alloc(request, blocks=[], num_external_tokens=0)
    _, response = connector.request_finished(request, [])

    assert response is not None
    assert response["daystrom"]["reason_code"] == "purge_inventory_empty"


def test_scheduler_refuses_purge_commit_when_shared_owner_expires(
    purge_env, secret_file, fixed_time
) -> None:
    connector_mod, _, VllmConfig, KVConnectorRole, extra = purge_env
    connector = connector_mod.DaystromCooperativeKVConnector(
        VllmConfig(extra_config=extra), KVConnectorRole.SCHEDULER
    )
    now = [fixed_time]
    connector.policy._time_fn = lambda: now[0]  # type: ignore[attr-defined]
    first = _HELPERS._digest("shared-owner-first")
    second = _HELPERS._digest("shared-owner-second")
    shared = _HELPERS._block_hashes(1)[0]
    first_only = _HELPERS._block_hashes(2)[1]
    second_only = _HELPERS._block_hashes(3)[2]
    for checkpoint, hashes, nonce in (
        (first, [shared, first_only], "save-shared-first"),
        (second, [shared, second_only], "save-shared-second"),
    ):
        params = build_kv_transfer_params(
            operation="save",
            checkpoint_digest=checkpoint,
            expires_at=fixed_time + 500,
            nonce=nonce,
            secret_path=secret_file,
        )
        assert connector.policy.evaluate(params, hashes).authorized

    connector.policy.begin_purge(
        first,
        purge_event=23,
        blocks_scheduled=1,
        shared_blocks=1,
        shared_hashes=(shared,),
    )

    target_block = types.SimpleNamespace(block_id=7, ref_cnt=1)
    freed: list[int] = []
    manager = connector.scheduler_manager
    manager.cpu_block_pool = types.SimpleNamespace(
        free_blocks=lambda blocks: freed.extend(block.block_id for block in blocks)
    )
    manager.update_connector_output = lambda output: None
    connector._purge_event_to_checkpoint[23] = first  # type: ignore[attr-defined]
    connector._purge_event_to_request[23] = "owner-race-purge"  # type: ignore[attr-defined]
    connector._purge_event_to_blocks[23] = [target_block]  # type: ignore[attr-defined]
    connector._expected_worker_count = 1  # type: ignore[attr-defined]

    now[0] = fixed_time + 600
    output = types.SimpleNamespace(
        kv_connector_worker_meta=connector_mod.DaystromPurgeWorkerMetadata(
            completed_store_events={},
            completed_purge_events={23: (1, 1, 2048)},
        )
    )
    connector.update_connector_output(output)

    assert freed == []
    assert target_block.ref_cnt == 1
    assert connector._purge_event_to_checkpoint[23] == first  # type: ignore[attr-defined]
    assert connector._purge_completion_errors[first] == (  # type: ignore[attr-defined]
        "purge_shared_ownership_changed"
    )
    status_params = build_kv_transfer_params(
        operation="status",
        checkpoint_digest=first,
        expires_at=fixed_time + 1000,
        nonce="shared-owner-status",
        secret_path=secret_file,
    )
    _, response = connector.request_finished(
        _HELPERS._FakeRequest(
            "shared-owner-status", kv_transfer_params=status_params, block_hashes=[]
        ),
        [],
    )
    assert response is not None
    assert response["daystrom"]["reason_code"] == (
        "purge_shared_ownership_changed"
    )
    assert response["daystrom"]["checkpoint_ready"] is False


def test_scheduler_protects_evicts_commits_and_denies_restore(
    purge_env, secret_file, fixed_time
) -> None:
    connector_mod, _, VllmConfig, KVConnectorRole, extra = purge_env
    connector = connector_mod.DaystromCooperativeKVConnector(
        VllmConfig(extra_config=extra), KVConnectorRole.SCHEDULER
    )
    connector.policy._time_fn = lambda: fixed_time  # type: ignore[attr-defined]
    block_hashes = _HELPERS._block_hashes(2)
    checkpoint = _HELPERS._digest("physical-purge")
    save_params = build_kv_transfer_params(
        operation="save",
        checkpoint_digest=checkpoint,
        expires_at=fixed_time + 500,
        nonce="save-purge",
        secret_path=secret_file,
    )
    assert connector.policy.evaluate(save_params, block_hashes, tokens=32).authorized
    exact_hashes = [block_hash + (0).to_bytes(4, "big") for block_hash in block_hashes]
    connector._checkpoint_stored_hashes[checkpoint] = set(exact_hashes)  # type: ignore[attr-defined]

    class Block:
        def __init__(self, block_id: int):
            self.block_id = block_id
            self.ref_cnt = 0

    blocks = [Block(7), Block(9)]

    class HashMap:
        def __init__(self):
            self.values = {
                exact_hashes[0]: blocks[0],
                exact_hashes[1]: blocks[1],
            }

        def get_one_block(self, key):
            return self.values.get(key)

    class Pool:
        def __init__(self):
            self.cached_block_hash_to_block = HashMap()
            self.evicted: set[int] = set()
            self.freed: list[int] = []

        def touch(self, selected):
            for block in selected:
                block.ref_cnt += 1

        def evict_blocks(self, block_ids):
            self.evicted.update(block_ids)
            self.cached_block_hash_to_block.values = {
                key: block
                for key, block in self.cached_block_hash_to_block.values.items()
                if block.block_id not in block_ids
            }

        def free_blocks(self, selected):
            for block in selected:
                block.ref_cnt -= 1
                self.freed.append(block.block_id)

    pool = Pool()
    manager = connector.scheduler_manager
    manager.cpu_block_pool = pool
    manager.cpu_kv_cache_config = types.SimpleNamespace(kv_cache_groups=[object()])
    manager.update_connector_output = lambda output: None

    purge_params = build_kv_transfer_params(
        operation="purge",
        checkpoint_digest=checkpoint,
        expires_at=fixed_time + 500,
        nonce="purge",
        secret_path=secret_file,
    )
    purge_request = _HELPERS._FakeRequest(
        "purge-request", kv_transfer_params=purge_params, block_hashes=[]
    )
    connector.update_state_after_alloc(
        purge_request, blocks=[], num_external_tokens=0
    )
    assert pool.evicted == {7, 9}
    assert [block.ref_cnt for block in blocks] == [1, 1]
    command = connector.build_connector_meta(types.SimpleNamespace())
    assert command.purge_event == 0
    assert command.purge_cpu_blocks == [7, 9]

    # vLLM builds request metadata before applying this step's worker output,
    # so the first response is honestly pending.
    _, pending_response = connector.request_finished(purge_request, [])
    assert pending_response["daystrom"]["reason_code"] == "purge_pending"

    output = types.SimpleNamespace(
        kv_connector_worker_meta=connector_mod.DaystromPurgeWorkerMetadata(
            completed_store_events={},
            completed_purge_events={0: (1, 2, 4096)},
        )
    )
    connector.update_connector_output(output)
    assert pool.freed == [7, 9]
    assert [block.ref_cnt for block in blocks] == [0, 0]

    status_request = _HELPERS._FakeRequest(
        "purge-status", kv_transfer_params=purge_params, block_hashes=[]
    )
    connector.update_state_after_alloc(
        status_request, blocks=[], num_external_tokens=0
    )
    assert connector._purge_unsent_events == []  # type: ignore[attr-defined]
    _, response = connector.request_finished(status_request, [])
    daystrom = response["daystrom"]
    assert daystrom["reason_code"] == "purge_complete"
    assert daystrom["purged_blocks"] == 2
    assert daystrom["purged_bytes"] == 4096

    restore_params = build_kv_transfer_params(
        operation="restore",
        checkpoint_digest=checkpoint,
        expires_at=fixed_time + 500,
        nonce="restore-after-purge",
        secret_path=secret_file,
    )
    restore_request = _HELPERS._FakeRequest(
        "restore-after-purge",
        kv_transfer_params=restore_params,
        block_hashes=block_hashes,
    )
    assert connector.get_num_new_matched_tokens(restore_request, 0) == (0, False)
    _, denied_response = connector.request_finished(restore_request, [])
    assert denied_response is not None
    assert denied_response["daystrom"]["reason_code"] == "purge_complete"
    assert denied_response["daystrom"]["matched_tokens"] == 0


def _setup_row_mismatch_connector(
    purge_env, secret_file, fixed_time, label, actual_rows
):
    """Shared setup for under-count and over-count mismatch tests.

    Returns ``(connector, checkpoint, block_hashes, blocks, freed, event)``
    where *blocks* are the two pinned CPU blocks and *freed* accumulates block
    ids that the scheduler attempted to return to the free queue.
    """

    connector_mod, _, VllmConfig, KVConnectorRole, extra = purge_env
    connector = connector_mod.DaystromCooperativeKVConnector(
        VllmConfig(extra_config=extra), KVConnectorRole.SCHEDULER
    )
    connector.policy._time_fn = lambda: fixed_time  # type: ignore[attr-defined]
    block_hashes = _HELPERS._block_hashes(2)
    checkpoint = _HELPERS._digest(label)
    save_params = build_kv_transfer_params(
        operation="save",
        checkpoint_digest=checkpoint,
        expires_at=fixed_time + 500,
        nonce=f"{label}-save",
        secret_path=secret_file,
    )
    assert connector.policy.evaluate(save_params, block_hashes, tokens=32).authorized

    connector.policy.begin_purge(
        checkpoint,
        purge_event=29,
        blocks_scheduled=2,
        shared_blocks=0,
    )

    target_block_a = types.SimpleNamespace(block_id=3, ref_cnt=1)
    target_block_b = types.SimpleNamespace(block_id=5, ref_cnt=1)
    blocks = [target_block_a, target_block_b]
    freed: list[int] = []
    manager = connector.scheduler_manager
    manager.cpu_block_pool = types.SimpleNamespace(
        free_blocks=lambda selected: freed.extend(
            block.block_id for block in selected
        )
    )
    manager.update_connector_output = lambda output: None
    event = 29
    connector._purge_event_to_checkpoint[event] = checkpoint  # type: ignore[attr-defined]
    connector._purge_event_to_request[event] = f"{label}-purge"  # type: ignore[attr-defined]
    connector._purge_event_to_blocks[event] = blocks  # type: ignore[attr-defined]
    connector._expected_worker_count = 1  # type: ignore[attr-defined]

    output = types.SimpleNamespace(
        kv_connector_worker_meta=connector_mod.DaystromPurgeWorkerMetadata(
            completed_store_events={},
            completed_purge_events={event: (1, actual_rows, actual_rows * 2048)},
        )
    )
    connector.update_connector_output(output)

    return connector, connector_mod, checkpoint, block_hashes, blocks, freed, event


def test_scheduler_row_undercount_is_terminal_and_does_not_requeue(
    purge_env, secret_file, fixed_time
) -> None:
    """Under-count: worker reports fewer rows than scheduled. The purge must
    record a terminal completion fault, keep rows protected, and never requeue
    the zero command or allow restore."""

    connector, _, checkpoint, block_hashes, blocks, freed, event = (
        _setup_row_mismatch_connector(
            purge_env, secret_file, fixed_time, "row-undercount", actual_rows=1
        )
    )

    # Rows remain protected — never freed.
    assert freed == []
    assert blocks[0].ref_cnt == 1
    assert blocks[1].ref_cnt == 1
    # Event maps preserved so rows stay pinned and purge is nonterminal.
    assert connector._purge_event_to_checkpoint[event] == checkpoint  # type: ignore[attr-defined]
    assert connector._purge_event_to_blocks[event] == blocks  # type: ignore[attr-defined]
    # Terminal completion fault recorded with safe evidence.
    assert connector._purge_completion_errors[checkpoint] == (  # type: ignore[attr-defined]
        "purge_worker_row_mismatch"
    )
    assert connector._purge_completion_evidence[checkpoint] == (  # type: ignore[attr-defined]
        {"purge_expected_rows": 2, "purge_actual_rows": 1}
    )

    # A subsequent purge status request must NOT requeue the zero command.
    purge_params = build_kv_transfer_params(
        operation="purge",
        checkpoint_digest=checkpoint,
        expires_at=fixed_time + 500,
        nonce="row-undercount-status",
        secret_path=secret_file,
    )
    status_request = _HELPERS._FakeRequest(
        "row-undercount-status", kv_transfer_params=purge_params, block_hashes=[]
    )
    connector.update_state_after_alloc(
        status_request, blocks=[], num_external_tokens=0
    )
    assert connector._purge_unsent_events == []  # type: ignore[attr-defined]

    _, response = connector.request_finished(status_request, [])
    assert response is not None
    daystrom = response["daystrom"]
    assert daystrom["reason_code"] == "purge_worker_row_mismatch"
    assert daystrom["purge_expected_rows"] == 2
    assert daystrom["purge_actual_rows"] == 1

    # Restore must be denied — fail-closed row pinning preserved.
    restore_params = build_kv_transfer_params(
        operation="restore",
        checkpoint_digest=checkpoint,
        expires_at=fixed_time + 500,
        nonce="row-undercount-restore",
        secret_path=secret_file,
    )
    restore_request = _HELPERS._FakeRequest(
        "row-undercount-restore",
        kv_transfer_params=restore_params,
        block_hashes=block_hashes,
    )
    assert connector.get_num_new_matched_tokens(restore_request, 0) == (0, False)
    _, denied = connector.request_finished(restore_request, [])
    assert denied is not None
    assert denied["daystrom"]["reason_code"] == "purge_pending"


def test_overlapping_purges_sharing_hash_serialize_and_neither_orphaned(
    purge_env, secret_file, fixed_time
) -> None:
    """Deterministic A={H,a1}, B={H,b1} overlapping-purge coverage proving:

    * H is not zeroed while still retained/shared by an in-flight purge
    * no row is zeroed twice
    * neither purge becomes permanently nonterminal
    * exclusive rows are freed exactly once
    * status/retry behavior is deterministic
    """

    connector_mod, _, VllmConfig, KVConnectorRole, extra = purge_env
    connector = connector_mod.DaystromCooperativeKVConnector(
        VllmConfig(extra_config=extra), KVConnectorRole.SCHEDULER
    )
    connector.policy._time_fn = lambda: fixed_time  # type: ignore[attr-defined]

    # Two checkpoints sharing hash H: A={H, a1}, B={H, b1}.
    shared_hash = _HELPERS._block_hashes(1)[0]
    a1_hash = _HELPERS._block_hashes(2)[1]
    b1_hash = _HELPERS._block_hashes(3)[2]
    first = _HELPERS._digest("overlap-A")
    second = _HELPERS._digest("overlap-B")
    for checkpoint, hashes, nonce in (
        (first, [shared_hash, a1_hash], "save-overlap-A"),
        (second, [shared_hash, b1_hash], "save-overlap-B"),
    ):
        params = build_kv_transfer_params(
            operation="save",
            checkpoint_digest=checkpoint,
            expires_at=fixed_time + 500,
            nonce=nonce,
            secret_path=secret_file,
        )
        assert connector.policy.evaluate(params, hashes).authorized

    # Exact hashes (block_hash + group_id bytes) and block objects.
    h_exact = shared_hash + (0).to_bytes(4, "big")
    a1_exact = a1_hash + (0).to_bytes(4, "big")
    b1_exact = b1_hash + (0).to_bytes(4, "big")

    class Block:
        def __init__(self, block_id: int):
            self.block_id = block_id
            self.ref_cnt = 0

    h_block = Block(5)
    a1_block = Block(7)
    b1_block = Block(9)

    class HashMap:
        def __init__(self):
            self.values = {
                h_exact: h_block,
                a1_exact: a1_block,
                b1_exact: b1_block,
            }

        def get_one_block(self, key):
            return self.values.get(key)

    class Pool:
        def __init__(self):
            self.cached_block_hash_to_block = HashMap()
            self.evicted: set[int] = set()
            self.freed: list[int] = []

        def touch(self, selected):
            for block in selected:
                block.ref_cnt += 1

        def evict_blocks(self, block_ids):
            self.evicted.update(block_ids)
            self.cached_block_hash_to_block.values = {
                key: block
                for key, block in self.cached_block_hash_to_block.values.items()
                if block.block_id not in block_ids
            }

        def free_blocks(self, selected):
            for block in selected:
                block.ref_cnt -= 1
                self.freed.append(block.block_id)

    pool = Pool()
    manager = connector.scheduler_manager
    manager.cpu_block_pool = pool
    manager.cpu_kv_cache_config = types.SimpleNamespace(kv_cache_groups=[object()])
    manager.update_connector_output = lambda output: None

    # Pre-set CPU row inventories (resident rows confirmed for each checkpoint).
    connector._checkpoint_stored_hashes[first] = {h_exact, a1_exact}  # type: ignore[attr-defined]
    connector._checkpoint_stored_hashes[second] = {h_exact, b1_exact}  # type: ignore[attr-defined]

    # --- Step 1: A's purge is scheduled. H is shared (B live), only a1 targeted.
    purge_a_params = build_kv_transfer_params(
        operation="purge",
        checkpoint_digest=first,
        expires_at=fixed_time + 500,
        nonce="purge-overlap-A",
        secret_path=secret_file,
    )
    purge_a_request = _HELPERS._FakeRequest(
        "purge-overlap-A", kv_transfer_params=purge_a_params, block_hashes=[]
    )
    connector.update_state_after_alloc(purge_a_request, blocks=[], num_external_tokens=0)

    # A's purge evicted only a1's block; H is retained as shared.
    assert pool.evicted == {7}
    assert a1_block.ref_cnt == 1
    assert h_block.ref_cnt == 0
    command_a = connector.build_connector_meta(types.SimpleNamespace())
    assert command_a.purge_event == 0
    assert command_a.purge_cpu_blocks == [7]

    _, pending_a = connector.request_finished(purge_a_request, [])
    assert pending_a["daystrom"]["reason_code"] == "purge_pending"

    # --- Step 2: B's purge is attempted while A is in-flight. Guard fires.
    purge_b_params = build_kv_transfer_params(
        operation="purge",
        checkpoint_digest=second,
        expires_at=fixed_time + 500,
        nonce="purge-overlap-B",
        secret_path=secret_file,
    )
    purge_b_request = _HELPERS._FakeRequest(
        "purge-overlap-B", kv_transfer_params=purge_b_params, block_hashes=[]
    )
    connector.update_state_after_alloc(purge_b_request, blocks=[], num_external_tokens=0)

    # B's purge was NOT scheduled: no eviction, no new event.
    assert pool.evicted == {7}
    assert connector._purge_unsent_events == []  # type: ignore[attr-defined]
    _, conflict_response = connector.request_finished(purge_b_request, [])
    assert conflict_response is not None
    assert conflict_response["daystrom"]["reason_code"] == "purge_ownership_conflict"

    # H was not zeroed: H's block is still in the hash map and untouched.
    assert h_block.ref_cnt == 0
    assert h_exact in pool.cached_block_hash_to_block.values

    # --- Step 3: A's purge completes. a1 freed, H still resident.
    output_a = types.SimpleNamespace(
        kv_connector_worker_meta=connector_mod.DaystromPurgeWorkerMetadata(
            completed_store_events={},
            completed_purge_events={0: (1, 1, 2048)},
        )
    )
    connector.update_connector_output(output_a)
    assert pool.freed == [7]
    assert a1_block.ref_cnt == 0
    assert connector.policy.record_for(first) is None
    # H is still in the hash map (was never evicted by A's purge).
    assert h_exact in pool.cached_block_hash_to_block.values

    # --- Step 4: B's purge is retried with a new request. No conflict (A done).
    purge_b_retry_params = build_kv_transfer_params(
        operation="purge",
        checkpoint_digest=second,
        expires_at=fixed_time + 500,
        nonce="purge-overlap-B-retry",
        secret_path=secret_file,
    )
    purge_b_retry_request = _HELPERS._FakeRequest(
        "purge-overlap-B-retry",
        kv_transfer_params=purge_b_retry_params,
        block_hashes=[],
    )
    connector.update_state_after_alloc(
        purge_b_retry_request, blocks=[], num_external_tokens=0
    )

    # B's purge now targets H and b1 (both unique after A completed).
    assert pool.evicted == {7, 5, 9}
    assert h_block.ref_cnt == 1
    assert b1_block.ref_cnt == 1
    command_b = connector.build_connector_meta(types.SimpleNamespace())
    assert command_b.purge_event == 1
    assert command_b.purge_cpu_blocks == [5, 9]

    _, pending_b = connector.request_finished(purge_b_retry_request, [])
    assert pending_b["daystrom"]["reason_code"] == "purge_pending"

    # --- Step 5: B's purge completes. H and b1 freed exactly once.
    output_b = types.SimpleNamespace(
        kv_connector_worker_meta=connector_mod.DaystromPurgeWorkerMetadata(
            completed_store_events={},
            completed_purge_events={1: (1, 2, 4096)},
        )
    )
    connector.update_connector_output(output_b)
    assert pool.freed == [7, 5, 9]
    assert h_block.ref_cnt == 0
    assert b1_block.ref_cnt == 0
    assert connector.policy.record_for(second) is None

    # --- Verify: no row was zeroed twice (each freed exactly once).
    assert pool.freed.count(5) == 1
    assert pool.freed.count(7) == 1
    assert pool.freed.count(9) == 1

    # --- Verify: neither purge is nonterminal (both completed, records gone).
    assert connector.policy.record_for(first) is None
    assert connector.policy.record_for(second) is None

    # --- Verify: restore denied for both purged checkpoints (fail-closed).
    for checkpoint, nonce, hashes in (
        (first, "restore-overlap-A", [shared_hash, a1_hash]),
        (second, "restore-overlap-B", [shared_hash, b1_hash]),
    ):
        restore_params = build_kv_transfer_params(
            operation="restore",
            checkpoint_digest=checkpoint,
            expires_at=fixed_time + 500,
            nonce=nonce,
            secret_path=secret_file,
        )
        restore_request = _HELPERS._FakeRequest(
            f"restore-{nonce}",
            kv_transfer_params=restore_params,
            block_hashes=hashes,
        )
        assert connector.get_num_new_matched_tokens(restore_request, 0) == (0, False)
        _, denied = connector.request_finished(restore_request, [])
        assert denied is not None
        assert denied["daystrom"]["reason_code"] == "purge_complete"

def test_scheduler_row_overcount_is_terminal_and_does_not_requeue(
    purge_env, secret_file, fixed_time
) -> None:
    """Over-count: worker reports more rows than scheduled. The purge must
    record a terminal completion fault, keep rows protected, and never requeue
    the zero command or allow restore."""

    connector, _, checkpoint, block_hashes, blocks, freed, event = (
        _setup_row_mismatch_connector(
            purge_env, secret_file, fixed_time, "row-overcount", actual_rows=3
        )
    )

    # Rows remain protected — never freed.
    assert freed == []
    assert blocks[0].ref_cnt == 1
    assert blocks[1].ref_cnt == 1
    # Event maps preserved so rows stay pinned and purge is nonterminal.
    assert connector._purge_event_to_checkpoint[event] == checkpoint  # type: ignore[attr-defined]
    assert connector._purge_event_to_blocks[event] == blocks  # type: ignore[attr-defined]
    # Terminal completion fault recorded with safe evidence.
    assert connector._purge_completion_errors[checkpoint] == (  # type: ignore[attr-defined]
        "purge_worker_row_mismatch"
    )
    assert connector._purge_completion_evidence[checkpoint] == (  # type: ignore[attr-defined]
        {"purge_expected_rows": 2, "purge_actual_rows": 3}
    )

    # A subsequent purge status request must NOT requeue the zero command.
    purge_params = build_kv_transfer_params(
        operation="purge",
        checkpoint_digest=checkpoint,
        expires_at=fixed_time + 500,
        nonce="row-overcount-status",
        secret_path=secret_file,
    )
    status_request = _HELPERS._FakeRequest(
        "row-overcount-status", kv_transfer_params=purge_params, block_hashes=[]
    )
    connector.update_state_after_alloc(
        status_request, blocks=[], num_external_tokens=0
    )
    assert connector._purge_unsent_events == []  # type: ignore[attr-defined]

    _, response = connector.request_finished(status_request, [])
    assert response is not None
    daystrom = response["daystrom"]
    assert daystrom["reason_code"] == "purge_worker_row_mismatch"
    assert daystrom["purge_expected_rows"] == 2
    assert daystrom["purge_actual_rows"] == 3

    # Restore must be denied — fail-closed row pinning preserved.
    restore_params = build_kv_transfer_params(
        operation="restore",
        checkpoint_digest=checkpoint,
        expires_at=fixed_time + 500,
        nonce="row-overcount-restore",
        secret_path=secret_file,
    )
    restore_request = _HELPERS._FakeRequest(
        "row-overcount-restore",
        kv_transfer_params=restore_params,
        block_hashes=block_hashes,
    )
    assert connector.get_num_new_matched_tokens(restore_request, 0) == (0, False)
    _, denied = connector.request_finished(restore_request, [])
    assert denied is not None
    assert denied["daystrom"]["reason_code"] == "purge_pending"
