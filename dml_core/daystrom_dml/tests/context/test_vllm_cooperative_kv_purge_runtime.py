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
    manager = connector.scheduler_manager
    manager.cpu_block_pool = types.SimpleNamespace(blocks=cpu_blocks)
    manager._store_event_pending_counts = {}
    manager._expected_worker_count = 1
    manager._store_event_to_blocks = {
        5: types.SimpleNamespace(cpu_block_ids=[2])
    }
    manager._store_event_to_reqs = {5: ["save-request"]}
    manager._reqs_to_store = {}
    manager.update_connector_output = lambda output: None

    connector.update_connector_output(
        types.SimpleNamespace(
            kv_connector_worker_meta=connector_mod.DaystromPurgeWorkerMetadata(
                completed_store_events={5: 1},
                completed_purge_events={},
            )
        )
    )

    assert connector._checkpoint_stored_hashes[checkpoint] == {exact_hash}  # type: ignore[attr-defined]


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
