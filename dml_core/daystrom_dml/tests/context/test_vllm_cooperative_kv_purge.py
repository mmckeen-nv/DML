from __future__ import annotations

import hashlib
from pathlib import Path

from daystrom_dml.context.vllm_bridge.policy import (
    DaystromKVPolicy,
    build_kv_transfer_params,
)


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _hashes(label: str, count: int) -> list[bytes]:
    return [hashlib.sha256(f"{label}-{index}".encode()).digest() for index in range(count)]


def _secret(tmp_path: Path) -> Path:
    path = tmp_path / "control.key"
    path.write_text("purge-test-secret\n")
    return path


def _params(
    secret: Path,
    *,
    operation: str,
    checkpoint: str,
    nonce: str,
) -> dict[str, object]:
    return build_kv_transfer_params(
        operation=operation,
        checkpoint_digest=checkpoint,
        expires_at=1_000_500.0,
        nonce=nonce,
        secret_path=secret,
    )


def test_purge_logically_invalidates_before_physical_completion(tmp_path: Path) -> None:
    secret = _secret(tmp_path)
    checkpoint = _digest("checkpoint")
    blocks = _hashes("saved", 3)
    policy = DaystromKVPolicy(secret, time_fn=lambda: 1_000_000.0)
    assert policy.evaluate(
        _params(secret, operation="save", checkpoint=checkpoint, nonce="save"),
        blocks,
        tokens=48,
    ).authorized

    decision = policy.evaluate(
        _params(secret, operation="purge", checkpoint=checkpoint, nonce="purge"),
        [],
    )
    assert decision.authorized
    assert decision.reason_code == "purge_authorized"

    policy.begin_purge(
        checkpoint,
        purge_event=7,
        blocks_scheduled=2,
        shared_blocks=1,
        shared_hashes=(blocks[0],),
    )
    restore = policy.evaluate(
        _params(secret, operation="restore", checkpoint=checkpoint, nonce="restore"),
        blocks,
    )
    assert not restore.authorized
    assert restore.reason_code == "purge_pending"

    replacement = policy.evaluate(
        _params(secret, operation="save", checkpoint=checkpoint, nonce="replace"),
        blocks,
    )
    assert not replacement.authorized
    assert replacement.reason_code == "purge_pending"


def test_purge_completion_removes_record_and_returns_physical_counters(tmp_path: Path) -> None:
    secret = _secret(tmp_path)
    checkpoint = _digest("checkpoint")
    blocks = _hashes("saved", 2)
    policy = DaystromKVPolicy(secret, time_fn=lambda: 1_000_000.0)
    policy.evaluate(
        _params(secret, operation="save", checkpoint=checkpoint, nonce="save"),
        blocks,
        tokens=32,
    )
    policy.begin_purge(
        checkpoint,
        purge_event=11,
        blocks_scheduled=2,
        shared_blocks=0,
    )

    completed = policy.complete_purge(11, blocks_zeroed=2, bytes_zeroed=4096)
    assert completed.reason_code == "purge_complete"
    assert completed.purged_blocks == 2
    assert completed.purged_bytes == 4096
    assert completed.shared_blocks == 0
    assert policy.record_for(checkpoint) is None

    status = policy.evaluate(
        _params(secret, operation="purge", checkpoint=checkpoint, nonce="status"),
        [],
    )
    assert status.authorized
    assert status.reason_code == "purge_complete"
    assert status.purged_bytes == 4096


def test_shared_prefixes_are_not_selected_for_zeroization(tmp_path: Path) -> None:
    secret = _secret(tmp_path)
    first = _digest("first")
    second = _digest("second")
    shared = _hashes("shared", 2)
    first_only = _hashes("first-only", 1)
    second_only = _hashes("second-only", 1)
    policy = DaystromKVPolicy(secret, time_fn=lambda: 1_000_000.0)
    policy.evaluate(
        _params(secret, operation="save", checkpoint=first, nonce="save-first"),
        shared + first_only,
    )
    policy.evaluate(
        _params(secret, operation="save", checkpoint=second, nonce="save-second"),
        shared + second_only,
    )

    unique, retained = policy.partition_purge_hashes(first)
    assert unique == tuple(first_only)
    assert retained == tuple(shared)


def test_purge_completion_revalidates_exact_live_shared_owners(tmp_path: Path) -> None:
    secret = _secret(tmp_path)
    first = _digest("owner-first")
    second = _digest("owner-second")
    shared = _hashes("owner-shared", 1)
    now = [1_000_000.0]
    policy = DaystromKVPolicy(secret, time_fn=lambda: now[0])
    policy.evaluate(
        _params(secret, operation="save", checkpoint=first, nonce="save-first"),
        shared + _hashes("owner-first-only", 1),
    )
    policy.evaluate(
        _params(secret, operation="save", checkpoint=second, nonce="save-second"),
        shared + _hashes("owner-second-only", 1),
    )
    _, retained = policy.partition_purge_hashes(first)
    policy.begin_purge(
        first,
        purge_event=17,
        blocks_scheduled=1,
        shared_blocks=2,
        shared_hashes=retained,
    )

    assert policy.purge_shared_owners_valid(first)
    now[0] = 1_000_600.0
    assert not policy.purge_shared_owners_valid(first)
