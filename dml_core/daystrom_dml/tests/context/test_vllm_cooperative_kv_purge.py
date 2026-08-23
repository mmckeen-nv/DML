from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from daystrom_dml.context.vllm_bridge.policy import (
    DaystromKVAuthorizationError,
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


def test_overlapping_purges_sharing_hash_serialize_via_ownership_guard(
    tmp_path: Path,
) -> None:
    """Overlapping purges of checkpoints sharing KV hash H must serialize.

    A={H,a1} begins purge and retains H as shared.  B={H,b1} attempts to
    purge while A is in-flight.  The narrow ownership conflict guard must
    defer B until A completes, proving:

    * H is not zeroed while still retained/shared by an in-flight purge
    * no row is zeroed twice
    * neither purge becomes permanently nonterminal
    * exclusive rows are freed exactly once
    * status/retry behavior is deterministic
    """

    secret = _secret(tmp_path)
    first = _digest("overlap-first")
    second = _digest("overlap-second")
    shared = _hashes("overlap-shared", 1)
    first_only = _hashes("overlap-first-only", 1)
    second_only = _hashes("overlap-second-only", 1)
    policy = DaystromKVPolicy(secret, time_fn=lambda: 1_000_000.0)

    assert policy.evaluate(
        _params(secret, operation="save", checkpoint=first, nonce="save-first"),
        shared + first_only,
    ).authorized
    assert policy.evaluate(
        _params(secret, operation="save", checkpoint=second, nonce="save-second"),
        shared + second_only,
    ).authorized

    # A begins purge: H is shared (B still live), A retains H.
    unique_a, shared_a = policy.partition_purge_hashes(first)
    assert unique_a == tuple(first_only)
    assert shared_a == tuple(shared)
    policy.begin_purge(
        first,
        purge_event=101,
        blocks_scheduled=len(unique_a),
        shared_blocks=len(shared_a),
        shared_hashes=shared_a,
    )

    # B attempts purge while A is in-flight: guard fires deterministically.
    with pytest.raises(DaystromKVAuthorizationError) as exc_info:
        policy.partition_purge_hashes(second)
    assert exc_info.value.reason_code == "purge_ownership_conflict"

    # begin_purge also rejects (defense in depth).
    with pytest.raises(DaystromKVAuthorizationError) as exc_info_begin:
        policy.begin_purge(
            second,
            purge_event=102,
            blocks_scheduled=1,
            shared_blocks=0,
        )
    assert exc_info_begin.value.reason_code == "purge_ownership_conflict"

    # A completion: shared owners still valid (B is live, not purging).
    assert policy.purge_shared_owners_valid(first)
    completed_a = policy.complete_purge(
        101, blocks_zeroed=len(unique_a), bytes_zeroed=2048
    )
    assert completed_a.reason_code == "purge_complete"
    assert policy.record_for(first) is None

    # B retries purge: no conflict (A completed), H is now unique to B.
    unique_b, shared_b = policy.partition_purge_hashes(second)
    assert unique_b == tuple(shared + second_only)
    assert shared_b == ()
    policy.begin_purge(
        second,
        purge_event=103,
        blocks_scheduled=len(unique_b),
        shared_blocks=0,
    )
    assert policy.purge_shared_owners_valid(second)
    completed_b = policy.complete_purge(
        103, blocks_zeroed=len(unique_b), bytes_zeroed=4096
    )
    assert completed_b.reason_code == "purge_complete"
    assert policy.record_for(second) is None

    # Neither purge is nonterminal: both completed and records removed.
    assert first not in policy._purges or policy._purges[first].completed
    assert second not in policy._purges or policy._purges[second].completed


def test_ownership_guard_does_not_fire_for_non_overlapping_purges(
    tmp_path: Path,
) -> None:
    """Purges of checkpoints with no shared hashes must not conflict."""

    secret = _secret(tmp_path)
    first = _digest("disjoint-first")
    second = _digest("disjoint-second")
    first_hashes = _hashes("disjoint-first", 2)
    second_hashes = _hashes("disjoint-second", 2)
    policy = DaystromKVPolicy(secret, time_fn=lambda: 1_000_000.0)

    policy.evaluate(
        _params(secret, operation="save", checkpoint=first, nonce="save-first"),
        first_hashes,
    )
    policy.evaluate(
        _params(secret, operation="save", checkpoint=second, nonce="save-second"),
        second_hashes,
    )

    unique_a, _ = policy.partition_purge_hashes(first)
    policy.begin_purge(
        first,
        purge_event=201,
        blocks_scheduled=len(unique_a),
        shared_blocks=0,
    )

    # B can partition and begin while A is in-flight: no shared hashes.
    unique_b, _ = policy.partition_purge_hashes(second)
    assert unique_b == tuple(second_hashes)
    policy.begin_purge(
        second,
        purge_event=202,
        blocks_scheduled=len(unique_b),
        shared_blocks=0,
    )

    assert policy.purge_shared_owners_valid(first)
    assert policy.purge_shared_owners_valid(second)
    assert policy.complete_purge(201, blocks_zeroed=len(unique_a), bytes_zeroed=2048).reason_code == "purge_complete"
    assert policy.complete_purge(202, blocks_zeroed=len(unique_b), bytes_zeroed=2048).reason_code == "purge_complete"
