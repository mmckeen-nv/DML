"""Tests for the experimental controller-gated vLLM 0.20 SimpleCPUOffload
Daystrom cooperative KV connector and its runtime-neutral authorization policy.

The policy tests run on any platform without vLLM/torch.  The connector tests
stub the minimal vLLM modules explicitly before importing the connector, so
they also run without vLLM installed.
"""
from __future__ import annotations

import hashlib
import json
import enum
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from daystrom_dml.context.vllm_bridge.policy import (
    DAYSTROM_KV_SCHEMA_VERSION,
    DaystromKVAuthorizationError,
    DaystromKVDecision,
    DaystromKVPolicy,
    DaystromKVRequest,
    build_daystrom_params,
    build_kv_transfer_params,
    canonical_message_for,
    compute_authorization,
    compute_block_hash_digest,
    to_json_telemetry,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _block_hashes(n: int, seed: int = 0) -> list[bytes]:
    return [hashlib.sha256(f"block-{seed}-{i}".encode()).digest() for i in range(n)]


def _nonce() -> str:
    return "nonce-abc123"


@pytest.fixture()
def secret_file(tmp_path: Path) -> Path:
    p = tmp_path / "secret.key"
    p.write_text("super-secret-key-for-hmac-testing\n", encoding="utf-8")
    p.chmod(0o600)
    return p


@pytest.fixture()
def policy(secret_file: Path) -> DaystromKVPolicy:
    return DaystromKVPolicy(secret_file, max_ttl_seconds=3600, max_records=16)


@pytest.fixture()
def fixed_time() -> float:
    return 1_000_000.0


@pytest.fixture()
def policy_fixed_time(secret_file: Path, fixed_time: float) -> DaystromKVPolicy:
    return DaystromKVPolicy(
        secret_file,
        max_ttl_seconds=3600,
        max_records=16,
        time_fn=lambda: fixed_time,
    )


def _make_params(
    *,
    operation: str,
    checkpoint_digest: str,
    expires_at: float,
    nonce: str,
    secret_file: Path,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a signed envelope with valid defaults, then apply overrides.

    The helper always mints a validly-signed envelope first (so the HMAC is
    correct for the default fields), then mutates individual fields per the
    overrides.  This lets tests feed malformed/invalid field values to the
    policy without the builder rejecting them.
    """

    params = build_kv_transfer_params(
        operation="save",
        checkpoint_digest=_digest("valid-default"),
        expires_at=1_000_100.0,
        nonce="valid-default-nonce",
        secret_path=secret_file,
    )
    daystrom = params["daystrom"]
    daystrom["operation"] = operation
    daystrom["checkpoint_digest"] = checkpoint_digest
    daystrom["expires_at"] = expires_at
    daystrom["nonce"] = nonce
    daystrom.update(overrides)
    return params


# --------------------------------------------------------------------------- #
# Policy: construction and secret loading
# --------------------------------------------------------------------------- #


class TestPolicyConstruction:
    def test_constructs_with_absolute_secret_path(self, secret_file: Path):
        pol = DaystromKVPolicy(secret_file)
        assert pol.max_ttl_seconds == 3600
        assert pol.max_records == 4096

    def test_rejects_relative_secret_path(self, tmp_path: Path):
        rel = Path("secret.key")
        with pytest.raises(DaystromKVAuthorizationError, match="secret_path_not_absolute"):
            DaystromKVPolicy(rel)

    def test_rejects_missing_secret_file(self, tmp_path: Path):
        with pytest.raises(DaystromKVAuthorizationError, match="secret_file_missing"):
            DaystromKVPolicy(tmp_path / "nope.key")

    def test_rejects_empty_secret(self, tmp_path: Path):
        p = tmp_path / "empty.key"
        p.write_text("   \n", encoding="utf-8")
        with pytest.raises(DaystromKVAuthorizationError, match="secret_empty"):
            DaystromKVPolicy(p)

    def test_rejects_symlink_secret(self, secret_file: Path, tmp_path: Path):
        link = tmp_path / "link.key"
        os.symlink(secret_file, link)
        with pytest.raises(DaystromKVAuthorizationError, match="secret_path_is_symlink"):
            DaystromKVPolicy(link)

    def test_rejects_invalid_max_ttl(self, secret_file: Path):
        with pytest.raises(ValueError):
            DaystromKVPolicy(secret_file, max_ttl_seconds=0)
        with pytest.raises(ValueError):
            DaystromKVPolicy(secret_file, max_ttl_seconds=-1)

    def test_rejects_invalid_max_records(self, secret_file: Path):
        with pytest.raises(ValueError):
            DaystromKVPolicy(secret_file, max_records=0)


# --------------------------------------------------------------------------- #
# Policy: field validation (reject unknown/malformed)
# --------------------------------------------------------------------------- #


class TestFieldValidation:
    def test_missing_kv_transfer_params(self, policy_fixed_time: DaystromKVPolicy):
        d = policy_fixed_time.evaluate(None, None)
        assert not d.authorized
        assert d.reason_code == "missing_kv_transfer_params"

    def test_missing_daystrom_mapping(self, policy_fixed_time: DaystromKVPolicy):
        d = policy_fixed_time.evaluate({"other": 1}, None)
        assert not d.authorized
        assert d.reason_code == "missing_daystrom_mapping"

    def test_unknown_field_rejected(self, policy_fixed_time, secret_file):
        params = _make_params(
            operation="save",
            checkpoint_digest=_digest("x"),
            expires_at=1_000_100.0,
            nonce=_nonce(),
            secret_file=secret_file,
            extra_field="bad",
        )
        d = policy_fixed_time.evaluate(params, _block_hashes(1))
        assert not d.authorized
        assert d.reason_code == "unknown_field"

    def test_missing_field_rejected(self, policy_fixed_time, secret_file):
        params = _make_params(
            operation="save",
            checkpoint_digest=_digest("x"),
            expires_at=1_000_100.0,
            nonce=_nonce(),
            secret_file=secret_file,
        )
        del params["daystrom"]["nonce"]
        d = policy_fixed_time.evaluate(params, _block_hashes(1))
        assert not d.authorized
        assert d.reason_code == "missing_field"

    def test_wrong_schema_version(self, policy_fixed_time, secret_file):
        params = _make_params(
            operation="save",
            checkpoint_digest=_digest("x"),
            expires_at=1_000_100.0,
            nonce=_nonce(),
            secret_file=secret_file,
            schema_version="daystrom-vllm-kv-v0",
        )
        d = policy_fixed_time.evaluate(params, _block_hashes(1))
        assert not d.authorized
        assert d.reason_code == "schema_version_mismatch"

    def test_invalid_operation(self, policy_fixed_time, secret_file):
        params = _make_params(
            operation="delete",
            checkpoint_digest=_digest("x"),
            expires_at=1_000_100.0,
            nonce=_nonce(),
            secret_file=secret_file,
        )
        d = policy_fixed_time.evaluate(params, _block_hashes(1))
        assert not d.authorized
        assert d.reason_code == "operation_invalid"

    def test_malformed_checkpoint_digest(self, policy_fixed_time, secret_file):
        params = _make_params(
            operation="save",
            checkpoint_digest="sha256:deadbeef",
            expires_at=1_000_100.0,
            nonce=_nonce(),
            secret_file=secret_file,
        )
        d = policy_fixed_time.evaluate(params, _block_hashes(1))
        assert not d.authorized
        assert d.reason_code == "checkpoint_digest_malformed"

    def test_non_finite_expires_at(self, policy_fixed_time, secret_file):
        params = _make_params(
            operation="save",
            checkpoint_digest=_digest("x"),
            expires_at=float("inf"),
            nonce=_nonce(),
            secret_file=secret_file,
        )
        d = policy_fixed_time.evaluate(params, _block_hashes(1))
        assert not d.authorized
        assert d.reason_code == "expires_at_not_finite"

    def test_nonce_too_long(self, policy_fixed_time, secret_file):
        params = _make_params(
            operation="save",
            checkpoint_digest=_digest("x"),
            expires_at=1_000_100.0,
            nonce="x" * 200,
            secret_file=secret_file,
        )
        d = policy_fixed_time.evaluate(params, _block_hashes(1))
        assert not d.authorized
        assert d.reason_code == "nonce_length_out_of_range"

    def test_nonce_with_control_chars(self, policy_fixed_time, secret_file):
        params = _make_params(
            operation="save",
            checkpoint_digest=_digest("x"),
            expires_at=1_000_100.0,
            nonce="bad\tnonce",
            secret_file=secret_file,
        )
        d = policy_fixed_time.evaluate(params, _block_hashes(1))
        assert not d.authorized
        assert d.reason_code == "nonce_not_printable_ascii"

    def test_authorization_not_hex64(self, policy_fixed_time, secret_file):
        params = _make_params(
            operation="save",
            checkpoint_digest=_digest("x"),
            expires_at=1_000_100.0,
            nonce=_nonce(),
            secret_file=secret_file,
            authorization="nothex",
        )
        d = policy_fixed_time.evaluate(params, _block_hashes(1))
        assert not d.authorized
        assert d.reason_code == "authorization_not_hex64"


# --------------------------------------------------------------------------- #
# Policy: HMAC authorization
# --------------------------------------------------------------------------- #


class TestAuthorization:
    def test_hmac_mismatch_rejected(self, policy_fixed_time, secret_file):
        bh = _block_hashes(2)
        digest = compute_block_hash_digest(bh)
        params = _make_params(
            operation="save",
            checkpoint_digest=digest,
            expires_at=1_000_100.0,
            nonce=_nonce(),
            secret_file=secret_file,
            authorization="a" * 64,  # wrong hmac
        )
        d = policy_fixed_time.evaluate(params, bh)
        assert not d.authorized
        assert d.reason_code == "authorization_hmac_mismatch"

    def test_canonical_message_is_exact_ordered(self, secret_file):
        req = DaystromKVRequest(
            schema_version=DAYSTROM_KV_SCHEMA_VERSION,
            operation="save",
            checkpoint_digest=_digest("c"),
            expires_at=123.456789,
            nonce="n1",
            authorization="a" * 64,
            block_hashes=(),
        )
        expected = canonical_message_for(
            operation="save",
            checkpoint_digest=_digest("c"),
            expires_at=123.456789,
            nonce="n1",
        )
        assert req.canonical_message == expected
        # Exact ordered concatenation with \n separators
        assert expected == b"daystrom-vllm-kv-v1\nsave\n" + _digest("c").encode() + b"\n123.456789\nn1"

    def test_compute_authorization_matches_build(self, secret_file):
        req = DaystromKVRequest(
            schema_version=DAYSTROM_KV_SCHEMA_VERSION,
            operation="restore",
            checkpoint_digest=_digest("d"),
            expires_at=999.0,
            nonce="nonce-x",
            authorization="b" * 64,
            block_hashes=(),
        )
        secret = secret_file.read_text().strip().encode()
        expected = compute_authorization(secret, req)
        params = build_daystrom_params(
            operation="restore",
            checkpoint_digest=_digest("d"),
            expires_at=999.0,
            nonce="nonce-x",
            secret_path=secret_file,
        )
        assert params["authorization"] == expected


# --------------------------------------------------------------------------- #
# Policy: save path (bind checkpoint_digest to exact block_hashes)
# --------------------------------------------------------------------------- #


class TestSavePath:
    def test_save_authorized_with_arbitrary_digest(self, policy_fixed_time, secret_file, fixed_time):
        """A controller cannot know vLLM block hashes in advance, so
        checkpoint_digest is an independent controller identity digest — it
        is NOT required to equal SHA256(block_hashes)."""
        bh = _block_hashes(3)
        digest = _digest("arbitrary-controller-identity")
        params = build_kv_transfer_params(
            operation="save",
            checkpoint_digest=digest,
            expires_at=fixed_time + 100.0,
            nonce=_nonce(),
            secret_path=secret_file,
        )
        d = policy_fixed_time.evaluate(params, bh, tokens=128)
        assert d.authorized
        assert d.operation == "save"
        assert d.reason_code == "save_authorized"
        assert d.saved_tokens == 128
        assert d.schema_version == DAYSTROM_KV_SCHEMA_VERSION
        record = policy_fixed_time.record_for(digest)
        assert record is not None
        # The digest is bound to the exact ordered block_hashes in the record.
        assert record.block_hashes == tuple(bh)

    def test_save_binds_arbitrary_digest_to_block_hashes(self, policy_fixed_time, secret_file, fixed_time):
        """Even when the digest is completely unrelated to the block hashes,
        the save must succeed and bind the digest to the block_hashes."""
        bh = _block_hashes(3)
        digest = _digest("unrelated-controller-checkpoint-id")
        assert digest != compute_block_hash_digest(bh)
        params = build_kv_transfer_params(
            operation="save",
            checkpoint_digest=digest,
            expires_at=fixed_time + 100.0,
            nonce=_nonce(),
            secret_path=secret_file,
        )
        d = policy_fixed_time.evaluate(params, bh, tokens=64)
        assert d.authorized
        record = policy_fixed_time.record_for(digest)
        assert record is not None
        assert record.block_hashes == tuple(bh)

    def test_save_rejected_when_ttl_exceeds_max(self, policy_fixed_time, secret_file, fixed_time):
        bh = _block_hashes(1)
        digest = _digest("ttl-test")
        params = build_kv_transfer_params(
            operation="save",
            checkpoint_digest=digest,
            expires_at=fixed_time + 10_000.0,  # > 3600
            nonce=_nonce(),
            secret_path=secret_file,
        )
        d = policy_fixed_time.evaluate(params, bh, tokens=10)
        assert not d.authorized
        assert d.reason_code == "ttl_exceeds_max"

    def test_save_rejected_when_expired(self, policy_fixed_time, secret_file, fixed_time):
        bh = _block_hashes(1)
        digest = _digest("expired-test")
        params = build_kv_transfer_params(
            operation="save",
            checkpoint_digest=digest,
            expires_at=fixed_time - 1.0,  # already expired
            nonce=_nonce(),
            secret_path=secret_file,
        )
        d = policy_fixed_time.evaluate(params, bh, tokens=10)
        assert not d.authorized
        assert d.reason_code == "expired"

    def test_save_cardinality_exceeded(self, secret_file, fixed_time):
        pol = DaystromKVPolicy(
            secret_file, max_ttl_seconds=3600, max_records=2, time_fn=lambda: fixed_time
        )
        for i in range(2):
            bh = _block_hashes(1, seed=i)
            digest = _digest(f"ckpt-{i}")
            params = build_kv_transfer_params(
                operation="save",
                checkpoint_digest=digest,
                expires_at=fixed_time + 100.0,
                nonce=f"n{i}",
                secret_path=secret_file,
            )
            d = pol.evaluate(params, bh, tokens=1)
            assert d.authorized
        # Third unique save should be rejected
        bh = _block_hashes(1, seed=99)
        digest = _digest("ckpt-99")
        params = build_kv_transfer_params(
            operation="save",
            checkpoint_digest=digest,
            expires_at=fixed_time + 100.0,
            nonce="n99",
            secret_path=secret_file,
        )
        d = pol.evaluate(params, bh, tokens=1)
        assert not d.authorized
        assert d.reason_code == "cardinality_exceeded"

    def test_save_overwrite_existing_digest_allowed(self, policy_fixed_time, secret_file, fixed_time):
        bh = _block_hashes(2)
        digest = _digest("overwrite-test")
        params = build_kv_transfer_params(
            operation="save",
            checkpoint_digest=digest,
            expires_at=fixed_time + 100.0,
            nonce=_nonce(),
            secret_path=secret_file,
        )
        d1 = policy_fixed_time.evaluate(params, bh, tokens=10)
        assert d1.authorized
        # Re-save same digest (e.g. refresh) should be allowed even at cap.
        d2 = policy_fixed_time.evaluate(params, bh, tokens=20)
        assert d2.authorized


# --------------------------------------------------------------------------- #
# Policy: restore path (record exists, unexpired, block-hash prefix match)
# --------------------------------------------------------------------------- #


class TestRestorePath:
    def test_restore_authorized_when_record_is_prefix_of_request(
        self, policy_fixed_time, secret_file, fixed_time
    ):
        """The saved record's block hashes must be a non-empty exact prefix of
        the new request's block hashes, because the restore request may extend
        the saved prefix."""
        bh = _block_hashes(2)
        digest = _digest("restore-prefix-test")
        save_params = build_kv_transfer_params(
            operation="save",
            checkpoint_digest=digest,
            expires_at=fixed_time + 500.0,
            nonce="save-nonce",
            secret_path=secret_file,
        )
        assert policy_fixed_time.evaluate(save_params, bh, tokens=256).authorized
        # Restore with MORE block hashes (request extends the saved prefix)
        restore_bh = bh + _block_hashes(2, seed=1)
        restore_params = build_kv_transfer_params(
            operation="restore",
            checkpoint_digest=digest,
            expires_at=fixed_time + 500.0,
            nonce="restore-nonce",
            secret_path=secret_file,
        )
        d = policy_fixed_time.evaluate(restore_params, restore_bh)
        assert d.authorized
        assert d.operation == "restore"
        assert d.reason_code == "restore_authorized"
        # Policy returns matched_tokens=0; connector gets native count from parent
        assert d.matched_tokens == 0

    def test_restore_authorized_when_request_equals_record(
        self, policy_fixed_time, secret_file, fixed_time
    ):
        """Restore with exactly the same block hashes is also authorized."""
        bh = _block_hashes(4)
        digest = _digest("restore-equal-test")
        save_params = build_kv_transfer_params(
            operation="save",
            checkpoint_digest=digest,
            expires_at=fixed_time + 500.0,
            nonce="save-nonce",
            secret_path=secret_file,
        )
        assert policy_fixed_time.evaluate(save_params, bh, tokens=256).authorized
        restore_params = build_kv_transfer_params(
            operation="restore",
            checkpoint_digest=digest,
            expires_at=fixed_time + 500.0,
            nonce="restore-nonce",
            secret_path=secret_file,
        )
        d = policy_fixed_time.evaluate(restore_params, bh)
        assert d.authorized
        assert d.reason_code == "restore_authorized"
        assert d.matched_tokens == 0

    def test_restore_rejected_when_record_not_found(self, policy_fixed_time, secret_file, fixed_time):
        bh = _block_hashes(2)
        digest = _digest("not-found-test")
        params = build_kv_transfer_params(
            operation="restore",
            checkpoint_digest=digest,
            expires_at=fixed_time + 100.0,
            nonce=_nonce(),
            secret_path=secret_file,
        )
        d = policy_fixed_time.evaluate(params, bh)
        assert not d.authorized
        assert d.reason_code == "record_not_found"

    def test_restore_rejected_when_record_expired(self, secret_file, fixed_time):
        times = [fixed_time]
        pol = DaystromKVPolicy(
            secret_file, max_ttl_seconds=3600, max_records=16, time_fn=lambda: times[0]
        )
        bh = _block_hashes(2)
        digest = _digest("expired-record-test")
        save_params = build_kv_transfer_params(
            operation="save",
            checkpoint_digest=digest,
            expires_at=fixed_time + 100.0,
            nonce="n1",
            secret_path=secret_file,
        )
        assert pol.evaluate(save_params, bh, tokens=10).authorized
        # Advance time past the record's expiry; the restore request itself
        # must still be unexpired (its own expires_at is far in the future).
        times[0] = fixed_time + 200.0
        restore_params = build_kv_transfer_params(
            operation="restore",
            checkpoint_digest=digest,
            expires_at=fixed_time + 500.0,
            nonce="n2",
            secret_path=secret_file,
        )
        d = pol.evaluate(restore_params, bh)
        assert not d.authorized
        assert d.reason_code == "record_expired"

    def test_restore_rejected_when_request_diverges_from_record_prefix(
        self, policy_fixed_time, secret_file, fixed_time
    ):
        bh = _block_hashes(3)
        digest = _digest("diverge-test")
        save_params = build_kv_transfer_params(
            operation="save",
            checkpoint_digest=digest,
            expires_at=fixed_time + 500.0,
            nonce="sn",
            secret_path=secret_file,
        )
        assert policy_fixed_time.evaluate(save_params, bh, tokens=10).authorized
        # Restore with block hashes that diverge from the saved prefix
        wrong_bh = _block_hashes(2, seed=999)
        restore_params = build_kv_transfer_params(
            operation="restore",
            checkpoint_digest=digest,
            expires_at=fixed_time + 500.0,
            nonce="rn",
            secret_path=secret_file,
        )
        d = policy_fixed_time.evaluate(restore_params, wrong_bh)
        assert not d.authorized
        assert d.reason_code == "block_hash_prefix_mismatch"

    def test_restore_rejected_when_request_shorter_than_record(
        self, policy_fixed_time, secret_file, fixed_time
    ):
        """The record's block hashes must be a prefix of the request's.  If
        the request is shorter than the record, the record cannot be a prefix
        and the restore is rejected."""
        bh = _block_hashes(4)
        digest = _digest("shorter-request-test")
        save_params = build_kv_transfer_params(
            operation="save",
            checkpoint_digest=digest,
            expires_at=fixed_time + 500.0,
            nonce="sn",
            secret_path=secret_file,
        )
        assert policy_fixed_time.evaluate(save_params, bh, tokens=10).authorized
        # Restore request has fewer block hashes than the record
        shorter_bh = bh[:2]
        restore_params = build_kv_transfer_params(
            operation="restore",
            checkpoint_digest=digest,
            expires_at=fixed_time + 500.0,
            nonce="rn",
            secret_path=secret_file,
        )
        d = policy_fixed_time.evaluate(restore_params, shorter_bh)
        assert not d.authorized
        assert d.reason_code == "block_hash_prefix_mismatch"

    def test_restore_authorized_with_extended_prefix(
        self, policy_fixed_time, secret_file, fixed_time
    ):
        """Explicit test: save with 2 blocks, restore with 5 blocks where the
        first 2 match the saved record and the last 3 are new."""
        saved_bh = _block_hashes(2, seed=42)
        digest = _digest("extended-prefix-test")
        save_params = build_kv_transfer_params(
            operation="save",
            checkpoint_digest=digest,
            expires_at=fixed_time + 500.0,
            nonce="sn",
            secret_path=secret_file,
        )
        assert policy_fixed_time.evaluate(save_params, saved_bh, tokens=64).authorized
        # Restore with extended prefix
        restore_bh = saved_bh + _block_hashes(3, seed=99)
        assert len(restore_bh) == 5
        restore_params = build_kv_transfer_params(
            operation="restore",
            checkpoint_digest=digest,
            expires_at=fixed_time + 500.0,
            nonce="rn",
            secret_path=secret_file,
        )
        d = policy_fixed_time.evaluate(restore_params, restore_bh)
        assert d.authorized
        assert d.reason_code == "restore_authorized"

    def test_restore_rejected_when_no_block_hashes(self, policy_fixed_time, secret_file, fixed_time):
        bh = _block_hashes(2)
        digest = _digest("empty-bh-test")
        save_params = build_kv_transfer_params(
            operation="save",
            checkpoint_digest=digest,
            expires_at=fixed_time + 500.0,
            nonce="sn",
            secret_path=secret_file,
        )
        assert policy_fixed_time.evaluate(save_params, bh, tokens=10).authorized
        restore_params = build_kv_transfer_params(
            operation="restore",
            checkpoint_digest=digest,
            expires_at=fixed_time + 500.0,
            nonce="rn",
            secret_path=secret_file,
        )
        d = policy_fixed_time.evaluate(restore_params, [])
        assert not d.authorized
        assert d.reason_code == "block_hash_prefix_mismatch"


# --------------------------------------------------------------------------- #
# Policy: reset_cache (fail-closed, no fake purge)
# --------------------------------------------------------------------------- #


class TestResetCache:
    def test_reset_cache_clears_authorization_index(self, policy_fixed_time, secret_file, fixed_time):
        bh = _block_hashes(2)
        digest = _digest("reset-test")
        params = build_kv_transfer_params(
            operation="save",
            checkpoint_digest=digest,
            expires_at=fixed_time + 100.0,
            nonce=_nonce(),
            secret_path=secret_file,
        )
        assert policy_fixed_time.evaluate(params, bh, tokens=10).authorized
        assert len(policy_fixed_time) == 1
        assert policy_fixed_time.reset_cache() is True
        assert len(policy_fixed_time) == 0
        # Subsequent restore fails closed
        restore_params = build_kv_transfer_params(
            operation="restore",
            checkpoint_digest=digest,
            expires_at=fixed_time + 100.0,
            nonce="r1",
            secret_path=secret_file,
        )
        d = policy_fixed_time.evaluate(restore_params, bh)
        assert not d.authorized
        assert d.reason_code == "record_not_found"


# --------------------------------------------------------------------------- #
# Policy: telemetry is payload-free
# --------------------------------------------------------------------------- #


class TestTelemetry:
    def test_decision_telemetry_has_no_secrets_or_hashes(self, policy_fixed_time, secret_file, fixed_time):
        bh = _block_hashes(2)
        digest = _digest("telemetry-test")
        params = build_kv_transfer_params(
            operation="save",
            checkpoint_digest=digest,
            expires_at=fixed_time + 100.0,
            nonce=_nonce(),
            secret_path=secret_file,
        )
        d = policy_fixed_time.evaluate(params, bh, tokens=64)
        tele = d.telemetry()
        assert set(tele) == {
            "schema_version", "authorized", "operation", "checkpoint_digest",
            "reason_code", "matched_tokens", "saved_tokens",
        }
        # No block-hash bytes, no nonce, no authorization, no secret
        blob = json.dumps(tele)
        assert _nonce() not in blob
        assert "authorization" not in blob
        assert "nonce" not in blob
        assert "block_hashes" not in blob
        # checkpoint_digest is allowed (it's a digest, not prompt/token data)
        assert tele["checkpoint_digest"] == digest
        assert tele["schema_version"] == DAYSTROM_KV_SCHEMA_VERSION

    def test_policy_telemetry_lists_only_digests(self, policy_fixed_time, secret_file, fixed_time):
        bh = _block_hashes(1)
        digest = _digest("policy-tele-test")
        params = build_kv_transfer_params(
            operation="save",
            checkpoint_digest=digest,
            expires_at=fixed_time + 100.0,
            nonce=_nonce(),
            secret_path=secret_file,
        )
        policy_fixed_time.evaluate(params, bh, tokens=1)
        tele = policy_fixed_time.telemetry()
        assert tele["num_records"] == 1
        assert digest in tele["checkpoint_digests"]
        assert "block_hashes" not in tele
        assert "secret" not in json.dumps(tele)

    def test_to_json_telemetry_is_compact(self, secret_file, fixed_time):
        d = DaystromKVDecision(
            authorized=True, operation="save",
            checkpoint_digest=_digest("z"), reason_code="save_authorized",
            saved_tokens=10,
        )
        s = to_json_telemetry(d)
        parsed = json.loads(s)
        assert parsed["authorized"] is True
        assert parsed["saved_tokens"] == 10


# --------------------------------------------------------------------------- #
# Policy: helper functions
# --------------------------------------------------------------------------- #


class TestHelpers:
    def test_compute_block_hash_digest_is_deterministic(self):
        bh = _block_hashes(3)
        d1 = compute_block_hash_digest(bh)
        d2 = compute_block_hash_digest(bh)
        assert d1 == d2
        assert d1.startswith("sha256:")
        assert len(d1) == 7 + 64

    def test_compute_block_hash_digest_order_matters(self):
        bh = _block_hashes(2)
        d1 = compute_block_hash_digest(bh)
        d2 = compute_block_hash_digest(list(reversed(bh)))
        assert d1 != d2

    def test_compute_block_hash_digest_rejects_non_bytes(self):
        with pytest.raises(DaystromKVAuthorizationError, match="block_hash_not_bytes"):
            compute_block_hash_digest(["not", "bytes"])  # type: ignore[list-item]

    def test_build_daystrom_params_validates_operation(self, secret_file):
        with pytest.raises(ValueError):
            build_daystrom_params(
                operation="bad", checkpoint_digest=_digest("x"),
                expires_at=1.0, nonce="n", secret_path=secret_file,
            )

    def test_build_daystrom_params_validates_digest(self, secret_file):
        with pytest.raises(ValueError):
            build_daystrom_params(
                operation="save", checkpoint_digest="nope",
                expires_at=1.0, nonce="n", secret_path=secret_file,
            )


# --------------------------------------------------------------------------- #
# Connector: stub minimal vLLM modules, then test gating
# --------------------------------------------------------------------------- #


def _install_vllm_stubs(monkeypatch, tmp_path: Path) -> None:
    """Install minimal vLLM module stubs into sys.modules so the connector
    module can be imported without vLLM/torch installed."""

    # torch stub
    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = type("Tensor", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    # vllm package
    vllm_mod = types.ModuleType("vllm")
    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)

    # vllm.config
    config_mod = types.ModuleType("vllm.config")

    class VllmConfig:
        def __init__(self, extra_config=None, world_size=1, enable_prefix_caching=True):
            self.parallel_config = types.SimpleNamespace(world_size=world_size)
            self.cache_config = types.SimpleNamespace(enable_prefix_caching=enable_prefix_caching)
            self.kv_transfer_config = types.SimpleNamespace(
                kv_connector_extra_config=extra_config or {},
                kv_connector_name="daystrom-cooperative-cpu-offload",
            )

    config_mod.VllmConfig = VllmConfig
    monkeypatch.setitem(sys.modules, "vllm.config", config_mod)

    # vllm.logger
    logger_mod = types.ModuleType("vllm.logger")

    def init_logger(name):
        import logging
        return logging.getLogger(name)

    logger_mod.init_logger = init_logger
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_mod)

    # vllm.distributed.kv_events
    kv_events_mod = types.ModuleType("vllm.distributed.kv_events")
    kv_events_mod.KVCacheEvent = type("KVCacheEvent", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.distributed.kv_events", kv_events_mod)

    # vllm.distributed package + subpackages
    for pkg in [
        "vllm.distributed",
        "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1",
    ]:
        monkeypatch.setitem(sys.modules, pkg, types.ModuleType(pkg))

    # vllm.distributed.kv_transfer.kv_connector.v1.base
    base_mod = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")

    class KVConnectorMetadata:  # noqa: B024
        pass

    class KVConnectorRole(enum.Enum):
        SCHEDULER = 0
        WORKER = 1

    class SupportsHMA:  # noqa: B024
        pass

    class KVConnectorBase_V1:
        def __init__(self, vllm_config, role, kv_cache_config=None):
            self._vllm_config = vllm_config
            self._kv_transfer_config = vllm_config.kv_transfer_config
            self._kv_cache_config = kv_cache_config
            self._role = role
            self.scheduler_manager = None
            self.worker_handler = None

        @property
        def role(self):
            return self._role

    base_mod.KVConnectorMetadata = KVConnectorMetadata
    base_mod.KVConnectorRole = KVConnectorRole
    base_mod.SupportsHMA = SupportsHMA
    base_mod.KVConnectorBase_V1 = KVConnectorBase_V1
    monkeypatch.setitem(sys.modules, "vllm.distributed.kv_transfer.kv_connector.v1.base", base_mod)

    # vllm.v1 package + subpackages
    for pkg in ["vllm.v1", "vllm.v1.core", "vllm.v1.core.sched", "vllm.v1.simple_kv_offload"]:
        monkeypatch.setitem(sys.modules, pkg, types.ModuleType(pkg))

    # vllm.v1.core.sched.output
    sched_out_mod = types.ModuleType("vllm.v1.core.sched.output")
    sched_out_mod.SchedulerOutput = type("SchedulerOutput", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.v1.core.sched.output", sched_out_mod)

    # vllm.v1.outputs
    outputs_mod = types.ModuleType("vllm.v1.outputs")
    outputs_mod.KVConnectorOutput = type("KVConnectorOutput", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.v1.outputs", outputs_mod)

    # vllm.v1.simple_kv_offload.metadata
    meta_mod = types.ModuleType("vllm.v1.simple_kv_offload.metadata")

    class SimpleCPUOffloadMetadata(KVConnectorMetadata):
        pass

    meta_mod.SimpleCPUOffloadMetadata = SimpleCPUOffloadMetadata
    monkeypatch.setitem(sys.modules, "vllm.v1.simple_kv_offload.metadata", meta_mod)

    # vllm.v1.simple_kv_offload.manager
    mgr_mod = types.ModuleType("vllm.v1.simple_kv_offload.manager")

    class SimpleCPUOffloadScheduler:
        def __init__(self, *args, **kwargs):
            self.calls = []

        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            self.calls.append(("get_num_new_matched_tokens", request, num_computed_tokens))
            return 5, True

        def update_state_after_alloc(self, request, blocks, num_external_tokens):
            self.calls.append(("update_state_after_alloc", request, blocks, num_external_tokens))

        def build_connector_meta(self, scheduler_output):
            self.calls.append(("build_connector_meta", scheduler_output))
            return SimpleCPUOffloadMetadata()

        def request_finished(self, request, block_ids):
            self.calls.append(("request_finished", request, block_ids))
            return False, None

        def request_finished_all_groups(self, request, block_ids):
            self.calls.append(("request_finished_all_groups", request, block_ids))
            return False, None

        def has_pending_stores(self):
            return False

        def take_events(self):
            return []

    mgr_mod.SimpleCPUOffloadScheduler = SimpleCPUOffloadScheduler
    monkeypatch.setitem(sys.modules, "vllm.v1.simple_kv_offload.manager", mgr_mod)

    # vllm.v1.simple_kv_offload.worker
    worker_mod = types.ModuleType("vllm.v1.simple_kv_offload.worker")

    class SimpleCPUOffloadWorker:
        def __init__(self, *args, **kwargs):
            pass

    worker_mod.SimpleCPUOffloadWorker = SimpleCPUOffloadWorker
    monkeypatch.setitem(sys.modules, "vllm.v1.simple_kv_offload.worker", worker_mod)

    # vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector
    conn_mod = types.ModuleType(
        "vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector"
    )

    class SimpleCPUOffloadConnector(KVConnectorBase_V1, SupportsHMA):
        def __init__(self, vllm_config, role, kv_cache_config=None):
            super().__init__(vllm_config, role, kv_cache_config)
            if vllm_config.cache_config.enable_prefix_caching and role == KVConnectorRole.SCHEDULER:
                self.scheduler_manager = SimpleCPUOffloadScheduler(
                    vllm_config, kv_cache_config, 0
                )
            else:
                self.scheduler_manager = None
            self.worker_handler = None

        def reset_cache(self) -> bool | None:
            raise NotImplementedError("SimpleCPUOffloadConnector does not support reset_cache().")

    conn_mod.SimpleCPUOffloadConnector = SimpleCPUOffloadConnector
    conn_mod.DEFAULT_CPU_CAPACITY_BYTES = 8 * (1024**3)
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector",
        conn_mod,
    )


class _FakeRequest:
    def __init__(self, request_id, kv_transfer_params=None, block_hashes=None, num_computed_tokens=0):
        self.request_id = request_id
        self.kv_transfer_params = kv_transfer_params
        self.block_hashes = block_hashes or []
        self.num_computed_tokens = num_computed_tokens


@pytest.fixture()
def connector_env(monkeypatch, tmp_path, secret_file):
    """Install vLLM stubs and build a connector instance."""

    _install_vllm_stubs(monkeypatch, tmp_path)
    # Import connector module AFTER stubs installed.
    # Remove any cached module first.
    sys.modules.pop("daystrom_dml.context.vllm_bridge.connector", None)
    import daystrom_dml.context.vllm_bridge.connector as conn_mod

    extra_config = {
        "daystrom_secret_path": str(secret_file),
        "daystrom_max_ttl_seconds": 3600,
        "daystrom_max_records": 16,
    }
    # Build a stub VllmConfig
    VllmConfig = sys.modules["vllm.config"].VllmConfig
    KVConnectorRole = sys.modules["vllm.distributed.kv_transfer.kv_connector.v1.base"].KVConnectorRole
    cfg = VllmConfig(extra_config=extra_config)
    connector = conn_mod.DaystromCooperativeKVConnector(cfg, KVConnectorRole.SCHEDULER)
    return connector, conn_mod


class TestConnectorGating:
    def test_get_num_new_matched_tokens_unapproved_returns_zero(self, connector_env, secret_file, fixed_time):
        connector, _ = connector_env
        connector.policy._time_fn = lambda: fixed_time  # type: ignore[attr-defined]
        req = _FakeRequest("r1", kv_transfer_params=None)
        matched, is_async = connector.get_num_new_matched_tokens(req, 0)
        assert matched == 0
        assert is_async is False

    def test_get_num_new_matched_tokens_approved_delegates_to_parent(
        self, connector_env, secret_file, fixed_time
    ):
        """For restore, delegate to parent and use parent's native token count
        directly — do NOT treat count of block hashes as token count."""
        connector, _ = connector_env
        connector.policy._time_fn = lambda: fixed_time  # type: ignore[attr-defined]
        bh = _block_hashes(2)
        digest = _digest("connector-restore-test")
        # First save
        save_params = build_kv_transfer_params(
            operation="save", checkpoint_digest=digest,
            expires_at=fixed_time + 500.0, nonce="sn",
            secret_path=secret_file,
        )
        save_req = _FakeRequest("r-save", kv_transfer_params=save_params, block_hashes=bh, num_computed_tokens=256)
        connector.request_finished(save_req, [1, 2])
        # Now restore with extended prefix (record is prefix of request)
        restore_bh = bh + _block_hashes(2, seed=1)
        restore_params = build_kv_transfer_params(
            operation="restore", checkpoint_digest=digest,
            expires_at=fixed_time + 500.0, nonce="rn",
            secret_path=secret_file,
        )
        restore_req = _FakeRequest("r-restore", kv_transfer_params=restore_params, block_hashes=restore_bh)
        matched, is_async = connector.get_num_new_matched_tokens(restore_req, 0)
        # Parent stub returns (5, True); connector uses parent's native count directly
        assert matched == 5
        assert is_async is True
        # Telemetry should reflect the native count, not block-hash count
        tele = connector.take_telemetry()
        assert tele["r-restore"]["matched_tokens"] == 5

    def test_update_state_after_alloc_unapproved_is_noop(self, connector_env):
        connector, _ = connector_env
        req = _FakeRequest("r2", kv_transfer_params=None)
        connector.update_state_after_alloc(req, blocks=[], num_external_tokens=10)
        # Parent scheduler_manager should not have been called
        calls = connector.scheduler_manager.calls
        assert not any(c[0] == "update_state_after_alloc" for c in calls)

    def test_update_state_after_alloc_approved_save_delegates_to_parent(
        self, connector_env, secret_file, fixed_time
    ):
        """Approved SAVE requests must be delegated to the parent scheduler
        manager's update_state_after_alloc so eager mode creates _reqs_to_store
        entries.  num_external_tokens is normally 0 for save."""
        connector, _ = connector_env
        connector.policy._time_fn = lambda: fixed_time  # type: ignore[attr-defined]
        bh = _block_hashes(2)
        digest = _digest("save-delegation-test")
        params = build_kv_transfer_params(
            operation="save", checkpoint_digest=digest,
            expires_at=fixed_time + 100.0, nonce="n1",
            secret_path=secret_file,
        )
        req = _FakeRequest("r-save-alloc", kv_transfer_params=params, block_hashes=bh, num_computed_tokens=64)
        connector.update_state_after_alloc(req, blocks=[], num_external_tokens=0)
        # Parent scheduler_manager MUST have been called for approved save
        calls = connector.scheduler_manager.calls
        alloc_calls = [c for c in calls if c[0] == "update_state_after_alloc"]
        assert len(alloc_calls) == 1
        assert alloc_calls[0][1] is req

    def test_update_state_after_alloc_unapproved_save_is_noop(
        self, connector_env
    ):
        """Unapproved SAVE requests must NOT be delegated to the parent."""
        connector, _ = connector_env
        req = _FakeRequest("r-unapproved-save", kv_transfer_params=None)
        connector.update_state_after_alloc(req, blocks=[], num_external_tokens=0)
        calls = connector.scheduler_manager.calls
        assert not any(c[0] == "update_state_after_alloc" for c in calls)

    def test_request_finished_unapproved_returns_false_none(self, connector_env):
        connector, _ = connector_env
        req = _FakeRequest("r3", kv_transfer_params=None, num_computed_tokens=10)
        result = connector.request_finished(req, [1, 2])
        assert result == (False, None)

    def test_request_finished_all_groups_unapproved_returns_false_none(self, connector_env):
        connector, _ = connector_env
        req = _FakeRequest("r4", kv_transfer_params=None, num_computed_tokens=10)
        result = connector.request_finished_all_groups(req, ([1], [2]))
        assert result == (False, None)

    def test_request_finished_approved_delegates_save(
        self, connector_env, secret_file, fixed_time
    ):
        """Approved save delegates to parent and returns daystrom response meta."""
        connector, _ = connector_env
        connector.policy._time_fn = lambda: fixed_time  # type: ignore[attr-defined]
        bh = _block_hashes(2)
        digest = _digest("save-finished-test")
        params = build_kv_transfer_params(
            operation="save", checkpoint_digest=digest,
            expires_at=fixed_time + 100.0, nonce="n1",
            secret_path=secret_file,
        )
        req = _FakeRequest("r5", kv_transfer_params=params, block_hashes=bh, num_computed_tokens=64)
        ok, meta = connector.request_finished(req, [1, 2])
        # Parent returns (False, None) in real vLLM 0.20
        assert ok is False
        # Parent was called
        calls = connector.scheduler_manager.calls
        assert any(c[0] == "request_finished" for c in calls)
        # Response meta has daystrom key with payload-free telemetry
        assert meta is not None
        assert "daystrom" in meta
        daystrom = meta["daystrom"]
        assert daystrom["schema_version"] == DAYSTROM_KV_SCHEMA_VERSION
        assert daystrom["operation"] == "save"
        assert daystrom["checkpoint_digest"] == digest
        assert daystrom["reason_code"] == "save_authorized"
        assert daystrom["saved_tokens"] == 64  # num_computed_tokens
        assert daystrom["matched_tokens"] == 0
        # No secrets or sensitive data
        blob = json.dumps(meta)
        assert "n1" not in blob  # nonce
        assert "authorization" not in blob
        assert "block_hashes" not in blob
        assert "request_id" not in blob

    def test_request_finished_all_groups_approved_returns_daystrom_meta(
        self, connector_env, secret_file, fixed_time
    ):
        """request_finished_all_groups also returns daystrom response meta."""
        connector, _ = connector_env
        connector.policy._time_fn = lambda: fixed_time  # type: ignore[attr-defined]
        bh = _block_hashes(2)
        digest = _digest("save-all-groups-test")
        params = build_kv_transfer_params(
            operation="save", checkpoint_digest=digest,
            expires_at=fixed_time + 100.0, nonce="n1",
            secret_path=secret_file,
        )
        req = _FakeRequest("r5g", kv_transfer_params=params, block_hashes=bh, num_computed_tokens=32)
        ok, meta = connector.request_finished_all_groups(req, ([1], [2]))
        assert ok is False
        assert meta is not None
        assert "daystrom" in meta
        assert meta["daystrom"]["operation"] == "save"
        assert meta["daystrom"]["saved_tokens"] == 32
        calls = connector.scheduler_manager.calls
        assert any(c[0] == "request_finished_all_groups" for c in calls)

    def test_reset_cache_fails_closed_and_clears_policy(self, connector_env):
        connector, _ = connector_env
        with pytest.raises(NotImplementedError, match="not safely implemented"):
            connector.reset_cache()
        # Policy index was cleared
        assert len(connector.policy) == 0

    def test_take_telemetry_is_payload_free(self, connector_env, secret_file, fixed_time):
        connector, _ = connector_env
        connector.policy._time_fn = lambda: fixed_time  # type: ignore[attr-defined]
        bh = _block_hashes(1)
        digest = _digest("tele-test")
        params = build_kv_transfer_params(
            operation="save", checkpoint_digest=digest,
            expires_at=fixed_time + 100.0, nonce="secret-nonce",
            secret_path=secret_file,
        )
        req = _FakeRequest("r6", kv_transfer_params=params, block_hashes=bh, num_computed_tokens=32)
        connector.request_finished(req, [1])
        tele = connector.take_telemetry()
        assert "r6" in tele
        blob = json.dumps(tele)
        assert "secret-nonce" not in blob
        assert "authorization" not in blob
        assert "block_hashes" not in blob
        assert tele["r6"]["checkpoint_digest"] == digest
        assert tele["r6"]["operation"] == "save"
        assert tele["r6"]["schema_version"] == DAYSTROM_KV_SCHEMA_VERSION

    def test_connector_is_subclass_of_simple_cpu_offload_and_supports_hma(self, connector_env):
        connector, conn_mod = connector_env
        base_mod = sys.modules["vllm.distributed.kv_transfer.kv_connector.v1.base"]
        simple_mod = sys.modules[
            "vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector"
        ]
        assert issubclass(conn_mod.DaystromCooperativeKVConnector, simple_mod.SimpleCPUOffloadConnector)
        assert issubclass(conn_mod.DaystromCooperativeKVConnector, base_mod.SupportsHMA)

    def test_restore_finish_delegates_cleanup_and_returns_native_hit_telemetry(
        self, connector_env, secret_file, fixed_time
    ):
        connector, _ = connector_env
        connector.policy._time_fn = lambda: fixed_time  # type: ignore[attr-defined]
        saved_hashes = _block_hashes(2)
        checkpoint_digest = _digest("controller-checkpoint")
        save_params = build_kv_transfer_params(
            operation="save",
            checkpoint_digest=checkpoint_digest,
            expires_at=fixed_time + 500.0,
            nonce="save-finish",
            secret_path=secret_file,
        )
        save_request = _FakeRequest(
            "save-finish",
            kv_transfer_params=save_params,
            block_hashes=saved_hashes,
            num_computed_tokens=64,
        )
        connector.update_state_after_alloc(save_request, blocks=[], num_external_tokens=0)
        connector.request_finished(save_request, [1, 2])

        restore_params = build_kv_transfer_params(
            operation="restore",
            checkpoint_digest=checkpoint_digest,
            expires_at=fixed_time + 500.0,
            nonce="restore-finish",
            secret_path=secret_file,
        )
        restore_request = _FakeRequest(
            "restore-finish",
            kv_transfer_params=restore_params,
            block_hashes=saved_hashes + _block_hashes(1, seed=99),
            num_computed_tokens=64,
        )
        assert connector.get_num_new_matched_tokens(restore_request, 0) == (5, True)
        retain, response_meta = connector.request_finished(restore_request, [3, 4])

        assert retain is False
        assert response_meta is not None
        assert response_meta["daystrom"]["operation"] == "restore"
        assert response_meta["daystrom"]["matched_tokens"] == 5
        assert any(
            call[0] == "request_finished" and call[1] is restore_request
            for call in connector.scheduler_manager.calls
        )
