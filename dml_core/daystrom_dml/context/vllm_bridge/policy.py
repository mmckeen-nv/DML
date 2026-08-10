"""Runtime-neutral authorization policy for the Daystrom vLLM KV connector.

This module is deliberately pure: it imports only the Python standard library
and the dependency-light ``daystrom_dml.api_contracts`` helpers.  It does NOT
import vLLM, torch, or any GPU/runtime code, so it is importable and testable
on macOS/Windows without vLLM installed.

Contract
--------
A request's ``kv_transfer_params`` carries a nested ``daystrom`` mapping with:

* ``schema_version`` == ``"daystrom-vllm-kv-v1"``
* ``operation`` in ``{"save", "restore", "purge", "status"}``
* ``checkpoint_digest`` == ``"sha256:" + 64 hex chars``
* ``expires_at`` finite epoch seconds (int/float)
* ``nonce`` bounded opaque string (1..MAX_NONCE_LEN, printable ASCII)
* ``authorization`` HMAC-SHA256 hex digest over the canonical message

The canonical HMAC message is the exact ordered concatenation of:
``schema_version | operation | checkpoint_digest | expires_at | nonce``
using a single ``\\n`` separator, ASCII-encoded.

The HMAC secret is loaded from a configured file path (read at policy
construction).  It must never come from CLI args, environment variables, or a
raw artifact embedded in the request.

Policy enforcement
------------------
* Cap TTL (``expires_at - now``) at ``max_ttl_seconds``.
* Cap cardinality of live records at ``max_records``.
* Cap field lengths (nonce, digest, schema version).
* Reject unknown or malformed fields (no extra keys in the ``daystrom`` map).
* On ``save``: bind the independent controller ``checkpoint_digest`` to the
  exact ordered ``request.block_hashes`` in the in-memory record.  The digest
  is NOT required to equal ``SHA256(block_hashes)`` — a controller cannot know
  vLLM block hashes in advance.  ``compute_block_hash_digest`` is kept only as
  an internal/telemetry helper, never as an API precondition.  Record the entry.
* On ``restore``: authorize (HMAC), require the record exists and is
  unexpired, and require the saved record's block-hash prefix to be a
  non-empty exact prefix of the new request's block hashes (the restore
  request may extend the saved prefix).  Reject when the request is shorter
  or diverges.  The matched token count is NOT derived from block-hash count;
  the connector obtains the native count from the parent scheduler manager.
* ``reset_cache`` fails closed: it does NOT claim physical purge.  The policy
  only drops its in-memory authorization index; physical deletion is the
  connector's responsibility and is not faked here.

Telemetry
---------
Decisions expose payload-free telemetry only: checkpoint digest, operation,
matched/saved token counts.  No prompt text, token ids, block-hash bytes, or
secrets are exposed.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DAYSTROM_KV_SCHEMA_VERSION = "daystrom-vllm-kv-v1"
DAYSTROM_KV_TRANSITION_SCHEMA_VERSION = "daystrom-vllm-kv-transition-v1"
_VALID_OPERATIONS = frozenset({"save", "restore", "purge", "status"})
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
# Printable ASCII, no control chars, no whitespace-only.
_NONCE_RE = re.compile(r"^[!-~]+$")

MAX_SCHEMA_VERSION_LEN = 64
MAX_NONCE_LEN = 128
MAX_AUTHORIZATION_LEN = 128
MAX_OPERATION_LEN = 16
MAX_CHECKPOINT_DIGEST_LEN = 80  # "sha256:" + 64

DEFAULT_MAX_TTL_SECONDS = 3600
DEFAULT_MAX_RECORDS = 4096

_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "operation",
        "checkpoint_digest",
        "expires_at",
        "nonce",
        "authorization",
    }
)
_TRANSITION_REQUIRED_FIELDS = _REQUIRED_FIELDS | {"child_checkpoint_digest"}


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class DaystromKVAuthorizationError(Exception):
    """Fail-closed authorization error for Daystrom KV transfer requests.

    ``reason_code`` is an inspectable, payload-free string.  No prompt text,
    token ids, block-hash bytes, or secrets are included in the message.
    """

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DaystromKVRequest:
    """Parsed, validated Daystrom KV transfer request envelope."""

    schema_version: str
    operation: str
    checkpoint_digest: str
    expires_at: float
    nonce: str
    authorization: str
    block_hashes: tuple[bytes, ...]
    child_checkpoint_digest: str = ""

    @property
    def canonical_message(self) -> bytes:
        """Exact ordered canonical message covered by the HMAC."""

        fields = [self.schema_version, self.operation, self.checkpoint_digest]
        if self.operation == "transition":
            fields.append(self.child_checkpoint_digest)
        fields.extend([f"{self.expires_at:.6f}", self.nonce])
        return "\n".join(fields).encode("ascii")

    def telemetry(self) -> dict[str, Any]:
        """Payload-free telemetry for this request (no secrets/hashes)."""

        return {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "checkpoint_digest": self.checkpoint_digest,
            "num_block_hashes": len(self.block_hashes),
        }


@dataclass
class DaystromKVRecord:
    """In-memory authorization record for a saved checkpoint."""

    checkpoint_digest: str
    block_hashes: tuple[bytes, ...]
    expires_at: float
    tokens_saved: int = 0
    created_at: float = field(default_factory=time.time)

    def is_expired(self, now: float) -> bool:
        return now >= self.expires_at

    def telemetry(self) -> dict[str, Any]:
        return {
            "checkpoint_digest": self.checkpoint_digest,
            "num_block_hashes": len(self.block_hashes),
            "tokens_saved": self.tokens_saved,
            "expires_at": self.expires_at,
        }


@dataclass
class DaystromKVPurgeState:
    """Payload-free lifecycle state for one selective physical purge."""

    checkpoint_digest: str
    purge_event: int
    blocks_scheduled: int
    shared_blocks: int
    shared_hashes: tuple[bytes, ...] = ()
    completed: bool = False
    blocks_zeroed: int = 0
    bytes_zeroed: int = 0


@dataclass
class DaystromKVDecision:
    """Outcome of a policy evaluation. Payload-free telemetry only."""

    authorized: bool
    operation: str
    checkpoint_digest: str
    reason_code: str
    schema_version: str = ""
    matched_tokens: int = 0
    saved_tokens: int = 0
    purged_blocks: int = 0
    purged_bytes: int = 0
    shared_blocks: int = 0
    child_checkpoint_digest: str = ""

    def telemetry(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "authorized": self.authorized,
            "operation": self.operation,
            "checkpoint_digest": self.checkpoint_digest,
            "reason_code": self.reason_code,
            "matched_tokens": self.matched_tokens,
            "saved_tokens": self.saved_tokens,
            "purged_blocks": self.purged_blocks,
            "purged_bytes": self.purged_bytes,
            "shared_blocks": self.shared_blocks,
            **(
                {"child_checkpoint_digest": self.child_checkpoint_digest}
                if self.child_checkpoint_digest
                else {}
            ),
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def compute_block_hash_digest(block_hashes: Sequence[bytes]) -> str:
    """Compute the canonical checkpoint digest from ordered block hashes.

    The digest is ``sha256:`` + the SHA-256 of the raw concatenation of the
    block-hash bytes in order.  Block hashes are vLLM ``BlockHash`` (``bytes``)
    values; their byte length is fixed by the block hasher.
    """

    hasher = hashlib.sha256()
    for bh in block_hashes:
        if not isinstance(bh, (bytes, bytearray)):
            raise DaystromKVAuthorizationError("block_hash_not_bytes")
        hasher.update(bh)
    return "sha256:" + hasher.hexdigest()


def compute_authorization(secret: bytes, request: DaystromKVRequest) -> str:
    """Compute the expected HMAC-SHA256 hex digest for a request."""

    return hmac.new(secret, request.canonical_message, hashlib.sha256).hexdigest()


def _load_secret(secret_path: Path) -> bytes:
    """Load the HMAC secret from a configured file path.

    The secret must come from a file, never CLI/env/raw artifact.  We strip
    trailing whitespace and require a non-empty secret.
    """

    if not isinstance(secret_path, Path):
        raise DaystromKVAuthorizationError("secret_path_not_path")
    if not secret_path.is_absolute():
        raise DaystromKVAuthorizationError("secret_path_not_absolute")
    if secret_path.is_symlink():
        raise DaystromKVAuthorizationError("secret_path_is_symlink")
    if not secret_path.is_file():
        raise DaystromKVAuthorizationError("secret_file_missing")
    try:
        raw = secret_path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - defensive
        raise DaystromKVAuthorizationError("secret_file_unreadable") from exc
    secret = raw.strip()
    if not secret:
        raise DaystromKVAuthorizationError("secret_empty")
    return secret.encode("utf-8")


def _parse_daystrom_mapping(
    kv_transfer_params: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Extract and validate the nested ``daystrom`` mapping."""

    if kv_transfer_params is None:
        raise DaystromKVAuthorizationError("missing_kv_transfer_params")
    if not isinstance(kv_transfer_params, Mapping):
        raise DaystromKVAuthorizationError("kv_transfer_params_not_mapping")
    daystrom = kv_transfer_params.get("daystrom")
    if daystrom is None:
        raise DaystromKVAuthorizationError("missing_daystrom_mapping")
    if not isinstance(daystrom, Mapping):
        raise DaystromKVAuthorizationError("daystrom_not_mapping")
    return daystrom


def _validate_fields(daystrom: Mapping[str, Any]) -> dict[str, Any]:
    """Validate field presence, types, lengths, and reject unknown fields."""

    operation_value = daystrom.get("operation")
    expected_fields = (
        _TRANSITION_REQUIRED_FIELDS
        if operation_value == "transition"
        else _REQUIRED_FIELDS
    )
    unknown = set(daystrom) - expected_fields
    if unknown:
        raise DaystromKVAuthorizationError("unknown_field")
    missing = expected_fields - set(daystrom)
    if missing:
        raise DaystromKVAuthorizationError("missing_field")

    schema_version = daystrom["schema_version"]
    if not isinstance(schema_version, str):
        raise DaystromKVAuthorizationError("schema_version_not_str")
    expected_schema = (
        DAYSTROM_KV_TRANSITION_SCHEMA_VERSION
        if operation_value == "transition"
        else DAYSTROM_KV_SCHEMA_VERSION
    )
    if schema_version != expected_schema:
        raise DaystromKVAuthorizationError("schema_version_mismatch")
    if len(schema_version) > MAX_SCHEMA_VERSION_LEN:
        raise DaystromKVAuthorizationError("schema_version_too_long")

    operation = daystrom["operation"]
    if not isinstance(operation, str):
        raise DaystromKVAuthorizationError("operation_not_str")
    if len(operation) > MAX_OPERATION_LEN:
        raise DaystromKVAuthorizationError("operation_too_long")
    if operation not in _VALID_OPERATIONS | {"transition"}:
        raise DaystromKVAuthorizationError("operation_invalid")

    checkpoint_digest = daystrom["checkpoint_digest"]
    if not isinstance(checkpoint_digest, str):
        raise DaystromKVAuthorizationError("checkpoint_digest_not_str")
    if len(checkpoint_digest) > MAX_CHECKPOINT_DIGEST_LEN:
        raise DaystromKVAuthorizationError("checkpoint_digest_too_long")
    if not _DIGEST_RE.match(checkpoint_digest):
        raise DaystromKVAuthorizationError("checkpoint_digest_malformed")

    child_checkpoint_digest = ""
    if operation == "transition":
        child_checkpoint_digest = daystrom["child_checkpoint_digest"]
        if not isinstance(child_checkpoint_digest, str):
            raise DaystromKVAuthorizationError("child_checkpoint_digest_not_str")
        if len(child_checkpoint_digest) > MAX_CHECKPOINT_DIGEST_LEN:
            raise DaystromKVAuthorizationError("child_checkpoint_digest_too_long")
        if not _DIGEST_RE.match(child_checkpoint_digest):
            raise DaystromKVAuthorizationError("child_checkpoint_digest_malformed")
        if child_checkpoint_digest == checkpoint_digest:
            raise DaystromKVAuthorizationError("transition_checkpoint_identity_reused")

    expires_at = daystrom["expires_at"]
    if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)):
        raise DaystromKVAuthorizationError("expires_at_not_number")
    expires_at = float(expires_at)
    if not _is_finite(expires_at):
        raise DaystromKVAuthorizationError("expires_at_not_finite")
    if expires_at <= 0:
        raise DaystromKVAuthorizationError("expires_at_non_positive")

    nonce = daystrom["nonce"]
    if not isinstance(nonce, str):
        raise DaystromKVAuthorizationError("nonce_not_str")
    if not (1 <= len(nonce) <= MAX_NONCE_LEN):
        raise DaystromKVAuthorizationError("nonce_length_out_of_range")
    if not _NONCE_RE.match(nonce):
        raise DaystromKVAuthorizationError("nonce_not_printable_ascii")

    authorization = daystrom["authorization"]
    if not isinstance(authorization, str):
        raise DaystromKVAuthorizationError("authorization_not_str")
    if not (1 <= len(authorization) <= MAX_AUTHORIZATION_LEN):
        raise DaystromKVAuthorizationError("authorization_length_out_of_range")
    if not _HEX64_RE.match(authorization):
        raise DaystromKVAuthorizationError("authorization_not_hex64")

    return {
        "schema_version": schema_version,
        "operation": operation,
        "checkpoint_digest": checkpoint_digest,
        "child_checkpoint_digest": child_checkpoint_digest,
        "expires_at": expires_at,
        "nonce": nonce,
        "authorization": authorization,
    }


def _is_finite(value: float) -> bool:
    import math

    return math.isfinite(value)


def _parse_request(
    kv_transfer_params: Mapping[str, Any] | None,
    block_hashes: Sequence[bytes] | None,
) -> DaystromKVRequest:
    """Parse and structurally validate a request envelope."""

    daystrom = _parse_daystrom_mapping(kv_transfer_params)
    fields = _validate_fields(daystrom)

    if block_hashes is None:
        bh: tuple[bytes, ...] = ()
    elif isinstance(block_hashes, (list, tuple)):
        bh = tuple(bytes(b) for b in block_hashes)
    else:
        raise DaystromKVAuthorizationError("block_hashes_not_sequence")
    for b in bh:
        if not isinstance(b, (bytes, bytearray)):
            raise DaystromKVAuthorizationError("block_hash_not_bytes")

    return DaystromKVRequest(
        schema_version=fields["schema_version"],
        operation=fields["operation"],
        checkpoint_digest=fields["checkpoint_digest"],
        expires_at=fields["expires_at"],
        nonce=fields["nonce"],
        authorization=fields["authorization"],
        block_hashes=bh,
        child_checkpoint_digest=fields["child_checkpoint_digest"],
    )


def _verify_authorization(secret: bytes, request: DaystromKVRequest) -> None:
    expected = compute_authorization(secret, request)
    if not hmac.compare_digest(expected, request.authorization):
        raise DaystromKVAuthorizationError("authorization_hmac_mismatch")


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #


class DaystromKVPolicy:
    """Controller-gated, runtime-neutral authorization policy.

    The policy is constructed with an absolute path to a secret file and
    optional limits.  It maintains an in-memory authorization index of saved
    checkpoint records.  All decisions are fail-closed.
    """

    def __init__(
        self,
        secret_path: Path,
        *,
        max_ttl_seconds: int = DEFAULT_MAX_TTL_SECONDS,
        max_records: int = DEFAULT_MAX_RECORDS,
        time_fn: Any = time.time,
    ) -> None:
        if not isinstance(max_ttl_seconds, int) or max_ttl_seconds <= 0:
            raise ValueError("max_ttl_seconds must be a positive integer")
        if not isinstance(max_records, int) or max_records <= 0:
            raise ValueError("max_records must be a positive integer")
        self._secret = _load_secret(Path(secret_path))
        self._max_ttl_seconds = max_ttl_seconds
        self._max_records = max_records
        self._time_fn = time_fn
        self._records: dict[str, DaystromKVRecord] = {}
        self._purges: dict[str, DaystromKVPurgeState] = {}
        self._purge_events: dict[int, str] = {}

    # -- introspection ------------------------------------------------------ #

    @property
    def max_ttl_seconds(self) -> int:
        return self._max_ttl_seconds

    @property
    def max_records(self) -> int:
        return self._max_records

    def __len__(self) -> int:
        return len(self._records)

    def record_for(self, checkpoint_digest: str) -> DaystromKVRecord | None:
        return self._records.get(checkpoint_digest)

    # -- enforcement -------------------------------------------------------- #

    def evaluate(
        self,
        kv_transfer_params: Mapping[str, Any] | None,
        block_hashes: Sequence[bytes] | None,
        *,
        tokens: int = 0,
    ) -> DaystromKVDecision:
        """Evaluate a save or restore request and return a decision.

        On ``save``: the HMAC must verify, the TTL must be within bounds, and
        cardinality must be within bounds.  The checkpoint_digest is an
        independent controller identity digest — it is NOT required to equal
        SHA256(block_hashes).  On success the digest is bound to the exact
        ordered ``block_hashes`` and a record is stored.

        On ``restore``: the HMAC must verify, the record must exist and be
        unexpired, and the saved record's block-hash prefix must be a non-empty
        exact prefix of the request's block hashes (the restore request may
        extend the saved prefix).

        ``tokens`` is the count of tokens to save (save) or that were matched
        (restore); it is used only for telemetry.  The policy does NOT treat
        the count of block hashes as a token count — the connector obtains the
        native matched token count from the parent scheduler manager.
        """

        now = float(self._time_fn())
        try:
            request = _parse_request(kv_transfer_params, block_hashes)
            _verify_authorization(self._secret, request)
        except DaystromKVAuthorizationError as exc:
            return DaystromKVDecision(
                authorized=False,
                operation="unknown",
                checkpoint_digest="",
                reason_code=exc.reason_code,
            )

        ttl = request.expires_at - now
        if ttl <= 0:
            self._evict_expired(now)
            return DaystromKVDecision(
                authorized=False,
                operation=request.operation,
                checkpoint_digest=request.checkpoint_digest,
                reason_code="expired",
                schema_version=request.schema_version,
            )
        if ttl > self._max_ttl_seconds:
            return DaystromKVDecision(
                authorized=False,
                operation=request.operation,
                checkpoint_digest=request.checkpoint_digest,
                reason_code="ttl_exceeds_max",
                schema_version=request.schema_version,
            )

        if request.operation == "save":
            return self._evaluate_save(request, now, tokens)
        if request.operation == "restore":
            return self._evaluate_restore(request, now, tokens)
        if request.operation == "transition":
            return self._evaluate_transition(request, now, tokens)
        if request.operation == "status":
            return self._evaluate_status(request, now)
        return self._evaluate_purge(request, now)

    def _evaluate_save(
        self, request: DaystromKVRequest, now: float, tokens: int
    ) -> DaystromKVDecision:
        # The checkpoint_digest is an independent controller identity digest.
        # A controller cannot know vLLM block hashes in advance, so we do NOT
        # require checkpoint_digest to equal SHA256(block_hashes).  On authorized
        # save we bind the digest to the exact ordered request.block_hashes in
        # the in-memory record.  compute_block_hash_digest is kept only as an
        # internal/telemetry helper, never as an API precondition.
        self._evict_expired(now)
        purge = self._purges.get(request.checkpoint_digest)
        if purge is not None and not purge.completed:
            return DaystromKVDecision(
                authorized=False,
                operation="save",
                checkpoint_digest=request.checkpoint_digest,
                reason_code="purge_pending",
                schema_version=request.schema_version,
            )
        if len(self._records) >= self._max_records and request.checkpoint_digest not in self._records:
            return DaystromKVDecision(
                authorized=False,
                operation="save",
                checkpoint_digest=request.checkpoint_digest,
                reason_code="cardinality_exceeded",
                schema_version=request.schema_version,
            )
        record = DaystromKVRecord(
            checkpoint_digest=request.checkpoint_digest,
            block_hashes=request.block_hashes,
            expires_at=request.expires_at,
            tokens_saved=tokens,
            created_at=now,
        )
        self._records[request.checkpoint_digest] = record
        previous = self._purges.pop(request.checkpoint_digest, None)
        if previous is not None:
            self._purge_events.pop(previous.purge_event, None)
        return DaystromKVDecision(
            authorized=True,
            operation="save",
            checkpoint_digest=request.checkpoint_digest,
            reason_code="save_authorized",
            schema_version=request.schema_version,
            saved_tokens=tokens,
        )

    def _evaluate_restore(
        self, request: DaystromKVRequest, now: float, tokens: int
    ) -> DaystromKVDecision:
        purge = self._purges.get(request.checkpoint_digest)
        if purge is not None:
            return DaystromKVDecision(
                authorized=False,
                operation="restore",
                checkpoint_digest=request.checkpoint_digest,
                reason_code="purge_complete" if purge.completed else "purge_pending",
                schema_version=request.schema_version,
                purged_blocks=purge.blocks_zeroed,
                purged_bytes=purge.bytes_zeroed,
                shared_blocks=purge.shared_blocks,
            )
        record = self._records.get(request.checkpoint_digest)
        if record is None:
            return DaystromKVDecision(
                authorized=False,
                operation="restore",
                checkpoint_digest=request.checkpoint_digest,
                reason_code="record_not_found",
                schema_version=request.schema_version,
            )
        if record.is_expired(now):
            del self._records[request.checkpoint_digest]
            return DaystromKVDecision(
                authorized=False,
                operation="restore",
                checkpoint_digest=request.checkpoint_digest,
                reason_code="record_expired",
                schema_version=request.schema_version,
            )
        # The saved record's block hashes must be a non-empty exact prefix of
        # the new request's block hashes, because the restore request may
        # extend the saved prefix.  Reject when the request is shorter or
        # diverges from the stored prefix.
        if not _block_hash_prefix_match(record.block_hashes, request.block_hashes):
            return DaystromKVDecision(
                authorized=False,
                operation="restore",
                checkpoint_digest=request.checkpoint_digest,
                reason_code="block_hash_prefix_mismatch",
                schema_version=request.schema_version,
            )
        # Do NOT treat the count of block hashes as a token count.  The
        # connector obtains the native matched token count from the parent
        # scheduler manager and updates the decision telemetry with that
        # actual count.  The policy returns matched_tokens=0 here.
        return DaystromKVDecision(
            authorized=True,
            operation="restore",
            checkpoint_digest=request.checkpoint_digest,
            reason_code="restore_authorized",
            schema_version=request.schema_version,
            matched_tokens=0,
        )

    def _evaluate_transition(
        self, request: DaystromKVRequest, now: float, tokens: int
    ) -> DaystromKVDecision:
        """Authorize one source restore and distinct child save as one request."""

        restored = self._evaluate_restore(request, now, tokens)
        child = request.child_checkpoint_digest
        if not restored.authorized:
            return DaystromKVDecision(
                authorized=False,
                operation="transition",
                checkpoint_digest=request.checkpoint_digest,
                child_checkpoint_digest=child,
                reason_code=restored.reason_code,
                schema_version=request.schema_version,
            )
        self._evict_expired(now)
        purge = self._purges.get(child)
        if purge is not None:
            return DaystromKVDecision(
                authorized=False,
                operation="transition",
                checkpoint_digest=request.checkpoint_digest,
                child_checkpoint_digest=child,
                reason_code=(
                    "child_purge_complete" if purge.completed else "child_purge_pending"
                ),
                schema_version=request.schema_version,
            )
        existing = self._records.get(child)
        if existing is not None and existing.block_hashes != request.block_hashes:
            return DaystromKVDecision(
                authorized=False,
                operation="transition",
                checkpoint_digest=request.checkpoint_digest,
                child_checkpoint_digest=child,
                reason_code="child_checkpoint_conflict",
                schema_version=request.schema_version,
            )
        if existing is None and len(self._records) >= self._max_records:
            return DaystromKVDecision(
                authorized=False,
                operation="transition",
                checkpoint_digest=request.checkpoint_digest,
                child_checkpoint_digest=child,
                reason_code="cardinality_exceeded",
                schema_version=request.schema_version,
            )
        self._records[child] = DaystromKVRecord(
            checkpoint_digest=child,
            block_hashes=request.block_hashes,
            expires_at=request.expires_at,
            tokens_saved=tokens,
            created_at=existing.created_at if existing is not None else now,
        )
        return DaystromKVDecision(
            authorized=True,
            operation="transition",
            checkpoint_digest=request.checkpoint_digest,
            child_checkpoint_digest=child,
            reason_code="transition_authorized",
            schema_version=request.schema_version,
            saved_tokens=tokens,
        )

    def _evaluate_status(
        self, request: DaystromKVRequest, now: float
    ) -> DaystromKVDecision:
        """Authorize a payload-free lifecycle query for one checkpoint.

        HMAC verification and TTL validation have already succeeded. Physical
        row readiness is runtime-owned and is added by the connector; the pure
        policy reports only logical record/purge state.
        """

        purge = self._purges.get(request.checkpoint_digest)
        if purge is not None:
            return DaystromKVDecision(
                authorized=True,
                operation="status",
                checkpoint_digest=request.checkpoint_digest,
                reason_code="purge_complete" if purge.completed else "purge_pending",
                schema_version=request.schema_version,
                purged_blocks=purge.blocks_zeroed,
                purged_bytes=purge.bytes_zeroed,
                shared_blocks=purge.shared_blocks,
            )
        record = self._records.get(request.checkpoint_digest)
        if record is None:
            return DaystromKVDecision(
                authorized=False,
                operation="status",
                checkpoint_digest=request.checkpoint_digest,
                reason_code="record_not_found",
                schema_version=request.schema_version,
            )
        if record.is_expired(now):
            del self._records[request.checkpoint_digest]
            return DaystromKVDecision(
                authorized=False,
                operation="status",
                checkpoint_digest=request.checkpoint_digest,
                reason_code="record_expired",
                schema_version=request.schema_version,
            )
        return DaystromKVDecision(
            authorized=True,
            operation="status",
            checkpoint_digest=request.checkpoint_digest,
            reason_code="status_authorized",
            schema_version=request.schema_version,
        )

    def _evaluate_purge(
        self, request: DaystromKVRequest, now: float
    ) -> DaystromKVDecision:
        purge = self._purges.get(request.checkpoint_digest)
        if purge is not None:
            return DaystromKVDecision(
                authorized=True,
                operation="purge",
                checkpoint_digest=request.checkpoint_digest,
                reason_code="purge_complete" if purge.completed else "purge_pending",
                schema_version=request.schema_version,
                purged_blocks=purge.blocks_zeroed,
                purged_bytes=purge.bytes_zeroed,
                shared_blocks=purge.shared_blocks,
            )
        record = self._records.get(request.checkpoint_digest)
        if record is None:
            return DaystromKVDecision(
                authorized=False,
                operation="purge",
                checkpoint_digest=request.checkpoint_digest,
                reason_code="record_not_found",
                schema_version=request.schema_version,
            )
        if record.is_expired(now):
            del self._records[request.checkpoint_digest]
            return DaystromKVDecision(
                authorized=False,
                operation="purge",
                checkpoint_digest=request.checkpoint_digest,
                reason_code="record_expired",
                schema_version=request.schema_version,
            )
        return DaystromKVDecision(
            authorized=True,
            operation="purge",
            checkpoint_digest=request.checkpoint_digest,
            reason_code="purge_authorized",
            schema_version=request.schema_version,
        )

    def partition_purge_hashes(
        self, checkpoint_digest: str
    ) -> tuple[tuple[bytes, ...], tuple[bytes, ...]]:
        """Partition one record's hashes into unique and still-shared prefixes."""

        record = self._records.get(checkpoint_digest)
        if record is None:
            raise DaystromKVAuthorizationError("record_not_found")
        other_hashes = {
            block_hash
            for digest, other in self._records.items()
            if digest != checkpoint_digest and digest not in self._purges
            for block_hash in other.block_hashes
        }
        unique = tuple(h for h in record.block_hashes if h not in other_hashes)
        shared = tuple(h for h in record.block_hashes if h in other_hashes)
        return unique, shared

    def begin_purge(
        self,
        checkpoint_digest: str,
        *,
        purge_event: int,
        blocks_scheduled: int,
        shared_blocks: int,
        shared_hashes: Sequence[bytes] = (),
    ) -> DaystromKVPurgeState:
        """Logically invalidate a checkpoint before worker-side zeroization."""

        if checkpoint_digest not in self._records:
            raise DaystromKVAuthorizationError("record_not_found")
        if purge_event < 0 or purge_event in self._purge_events:
            raise DaystromKVAuthorizationError("purge_event_invalid")
        if blocks_scheduled < 0 or shared_blocks < 0:
            raise DaystromKVAuthorizationError("purge_counter_invalid")
        try:
            exact_shared_hashes = tuple(bytes(value) for value in shared_hashes)
        except (TypeError, ValueError) as exc:
            raise DaystromKVAuthorizationError("purge_shared_hash_invalid") from exc
        if (shared_blocks == 0) != (len(exact_shared_hashes) == 0):
            raise DaystromKVAuthorizationError("purge_shared_hash_count_mismatch")
        if shared_blocks < len(exact_shared_hashes):
            raise DaystromKVAuthorizationError("purge_shared_hash_count_mismatch")
        record_hashes = set(self._records[checkpoint_digest].block_hashes)
        if any(value not in record_hashes for value in exact_shared_hashes):
            raise DaystromKVAuthorizationError("purge_shared_hash_invalid")
        state = DaystromKVPurgeState(
            checkpoint_digest=checkpoint_digest,
            purge_event=purge_event,
            blocks_scheduled=blocks_scheduled,
            shared_blocks=shared_blocks,
            shared_hashes=exact_shared_hashes,
        )
        self._purges[checkpoint_digest] = state
        self._purge_events[purge_event] = checkpoint_digest
        return state

    def purge_shared_owners_valid(self, checkpoint_digest: str) -> bool:
        """Return whether every retained hash still has another live owner."""

        state = self._purges.get(checkpoint_digest)
        if state is None or state.completed:
            return False
        if not state.shared_hashes:
            return state.shared_blocks == 0
        now = float(self._time_fn())
        other_hashes = {
            block_hash
            for digest, record in self._records.items()
            if digest != checkpoint_digest
            and digest not in self._purges
            and not record.is_expired(now)
            for block_hash in record.block_hashes
        }
        return all(value in other_hashes for value in state.shared_hashes)

    def complete_purge(
        self, purge_event: int, *, blocks_zeroed: int, bytes_zeroed: int
    ) -> DaystromKVDecision:
        """Commit physical purge only after all runtime workers acknowledge it."""

        checkpoint_digest = self._purge_events.get(purge_event)
        if checkpoint_digest is None:
            raise DaystromKVAuthorizationError("purge_event_not_found")
        state = self._purges[checkpoint_digest]
        if blocks_zeroed != state.blocks_scheduled:
            raise DaystromKVAuthorizationError("purge_block_count_mismatch")
        if bytes_zeroed < 0:
            raise DaystromKVAuthorizationError("purge_counter_invalid")
        state.completed = True
        state.blocks_zeroed = blocks_zeroed
        state.bytes_zeroed = bytes_zeroed
        self._records.pop(checkpoint_digest, None)
        self._trim_completed_purges()
        return DaystromKVDecision(
            authorized=True,
            operation="purge",
            checkpoint_digest=checkpoint_digest,
            reason_code="purge_complete",
            schema_version=DAYSTROM_KV_SCHEMA_VERSION,
            purged_blocks=blocks_zeroed,
            purged_bytes=bytes_zeroed,
            shared_blocks=state.shared_blocks,
        )

    def _trim_completed_purges(self) -> None:
        while len(self._purges) > self._max_records:
            oldest_completed = next(
                (
                    (digest, state)
                    for digest, state in self._purges.items()
                    if state.completed
                ),
                None,
            )
            if oldest_completed is None:
                return
            digest, state = oldest_completed
            self._purges.pop(digest, None)
            self._purge_events.pop(state.purge_event, None)

    def _evict_expired(self, now: float) -> None:
        expired = [d for d, r in self._records.items() if r.is_expired(now)]
        for d in expired:
            del self._records[d]

    # -- cache reset (fail-closed, no fake purge) --------------------------- #

    def reset_cache(self) -> bool:
        """Drop the in-memory authorization index.

        This does NOT claim physical purge of offloaded KV bytes; it only
        clears the authorization records so subsequent restores fail closed.
        Returns ``True`` if the index was cleared.
        """

        self._records.clear()
        pending = {
            digest: state for digest, state in self._purges.items() if not state.completed
        }
        self._purges = pending
        self._purge_events = {
            state.purge_event: digest for digest, state in pending.items()
        }
        return True

    # -- telemetry ---------------------------------------------------------- #

    def telemetry(self) -> dict[str, Any]:
        """Aggregate, payload-free telemetry over the policy state."""

        now = float(self._time_fn())
        self._evict_expired(now)
        return {
            "num_records": len(self._records),
            "max_records": self._max_records,
            "max_ttl_seconds": self._max_ttl_seconds,
            "checkpoint_digests": sorted(self._records),
        }


def _block_hash_prefix_match(
    record_hashes: tuple[bytes, ...], request_hashes: tuple[bytes, ...]
) -> bool:
    """Return True iff the saved record's block hashes are a non-empty exact
    prefix of the new request's block hashes.

    The restore request may extend the saved prefix, so the record's hashes
    must be a prefix of (not equal to or longer than) the request's hashes.
    Reject when the request is shorter than the record or diverges.
    """

    if not record_hashes:
        return False
    if len(record_hashes) > len(request_hashes):
        return False
    for i, bh in enumerate(record_hashes):
        if not hmac.compare_digest(bytes(bh), bytes(request_hashes[i])):
            return False
    return True


# --------------------------------------------------------------------------- #
# Convenience: build a signed envelope (for tests / controller integration)
# --------------------------------------------------------------------------- #


def build_daystrom_params(
    *,
    operation: str,
    checkpoint_digest: str,
    expires_at: float,
    nonce: str,
    secret_path: Path,
) -> dict[str, Any]:
    """Build a signed ``daystrom`` sub-mapping for ``kv_transfer_params``.

    This is intended for controllers/tests that need to mint valid envelopes.
    The secret is loaded from ``secret_path``.
    """

    secret = _load_secret(Path(secret_path))
    if operation not in _VALID_OPERATIONS:
        raise ValueError(f"operation must be one of {_VALID_OPERATIONS}")
    if not _DIGEST_RE.match(checkpoint_digest):
        raise ValueError("checkpoint_digest must be sha256:+64 hex")
    if not _NONCE_RE.match(nonce) or not (1 <= len(nonce) <= MAX_NONCE_LEN):
        raise ValueError("nonce invalid")
    import math

    if not isinstance(expires_at, (int, float)) or isinstance(expires_at, bool) or not math.isfinite(float(expires_at)):
        raise ValueError("expires_at must be finite")
    expires_at = float(expires_at)
    canonical = "\n".join(
        [
            DAYSTROM_KV_SCHEMA_VERSION,
            operation,
            checkpoint_digest,
            f"{expires_at:.6f}",
            nonce,
        ]
    ).encode("ascii")
    authorization = hmac.new(secret, canonical, hashlib.sha256).hexdigest()
    return {
        "schema_version": DAYSTROM_KV_SCHEMA_VERSION,
        "operation": operation,
        "checkpoint_digest": checkpoint_digest,
        "expires_at": expires_at,
        "nonce": nonce,
        "authorization": authorization,
    }


def build_kv_transfer_params(
    *,
    operation: str,
    checkpoint_digest: str,
    expires_at: float,
    nonce: str,
    secret_path: Path,
) -> dict[str, Any]:
    """Build the full ``kv_transfer_params`` dict with a nested ``daystrom``."""

    return {
        "daystrom": build_daystrom_params(
            operation=operation,
            checkpoint_digest=checkpoint_digest,
            expires_at=expires_at,
            nonce=nonce,
            secret_path=secret_path,
        )
    }


def build_kv_transition_params(
    *,
    parent_checkpoint_digest: str,
    child_checkpoint_digest: str,
    expires_at: float,
    nonce: str,
    secret_path: Path,
) -> dict[str, Any]:
    """Build a dual-digest envelope for one restore-and-save request."""

    secret = _load_secret(Path(secret_path))
    for label, digest in (
        ("parent_checkpoint_digest", parent_checkpoint_digest),
        ("child_checkpoint_digest", child_checkpoint_digest),
    ):
        if not _DIGEST_RE.match(digest):
            raise ValueError(f"{label} must be sha256:+64 hex")
    if parent_checkpoint_digest == child_checkpoint_digest:
        raise ValueError("parent and child checkpoint digests must differ")
    if not _NONCE_RE.match(nonce) or not (1 <= len(nonce) <= MAX_NONCE_LEN):
        raise ValueError("nonce invalid")
    import math

    if (
        not isinstance(expires_at, (int, float))
        or isinstance(expires_at, bool)
        or not math.isfinite(float(expires_at))
    ):
        raise ValueError("expires_at must be finite")
    parsed_expiry = float(expires_at)
    transition = DaystromKVRequest(
        schema_version=DAYSTROM_KV_TRANSITION_SCHEMA_VERSION,
        operation="transition",
        checkpoint_digest=parent_checkpoint_digest,
        child_checkpoint_digest=child_checkpoint_digest,
        expires_at=parsed_expiry,
        nonce=nonce,
        authorization="0" * 64,
        block_hashes=(),
    )
    return {
        "daystrom": {
            "schema_version": transition.schema_version,
            "operation": transition.operation,
            "checkpoint_digest": parent_checkpoint_digest,
            "child_checkpoint_digest": child_checkpoint_digest,
            "expires_at": parsed_expiry,
            "nonce": nonce,
            "authorization": compute_authorization(secret, transition),
        }
    }


def canonical_message_for(
    *,
    operation: str,
    checkpoint_digest: str,
    expires_at: float,
    nonce: str,
) -> bytes:
    """Return the canonical HMAC message bytes for the given fields."""

    return "\n".join(
        [
            DAYSTROM_KV_SCHEMA_VERSION,
            operation,
            checkpoint_digest,
            f"{float(expires_at):.6f}",
            nonce,
        ]
    ).encode("ascii")


def to_json_telemetry(decision: DaystromKVDecision) -> str:
    """Serialize a decision's telemetry to compact JSON (payload-free)."""

    return json.dumps(decision.telemetry(), sort_keys=True, separators=(",", ":"))
