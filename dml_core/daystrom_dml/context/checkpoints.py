"""Fail-closed identity binding and durable metadata for model KV checkpoints.

Checkpoint files remain owned by the model runtime.  This module persists only
bounded metadata needed to authorize a save/restore operation; it never stores
prompt text, generated text, credentials, or checkpoint bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
from pathlib import Path, PurePath
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Protocol

from daystrom_dml.api_contracts import DaystromScope, SerializableDataclass, _serialize
from daystrom_dml.atomic_io import atomic_write_text
from daystrom_dml.context.execution import (
    RuntimeCacheOperation,
    RuntimeCacheOperationResult,
    RuntimeCheckpointDeleteResult,
    RuntimeExecutionCapabilities,
    RuntimeExecutionError,
)

EXECUTION_CHECKPOINT_IDENTITY_V1 = "daystrom-execution-checkpoint-identity-v1"
EXECUTION_CHECKPOINT_IDENTITY_V2 = "daystrom-execution-checkpoint-identity-v2"
EXECUTION_CHECKPOINT_RECORD_V1 = "daystrom-execution-checkpoint-record-v1"
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ExecutionStateAdapter(Protocol):
    """Narrow capability required by the bound checkpoint controller."""

    def capabilities(self) -> RuntimeExecutionCapabilities: ...

    def save_slot(self, slot_id: int, filename: str) -> RuntimeCacheOperationResult: ...

    def restore_slot(self, slot_id: int, filename: str) -> RuntimeCacheOperationResult: ...

    def delete_checkpoint(self, filename: str) -> RuntimeCheckpointDeleteResult: ...


class CheckpointSelectionError(RuntimeExecutionError):
    """Bounded checkpoint selection failed without exposing record identifiers."""

    def __init__(self, reason_code: str):
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass
class ExecutionCheckpointIdentity(SerializableDataclass):
    """Complete compatibility and authority identity for one materialized prefix."""

    scope: DaystromScope
    model_id: str
    model_digest: str
    tokenizer_digest: str
    positional_config_digest: str
    immutable_prefix_digest: str
    packet_digest: str
    manifest_digest: str
    runtime_id: str
    runtime_version: str
    adapter_id: str
    identity_version: str = EXECUTION_CHECKPOINT_IDENTITY_V2
    runtime_endpoint_digest: str = ""
    binding_digest: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.scope, dict):
            self.scope = DaystromScope.from_dict(self.scope)
        if not isinstance(self.scope, DaystromScope):
            raise RuntimeExecutionError("scope must be a DaystromScope")
        if not isinstance(self.scope.session_id, str) or not self.scope.session_id:
            raise RuntimeExecutionError("scope.session_id must be non-empty")
        for name in (
            "tenant_id",
            "client_id",
            "session_id",
            "instance_id",
            "thread_id",
            "project_id",
            "relationship_id",
        ):
            value = getattr(self.scope, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise RuntimeExecutionError(f"scope.{name} must be null or a non-empty string")
        for name in ("model_id", "runtime_id", "runtime_version", "adapter_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise RuntimeExecutionError(f"{name} must be non-empty")
        if self.runtime_version.casefold() == "unknown":
            raise RuntimeExecutionError("runtime_version must be exact")
        for name in (
            "model_digest",
            "tokenizer_digest",
            "positional_config_digest",
            "immutable_prefix_digest",
            "packet_digest",
            "manifest_digest",
        ):
            _strong_digest(name, getattr(self, name))
        if self.identity_version == EXECUTION_CHECKPOINT_IDENTITY_V2:
            _strong_digest("runtime_endpoint_digest", self.runtime_endpoint_digest)
        elif self.identity_version == EXECUTION_CHECKPOINT_IDENTITY_V1:
            if self.runtime_endpoint_digest:
                raise RuntimeExecutionError("v1 checkpoint identity cannot bind a runtime endpoint")
        else:
            raise RuntimeExecutionError("unsupported checkpoint identity version")
        computed = self.compute_binding_digest()
        if self.binding_digest and self.binding_digest != computed:
            raise RuntimeExecutionError("checkpoint identity integrity check failed")
        self.binding_digest = computed

    @classmethod
    def from_packet(
        cls,
        packet: Any,
        runtime_capabilities: RuntimeExecutionCapabilities,
        *,
        model_digest: str,
        tokenizer_digest: str,
        positional_config: Mapping[str, Any],
        immutable_prefix: str,
        runtime_endpoint_url: str,
    ) -> "ExecutionCheckpointIdentity":
        """Build a binding from a revalidated packet and the exact rendered prefix.

        Raw prefix and positional configuration are hashed in memory and are not
        retained in the resulting identity or registry record.
        """
        from daystrom_dml.context.manifest import ContextPacket

        if not isinstance(packet, ContextPacket):
            raise RuntimeExecutionError("packet must be a ContextPacket")
        try:
            validated = ContextPacket.from_dict(packet.to_dict())
        except Exception as exc:
            raise RuntimeExecutionError("context packet integrity check failed") from exc
        if not isinstance(runtime_capabilities, RuntimeExecutionCapabilities):
            raise RuntimeExecutionError("runtime capabilities are required")
        if validated.capabilities.backend_id != runtime_capabilities.runtime_id:
            raise RuntimeExecutionError("packet runtime identity mismatch")
        if not runtime_capabilities.supports_kv_checkpoint or not runtime_capabilities.supports_kv_restore:
            raise RuntimeExecutionError("runtime lacks checkpoint save/restore capabilities")
        if not isinstance(immutable_prefix, str) or not immutable_prefix:
            raise RuntimeExecutionError("immutable_prefix must be non-empty")
        if not isinstance(positional_config, Mapping) or not positional_config:
            raise RuntimeExecutionError("positional_config must be non-empty")
        from daystrom_dml.context.probe import endpoint_origin_identity_digest

        try:
            runtime_endpoint_digest = "sha256:" + endpoint_origin_identity_digest(runtime_endpoint_url)
        except (TypeError, ValueError) as exc:
            raise RuntimeExecutionError("runtime_endpoint_url is invalid") from exc
        expected_endpoint_digest = runtime_capabilities.metadata.get("endpoint_origin_digest")
        if expected_endpoint_digest != runtime_endpoint_digest:
            raise RuntimeExecutionError("runtime endpoint identity mismatch")
        try:
            positional_payload = json.dumps(
                dict(positional_config), sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeExecutionError("positional_config must be JSON serializable") from exc
        return cls(
            scope=validated.scope,
            model_id=validated.capabilities.model_id,
            model_digest=model_digest,
            tokenizer_digest=tokenizer_digest,
            positional_config_digest=_text_digest(positional_payload),
            immutable_prefix_digest=_text_digest(immutable_prefix),
            packet_digest="sha256:" + validated.packet_content_digest,
            manifest_digest="sha256:" + validated.manifest.content_digest,
            runtime_id=runtime_capabilities.runtime_id,
            runtime_version=runtime_capabilities.runtime_version,
            adapter_id=runtime_capabilities.adapter_id,
            runtime_endpoint_digest=runtime_endpoint_digest,
        )

    def compute_binding_digest(self) -> str:
        stable = {
            "scope": _serialize(self.scope),
            "model_id": self.model_id,
            "model_digest": self.model_digest,
            "tokenizer_digest": self.tokenizer_digest,
            "positional_config_digest": self.positional_config_digest,
            "immutable_prefix_digest": self.immutable_prefix_digest,
            "packet_digest": self.packet_digest,
            "manifest_digest": self.manifest_digest,
            "runtime_id": self.runtime_id,
            "runtime_version": self.runtime_version,
            "adapter_id": self.adapter_id,
            "identity_version": self.identity_version,
        }
        if self.identity_version == EXECUTION_CHECKPOINT_IDENTITY_V2:
            stable["runtime_endpoint_digest"] = self.runtime_endpoint_digest
        return _json_digest(stable)

    def to_dict(self) -> Dict[str, Any]:
        if self.binding_digest != self.compute_binding_digest():
            raise RuntimeExecutionError("checkpoint identity integrity check failed")
        payload = super().to_dict()
        if self.identity_version == EXECUTION_CHECKPOINT_IDENTITY_V1:
            payload.pop("runtime_endpoint_digest", None)
        return payload

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ExecutionCheckpointIdentity":
        if not isinstance(data, dict):
            raise RuntimeExecutionError("checkpoint identity must be an object")
        if not data.get("binding_digest"):
            raise RuntimeExecutionError("checkpoint identity binding_digest is required")
        payload = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        if "scope" in payload:
            scope = payload["scope"]
            if not isinstance(scope, dict):
                raise RuntimeExecutionError("checkpoint identity scope must be an object")
            payload["scope"] = DaystromScope.from_dict(scope)
        try:
            return cls(**payload)
        except TypeError as exc:
            raise RuntimeExecutionError("invalid checkpoint identity") from exc


@dataclass
class ExecutionCheckpointRecord(SerializableDataclass):
    """Payload-free durable record authorizing a runtime-owned checkpoint."""

    checkpoint_id: str
    checkpoint_name: str
    identity: ExecutionCheckpointIdentity
    tokens_saved: int
    bytes_saved: int
    created_at: float
    expires_at: float
    record_version: str = EXECUTION_CHECKPOINT_RECORD_V1
    record_digest: str = ""

    def __post_init__(self) -> None:
        _safe_id("checkpoint_id", self.checkpoint_id)
        _checkpoint_name(self.checkpoint_name)
        if isinstance(self.identity, dict):
            self.identity = ExecutionCheckpointIdentity.from_dict(self.identity)
        if not isinstance(self.identity, ExecutionCheckpointIdentity):
            raise RuntimeExecutionError("identity must be an ExecutionCheckpointIdentity")
        for name in ("tokens_saved", "bytes_saved"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise RuntimeExecutionError(f"{name} must be a positive integer")
        for name in ("created_at", "expires_at"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise RuntimeExecutionError(f"{name} must be numeric")
        if self.expires_at <= self.created_at:
            raise RuntimeExecutionError("expires_at must be later than created_at")
        if self.record_version != EXECUTION_CHECKPOINT_RECORD_V1:
            raise RuntimeExecutionError("unsupported checkpoint record version")
        computed = self.compute_record_digest()
        if self.record_digest and self.record_digest != computed:
            raise RuntimeExecutionError("checkpoint record integrity check failed")
        self.record_digest = computed

    def compute_record_digest(self) -> str:
        stable = {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_name": self.checkpoint_name,
            "identity": self.identity.to_dict(),
            "tokens_saved": self.tokens_saved,
            "bytes_saved": self.bytes_saved,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "record_version": self.record_version,
        }
        return _json_digest(stable)

    def to_dict(self) -> Dict[str, Any]:
        if self.record_digest != self.compute_record_digest():
            raise RuntimeExecutionError("checkpoint record integrity check failed")
        return super().to_dict()

    def to_telemetry(self) -> Dict[str, Any]:
        return {
            "checkpoint_id_digest": _text_digest(self.checkpoint_id),
            "binding_digest": self.identity.binding_digest,
            "tokens_saved": self.tokens_saved,
            "bytes_saved": self.bytes_saved,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "record_version": self.record_version,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ExecutionCheckpointRecord":
        if not isinstance(data, dict):
            raise RuntimeExecutionError("checkpoint record must be an object")
        if not data.get("record_digest"):
            raise RuntimeExecutionError("checkpoint record record_digest is required")
        payload = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        identity = payload.get("identity")
        if not isinstance(identity, dict):
            raise RuntimeExecutionError("checkpoint record identity must be an object")
        payload["identity"] = ExecutionCheckpointIdentity.from_dict(identity)
        try:
            return cls(**payload)
        except TypeError as exc:
            raise RuntimeExecutionError("invalid checkpoint record") from exc


@dataclass(frozen=True)
class CheckpointPurgeResult:
    checkpoint_id_digest: str
    binding_digest: str
    checkpoint_name_digest: str
    bytes_deleted: int
    existed: bool
    elapsed_ms: float


@dataclass(frozen=True)
class CheckpointRestoreResult:
    """Payload-free evidence for one identity-selected checkpoint restore."""

    checkpoint_id_digest: str
    binding_digest: str
    tokens_restored: int
    bytes_restored: int
    elapsed_ms: float
    slot_id: int

    def to_telemetry(self) -> Dict[str, Any]:
        return {
            "checkpoint_id_digest": self.checkpoint_id_digest,
            "binding_digest": self.binding_digest,
            "tokens_restored": self.tokens_restored,
            "bytes_restored": self.bytes_restored,
            "elapsed_ms": self.elapsed_ms,
            "slot_id": self.slot_id,
        }


class FileExecutionCheckpointRegistry:
    """Atomic, bounded metadata registry keyed by a safe checkpoint identifier."""

    def __init__(
        self,
        root: str | Path,
        *,
        clock: Callable[[], float] = time.time,
        max_record_bytes: int = 64 * 1024,
        lock_timeout_seconds: float = 2.0,
    ) -> None:
        if max_record_bytes <= 0 or lock_timeout_seconds <= 0:
            raise RuntimeExecutionError("registry limits must be positive")
        self.root = Path(root)
        self.clock = clock
        self.max_record_bytes = max_record_bytes
        self.lock_timeout_seconds = lock_timeout_seconds
        self.root.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            os.chmod(self.root, 0o700)

    def put(self, record: ExecutionCheckpointRecord) -> None:
        if not isinstance(record, ExecutionCheckpointRecord):
            raise RuntimeExecutionError("record must be an ExecutionCheckpointRecord")
        self.create(record.checkpoint_id, lambda: record)

    def create(
        self,
        checkpoint_id: str,
        build_record: Callable[[], ExecutionCheckpointRecord],
    ) -> ExecutionCheckpointRecord:
        """Create one record while holding its interprocess reservation.

        The callback may perform the runtime save. Holding the reservation across
        that call prevents a duplicate identifier from overwriting an existing
        runtime checkpoint before registry rejection.
        """
        path = self._path(checkpoint_id)
        with self._lock(checkpoint_id):
            if path.exists():
                raise RuntimeExecutionError("checkpoint record already exists")
            record = build_record()
            if not isinstance(record, ExecutionCheckpointRecord) or record.checkpoint_id != checkpoint_id:
                raise RuntimeExecutionError("checkpoint record builder returned an invalid record")
            if record.expires_at <= self.clock():
                raise RuntimeExecutionError("checkpoint record is already expired")
            payload = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            encoded = payload.encode("utf-8")
            if len(encoded) > self.max_record_bytes:
                raise RuntimeExecutionError("checkpoint record exceeded byte limit")
            atomic_write_text(path, payload)
            if os.name != "nt":
                os.chmod(path, 0o600)
        return record

    def require(
        self,
        checkpoint_id: str,
        expected_identity: ExecutionCheckpointIdentity,
    ) -> ExecutionCheckpointRecord:
        record = self._read(checkpoint_id)
        self._validate_identity(record, expected_identity)
        if self.clock() >= record.expires_at:
            raise RuntimeExecutionError("checkpoint record expired")
        return record

    def select(
        self,
        expected_identity: ExecutionCheckpointIdentity,
        *,
        max_records: int = 1024,
    ) -> ExecutionCheckpointRecord:
        """Select exactly one live record matching the complete bound identity."""

        if not isinstance(max_records, int) or isinstance(max_records, bool) or max_records <= 0:
            raise RuntimeExecutionError("max_records must be a positive integer")
        expected_identity.to_dict()
        matches: list[ExecutionCheckpointRecord] = []
        paths = list(islice(self.root.glob("*.json"), max_records + 1))
        if len(paths) > max_records:
            raise CheckpointSelectionError("checkpoint_selection_scan_limit")
        for path in sorted(paths):
            checkpoint_id = path.stem
            _safe_id("checkpoint_id", checkpoint_id)
            record = self._read(checkpoint_id)
            if self.clock() >= record.expires_at:
                continue
            if record.identity.binding_digest == expected_identity.binding_digest and record.identity == expected_identity:
                matches.append(record)
                if len(matches) > 1:
                    raise CheckpointSelectionError("checkpoint_selection_ambiguous")
        if not matches:
            raise CheckpointSelectionError("checkpoint_selection_not_found")
        return matches[0]

    def expired_records(self, *, max_records: int) -> list[ExecutionCheckpointRecord]:
        if not isinstance(max_records, int) or isinstance(max_records, bool) or max_records <= 0:
            raise RuntimeExecutionError("max_records must be a positive integer")
        records: list[ExecutionCheckpointRecord] = []
        for path in sorted(self.root.glob("*.json")):
            if len(records) >= max_records:
                break
            checkpoint_id = path.stem
            _safe_id("checkpoint_id", checkpoint_id)
            record = self._read(checkpoint_id)
            if self.clock() >= record.expires_at:
                records.append(record)
        return records

    def purge(
        self,
        checkpoint_id: str,
        expected_identity: ExecutionCheckpointIdentity,
        delete_runtime: Callable[[ExecutionCheckpointRecord], RuntimeCheckpointDeleteResult],
    ) -> RuntimeCheckpointDeleteResult:
        path = self._path(checkpoint_id)
        with self._lock(checkpoint_id):
            record = self._read(checkpoint_id)
            self._validate_identity(record, expected_identity)
            result = delete_runtime(record)
            if not isinstance(result, RuntimeCheckpointDeleteResult):
                raise RuntimeExecutionError("runtime checkpoint delete result is invalid")
            try:
                path.unlink()
            except OSError as exc:
                raise RuntimeExecutionError("checkpoint metadata could not be deleted") from exc
        return result

    def _read(self, checkpoint_id: str) -> ExecutionCheckpointRecord:
        path = self._path(checkpoint_id)
        try:
            with path.open("rb") as handle:
                raw = handle.read(self.max_record_bytes + 1)
        except FileNotFoundError as exc:
            raise RuntimeExecutionError("checkpoint record not found") from exc
        except OSError as exc:
            raise RuntimeExecutionError("checkpoint record could not be read") from exc
        if len(raw) > self.max_record_bytes:
            raise RuntimeExecutionError("checkpoint record exceeded byte limit")
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeExecutionError("checkpoint record integrity check failed") from exc
        try:
            record = ExecutionCheckpointRecord.from_dict(data)
        except RuntimeExecutionError as exc:
            raise RuntimeExecutionError("checkpoint record integrity check failed") from exc
        if record.checkpoint_id != checkpoint_id:
            raise RuntimeExecutionError("checkpoint record integrity check failed")
        return record

    @staticmethod
    def _validate_identity(
        record: ExecutionCheckpointRecord,
        expected_identity: ExecutionCheckpointIdentity,
    ) -> None:
        expected_identity.to_dict()
        if record.identity.binding_digest != expected_identity.binding_digest or record.identity != expected_identity:
            raise RuntimeExecutionError("checkpoint identity mismatch")

    def _path(self, checkpoint_id: str) -> Path:
        _safe_id("checkpoint_id", checkpoint_id)
        return self.root / f"{checkpoint_id}.json"

    @contextmanager
    def _lock(self, checkpoint_id: str) -> Iterator[None]:
        lock_path = self.root / f".{checkpoint_id}.lock"
        deadline = time.monotonic() + self.lock_timeout_seconds
        while True:
            try:
                lock_path.mkdir(mode=0o700)
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise RuntimeExecutionError("checkpoint registry lock timed out")
                time.sleep(0.01)
        try:
            yield
        finally:
            try:
                lock_path.rmdir()
            except OSError:
                pass


class ExecutionCheckpointController:
    """Authorize runtime save/restore operations against exact durable bindings."""

    def __init__(
        self,
        *,
        adapter: ExecutionStateAdapter,
        registry: FileExecutionCheckpointRegistry,
        clock: Callable[[], float] = time.time,
        max_ttl_seconds: float = 24 * 60 * 60,
    ) -> None:
        if max_ttl_seconds <= 0:
            raise RuntimeExecutionError("max_ttl_seconds must be positive")
        self.adapter = adapter
        self.registry = registry
        self.clock = clock
        self.max_ttl_seconds = max_ttl_seconds

    def save(
        self,
        checkpoint_id: str,
        identity: ExecutionCheckpointIdentity,
        *,
        slot_id: int,
        ttl_seconds: float,
    ) -> ExecutionCheckpointRecord:
        _safe_id("checkpoint_id", checkpoint_id)
        if not isinstance(ttl_seconds, (int, float)) or isinstance(ttl_seconds, bool):
            raise RuntimeExecutionError("ttl_seconds must be numeric")
        if ttl_seconds <= 0 or ttl_seconds > self.max_ttl_seconds:
            raise RuntimeExecutionError("ttl_seconds is outside the allowed retention bound")
        caps = self._authorize(identity, operation="save")
        if not caps.supports_kv_checkpoint:
            raise RuntimeExecutionError("runtime does not support checkpoint save")
        if not caps.supports_kv_checkpoint_delete:
            raise RuntimeExecutionError("runtime does not support checkpoint deletion")
        checkpoint_name = _runtime_checkpoint_name(checkpoint_id, identity.binding_digest)
        saved = False

        def save_and_build_record() -> ExecutionCheckpointRecord:
            nonlocal saved
            result = self.adapter.save_slot(slot_id, checkpoint_name)
            self._validate_result(result, RuntimeCacheOperation.SAVE, identity, slot_id, checkpoint_name)
            saved = True
            created_at = float(self.clock())
            return ExecutionCheckpointRecord(
                checkpoint_id=checkpoint_id,
                checkpoint_name=checkpoint_name,
                identity=identity,
                tokens_saved=result.tokens_affected,
                bytes_saved=result.bytes_affected,
                created_at=created_at,
                expires_at=created_at + float(ttl_seconds),
            )

        try:
            return self.registry.create(checkpoint_id, save_and_build_record)
        except Exception:
            if saved:
                try:
                    deleted = self.adapter.delete_checkpoint(checkpoint_name)
                    self._validate_delete_result(deleted, identity, checkpoint_name)
                except Exception as delete_exc:
                    raise RuntimeExecutionError("checkpoint save failed and compensation deletion failed") from delete_exc
            raise

    def restore(
        self,
        checkpoint_id: str,
        identity: ExecutionCheckpointIdentity,
        *,
        slot_id: int,
    ) -> RuntimeCacheOperationResult:
        caps = self._authorize(identity, operation="restore")
        if not caps.supports_kv_restore:
            raise RuntimeExecutionError("runtime does not support checkpoint restore")
        record = self.registry.require(checkpoint_id, identity)
        result = self.adapter.restore_slot(slot_id, record.checkpoint_name)
        self._validate_result(result, RuntimeCacheOperation.RESTORE, identity, slot_id, record.checkpoint_name)
        if result.tokens_affected != record.tokens_saved or result.bytes_affected != record.bytes_saved:
            raise RuntimeExecutionError("runtime restore result did not match checkpoint record")
        return result

    def restore_matching(
        self,
        identity: ExecutionCheckpointIdentity,
        *,
        slot_id: int,
        max_records: int = 1024,
    ) -> CheckpointRestoreResult:
        """Select one exact live identity match, revalidate it, and restore it."""

        self._authorize(identity, operation="restore")
        record = self.registry.select(identity, max_records=max_records)
        restored = self.restore(record.checkpoint_id, identity, slot_id=slot_id)
        return CheckpointRestoreResult(
            checkpoint_id_digest=_text_digest(record.checkpoint_id),
            binding_digest=identity.binding_digest,
            tokens_restored=restored.tokens_affected,
            bytes_restored=restored.bytes_affected,
            elapsed_ms=restored.elapsed_ms,
            slot_id=slot_id,
        )

    def purge(
        self,
        checkpoint_id: str,
        identity: ExecutionCheckpointIdentity,
    ) -> CheckpointPurgeResult:
        caps = self._authorize(identity, operation="purge")
        if not caps.supports_kv_checkpoint_delete:
            raise RuntimeExecutionError("runtime does not support checkpoint deletion")

        def delete_runtime(record: ExecutionCheckpointRecord) -> RuntimeCheckpointDeleteResult:
            deleted = self.adapter.delete_checkpoint(record.checkpoint_name)
            self._validate_delete_result(deleted, identity, record.checkpoint_name)
            return deleted

        deleted = self.registry.purge(checkpoint_id, identity, delete_runtime)
        return CheckpointPurgeResult(
            checkpoint_id_digest=_text_digest(checkpoint_id),
            binding_digest=identity.binding_digest,
            checkpoint_name_digest=_text_digest(deleted.checkpoint_name),
            bytes_deleted=deleted.bytes_deleted,
            existed=deleted.existed,
            elapsed_ms=deleted.elapsed_ms,
        )

    def purge_expired(self, *, max_records: int = 100) -> list[CheckpointPurgeResult]:
        results: list[CheckpointPurgeResult] = []
        for record in self.registry.expired_records(max_records=max_records):
            results.append(self.purge(record.checkpoint_id, record.identity))
        return results

    def _authorize(
        self,
        identity: ExecutionCheckpointIdentity,
        *,
        operation: str,
    ) -> RuntimeExecutionCapabilities:
        if not isinstance(identity, ExecutionCheckpointIdentity):
            raise RuntimeExecutionError("checkpoint identity is required")
        identity.to_dict()
        caps = self.adapter.capabilities()
        legacy_v1_purge = identity.identity_version == EXECUTION_CHECKPOINT_IDENTITY_V1 and operation == "purge"
        if identity.identity_version == EXECUTION_CHECKPOINT_IDENTITY_V1 and not legacy_v1_purge:
            raise RuntimeExecutionError("legacy checkpoint identity is purge-only")
        if (
            caps.runtime_id != identity.runtime_id
            or caps.runtime_version != identity.runtime_version
            or caps.adapter_id != identity.adapter_id
            or (
                not legacy_v1_purge
                and caps.metadata.get("endpoint_origin_digest") != identity.runtime_endpoint_digest
            )
        ):
            raise RuntimeExecutionError(f"runtime identity mismatch before checkpoint {operation}")
        return caps

    @staticmethod
    def _validate_delete_result(
        result: RuntimeCheckpointDeleteResult,
        identity: ExecutionCheckpointIdentity,
        checkpoint_name: str,
    ) -> None:
        if (
            not isinstance(result, RuntimeCheckpointDeleteResult)
            or result.runtime_id != identity.runtime_id
            or result.checkpoint_name != checkpoint_name
            or not isinstance(result.bytes_deleted, int)
            or isinstance(result.bytes_deleted, bool)
            or result.bytes_deleted < 0
            or not isinstance(result.existed, bool)
            or (result.existed and result.bytes_deleted <= 0)
            or not isinstance(result.elapsed_ms, (int, float))
            or isinstance(result.elapsed_ms, bool)
            or result.elapsed_ms < 0
        ):
            raise RuntimeExecutionError("runtime checkpoint delete result failed validation")

    @staticmethod
    def _validate_result(
        result: RuntimeCacheOperationResult,
        operation: RuntimeCacheOperation,
        identity: ExecutionCheckpointIdentity,
        slot_id: int,
        checkpoint_name: str,
    ) -> None:
        if (
            not isinstance(result, RuntimeCacheOperationResult)
            or result.operation is not operation
            or result.runtime_id != identity.runtime_id
            or result.slot_id != slot_id
            or result.checkpoint_name != checkpoint_name
            or not isinstance(result.tokens_affected, int)
            or isinstance(result.tokens_affected, bool)
            or result.tokens_affected <= 0
            or not isinstance(result.bytes_affected, int)
            or isinstance(result.bytes_affected, bool)
            or result.bytes_affected <= 0
        ):
            raise RuntimeExecutionError(f"runtime {operation.value} result failed validation")


def _strong_digest(name: str, value: Any) -> None:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise RuntimeExecutionError(f"{name} must be a sha256 digest")


def _safe_id(name: str, value: Any) -> None:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise RuntimeExecutionError(f"{name} must be a safe identifier")


def _checkpoint_name(filename: Any) -> None:
    if (
        not isinstance(filename, str)
        or not filename
        or PurePath(filename).name != filename
        or filename in {".", ".."}
    ):
        raise RuntimeExecutionError("checkpoint_name must be a basename")


def _runtime_checkpoint_name(checkpoint_id: str, binding_digest: str) -> str:
    material = f"{checkpoint_id}:{binding_digest}".encode("utf-8")
    return f"dcm-kv-{hashlib.sha256(material).hexdigest()[:32]}.bin"


def _text_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_digest(value: Dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _text_digest(payload)
