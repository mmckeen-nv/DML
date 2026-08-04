"""Provider-neutral execution-state contracts for materialized model context."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from daystrom_dml.api_contracts import SerializableDataclass


class RuntimeExecutionError(RuntimeError):
    """Fail-closed runtime execution or cache-control error."""


class RuntimeCacheOperation(str, Enum):
    SAVE = "save"
    RESTORE = "restore"
    ERASE = "erase"


@dataclass
class RuntimeExecutionCapabilities(SerializableDataclass):
    runtime_id: str
    adapter_id: str
    runtime_version: str = "unknown"
    supports_prompt_cache: bool = False
    supports_kv_checkpoint: bool = False
    supports_kv_restore: bool = False
    supports_kv_erase: bool = False
    supports_kv_checkpoint_delete: bool = False
    supports_slot_affinity: bool = False
    supports_metrics: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeCompletionTrace:
    runtime_id: str
    slot_id: int
    prompt_tokens_total: int
    prompt_tokens_processed: int
    prompt_tokens_reused: int
    prompt_ms: float
    predicted_tokens: int
    output_text: str
    output_token_ids: List[int] = field(default_factory=list)
    truncated: bool = False

    def to_telemetry(self) -> Dict[str, Any]:
        return {
            "runtime_id": self.runtime_id,
            "slot_id": self.slot_id,
            "prompt_tokens_total": self.prompt_tokens_total,
            "prompt_tokens_processed": self.prompt_tokens_processed,
            "prompt_tokens_reused": self.prompt_tokens_reused,
            "prompt_ms": self.prompt_ms,
            "predicted_tokens": self.predicted_tokens,
            "output_digest": hashlib.sha256(self.output_text.encode("utf-8")).hexdigest(),
            "output_token_ids": list(self.output_token_ids),
            "truncated": self.truncated,
        }


@dataclass
class RuntimeCacheOperationResult(SerializableDataclass):
    runtime_id: str
    slot_id: int
    operation: RuntimeCacheOperation
    tokens_affected: int = 0
    bytes_affected: int = 0
    elapsed_ms: float = 0.0
    checkpoint_name: Optional[str] = None


@dataclass
class RuntimeCheckpointDeleteResult(SerializableDataclass):
    """Result of deleting runtime-owned checkpoint bytes outside a slot."""

    runtime_id: str
    checkpoint_name: str
    bytes_deleted: int = 0
    existed: bool = False
    elapsed_ms: float = 0.0
