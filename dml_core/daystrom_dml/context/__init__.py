"""Daystrom Context Manager contract primitives."""
from daystrom_dml.context.admission import ACTIVE_ADMISSION_MODE, ADMISSION_MODES, OBSERVE_ONLY_MODE, admit_context_segments
from daystrom_dml.context.budget import ContextBudget
from daystrom_dml.context.capabilities import RUNTIME_CAPABILITIES_V1, RuntimeCapabilities
from daystrom_dml.context.checkpoints import (
    EXECUTION_CHECKPOINT_IDENTITY_V1,
    EXECUTION_CHECKPOINT_IDENTITY_V2,
    EXECUTION_CHECKPOINT_RECORD_V1,
    CheckpointPurgeResult,
    CheckpointRestoreResult,
    CheckpointSelectionError,
    ExecutionCheckpointController,
    ExecutionCheckpointIdentity,
    ExecutionCheckpointRecord,
    FileExecutionCheckpointRegistry,
)
from daystrom_dml.context.execution import (
    RuntimeCacheOperation,
    RuntimeCacheOperationResult,
    RuntimeCheckpointDeleteResult,
    RuntimeCompletionTrace,
    RuntimeExecutionCapabilities,
    RuntimeExecutionError,
)
from daystrom_dml.context.manifest import CONTEXT_MANIFEST_V1, CONTEXT_PACKET_V1, ContextManifest, ContextPacket
from daystrom_dml.context.recovery import (
    AutonomousFaultRetryRunner,
    CheckpointRecoveryPlan,
    FaultRetryPolicy,
    FaultRetryResult,
    RecoveryStatus,
)
from daystrom_dml.context.schema import ContextAuthority, ContextPriority, ContextSegment

__all__ = [
    "ACTIVE_ADMISSION_MODE",
    "ADMISSION_MODES",
    "AutonomousFaultRetryRunner",
    "CheckpointRecoveryPlan",
    "CONTEXT_MANIFEST_V1",
    "CONTEXT_PACKET_V1",
    "ContextAuthority",
    "ContextBudget",
    "CheckpointPurgeResult",
    "CheckpointRestoreResult",
    "CheckpointSelectionError",
    "ContextManifest",
    "ContextPacket",
    "ContextPriority",
    "ContextSegment",
    "EXECUTION_CHECKPOINT_IDENTITY_V1",
    "EXECUTION_CHECKPOINT_IDENTITY_V2",
    "EXECUTION_CHECKPOINT_RECORD_V1",
    "ExecutionCheckpointController",
    "ExecutionCheckpointIdentity",
    "ExecutionCheckpointRecord",
    "FaultRetryPolicy",
    "FaultRetryResult",
    "FileExecutionCheckpointRegistry",
    "OBSERVE_ONLY_MODE",
    "RUNTIME_CAPABILITIES_V1",
    "RuntimeCacheOperation",
    "RuntimeCacheOperationResult",
    "RuntimeCheckpointDeleteResult",
    "RuntimeCompletionTrace",
    "RecoveryStatus",
    "RuntimeCapabilities",
    "RuntimeExecutionCapabilities",
    "RuntimeExecutionError",
    "admit_context_segments",
]
