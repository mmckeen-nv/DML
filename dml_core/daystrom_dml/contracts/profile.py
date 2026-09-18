"""Admission contract for the candidate local receipt profile.

Configuration admission freezes a support boundary; it does not qualify a
platform, an embedding provider, or the wider repository for production use.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from pathlib import Path
import platform
import sys
from types import UnionType
from typing import Union, get_args, get_origin, get_type_hints


PROFILE_ID = "dml-receipted-local-v1"


class ProductionProfileError(ValueError):
    """The requested configuration or runtime is outside the frozen profile."""


_REQUIRED_VALUES = {
    "model_name": "dummy",
    "llm_backend": "dummy",
    "strict_embedding_required": True,
    "strict_llm_required": False,
    "persistence.enable": False,
    "persistence.interval_sec": 0,
    "persistence.journal": True,
    "persistence.receipts": True,
    "rag_store.enable": False,
    "ann_min_items": 0,
    "checkpoint_interval_seconds": 0,
    "skip_rag_state_import": True,
    "survival_ledger_enabled": False,
    "background_processing_enabled": False,
    "enable_stm_controller": False,
    "enable_workflow_cache": False,
    "enable_quality_on_retrieval": False,
    "mirror_agentic_memory_to_rag": False,
    "gpu_acceleration": False,
    "dpm.enable": False,
    "dpm.mode": "disabled",
    "dpm.include_in_context": False,
    "dpm.include_in_preamble": False,
}

# These existing adapter extras have no Pydantic fields. Restrict each to its
# documented inactive shape rather than accepting arbitrary experimental config.
_EXTRA_SHAPES = {
    "skip_rag_state_import": bool,
    "survival_ledger_enabled": bool,
    "background_processing_enabled": bool,
    "agentic_mode": {"enabled": bool, "router": {"enabled": bool}},
    "router": {"enabled": bool},
    "dml": {
        "agentic_mode": {"enabled": bool, "router": {"enabled": bool}},
        "router": {"enabled": bool},
    },
    "dml.agentic_mode.enabled": bool,
    "dml.router.enabled": bool,
}


def _bool_paths(shapes, prefix=()):
    for name, shape in shapes.items():
        path = (*prefix, name)
        if shape is bool:
            yield path
        else:
            yield from _bool_paths(shape, path)


PROFILE_EXTRA_BOOL_PATHS = tuple(_bool_paths(_EXTRA_SHAPES))

_PYTHON_APIS = [
    "ingest_memory_receipted", "retire_memory_receipted",
    "supersede_memory_receipted", "update_memory_receipted",
    "promote_memories_receipted", "retrieve_context",
    "inspect_memory_retention", "durability_status",
    "production_profile_status", "close",
]
_HTTP_APIS = [
    ["GET", "/health"], ["GET", "/api/contracts"],
    ["POST", "/api/recall"], ["POST", "/api/remember/receipt"],
    ["POST", "/api/memory/retention/inspect"],
    ["POST", "/api/memory/retire/receipt"],
    ["POST", "/api/memory/supersede/receipt"],
    ["POST", "/api/memory/update/receipt"],
    ["POST", "/api/memory/promote/receipt"],
]

_PROFILE = {
    "schema_version": "dml-production-profile-v1",
    "profile_id": PROFILE_ID,
    "maturity": "candidate",
    "production_ready": False,
    "qualification_pending": True,
    "configuration": {
        "selector": "production_profile",
        "default": None,
        "required_values": _REQUIRED_VALUES,
        "unknown_keys": "rejected",
        "coercion": "canonical_environment_values_only",
        "embedding_identity": "explicit_nonempty_immutable_operator_assertion",
        "embedding_identity_is_weight_attestation": False,
    },
    "apis": {"python": _PYTHON_APIS, "http": _HTTP_APIS},
    "authority": {
        "path": "storage_dir/dml_state.sqlite3",
        "journal_schema_versions": [2, 3, 4],
        "fresh_schema_without_outbox": 2,
        "fresh_schema_with_outbox": 3,
        "existing_schema_without_outbox": [2],
        "existing_schema_with_outbox": [3, 4],
        "schema_4_creation": "explicit_side_by_side_schema_2_migration_only",
        "receipt_schema_version": 1,
        "state_record_identity_schema_version": 1,
        "embedding_contract": "dml-embedding-contract-v1",
        "outbox_formats": {"3": "dml-journal-outbox-v1", "4": "dml-journal-outbox-v2"},
        "automatic_migration": False,
        "legacy_json_and_rag_import": False,
        "required_coordination_files": [
            "dml_state.sqlite3.identity.json", "dml_state.sqlite3.init.lock",
            ".dml_store.lock",
        ],
        "runtime_or_recovery_sidecars": [
            "dml_state.sqlite3-wal", "dml_state.sqlite3-shm",
            "dml_state.sqlite3.migration.json", ".dml_store.lock.json",
        ],
        "backup": "consistent_full_sqlite_backup_and_identity_with_writers_stopped",
        "history_retention": "indefinite_no_erasure_or_compaction",
    },
    "scope": {
        "callers": "trusted_cooperating_local_service_operators",
        "fields": ["tenant_id", "client_id", "session_id", "instance_id"],
        "matching": "exact_including_null_members",
        "tenant_required": True,
        "per_token_tenant_authorization": False,
    },
    "platform": {
        "admitted_operating_systems": ["Linux", "Darwin", "Windows"],
        "python_implementation": "CPython",
        "python_versions": ["3.10", "3.11", "3.12", "3.13"],
        "host_count": 1,
        "filesystem": "local_storage_with_sqlite_wal_advisory_locks_atomic_replace_and_sync",
        "network_filesystems": False,
        "noncooperating_writers": False,
        "filesystem_qualification_complete": False,
        "power_loss_qualified": False,
        "hardware_combinations_qualified": False,
    },
    "dependencies": {
        "installation": "project_base_dependencies_in_pyproject.toml",
        "sqlite": "python_standard_library_sqlite3_with_wal_support",
        "embedding": "operator_supplied_native_provider_with_explicit_immutable_identity",
        "embedding_provider_packages": "required_by_selected_provider",
        "random_embedding_fallback": False,
        "generation_model_required": False,
        "faiss_ann_cuda_native_kv_required": False,
        "dependency_combinations_qualified": False,
    },
    "limits": {
        "canonical_request_bytes": 1024 * 1024,
        "idempotency_key_utf8_bytes": 256,
        "scope_member_utf8_bytes": 256,
        "kind_utf8_bytes": 256,
        "reason_utf8_bytes": 1024,
        "promotion_source_count": [1, 32],
        "retrieval_top_k": [1, 10],
        "capacity": "configured_positive_integer_no_implicit_eviction",
        "retirement_releases_capacity": False,
        "context_token_counts": "estimates_not_final_model_input_budget_proof",
        "full_history_validation": "cost_grows_with_retained_history",
        "throughput_latency_or_store_size_slo": None,
    },
    "retry": {
        "identity": "identical_canonical_request_and_full_scoped_idempotency_key",
        "receipt": "immutable_historical_commit_not_current_liveness",
        "timeout_or_disconnect": "does_not_establish_noncommit",
        "uncertain_outcome": "recover_authority_then_retry_identical_request_and_key",
        "conflicting_key": "reject_without_mutation",
    },
    "excluded": [
        "legacy_mutation_and_hybrid_retrieval", "llm_generation",
        "persistent_rag", "projection_and_outbox_backend_apis", "ann", "dpm", "dcn",
        "native_kv", "kv_fabric", "automatic_aging_or_promotion", "semantic_checkpoints",
        "physical_erasure", "recursive_derivation", "cascading_invalidation",
        "all_component_transactions", "durable_full_context_replay",
    ],
}


def production_profile_contract() -> dict:
    """Return detached, data-independent capability and qualification metadata."""
    return deepcopy(_PROFILE)


def _selected(profile_id) -> bool:
    if profile_id is None:
        return False
    if type(profile_id) is not str or profile_id != PROFILE_ID:
        raise ProductionProfileError("Unsupported production_profile")
    return True


def _fields(model) -> dict:
    return getattr(model, "model_fields", None) or model.__fields__


def _check_value(value, annotation, path: str) -> None:
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        for choice in get_args(annotation):
            try:
                _check_value(value, choice, path)
                return
            except ProductionProfileError:
                pass
        raise ProductionProfileError(f"Invalid profile configuration type: {path}")
    if annotation is type(None):
        valid = value is None
    elif annotation is bool:
        valid = type(value) is bool
    elif annotation is int:
        valid = type(value) is int
    elif annotation is float:
        try:
            valid = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            valid = False
    elif annotation is str:
        valid = type(value) is str
    elif annotation is Path:
        valid = (type(value) is str and bool(value.strip())) or isinstance(value, Path)
    elif isinstance(annotation, type) and (hasattr(annotation, "model_fields") or hasattr(annotation, "__fields__")):
        _check_model(value, annotation, path)
        return
    else:
        raise ProductionProfileError(f"Unrecognized profile configuration field: {path}")
    if not valid:
        raise ProductionProfileError(f"Invalid profile configuration type: {path}")


def _check_model(payload, model, path: str = "") -> None:
    if not isinstance(payload, Mapping):
        raise ProductionProfileError(f"Profile configuration requires an object: {path}")
    fields = _fields(model)
    annotations = get_type_hints(model)
    for name, value in payload.items():
        qualified = f"{path}.{name}" if path else str(name)
        if type(name) is not str or name not in fields:
            raise ProductionProfileError(f"Unknown profile configuration field: {qualified}")
        _check_value(value, annotations[name], qualified)
        if value is not None:
            field = fields[name]
            constraints = getattr(field, "metadata", [getattr(field, "field_info", field)])
            for constraint in constraints:
                for key, compare in (("ge", lambda x, y: x >= y), ("gt", lambda x, y: x > y),
                                     ("le", lambda x, y: x <= y), ("lt", lambda x, y: x < y)):
                    bound = getattr(constraint, key, None)
                    if bound is not None and not compare(value, bound):
                        raise ProductionProfileError(f"Invalid profile configuration range: {qualified}")


def _check_extra(value, shape, path: str) -> None:
    if shape is bool:
        if type(value) is not bool:
            raise ProductionProfileError(f"Invalid profile configuration type: {path}")
        if path not in {"skip_rag_state_import", "survival_ledger_enabled", "background_processing_enabled"} and value:
            raise ProductionProfileError(f"Excluded profile feature: {path}")
        return
    if not isinstance(value, Mapping):
        raise ProductionProfileError(f"Profile configuration requires an object: {path}")
    for key, item in value.items():
        if type(key) is not str or key not in shape:
            raise ProductionProfileError(f"Unknown profile configuration field: {path}")
        _check_extra(item, shape[key], f"{path}.{key}")


def _merge(base: dict, override: Mapping) -> dict:
    result = deepcopy(base)
    for name, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(name), dict):
            result[name] = _merge(result[name], value)
        else:
            result[name] = value
    return result


def _defaults(model) -> dict:
    """Read declared defaults without constructing or coercing a settings model."""
    result = {}
    for name, field in _fields(model).items():
        value = field.default
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        elif hasattr(value, "dict"):
            value = value.dict()
        result[name] = deepcopy(value)
    return result


def validate_profile_config(config: Mapping) -> str | None:
    """Check raw or resolved settings without coercion, I/O, or model loading.

Absent declared fields use DMLSettings defaults. Environment normalization is
performed by the loader before this function; direct/YAML strings cannot stand
in for numeric or boolean settings. Legacy configurations remain unchanged.
    """
    if not isinstance(config, Mapping):
        raise ProductionProfileError("Profile configuration requires an object")
    profile_id = config.get("production_profile")
    if not _selected(profile_id):
        return None
    from ..settings import DMLSettings

    declared = {}
    for name, value in config.items():
        if name == "production_profile":
            continue
        if name in _EXTRA_SHAPES:
            _check_extra(value, _EXTRA_SHAPES[name], name)
        else:
            declared[name] = value
    _check_model(declared, DMLSettings)
    effective = _merge(_defaults(DMLSettings), config)
    for path, required in _REQUIRED_VALUES.items():
        value = effective
        for member in path.split("."):
            value = value.get(member) if isinstance(value, Mapping) else None
        if type(value) is not type(required) or value != required:
            raise ProductionProfileError(f"Unsupported profile configuration: {path}")
    identity = effective["persistence"].get("receipt_embedding_identity")
    _validate_identity(identity)
    budget = effective["budgets"]
    if sum(budget[name] for name in ("semantic_pct", "literal_pct", "free_pct")) > 1.0 + 1e-6:
        raise ProductionProfileError("Profile token budget percentages exceed 1.0")
    return PROFILE_ID


def _validate_identity(identity) -> None:
    try:
        valid = type(identity) is str and bool(identity.strip()) and len(identity.encode("utf-8")) <= 1024
    except UnicodeError:
        valid = False
    if not valid:
        raise ProductionProfileError("Profile requires explicit persistence.receipt_embedding_identity")


def validate_profile_embedding(profile_id, embedder, explicit_identity) -> None:
    """Check startup embedding admission without preparing any vector."""
    if not _selected(profile_id):
        return
    from ..embeddings import RandomEmbedder, SentenceTransformerEmbedder
    from ..services.receipt_ingestion import embedding_identity

    _validate_identity(explicit_identity)
    if isinstance(embedder, RandomEmbedder):
        raise ProductionProfileError("Random embeddings are excluded from the production profile")
    if isinstance(embedder, SentenceTransformerEmbedder) and getattr(embedder, "_model", None) is None:
        raise ProductionProfileError("Fallback embeddings are excluded from the production profile")
    if not callable(getattr(embedder, "embed", None)):
        raise ProductionProfileError("Profile requires a native embedding provider")
    try:
        identity = embedding_identity(embedder, explicit_identity)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ProductionProfileError("Invalid production profile embedding identity") from exc
    if identity.get("mode") != "native":
        raise ProductionProfileError("Profile requires a native embedding provider")


def validate_profile_authority(profile_id, schema_version, outbox_enabled) -> None:
    """Require explicit format selection; never silently adopt outbox authority."""
    if not _selected(profile_id):
        return
    if type(outbox_enabled) is not bool or type(schema_version) is not int:
        raise ProductionProfileError("Invalid production profile authority")
    allowed = (3, 4) if outbox_enabled else (2,)
    if schema_version not in allowed:
        raise ProductionProfileError("Journal format does not match production profile outbox selection")


def validate_profile_platform(profile_id) -> None:
    """Check candidate runtime admission, without asserting qualification."""
    if not _selected(profile_id):
        return
    if (platform.python_implementation() != "CPython"
            or platform.system() not in {"Linux", "Darwin", "Windows"}
            or sys.version_info[:2] not in {(3, 10), (3, 11), (3, 12), (3, 13)}):
        raise ProductionProfileError("Runtime is outside the candidate production profile")
