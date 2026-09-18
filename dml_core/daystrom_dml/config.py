"""Configuration loader with YAML defaults and environment overrides."""
from __future__ import annotations

import contextlib
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, get_args, get_type_hints

import yaml  # type: ignore[import-untyped]

from .settings import DMLSettings

ENV_PREFIX = "DML_"
_RESERVED_ENV_KEYS = {
    f"{ENV_PREFIX}HOST",
    f"{ENV_PREFIX}PORT",
    f"{ENV_PREFIX}CONFIG",
    f"{ENV_PREFIX}CONFIG_PATH",
}
_NESTED_ROOTS = {"persistence", "rag_store", "literal", "budgets", "dpm"}
# These names are consumed outside DMLSettings. Keep this explicit: an unrecognised
# DML_* name must still reach the profile validator and fail closed.
_OPERATIONAL_ENV_KEYS = {
    # Authentication and remote provider clients.
    "DML_API_TOKEN", "DML_ADMIN_TOKEN", "DML_API_BASE", "DML_API_KEY",
    "DML_PROVIDER_URL", "DML_TENANT_ID", "DML_SESSION_ID",
    # server.py and ingestion_limits.py.
    "DML_LOG_LEVEL", "DML_VISUALIZER_URL", "DML_VISUALIZER_PORT", "DML_VISUALIZER_PATH",
    "DML_PERSIST_NGC_KEY", "DML_NVIDIA_API_KEY", "DML_NVIDIA_INFERENCE_URL",
    "DML_NVIDIA_INFERENCE_TIMEOUT", "DML_MAX_UPLOAD_BYTES", "DML_MAX_UPLOAD_FILE_SIZE",
    "DML_MAX_DECOMPRESSED_BYTES", "DML_MAX_ARCHIVE_MEMBER_SIZE", "DML_MAX_ARCHIVE_MEMBERS",
    "DML_MAX_ARCHIVE_DEPTH", "DML_MAX_INGEST_DOCUMENTS", "DML_MAX_INGEST_CHUNKS",
    "DML_MAX_INGEST_TOKENS",
    # Installer, wrapper, worker, and example-client entry points.
    "DML_BUILD_CUDA", "DML_STORE", "DML_SKIP_VENV_REEXEC", "DML_AUDIT_ACTOR",
    "DML_SKILL_TARGET", "DML_INSTALL_EXTRAS", "DML_SCRIPT", "DML_REQUIRE_GPU",
    "DML_CLIENT_ID", "DML_INSTANCE_ID", "DML_FRESH_STORE_DEFAULT", "DML_LIVE_STORE_DEFAULT",
    "DML_FRESH_STORE", "DML_LIVE_STORE", "DML_PLAYGROUND_STORAGE", "DML_BASE_URL",
    "DML_SERVICE_URL",
}
_ENV_INTEGER = re.compile(r"(?:0|-?[1-9][0-9]*)\Z")
_ENV_FLOAT = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")
# Preserve ambiguity until the final merge so a higher-priority source may
# replace it, while a surviving conflict fails raw type validation.
_CONFLICTING_ENV_VALUES = object()


def _profile_environment_fields() -> Dict[tuple[str, ...], Any]:
    """Derive environment scalar types without constructing settings models."""
    from .contracts.profile import PROFILE_EXTRA_BOOL_PATHS

    fields: Dict[tuple[str, ...], Any] = {}

    def visit(model: type, prefix: tuple[str, ...] = ()) -> None:
        declared = getattr(model, "model_fields", None)
        if declared is None:  # pragma: no cover - Pydantic v1 compatibility
            declared = getattr(model, "__fields__")
        annotations = get_type_hints(model)
        for name in declared:
            annotation = annotations[name]
            path = (*prefix, name)
            if hasattr(annotation, "model_fields") or hasattr(annotation, "__fields__"):
                visit(annotation, path)
            else:
                fields[path] = annotation

    visit(DMLSettings)
    fields.update({path: bool for path in PROFILE_EXTRA_BOOL_PATHS})
    return fields


def _normalise_environment_scalar(value: str, annotation: Any) -> Any:
    """Only environment strings may become canonical typed profile values."""
    allowed = get_args(annotation) or (annotation,)
    if value == "null" and type(None) in allowed:
        return None
    if bool in allowed and value in {"true", "false"}:
        return value == "true"
    if int in allowed and _ENV_INTEGER.fullmatch(value):
        try:
            return int(value)
        except ValueError:  # Excessively long integers remain invalid typed values.
            return value
    if float in allowed and _ENV_FLOAT.fullmatch(value):
        result = float(value)
        if math.isfinite(result):
            return result
    return value


def load_config(
    path: str | os.PathLike | None = None,
    *,
    overrides: Dict[str, Any] | None = None,
) -> DMLSettings:
    """Load the DML configuration and validate it."""

    config_file = _resolve_config_path(path)
    base_config = _read_yaml(config_file)
    env_file_vars = _load_env_files(config_file)
    # Take one environment snapshot so both merge passes observe the same input.
    process_environment = dict(os.environ)
    environment = {**env_file_vars, **process_environment}
    env_overrides = _collect_env_overrides(environment, include_process_environment=False)
    combined = _deep_merge(base_config, env_overrides)
    if overrides:
        combined = _deep_merge(combined, overrides)
    if combined.get("production_profile") is not None:
        env_overrides = _deep_merge(
            _collect_env_overrides(
                env_file_vars, strict_profile=True, include_process_environment=False,
            ),
            _collect_env_overrides(
                process_environment, strict_profile=True, include_process_environment=False,
            ),
        )
        combined = _deep_merge(base_config, env_overrides)
        if overrides:
            combined = _deep_merge(combined, overrides)
    settings = _validate_settings(combined)
    return settings


def _resolve_config_path(path: str | os.PathLike | None) -> Path:
    if path is not None:
        return Path(path)
    env_override = os.environ.get(f"{ENV_PREFIX}CONFIG_PATH") or os.environ.get(
        f"{ENV_PREFIX}CONFIG"
    )
    if env_override:
        return Path(env_override)
    return Path(__file__).with_name("config.yaml")


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):  # pragma: no cover - defensive
        raise ValueError("Configuration root must be a mapping")
    return data


def _collect_env_overrides(
    env_values: Mapping[str, str] | None = None,
    *,
    strict_profile: bool = False,
    include_process_environment: bool = True,
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    merged = dict(env_values or {})
    if include_process_environment:
        merged.update(os.environ)
    fields = _profile_environment_fields() if strict_profile else {}
    aliases = {}
    for known_path in fields:
        aliases["_".join(known_path).upper()] = known_path
        aliases["__".join(known_path).upper()] = known_path
    for key, value in merged.items():
        if not key.startswith(ENV_PREFIX):
            continue
        if key in _RESERVED_ENV_KEYS:
            continue
        if strict_profile and key in _OPERATIONAL_ENV_KEYS:
            continue
        path = _decode_env_path(key[len(ENV_PREFIX) :])
        field_path = aliases.get(key[len(ENV_PREFIX) :])
        if field_path is not None:
            path = list(field_path)
        elif strict_profile:
            # Do not turn misspelled environment names into recognised fields by
            # removing separators.
            path = [key[len(ENV_PREFIX) :].lower()]
        if not path:
            continue
        converted = _normalise_environment_scalar(value, fields[tuple(path)]) if tuple(path) in fields else value
        _assign_path(overrides, path, converted, reject_conflicts=strict_profile)
    return overrides


def _decode_env_path(raw_key: str) -> List[str]:
    if not raw_key:
        return []
    lowered = raw_key.lower()
    if "__" in raw_key:
        segments = [segment.lower() for segment in raw_key.split("__") if segment]
        return segments
    segments = [segment for segment in lowered.split("_") if segment]
    if not segments:
        return []
    root = segments[0]
    if root in _NESTED_ROOTS and len(segments) > 1:
        tail = "_".join(segments[1:])
        return [root, tail]
    return ["_".join(segments)]


def _assign_path(
    container: Dict[str, Any], path: Iterable[str], value: Any, *, reject_conflicts: bool = False,
) -> None:
    iterator = iter(path)
    current = container
    try:
        first = next(iterator)
    except StopIteration:
        return
    key = first
    for segment in iterator:
        existing = current.get(key)
        if not isinstance(existing, dict):
            if reject_conflicts and key in current:
                current[key] = _CONFLICTING_ENV_VALUES
                return
            existing = {}
            current[key] = existing
        current = existing
        key = segment
    existing = current.get(key)
    if reject_conflicts and key in current and (type(existing) is not type(value) or existing != value):
        current[key] = _CONFLICTING_ENV_VALUES
    else:
        current[key] = value


def _deep_merge(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_settings(payload: Dict[str, Any]) -> DMLSettings:
    from .contracts.profile import validate_profile_config

    # Validate before Pydantic can coerce types or discard nested unknown keys.
    validate_profile_config(payload)
    try:
        if hasattr(DMLSettings, "model_validate"):
            settings = DMLSettings.model_validate(payload)  # type: ignore[attr-defined]
        else:  # pragma: no cover - Pydantic v1 fallback
            settings = DMLSettings(**payload)
    except Exception as exc:  # pragma: no cover - validation errors bubble to caller
        raise ValueError(f"Invalid configuration: {exc}") from exc
    if hasattr(settings, "budgets"):
        with contextlib.suppress(Exception):
            settings.budgets.validate_totals()
    validate_profile_config(settings.as_dict())
    return settings


def _load_env_files(config_file: Path | None) -> Dict[str, str]:
    env_vars: Dict[str, str] = {}
    candidates = _discover_env_files(config_file)
    for candidate in candidates:
        try:
            file_vars = _parse_env_file(candidate)
        except OSError:  # pragma: no cover - filesystem issues bubble up elsewhere
            continue
        for key, value in file_vars.items():
            env_vars.setdefault(key, value)
    return env_vars


def _discover_env_files(config_file: Path | None) -> List[Path]:
    candidates: List[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen or not path.exists() or not path.is_file():
            return
        seen.add(resolved)
        candidates.append(path)

    _add(Path.cwd() / ".env")
    _add(Path.cwd() / ".env.local")
    if config_file is not None:
        parent = config_file.parent
        _add(parent / ".env")
        _add(parent / ".env.local")
    return candidates


def _parse_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.lower().startswith("export "):
                stripped = stripped[7:].lstrip()
            if "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", 1)
            key = key.strip()
            value = raw_value.strip()
            if not key:
                continue
            if value and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            values[key] = value
    return values
