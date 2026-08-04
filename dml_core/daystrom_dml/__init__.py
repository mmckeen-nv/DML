"""Daystrom Memory Lattice package with lightweight, lazy public imports."""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .api_client import DMLClient
    from .config import load_config
    from .dml_adapter import DMLAdapter, PersistenceCommitError
    from .personality_matrix import PersonalityMatrix

__all__ = [
    "DMLAdapter",
    "DMLClient",
    "PersistenceCommitError",
    "PersonalityMatrix",
    "load_config",
    "utils",
]

_LAZY_EXPORTS = {
    "DMLAdapter": (".dml_adapter", "DMLAdapter"),
    "DMLClient": (".api_client", "DMLClient"),
    "PersistenceCommitError": (".dml_adapter", "PersistenceCommitError"),
    "PersonalityMatrix": (".personality_matrix", "PersonalityMatrix"),
    "load_config": (".config", "load_config"),
}


def __getattr__(name: str) -> Any:
    if name == "utils":
        value = import_module(".utils", __name__)
    elif name in _LAZY_EXPORTS:
        module_name, attribute = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attribute)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
