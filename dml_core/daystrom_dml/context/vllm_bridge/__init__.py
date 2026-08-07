"""Experimental vLLM 0.20 KV-connector bridge for Daystrom cooperative KV offload.

This package provides a controller-gated vertical slice that subclasses vLLM's
``SimpleCPUOffloadConnector`` and gates save/restore by request-level
``kv_transfer_params`` and exact checkpoint identity.

* :mod:`policy` is a pure, runtime-neutral authorization policy.  It imports
  only the Python standard library and the dependency-light
  ``daystrom_dml.api_contracts`` helpers, so it is importable and testable on
  macOS/Windows without vLLM or torch installed.
* :mod:`connector` imports vLLM and torch and is therefore dependency-gated:
  importing this package does not import the connector module.  Tests that
  exercise the connector stub the minimal vLLM modules explicitly.

The contract is version-pinned to vLLM 0.20.0 and the
``daystrom-vllm-kv-v1`` schema.
"""
from __future__ import annotations

from daystrom_dml.context.vllm_bridge.policy import (
    DAYSTROM_KV_SCHEMA_VERSION,
    DaystromKVAuthorizationError,
    DaystromKVPolicy,
    DaystromKVRecord,
    DaystromKVRequest,
    DaystromKVDecision,
)

__all__ = [
    "DAYSTROM_KV_SCHEMA_VERSION",
    "DaystromKVAuthorizationError",
    "DaystromKVDecision",
    "DaystromKVPolicy",
    "DaystromKVRecord",
    "DaystromKVRequest",
]
