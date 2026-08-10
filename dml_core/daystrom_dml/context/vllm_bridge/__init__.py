"""Experimental vLLM 0.20 KV-connector bridge for Daystrom cooperative KV offload.

This package provides a GPU-first hybrid vertical slice. vLLM checks local GPU
Automatic Prefix Caching before the controller-gated
``SimpleCPUOffloadConnector`` fallback, which gates managed save/restore by
request-level ``kv_transfer_params`` and exact checkpoint identity. GPU APC is
opportunistic and is not a controller authorization boundary.

* :mod:`policy` is a pure, runtime-neutral authorization policy.  It imports
  only the Python standard library and the dependency-light
  ``daystrom_dml.api_contracts`` helpers, so it is importable and testable on
  macOS/Windows without vLLM or torch installed.
* :mod:`connector` imports vLLM and torch and is therefore dependency-gated:
  importing this package does not import the connector module.  Tests that
  exercise the connector stub the minimal vLLM modules explicitly.

The contract is version-pinned to vLLM 0.20.0. Legacy single-operation requests
use ``daystrom-vllm-kv-v1``; compound restore-and-save requests use the distinct
``daystrom-vllm-kv-transition-v1`` schema.
"""
from __future__ import annotations

from daystrom_dml.context.vllm_bridge.policy import (
    DAYSTROM_KV_SCHEMA_VERSION,
    DAYSTROM_KV_TRANSITION_SCHEMA_VERSION,
    DaystromKVAuthorizationError,
    DaystromKVPolicy,
    DaystromKVRecord,
    DaystromKVRequest,
    DaystromKVDecision,
    build_kv_transition_params,
)

__all__ = [
    "DAYSTROM_KV_SCHEMA_VERSION",
    "DAYSTROM_KV_TRANSITION_SCHEMA_VERSION",
    "DaystromKVAuthorizationError",
    "DaystromKVDecision",
    "DaystromKVPolicy",
    "DaystromKVRecord",
    "DaystromKVRequest",
    "build_kv_transition_params",
]
