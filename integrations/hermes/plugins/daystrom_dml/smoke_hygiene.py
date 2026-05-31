#!/usr/bin/env python3
"""Smoke checks for Daystrom DML Hermes memory hygiene."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


PLUGIN_PATH = Path(__file__).with_name("__init__.py")


def _install_stubs() -> None:
    agent = types.ModuleType("agent")
    memory_provider = types.ModuleType("agent.memory_provider")

    class MemoryProvider:
        pass

    memory_provider.MemoryProvider = MemoryProvider
    sys.modules.setdefault("agent", agent)
    sys.modules.setdefault("agent.memory_provider", memory_provider)

    hermes_constants = types.ModuleType("hermes_constants")
    hermes_constants.get_hermes_home = lambda: Path("/Users/markmckeen/.hermes")
    sys.modules.setdefault("hermes_constants", hermes_constants)

    hermes_cli = types.ModuleType("hermes_cli")
    config = types.ModuleType("hermes_cli.config")
    config.cfg_get = lambda cfg, *keys: {}
    config.load_config = lambda: {}
    sys.modules.setdefault("hermes_cli", hermes_cli)
    sys.modules.setdefault("hermes_cli.config", config)


def _load_plugin():
    _install_stubs()
    spec = importlib.util.spec_from_file_location("daystrom_dml_plugin_smoke", PLUGIN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load plugin from {PLUGIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    plugin = _load_plugin()

    durable = plugin._classify_turn_memory(
        "Remember: Mark prefers concise status updates and no inference/frontier routing.",
        "Updated plugins/daystrom_dml/__init__.py and ran py_compile successfully.",
    )
    assert durable["keep"] is True, durable
    assert durable["memory_class"] in {"preference", "constraint"}, durable
    assert "hygiene_score" not in durable

    noisy = plugin._classify_turn_memory(
        "<memory-context>secret-looking context</memory-context>\nsmoke-test record",
        "assistant: | assistant: I’ll inspect files.\nChunk ID: abc\nProcess exited with code 0",
    )
    assert noisy["keep"] is False, noisy
    assert "smoke_or_self_test" in noisy["reasons"], noisy

    handoff = plugin._handoff_fragment(
        "assistant",
        "=== Daystrom DML Retrieved Memory ===\nold scaffold\n=== Other ===\nDone. Tests passed.",
    )
    assert "Daystrom DML Retrieved Memory" not in handoff, handoff
    assert handoff, handoff

    print("daystrom_dml hygiene smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
