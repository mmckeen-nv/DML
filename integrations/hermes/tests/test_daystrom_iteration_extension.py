"""Deterministic bounded iteration-extension policy tests."""
from __future__ import annotations

import sys
import types
from pathlib import Path


# The provider is shipped into Hermes, but this repository's CI intentionally
# tests its deterministic policy without installing Hermes itself. Supply the
# minimum host interfaces needed to import the plugin module.
agent_module = types.ModuleType("agent")
memory_provider_module = types.ModuleType("agent.memory_provider")
setattr(memory_provider_module, "MemoryProvider", object)
setattr(agent_module, "memory_provider", memory_provider_module)
sys.modules.setdefault("agent", agent_module)
sys.modules.setdefault("agent.memory_provider", memory_provider_module)

constants_module = types.ModuleType("hermes_constants")
setattr(constants_module, "get_default_hermes_root", lambda: Path.home() / ".hermes")
setattr(constants_module, "get_hermes_home", lambda: Path.home() / ".hermes")
sys.modules.setdefault("hermes_constants", constants_module)

hermes_cli_module = types.ModuleType("hermes_cli")
config_module = types.ModuleType("hermes_cli.config")
setattr(config_module, "cfg_get", lambda *_args, **_kwargs: None)
setattr(config_module, "load_config", lambda: {})
setattr(hermes_cli_module, "config", config_module)
sys.modules.setdefault("hermes_cli", hermes_cli_module)
sys.modules.setdefault("hermes_cli.config", config_module)

from integrations.hermes.plugins.daystrom_dml import _iteration_extension_decision


def _state(**overrides):
    state = {
        "user_message": "Implement the remaining change",
        "recent_execution_text": "Updated the parser and verified the focused test passed.",
        "recent_tool_call_count": 3,
        "recent_tool_result_count": 3,
        "requested_extra_iterations": 30,
        "pending_verification": True,
        "recent_progress": True,
    }
    state.update(overrides)
    return state


def test_iteration_extension_allows_progress_and_pending_verification():
    decision = _iteration_extension_decision(_state(requested_extra_iterations=18))

    assert decision == {
        "decision": "grant",
        "source": "daystrom_dml_deterministic",
        "reason_codes": ["recent_concrete_progress", "bounded_verification_pending"],
        "extra_iterations": 18,
    }


def test_iteration_extension_caps_each_grant_at_thirty():
    decision = _iteration_extension_decision(_state(requested_extra_iterations=200))
    assert decision["decision"] == "grant"
    assert decision["extra_iterations"] == 30


def test_iteration_extension_respects_remaining_hard_cap_capacity():
    decision = _iteration_extension_decision(
        _state(requested_extra_iterations=30, budget_used=292, hard_cap=300)
    )
    assert decision["decision"] == "grant"
    assert decision["extra_iterations"] == 8


def test_iteration_extension_denies_zero_request_and_reached_hard_cap():
    zero = _iteration_extension_decision(_state(requested_extra_iterations=0))
    capped = _iteration_extension_decision(_state(budget_used=300, hard_cap=300))
    assert zero["reason_codes"] == ["no_extension_requested"]
    assert zero["extra_iterations"] == 0
    assert capped["reason_codes"] == ["hard_cap_reached"]
    assert capped["extra_iterations"] == 0


def test_iteration_extension_denies_explicit_stop():
    decision = _iteration_extension_decision(_state(user_message="Stop now and cancel the run"))
    assert decision["decision"] == "deny"
    assert decision["extra_iterations"] == 0
    assert "explicit_stop_or_cancel" in decision["reason_codes"]


def test_iteration_extension_denies_repeated_identical_failure_without_progress():
    failure = "Error: connection refused"
    decision = _iteration_extension_decision(
        _state(
            recent_execution_text="\n".join([failure, failure, failure]),
            recent_progress=False,
            pending_verification=False,
        )
    )
    assert decision["decision"] == "deny"
    assert "repeated_identical_failure" in decision["reason_codes"]


def test_iteration_extension_denies_insufficient_progress():
    decision = _iteration_extension_decision(
        _state(recent_execution_text="Waiting for another attempt", recent_progress=False)
    )
    assert decision["decision"] == "deny"
    assert decision["reason_codes"] == ["insufficient_progress_evidence"]
