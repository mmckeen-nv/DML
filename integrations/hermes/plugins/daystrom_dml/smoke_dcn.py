#!/usr/bin/env python3
"""Smoke checks for Daystrom DCN Hermes integration."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path


PLUGIN_PATH = Path(__file__).with_name("__init__.py")


def _install_stubs(config_block=None) -> None:
    agent = types.ModuleType("agent")
    memory_provider = types.ModuleType("agent.memory_provider")

    class MemoryProvider:
        pass

    setattr(memory_provider, "MemoryProvider", MemoryProvider)
    sys.modules["agent"] = agent
    sys.modules["agent.memory_provider"] = memory_provider

    hermes_constants = types.ModuleType("hermes_constants")
    setattr(hermes_constants, "get_hermes_home", lambda: Path("/Users/markmckeen/.hermes"))
    sys.modules["hermes_constants"] = hermes_constants

    hermes_cli = types.ModuleType("hermes_cli")
    config = types.ModuleType("hermes_cli.config")
    cfg = {"memory": {"daystrom_dml": config_block or {}}}
    setattr(config, "cfg_get", lambda data, *keys: data.get(keys[0], {}).get(keys[1], {}) if len(keys) == 2 else {})
    setattr(config, "load_config", lambda: cfg)
    sys.modules["hermes_cli"] = hermes_cli
    sys.modules["hermes_cli.config"] = config


def _load_plugin(config_block=None):
    _install_stubs(config_block)
    spec = importlib.util.spec_from_file_location("daystrom_dml_plugin_dcn_smoke", PLUGIN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load plugin from {PLUGIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _provider(plugin, *, mode="disabled", decision=None, fail_dcn=False, promotion=None):
    class FakeProvider(plugin.DaystromDMLProvider):
        def __init__(self):
            super().__init__()
            self.overlay_calls = 0
            self.resume_calls = 0
            self.retrieve_calls = 0
            self.policy_calls = 0

        def _personality_overlay(self, prompt):
            self.overlay_calls += 1
            return {
                "overlay": {
                    "rendered_text": "Identity: Citizen Snips.",
                    "style_directives": [],
                    "do_not_do": [],
                }
            }

        def _resume_block(self, session_id):
            self.resume_calls += 1
            return "=== Daystrom DML Active Continuity ===\n- Current focus: compact state."

        def _retrieve_block(self, query, session_id):
            self.retrieve_calls += 1
            return "=== Daystrom DML Retrieved Memory ===\n- Memory: compact semantic state."

        def _dcn_policy_decision(self, query):
            self.policy_calls += 1
            if fail_dcn:
                raise RuntimeError("forced dcn failure")
            if decision is not None:
                return decision
            return super()._dcn_policy_decision(query)

    old = os.environ.get("DAYSTROM_DCN_MODE")
    old_promotion = os.environ.get("DAYSTROM_DCN_PROMOTION_EVIDENCE")
    os.environ["DAYSTROM_DCN_MODE"] = mode
    if promotion is None:
        os.environ.pop("DAYSTROM_DCN_PROMOTION_EVIDENCE", None)
    else:
        import json
        os.environ["DAYSTROM_DCN_PROMOTION_EVIDENCE"] = json.dumps(promotion, sort_keys=True)
    try:
        return FakeProvider()
    finally:
        if old is None:
            os.environ.pop("DAYSTROM_DCN_MODE", None)
        else:
            os.environ["DAYSTROM_DCN_MODE"] = old
        if old_promotion is None:
            os.environ.pop("DAYSTROM_DCN_PROMOTION_EVIDENCE", None)
        else:
            os.environ["DAYSTROM_DCN_PROMOTION_EVIDENCE"] = old_promotion


def main() -> int:
    plugin = _load_plugin()

    assert plugin._DCN_MODES == {"disabled", "observe_only", "active_read", "active_learn"}

    # Disabled preserves the legacy DPM + heuristic DML path with no DCN telemetry.
    disabled = _provider(plugin, mode="disabled")
    disabled_prefetch = disabled.prefetch("resume from the previous task where we left off")
    assert "Daystrom Personality Matrix Overlay" in disabled_prefetch, disabled_prefetch
    assert "DML Active Continuity" in disabled_prefetch, disabled_prefetch
    assert disabled.resume_calls == 1, disabled.resume_calls
    assert disabled.retrieve_calls == 1, disabled.retrieve_calls
    assert disabled.policy_calls == 0, disabled.policy_calls
    assert disabled.dcn_observations() == [], disabled.dcn_observations()

    # Observe-only records intent but returns byte-identical legacy behavior.
    observe = _provider(plugin, mode="observe_only")
    observe_prefetch = observe.prefetch("resume from the previous task where we left off")
    assert observe_prefetch == disabled_prefetch, (observe_prefetch, disabled_prefetch)
    assert observe.overlay_calls == 1, observe.overlay_calls
    assert observe.resume_calls == 1, observe.resume_calls
    assert observe.retrieve_calls == 1, observe.retrieve_calls
    observations = observe.dcn_observations()
    assert len(observations) == 1, observations
    assert observations[0]["event"] == "dcn.observe", observations
    assert observations[0]["mode"] == "observe_only", observations
    assert observations[0]["would_apply_dpm"] is True, observations
    assert observations[0]["would_inject_dml"] is True, observations
    assert "resume from the previous task" not in str(observations), observations

    observe_greeting = _provider(plugin, mode="observe_only")
    greeting_prefetch = observe_greeting.prefetch("hello")
    assert "Daystrom Personality Matrix Overlay" in greeting_prefetch, greeting_prefetch
    assert "DML Active Continuity" not in greeting_prefetch, greeting_prefetch
    assert observe_greeting.resume_calls == 0, observe_greeting.resume_calls
    assert observe_greeting.retrieve_calls == 0, observe_greeting.retrieve_calls
    greeting_observations = observe_greeting.dcn_observations()
    assert greeting_observations[0]["would_inject_dml"] is False, greeting_observations
    assert greeting_observations[0]["would_call_retrieve"] is False, greeting_observations

    # Active-read casual turn: DCN chooses bounded DPM overlay only.
    active_casual = _provider(plugin, mode="active_read")
    active_casual_prefetch = active_casual.prefetch("hello")
    assert "Daystrom Personality Matrix Overlay" in active_casual_prefetch, active_casual_prefetch
    assert "DML Active Continuity" not in active_casual_prefetch, active_casual_prefetch
    assert active_casual.overlay_calls == 1, active_casual.overlay_calls
    assert active_casual.resume_calls == 0, active_casual.resume_calls
    assert active_casual.retrieve_calls == 0, active_casual.retrieve_calls
    assert active_casual.policy_calls == 1, active_casual.policy_calls
    active_casual_event = active_casual.dcn_observations()[0]
    assert active_casual_event["event"] == "dcn.active_read", active_casual_event
    assert active_casual_event["decision"] == "overlay_only", active_casual_event
    assert active_casual_event["include_dpm"] is True, active_casual_event
    assert active_casual_event["retrieve_dml"] is False, active_casual_event

    # Active-read long-horizon turn: DCN chooses DPM + DML continuity/retrieval.
    active_resume = _provider(plugin, mode="active_read")
    active_resume_prefetch = active_resume.prefetch("resume from the previous task where we left off")
    assert "Daystrom Personality Matrix Overlay" in active_resume_prefetch, active_resume_prefetch
    assert "DML Active Continuity" in active_resume_prefetch, active_resume_prefetch
    assert "DML Retrieved Memory" in active_resume_prefetch, active_resume_prefetch
    assert active_resume.resume_calls == 1, active_resume.resume_calls
    assert active_resume.retrieve_calls == 1, active_resume.retrieve_calls
    assert active_resume.dcn_observations()[0]["decision"] == "retrieve", active_resume.dcn_observations()

    # Active-read contradiction: DCN suppresses stale DPM and skips DML.
    active_suppress = _provider(plugin, mode="active_read")
    suppressed = active_suppress.prefetch("Current-turn contradiction: do not use personality overlay for this reply")
    assert "Daystrom Personality Matrix Overlay" not in suppressed, suppressed
    assert "DML Active Continuity" not in suppressed, suppressed
    assert active_suppress.overlay_calls == 0, active_suppress.overlay_calls
    assert active_suppress.resume_calls == 0, active_suppress.resume_calls
    assert active_suppress.retrieve_calls == 0, active_suppress.retrieve_calls
    assert active_suppress.dcn_observations()[0]["decision"] == "suppress_overlay", active_suppress.dcn_observations()

    # Active-read DCN failure falls back to legacy behavior.
    active_fallback = _provider(plugin, mode="active_read", fail_dcn=True)
    fallback_prefetch = active_fallback.prefetch("resume from the previous task where we left off")
    assert fallback_prefetch == disabled_prefetch, (fallback_prefetch, disabled_prefetch)
    assert active_fallback.resume_calls == 1, active_fallback.resume_calls
    assert active_fallback.retrieve_calls == 1, active_fallback.retrieve_calls
    fallback_event = active_fallback.dcn_observations()[0]
    assert fallback_event["event"] == "dcn.active_read_fallback", fallback_event
    assert fallback_event["fallback"] is True, fallback_event

    # Active-learn fails closed to active-read without governed promotion evidence.
    active_learn_ungated = _provider(plugin, mode="active_learn")
    ungated_prefetch = active_learn_ungated.prefetch("hello")
    assert "Daystrom Personality Matrix Overlay" in ungated_prefetch, ungated_prefetch
    assert active_learn_ungated.dcn_requested_mode == "active_learn", active_learn_ungated.dcn_requested_mode
    assert active_learn_ungated.dcn_mode == "active_read", active_learn_ungated.dcn_mode
    ungated_event = active_learn_ungated.dcn_observations()[0]
    assert ungated_event["event"] == "dcn.active_read", ungated_event
    assert ungated_event["requested_mode"] == "active_learn", ungated_event

    promotion = {
        "promoted": True,
        "runtime_mode": "active_learn",
        "promotion_id": "promotion-123",
        "checkpoint_id": "chk-123",
        "rollback_command": "dml dcn policy rollback --checkpoint-id chk-123",
        "eval": {"passed": True, "deterministic_hash": "evalhash", "summary": {"failed_count": 0}},
        "hygiene": {"passed": True, "artifact_hash": "hyghash"},
    }
    active_learn = _provider(plugin, mode="active_learn", promotion=promotion)
    learned_prefetch = active_learn.prefetch("resume from the previous task where we left off")
    assert "Daystrom Personality Matrix Overlay" in learned_prefetch, learned_prefetch
    assert "DML Active Continuity" in learned_prefetch, learned_prefetch
    assert active_learn.dcn_mode == "active_learn", active_learn.dcn_mode
    assert active_learn.policy_calls == 1, active_learn.policy_calls
    learn_event = active_learn.dcn_observations()[0]
    assert learn_event["event"] == "dcn.active_learn", learn_event
    assert learn_event["promotion_id"] == "promotion-123", learn_event
    assert learn_event["checkpoint_id"] == "chk-123", learn_event
    assert learn_event["promotion_gate"] == "ok", learn_event

    old = os.environ.get("DAYSTROM_DCN_MODE")
    os.environ["DAYSTROM_DCN_MODE"] = "banana"
    try:
        try:
            plugin.DaystromDMLProvider()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid DCN mode was not rejected")
    finally:
        if old is None:
            os.environ.pop("DAYSTROM_DCN_MODE", None)
        else:
            os.environ["DAYSTROM_DCN_MODE"] = old

    plugin_yaml = PLUGIN_PATH.with_name("plugin.yaml").read_text()
    for fragment in ("observe_only", "active_read", "active_learn", "default: disabled"):
        assert fragment in plugin_yaml, fragment

    print("daystrom_dml dcn active-read smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
