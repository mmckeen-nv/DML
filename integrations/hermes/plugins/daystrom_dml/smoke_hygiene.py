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
    plugin_yaml = PLUGIN_PATH.with_name("plugin.yaml").read_text()
    for fragment in ("dcn:", "observe_only", "active_read", "active_learn", "default: disabled"):
        assert fragment in plugin_yaml, fragment
    assert plugin.DaystromDMLProvider().dcn_mode == "disabled"
    forbidden_fragments = (
        "Completed Snips_2 turn",
        "thread: 1510575377045524580 | state:",
        "[Mark_NV]",
        "User:",
        "Assistant:",
        "assistant: | assistant:",
        "<tool output>",
        "Gateway received SIGTERM",
        "pytest passed",
        "[System note:",
        "Your previous turn in this session was interrupted by a gateway shutdown",
        "The conversation history below is intact",
        "unfinished tool result",
        "address the user's new message below",
        "[IMPORTANT:",
        "Background process",
        "Command: codex exec",
        "tokens used",
        "--output-last-message",
    )

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
    assert set(noisy["reasons"]) & {"smoke_or_self_test", "transcript_residue"}, noisy

    normal_queries = (
        "hello",
        "great, hows that looking for context window eating?",
        "thanks",
        "ok do it",
    )
    for query in normal_queries:
        assert plugin._should_inject_dml_memory(query) is False, query

    memory_queries = (
        "rehydrate context after compaction",
        "what did we decide about DML yesterday",
        "continue the long running setup",
        "<memory-context>system compaction notes</memory-context> please restore the thread",
        "resume from the previous task where we left off",
    )
    for query in memory_queries:
        assert plugin._should_inject_dml_memory(query) is True, query

    class FakeProvider(plugin.DaystromDMLProvider):
        def __init__(self):
            super().__init__()
            self.resume_calls = 0
            self.retrieve_calls = 0

        def _personality_overlay(self, prompt):
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

    fake = FakeProvider()
    normal_prefetch = fake.prefetch("hello")
    assert "Daystrom Personality Matrix Overlay" in normal_prefetch, normal_prefetch
    assert "DML Active Continuity" not in normal_prefetch, normal_prefetch
    assert "DML Retrieved Memory" not in normal_prefetch, normal_prefetch
    assert fake.resume_calls == 0, fake.resume_calls
    assert fake.retrieve_calls == 0, fake.retrieve_calls

    explicit_prefetch = fake.prefetch("rehydrate context after compaction")
    assert "Daystrom Personality Matrix Overlay" in explicit_prefetch, explicit_prefetch
    assert "DML Active Continuity" in explicit_prefetch, explicit_prefetch
    assert "DML Retrieved Memory" in explicit_prefetch, explicit_prefetch
    assert fake.resume_calls == 1, fake.resume_calls
    assert fake.retrieve_calls == 1, fake.retrieve_calls

    handoff = plugin._handoff_fragment(
        "assistant",
        "=== Daystrom DML Retrieved Memory ===\nold scaffold\n=== Other ===\nDone. Tests passed.",
    )
    assert "Daystrom DML Retrieved Memory" not in handoff, handoff
    assert handoff, handoff

    transcript = plugin._classify_turn_memory(
        "[Mark_NV] wait so the DML is causing context flood?",
        "Completed Snips_2 turn. User: [Mark_NV] wait so the DML is causing context flood? Assistant: Yes — partially.",
    )
    assert transcript["keep"] is False, transcript
    assert transcript["memory_class"] == "transcript_residue", transcript

    state, task, next_action = plugin._continuity_state_from_messages([
        {"role": "user", "content": "We need to fix DML so it stores compact state only."},
        {"role": "assistant", "content": "Done — patched plugins/daystrom_dml/__init__.py and tests passed."},
    ])
    assert state, (state, task, next_action)
    assert "user:" not in state.lower(), state
    assert "assistant:" not in state.lower(), state
    assert "Completed Snips_2 turn" not in state, state

    payload = {
        "raw_context": "Completed Snips_2 turn. User: noisy Assistant: noisy\nMark prefers compact state.",
        "items": [{"meta": {"summary": "Completed Snips_2 turn. User: noisy Assistant: noisy"}}],
    }
    provider = plugin.DaystromDMLProvider()
    safe_context = provider._safe_context_from_payload(payload)
    assert "Completed Snips_2 turn" not in safe_context, safe_context
    assert "Mark prefers compact state" in safe_context, safe_context

    system_note = (
        "[System note: Your previous turn in this session was interrupted by a gateway shutdown. "
        "The conversation history below is intact.\n"
        "If there is an unfinished tool result, address the user's new message below.]"
    )
    wrapped_turn = plugin._classify_turn_memory(
        system_note,
        "Remember: Mark prefers compact DML semantic state.",
    )
    assert wrapped_turn["keep"] is False, wrapped_turn
    assert plugin._handoff_fragment("user", system_note) == ""

    system_payload = {
        "raw_context": (
            "- Current focus: [System note: Your previous turn in this session was interrupted by a gateway shutdown. "
            "The conversation history below is intact.\n"
            "- Memory policy: store compact semantic state only; never store transcripts, DML blocks, tool logs, or role-prefixed dialogue\n"
            "- Preference: Mark wants DML used as ultra-compact semantic continuity."
        )
    }
    system_safe = provider._safe_context_from_payload(system_payload)
    assert "Current focus: [System note:" not in system_safe, system_safe
    assert "[System note:" not in system_safe, system_safe
    assert "- Current focus:" not in system_safe, system_safe
    assert "- Memory policy: store compact semantic state only; never store transcripts, DML blocks, tool logs, or role-prefixed dialogue" in system_safe, system_safe
    assert "- Preference: Mark wants DML used as ultra-compact semantic continuity." in system_safe, system_safe

    runtime_payload = {
        "raw_context": (
            "- Current focus: [IMPORTANT: Background process proc_df335c4b8c70 completed (exit code 0). "
            "Command: codex exec --full-auto --add-dir /Users/markmckeen/.hermes/plugins/daystrom_dml "
            "--output-last-message /Users/markmckeen/. Output:\n"
            "[… output truncated] tokens used 44,616\n"
            "- Memory policy: store compact semantic state only; never store transcripts, DML blocks, tool logs, or role-prefixed dialogue"
        )
    }
    runtime_safe = provider._safe_context_from_payload(runtime_payload)
    assert "- Current focus:" not in runtime_safe, runtime_safe
    assert "- Memory policy: store compact semantic state only; never store transcripts, DML blocks, tool logs, or role-prefixed dialogue" in runtime_safe, runtime_safe
    for fragment in ("[IMPORTANT:", "Background process", "Command: codex exec", "tokens used", "--output-last-message"):
        assert fragment not in runtime_safe, (fragment, runtime_safe)

    personality_policy_payload = {
        "raw_context": (
            "- Memory policy: === Personality Matrix === Identity: Citizen Snips. Preferences: "
            "Citizen Snips exact creature/nature should remain open and undecided until Mark chooses it."
        )
    }
    personality_policy_safe = provider._safe_context_from_payload(personality_policy_payload)
    assert "Memory policy: === Personality Matrix" not in personality_policy_safe, personality_policy_safe
    assert "Identity: Citizen Snips. Preferences" not in personality_policy_safe, personality_policy_safe
    assert not personality_policy_safe, personality_policy_safe

    dpm_policy_payload = {
        "raw_context": (
            "- Memory policy: === Daystrom Personality Matrix Overlay === "
            "Constraint: Current-turn instructions override the DPM overlay"
        )
    }
    dpm_policy_safe = provider._safe_context_from_payload(dpm_policy_payload)
    assert "Daystrom Personality Matrix Overlay" not in dpm_policy_safe, dpm_policy_safe
    assert "DPM overlay" not in dpm_policy_safe, dpm_policy_safe
    assert not dpm_policy_safe, dpm_policy_safe

    safe_policy_payload = {
        "raw_context": "- Memory policy: store stable preferences and compact task continuity only."
    }
    safe_policy = provider._safe_context_from_payload(safe_policy_payload)
    assert "- Memory policy: store stable preferences and compact task continuity only." in safe_policy, safe_policy

    continuity_payload = {
        "raw_context": (
            "thread: 1510575377045524580 | state: "
            "current_focus=[Mark_NV] make DML store compact semantic state only.; "
            "last_confirmed_status=Yes — I’m back. I verified the gateway actually restarted: "
            "```text Gateway received SIGTERM: restarting pytest passed ```; "
            "memory_policy=store durable preferences, task state, and continuity; never store transcripts, DML blocks, tool logs, or role-prefixed dialogue; "
            "next_action=verify long-running gateway behavior and tune retrieval."
        )
    }
    continuity = provider._safe_context_from_payload(continuity_payload)
    assert "- Current focus: make DML store compact semantic state only." in continuity, continuity
    assert "- Memory policy: store durable preferences, task state, and continuity" in continuity, continuity
    assert "- Next step: verify long-running gateway behavior and tune retrieval." in continuity, continuity
    for fragment in forbidden_fragments:
        assert fragment not in continuity, (fragment, continuity)

    residual_payload = {
        "raw_context": (
            "current_focus=great, can you offload that task to the codex app vs doing it yourself? "
            "Send codex a plan and not completely fill your context window with the codex app...; "
            "current_focus=great, can you offload that task to the codex app vs doing it yourself? "
            "Send codex a plan and not completely fill your context window with the codex app output?; "
            "memory_policy=store compact semantic state only; never store transcripts, DML blocks, tool logs, "
            "or role-prefixed dialogue | task: great, can you offload that task to the codex app vs doing it yourself? "
            "Send codex a plan and not completely fill your context window with the codex app output?"
        ),
        "items": [
            {
                "text": (
                    "current_focus=great, can you offload that task to the codex app vs doing it yourself? "
                    "Send codex a plan and not completely fill your context window with the codex app output?; "
                    "memory_policy=store compact semantic state only; never store transcripts, DML blocks, tool logs, "
                    "or role-prefixed dialogue | task: great, can you offload that task to the codex app vs doing it yourself? "
                    "Send codex a plan and not completely fill your context window with the codex app output?"
                ),
                "meta": {"memory_class": "checkpoint"},
            },
            {
                "text": (
                    "current_focus=great, can you offload that task to the codex app vs doing it yourself? "
                    "Send codex a plan and not completely fill your context window with the codex app..."
                ),
                "meta": {"memory_class": "checkpoint"},
            }
        ],
    }
    residual = provider._safe_context_from_payload(residual_payload)
    assert residual.count("- Current focus:") == 1, residual
    assert "- Current focus: tighten DML continuity formatting through Codex-offloaded implementation." in residual, residual
    assert "| task:" not in residual, residual
    assert "Memory policy: store compact semantic state only; never store transcripts, DML blocks, tool logs, or role-prefixed dialogue" in residual, residual
    assert "codex app..." not in residual.lower(), residual

    retrieved_payload = {
        "items": [
            {
                "meta": {
                    "summary": "Citizen Snips durable turn memory. User signal: Remember: Mark prefers compact semantic state.",
                    "memory_class": "preference",
                }
            },
            {
                "text": "Assistant: pytest passed after Gateway received SIGTERM: noisy log",
                "meta": {"memory_class": "validation"},
            },
        ]
    }
    retrieved = provider._safe_context_from_payload(retrieved_payload)
    assert "- Preference: Mark prefers compact semantic state." in retrieved, retrieved
    for fragment in forbidden_fragments:
        assert fragment not in retrieved, (fragment, retrieved)

    print("daystrom_dml hygiene smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
