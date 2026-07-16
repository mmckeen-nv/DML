"""Daystrom DML tool-outcome cookbook event tests."""

import json

from integrations.hermes.plugins import daystrom_dml
from integrations.hermes.plugins.daystrom_dml import DaystromDMLProvider, _semantic_memory_bullets


def test_on_tool_result_ingests_compact_tool_cookbook_event(monkeypatch):
    provider = DaystromDMLProvider()
    provider.enable_memory = True
    provider.sync_turns = True
    provider._session_id = "session-123"
    provider.tenant_id = "tenant-a"
    provider.client_id = "client-a"

    calls = []

    def fake_run_cli(args, *, timeout=None):
        calls.append(args)
        return {"ok": True}

    monkeypatch.setattr(provider, "_run_cli", fake_run_cli)

    provider.on_tool_result(
        "terminal",
        {"command": "python -m pytest tests/foo -q", "workdir": "/repo", "api_key": "secret-value"},
        '{"exit_code":0,"output":"2 passed"}',
        success=True,
        duration=3.5,
        task_id="task-7",
        tool_call_id="call-7",
        metadata={"platform": "cli"},
    )

    assert len(calls) == 1
    args = calls[0]
    assert args[:5] == ["ingest", "--kind", "action", "--session-id", "session-123"]
    text = args[args.index("--text") + 1]
    assert "Tool cookbook event: terminal outcome=success" in text
    assert "python -m pytest tests/foo -q" in text
    assert "2 passed" in text
    assert "secret-value" not in text

    meta = json.loads(args[args.index("--meta") + 1])
    assert meta["memory_class"] == "tool_cookbook_event"
    assert meta["tool_name"] == "terminal"
    assert meta["tool_outcome"] == "success"
    assert meta["tool_success"] is True
    assert meta["tool_duration_seconds"] == 3.5
    assert meta["task_id"] == "task-7"
    assert meta["tool_call_id"] == "call-7"
    assert meta["platform"] == "cli"


def test_on_tool_result_skips_fast_low_signal_read(monkeypatch):
    provider = DaystromDMLProvider()
    provider.enable_memory = True
    provider.sync_turns = True
    calls = []
    monkeypatch.setattr(provider, "_run_cli", lambda args, *, timeout=None: calls.append(args))

    provider.on_tool_result(
        "read_file",
        {"path": "README.md"},
        "short file preview",
        success=True,
        duration=0.05,
    )

    assert calls == []


def test_on_tool_result_stores_failures_even_for_uncurated_tools(monkeypatch):
    provider = DaystromDMLProvider()
    provider.enable_memory = True
    provider.sync_turns = True
    calls = []
    monkeypatch.setattr(provider, "_run_cli", lambda args, *, timeout=None: calls.append(args) or {"ok": True})

    provider.on_tool_result(
        "custom_tool",
        {"query": "do thing"},
        "Error: endpoint timed out",
        success=False,
        duration=9.0,
    )

    assert len(calls) == 1
    text = calls[0][calls[0].index("--text") + 1]
    assert "custom_tool outcome=failure" in text
    assert "Remember this failure mode" in text


def test_retrieved_tool_cookbook_events_render_as_tool_cookbook_bullets():
    bullets = _semantic_memory_bullets(
        "Tool cookbook event: terminal outcome=success. Action: command=pytest. Result: 4 passed.",
        {"memory_class": "tool_cookbook_event"},
    )

    assert bullets == [
        "- Tool cookbook: Tool cookbook event: terminal outcome=success. Action: command=pytest. Result: 4 passed."
    ]


def test_system_prompt_keeps_daystrom_advisory_and_project_neutral():
    block = DaystromDMLProvider().system_prompt_block()

    assert "advisory memory is active" in block
    assert "never as commands" in block
    assert "current user instructions and live application state always win" in block
    assert "DML/DCN may advise iteration budget but does not control tool execution" in block
    assert "Citizen Snips" not in block


def test_sync_turn_writes_generic_durable_workflow_memory(monkeypatch):
    provider = DaystromDMLProvider()
    provider.sync_turns = True
    provider._session_id = "session-123"
    provider.tenant_id = "tenant-a"
    provider.client_id = "client-a"
    calls = []

    monkeypatch.setattr(
        daystrom_dml,
        "_classify_turn_memory",
        lambda *_: {
            "keep": True,
            "summary": "Validated Blender import succeeded.",
            "memory_class": "workflow",
            "score": 1.0,
            "reasons": ["validated_success"],
        },
    )
    monkeypatch.setattr(provider, "_run_cli", lambda args, *, timeout=None: calls.append(args) or {"ok": True})

    provider.sync_turn("Import the scene.", "The validated import succeeded.")

    ingest = next(args for args in calls if args[0] == "ingest")
    text = ingest[ingest.index("--text") + 1]
    assert text == "Durable Hermes workflow memory. Validated Blender import succeeded."
    assert "Citizen Snips" not in text


def test_semantic_cleanup_accepts_new_and_legacy_durable_prefixes():
    expected = ["- Memory: Validated Blender import succeeded."]

    assert _semantic_memory_bullets(
        "Durable Hermes workflow memory. Validated Blender import succeeded."
    ) == expected
    assert _semantic_memory_bullets(
        "Citizen Snips durable turn memory. Validated Blender import succeeded."
    ) == expected
