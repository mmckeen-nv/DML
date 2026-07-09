"""Tests for the Daystrom DML Ollama cookbook worker."""

import json
from pathlib import Path

from plugins.memory.daystrom_dml import cookbook_worker as cw


def _write_state(path: Path, records):
    path.mkdir(parents=True, exist_ok=True)
    state = path / "dml_state.jsonl"
    with state.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "daystrom_dml.memory", "count": len(records)}) + "\n")
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_pending_tool_events_skips_events_already_covered_by_recipe(tmp_path):
    event = {
        "text": "Tool cookbook event: terminal outcome=success. Action: command=pytest. Result: 2 passed.",
        "timestamp": "1",
        "meta": {"memory_class": "tool_cookbook_event", "tool_name": "terminal", "tool_outcome": "success"},
    }
    eid = cw.event_id(event)
    _write_state(tmp_path, [
        event,
        {"text": "# Tool Cookbook", "meta": {"memory_class": "tool_cookbook_recipe", "source_event_ids": [eid]}},
    ])

    assert cw.pending_tool_events(cw.load_state_items(tmp_path)) == []


def test_build_cookbook_prompt_redacts_secrets():
    prompt = cw.build_cookbook_prompt([
        {
            "event_id": "evt1",
            "tool_name": "terminal",
            "outcome": "failure",
            "duration": 1.2,
            "text": cw.clean_text("Tool cookbook event: terminal. Action: command=run --api_key=abc123. Result: failed"),
        }
    ])

    assert "abc123" not in prompt
    assert "api_key=[REDACTED]" in prompt
    assert "# Tool Cookbook" in prompt
    assert "Evidence Event IDs" in prompt


def test_run_worker_dry_run_uses_ollama_and_does_not_ingest(monkeypatch, tmp_path):
    _write_state(tmp_path, [
        {
            "text": "Tool cookbook event: terminal outcome=success. Action: command=pytest. Result: 2 passed.",
            "timestamp": "1",
            "meta": {"memory_class": "tool_cookbook_event", "tool_name": "terminal", "tool_outcome": "success"},
        }
    ])
    prompts = []
    monkeypatch.setattr(cw, "call_ollama", lambda prompt, **kwargs: prompts.append((prompt, kwargs)) or "# Tool Cookbook\n## Situation\nTest\n## Evidence Event IDs\nevt")
    monkeypatch.setattr(cw, "ingest_cookbook", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not ingest")))

    result = cw.run_worker(storage_dir=tmp_path, dry_run=True, model="llama3:8b")

    assert result["ok"] is True
    assert result["distilled"] is True
    assert result["dry_run"] is True
    assert result["model"] == "llama3:8b"
    assert prompts
    assert "pytest" in prompts[0][0]


def test_run_worker_ingests_recipe_with_source_event_ids(monkeypatch, tmp_path):
    _write_state(tmp_path, [
        {
            "text": "Tool cookbook event: terminal outcome=failure. Action: command=bad. Result: Error: missing file.",
            "timestamp": "1",
            "meta": {"memory_class": "tool_cookbook_event", "tool_name": "terminal", "tool_outcome": "failure"},
        }
    ])
    monkeypatch.setattr(cw, "call_ollama", lambda prompt, **kwargs: "# Tool Cookbook\n## Failed / Blocked\nMissing file")
    ingests = []
    monkeypatch.setattr(cw, "ingest_cookbook", lambda cookbook, **kwargs: ingests.append((cookbook, kwargs)))

    result = cw.run_worker(storage_dir=tmp_path, launcher=Path("/fake/launcher"), dry_run=False, model="llama3:8b")

    assert result["distilled"] is True
    assert ingests
    cookbook, kwargs = ingests[0]
    assert "Missing file" in cookbook
    assert kwargs["source_event_ids"] == result["source_event_ids"]
    assert kwargs["model"] == "llama3:8b"
