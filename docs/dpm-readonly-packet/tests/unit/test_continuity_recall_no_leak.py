import json
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "continuity_recall.py"
spec = importlib.util.spec_from_file_location("continuity_recall", MODULE_PATH)
continuity_recall = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(continuity_recall)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text())


def test_exact_thread_key_hit_keeps_checkpoint_and_handoff_when_relevant(monkeypatch, tmp_path):
    threads = load_fixture("thread_registry.fixture.json")
    loops = load_fixture("open_loops.fixture.json")
    self_state = load_fixture("self_state.fixture.json")

    checkpoint = tmp_path / "2026-04-05_demo-thread.md"
    checkpoint.write_text("Demo thread checkpoint\nnext step here")
    handoff = tmp_path / "work-loop-handoff.md"
    handoff.write_text(
        "# Work Loop Handoff\n\n"
        "- task: Milestone 5 Bundle 2 smoke\n\n"
        "## Next Action\n"
        "inspect operator loop report\n"
    )
    threads["threads"]["thread:demo-thread"]["latest_checkpoint"] = str(checkpoint)

    monkeypatch.setattr(continuity_recall, "THREADS", tmp_path / "thread_registry.json")
    monkeypatch.setattr(continuity_recall, "LOOPS", tmp_path / "open_loops.json")
    monkeypatch.setattr(continuity_recall, "SELF", tmp_path / "self_state.json")
    monkeypatch.setattr(continuity_recall, "WORK_LOOP_HANDOFF", handoff)

    (tmp_path / "thread_registry.json").write_text(json.dumps(threads))
    (tmp_path / "open_loops.json").write_text(json.dumps(loops))
    (tmp_path / "self_state.json").write_text(json.dumps(self_state))

    payload = continuity_recall.build_payload("demo-thread", None)

    assert payload["retrieval_status"] == "hit"
    assert payload["thread"]["id"] == "thread:demo-thread"
    assert "Demo thread checkpoint" in payload["checkpoint_excerpt"]
    assert payload["handoff"]["is_relevant"] is True
    assert payload["handoff"]["path"] == str(handoff)
    assert payload["thread"]["provider"] == "discord"
    assert payload["thread"]["channel"] == "discord"
    assert payload["thread"]["chat_id"] == "thread:demo-thread"
    assert payload["thread"]["topic_id"] == "topic-demo"
    assert payload["thread"]["thread_label"] == "Milestone 5 Bundle 2 smoke"
    assert payload["thread"]["session_scope"] == "thread"
    assert payload["dpm_overlay"] is None


def test_disabled_mode_leaves_core_payload_unchanged(monkeypatch, tmp_path):
    threads = load_fixture("thread_registry.fixture.json")
    loops = load_fixture("open_loops.fixture.json")
    self_state = load_fixture("self_state.fixture.json")

    checkpoint = tmp_path / "2026-04-05_demo-thread.md"
    checkpoint.write_text("Demo thread checkpoint\nnext step here")
    threads["threads"]["thread:demo-thread"]["latest_checkpoint"] = str(checkpoint)

    config = tmp_path / "config.disabled.json"
    config.write_text(json.dumps({
        "version": 1,
        "plugin": "dpm",
        "mode": "disabled",
        "read": {"allow_thread": True},
        "write": {"enabled": False, "require_explicit_runtime_support": True, "allowed_scopes": []},
        "audit": {"enabled": True, "include_excluded_sources": True, "max_sources": 20, "max_overlay_chars": 32},
        "safety": {},
    }))

    monkeypatch.setattr(continuity_recall, "THREADS", tmp_path / "thread_registry.json")
    monkeypatch.setattr(continuity_recall, "LOOPS", tmp_path / "open_loops.json")
    monkeypatch.setattr(continuity_recall, "SELF", tmp_path / "self_state.json")
    monkeypatch.setattr(continuity_recall, "WORK_LOOP_HANDOFF", tmp_path / "missing-handoff.md")
    monkeypatch.setattr(continuity_recall, "DPM_CONFIG", config)
    monkeypatch.setattr(continuity_recall, "PROJECT_STATE", tmp_path / "missing-project-state.json")

    (tmp_path / "thread_registry.json").write_text(json.dumps(threads))
    (tmp_path / "open_loops.json").write_text(json.dumps(loops))
    (tmp_path / "self_state.json").write_text(json.dumps(self_state))

    payload = continuity_recall.build_payload("demo-thread", None)

    assert payload["retrieval_status"] == "hit"
    assert payload["thread"]["id"] == "thread:demo-thread"
    assert payload["thread"]["provider"] == "discord"
    assert payload["thread"]["thread_label"] == "Milestone 5 Bundle 2 smoke"
    assert "Demo thread checkpoint" in payload["checkpoint_excerpt"]
    assert payload["dpm_overlay"] is None


def test_active_read_exposes_bounded_thread_overlay_via_test_seam(monkeypatch, tmp_path):
    threads = load_fixture("thread_registry.fixture.json")
    loops = load_fixture("open_loops.fixture.json")
    self_state = load_fixture("self_state.fixture.json")

    checkpoint = tmp_path / "2026-04-05_demo-thread.md"
    checkpoint.write_text("Demo thread checkpoint\nnext step here")
    threads["threads"]["thread:demo-thread"]["latest_checkpoint"] = str(checkpoint)

    config = tmp_path / "config.active-read.json"
    config.write_text(json.dumps({
        "version": 1,
        "plugin": "dpm",
        "mode": "active-read",
        "read": {"allow_thread": True},
        "write": {"enabled": False, "require_explicit_runtime_support": True, "allowed_scopes": []},
        "audit": {"enabled": True, "include_excluded_sources": True, "max_sources": 20, "max_overlay_chars": 32},
        "safety": {},
    }))

    monkeypatch.setattr(continuity_recall, "THREADS", tmp_path / "thread_registry.json")
    monkeypatch.setattr(continuity_recall, "LOOPS", tmp_path / "open_loops.json")
    monkeypatch.setattr(continuity_recall, "SELF", tmp_path / "self_state.json")
    monkeypatch.setattr(continuity_recall, "WORK_LOOP_HANDOFF", tmp_path / "missing-handoff.md")
    monkeypatch.setattr(continuity_recall, "DPM_CONFIG", config)
    monkeypatch.setattr(continuity_recall, "PROJECT_STATE", tmp_path / "missing-project-state.json")

    (tmp_path / "thread_registry.json").write_text(json.dumps(threads))
    (tmp_path / "open_loops.json").write_text(json.dumps(loops))
    (tmp_path / "self_state.json").write_text(json.dumps(self_state))

    payload = continuity_recall.build_payload("demo-thread", None)

    assert payload["retrieval_status"] == "hit"
    overlay = payload["dpm_overlay"]
    assert overlay["schema_version"] == "dpm.replay-overlay.v1"
    assert overlay["mode"] == "active-read"
    assert overlay["overlay_id"] == "overlay:thread:demo-thread:active-read"
    assert overlay["scope"] == {
        "primary": "thread",
        "thread_id": "demo-thread",
        "project_id": "project:dpm",
        "relationship_id": "relationship:runtime",
    }
    assert overlay["overlay"]["max_chars"] == 32
    assert overlay["overlay"]["rendered_text"] == "Thread continuity for Milestone "
    assert overlay["sources"][0]["source_id"] == "demo-thread"
    assert overlay["audit"]["included_source_ids"] == ["demo-thread"]
    assert overlay["override_state"]["override_applied"] is False


def test_active_read_exposes_bounded_thread_plus_project_overlay_when_compatible(monkeypatch, tmp_path):
    threads = load_fixture("thread_registry.fixture.json")
    loops = load_fixture("open_loops.fixture.json")
    self_state = load_fixture("self_state.fixture.json")

    checkpoint = tmp_path / "2026-04-05_demo-thread.md"
    checkpoint.write_text("Demo thread checkpoint\nnext step here")
    threads["threads"]["thread:demo-thread"]["latest_checkpoint"] = str(checkpoint)

    config = tmp_path / "config.active-read.json"
    config.write_text(json.dumps({
        "version": 1,
        "plugin": "dpm",
        "mode": "active-read",
        "read": {"allow_thread": True, "allow_project": True},
        "write": {"enabled": False, "require_explicit_runtime_support": True, "allowed_scopes": []},
        "audit": {"enabled": True, "include_excluded_sources": True, "max_sources": 20, "max_overlay_chars": 200},
        "safety": {},
    }))

    project_state = tmp_path / "project_state.json"
    project_state.write_text(json.dumps({
        "project_id": "project:dpm",
        "label": "DPM runtime seam",
        "summary": "Honor bounded project defaults only when they are compatible with the matched thread.",
        "updated_at": "2026-04-14T18:00:00Z",
        "compatible_thread_keys": ["demo-thread"],
    }))

    monkeypatch.setattr(continuity_recall, "THREADS", tmp_path / "thread_registry.json")
    monkeypatch.setattr(continuity_recall, "LOOPS", tmp_path / "open_loops.json")
    monkeypatch.setattr(continuity_recall, "SELF", tmp_path / "self_state.json")
    monkeypatch.setattr(continuity_recall, "WORK_LOOP_HANDOFF", tmp_path / "missing-handoff.md")
    monkeypatch.setattr(continuity_recall, "DPM_CONFIG", config)
    monkeypatch.setattr(continuity_recall, "PROJECT_STATE", project_state)

    (tmp_path / "thread_registry.json").write_text(json.dumps(threads))
    (tmp_path / "open_loops.json").write_text(json.dumps(loops))
    (tmp_path / "self_state.json").write_text(json.dumps(self_state))

    payload = continuity_recall.build_payload("demo-thread", None)

    overlay = payload["dpm_overlay"]
    assert overlay["retrieval_order_applied"] == ["thread", "project"]
    assert overlay["sources"][0]["source_id"] == "demo-thread"
    assert overlay["sources"][1]["source_id"] == "project:dpm"
    assert overlay["audit"]["included_source_ids"] == ["demo-thread", "project:dpm"]
    assert "Project continuity for DPM runtime seam" in overlay["overlay"]["rendered_text"]
    assert overlay["effective_constraints"]["cross_scope_fallback_requires_compatibility"] is True


def test_active_read_fails_closed_on_incompatible_project_source(monkeypatch, tmp_path):
    threads = load_fixture("thread_registry.fixture.json")
    loops = load_fixture("open_loops.fixture.json")
    self_state = load_fixture("self_state.fixture.json")

    checkpoint = tmp_path / "2026-04-05_demo-thread.md"
    checkpoint.write_text("Demo thread checkpoint\nnext step here")
    threads["threads"]["thread:demo-thread"]["latest_checkpoint"] = str(checkpoint)

    config = tmp_path / "config.active-read.json"
    config.write_text(json.dumps({
        "version": 1,
        "plugin": "dpm",
        "mode": "active-read",
        "read": {"allow_thread": True, "allow_project": True},
        "write": {"enabled": False, "require_explicit_runtime_support": True, "allowed_scopes": []},
        "audit": {"enabled": True, "include_excluded_sources": True, "max_sources": 20, "max_overlay_chars": 80},
        "safety": {},
    }))

    project_state = tmp_path / "project_state.json"
    project_state.write_text(json.dumps({
        "project_id": "project:dpm",
        "label": "DPM runtime seam",
        "summary": "This project source should stay hidden for incompatible threads.",
        "updated_at": "2026-04-14T18:00:00Z",
        "compatible_thread_keys": ["claw-code"],
    }))

    monkeypatch.setattr(continuity_recall, "THREADS", tmp_path / "thread_registry.json")
    monkeypatch.setattr(continuity_recall, "LOOPS", tmp_path / "open_loops.json")
    monkeypatch.setattr(continuity_recall, "SELF", tmp_path / "self_state.json")
    monkeypatch.setattr(continuity_recall, "WORK_LOOP_HANDOFF", tmp_path / "missing-handoff.md")
    monkeypatch.setattr(continuity_recall, "DPM_CONFIG", config)
    monkeypatch.setattr(continuity_recall, "PROJECT_STATE", project_state)

    (tmp_path / "thread_registry.json").write_text(json.dumps(threads))
    (tmp_path / "open_loops.json").write_text(json.dumps(loops))
    (tmp_path / "self_state.json").write_text(json.dumps(self_state))

    payload = continuity_recall.build_payload("demo-thread", None)

    overlay = payload["dpm_overlay"]
    assert overlay["retrieval_order_applied"] == ["thread"]
    assert [source["source_id"] for source in overlay["sources"]] == ["demo-thread"]
    assert "Project continuity" not in overlay["overlay"]["rendered_text"]
    assert overlay["audit"]["included_source_ids"] == ["demo-thread"]


def test_partial_query_fails_closed_without_wrong_thread_leak(monkeypatch, tmp_path):
    threads = load_fixture("thread_registry.fixture.json")
    loops = load_fixture("open_loops.fixture.json")
    self_state = load_fixture("self_state.fixture.json")

    checkpoint = tmp_path / "2026-03-31_claw-code.md"
    checkpoint.write_text("Claw code checkpoint should not leak")
    handoff = tmp_path / "work-loop-handoff.md"
    handoff.write_text(
        "# Work Loop Handoff\n\n"
        "- task: Milestone 5 Bundle 2 smoke\n\n"
        "## Next Action\n"
        "inspect operator loop report\n"
    )
    threads["threads"]["thread:claw-code"]["latest_checkpoint"] = str(checkpoint)

    monkeypatch.setattr(continuity_recall, "THREADS", tmp_path / "thread_registry.json")
    monkeypatch.setattr(continuity_recall, "LOOPS", tmp_path / "open_loops.json")
    monkeypatch.setattr(continuity_recall, "SELF", tmp_path / "self_state.json")
    monkeypatch.setattr(continuity_recall, "WORK_LOOP_HANDOFF", handoff)

    (tmp_path / "thread_registry.json").write_text(json.dumps(threads))
    (tmp_path / "open_loops.json").write_text(json.dumps(loops))
    (tmp_path / "self_state.json").write_text(json.dumps(self_state))

    payload = continuity_recall.build_payload("claw", None)

    assert payload["retrieval_status"] == "miss"
    assert payload["thread"]["id"] is None
    assert payload["thread"]["key"] is None
    assert payload["checkpoint_excerpt"] is None
    assert payload["thread"]["provider"] is None
    assert payload["thread"]["session_scope"] is None
    assert payload["open_loops"] == []
    assert payload["handoff"]["is_relevant"] is False
    assert payload["handoff"]["path"] is None
    assert payload["handoff"]["excerpt"] is None
    assert payload["dpm_overlay"] is None


def test_ambiguous_query_fails_closed_without_selecting_a_thread(monkeypatch, tmp_path):
    threads = load_fixture("thread_registry.fixture.json")
    loops = load_fixture("open_loops.fixture.json")
    self_state = load_fixture("self_state.fixture.json")

    checkpoint_a = tmp_path / "2026-03-31_self-architecture-portrait.md"
    checkpoint_b = tmp_path / "2026-04-03_self-architecture-portrait.md"
    checkpoint_a.write_text("Older checkpoint")
    checkpoint_b.write_text("Newer checkpoint")
    threads["threads"]["1488668819655495782"]["latest_checkpoint"] = str(checkpoint_a)
    threads["threads"]["thread:self-architecture-portrait"]["latest_checkpoint"] = str(checkpoint_b)

    monkeypatch.setattr(continuity_recall, "THREADS", tmp_path / "thread_registry.json")
    monkeypatch.setattr(continuity_recall, "LOOPS", tmp_path / "open_loops.json")
    monkeypatch.setattr(continuity_recall, "SELF", tmp_path / "self_state.json")
    monkeypatch.setattr(continuity_recall, "WORK_LOOP_HANDOFF", tmp_path / "missing-handoff.md")

    (tmp_path / "thread_registry.json").write_text(json.dumps(threads))
    (tmp_path / "open_loops.json").write_text(json.dumps(loops))
    (tmp_path / "self_state.json").write_text(json.dumps(self_state))

    payload = continuity_recall.build_payload("self-architecture-portrait", None)

    assert payload["retrieval_status"] == "miss"
    assert payload["thread"]["id"] is None
    assert payload["checkpoint_excerpt"] is None
    assert payload["thread"]["provider"] is None
    assert payload["thread"]["thread_label"] is None
    assert payload["open_loops"] == []
    assert payload["dpm_overlay"] is None
