import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
MODULE_PATH = ROOT / "scripts" / "continuity_recall.py"

spec = importlib.util.spec_from_file_location("continuity_recall", MODULE_PATH)
continuity_recall = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(continuity_recall)


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text())


def _seed_runtime_files(tmp_path, monkeypatch, *, project_state=None, config_mode="active-read", max_overlay_chars=200):
    threads = load_fixture("thread_registry.fixture.json")
    loops = load_fixture("open_loops.fixture.json")
    self_state = load_fixture("self_state.fixture.json")

    checkpoint = tmp_path / "2026-04-05_demo-thread.md"
    checkpoint.write_text("Demo thread checkpoint\nnext step here")
    threads["threads"]["thread:demo-thread"]["latest_checkpoint"] = str(checkpoint)

    config = tmp_path / f"config.{config_mode}.json"
    config.write_text(json.dumps({
        "version": 1,
        "plugin": "dpm",
        "mode": config_mode,
        "read": {"allow_thread": True, "allow_project": True},
        "write": {"enabled": False, "require_explicit_runtime_support": True, "allowed_scopes": []},
        "audit": {
            "enabled": True,
            "include_excluded_sources": True,
            "max_sources": 20,
            "max_overlay_chars": max_overlay_chars,
        },
        "safety": {},
    }))

    thread_registry = tmp_path / "thread_registry.json"
    open_loops = tmp_path / "open_loops.json"
    self_state_path = tmp_path / "self_state.json"
    thread_registry.write_text(json.dumps(threads))
    open_loops.write_text(json.dumps(loops))
    self_state_path.write_text(json.dumps(self_state))

    monkeypatch.setattr(continuity_recall, "THREADS", thread_registry)
    monkeypatch.setattr(continuity_recall, "LOOPS", open_loops)
    monkeypatch.setattr(continuity_recall, "SELF", self_state_path)
    monkeypatch.setattr(continuity_recall, "WORK_LOOP_HANDOFF", tmp_path / "missing-handoff.md")
    monkeypatch.setattr(continuity_recall, "DPM_CONFIG", config)

    if project_state is None:
        monkeypatch.setattr(continuity_recall, "PROJECT_STATE", tmp_path / "missing-project-state.json")
    else:
        project_state_path = tmp_path / "project_state.json"
        project_state_path.write_text(json.dumps(project_state))
        monkeypatch.setattr(continuity_recall, "PROJECT_STATE", project_state_path)


def test_runtime_overlay_shape_matches_schema_subset(monkeypatch, tmp_path):
    project_state = {
        "project_id": "project:dpm",
        "label": "DPM runtime seam",
        "summary": "Honor bounded project defaults only when they are compatible with the matched thread.",
        "updated_at": "2026-04-14T18:00:00Z",
        "compatible_thread_keys": ["demo-thread"],
    }
    _seed_runtime_files(tmp_path, monkeypatch, project_state=project_state, max_overlay_chars=200)

    payload = continuity_recall.build_payload("demo-thread", None)
    overlay = payload["dpm_overlay"]

    expected_keys = {
        "schema_version",
        "overlay_id",
        "mode",
        "generated_at",
        "scope",
        "retrieval_order_applied",
        "overlay",
        "effective_constraints",
        "sources",
        "audit",
        "override_state",
    }
    assert set(overlay.keys()) == expected_keys
    assert overlay["schema_version"] == "dpm.replay-overlay.v1"
    assert overlay["mode"] == continuity_recall.load_dpm_config()["mode"] == "active-read"
    assert overlay["overlay_id"] == "overlay:thread:demo-thread:active-read"
    assert overlay["retrieval_order_applied"] == ["thread", "project"]
    assert set(overlay["scope"].keys()) == {"primary", "thread_id", "project_id", "relationship_id"}
    assert set(overlay["overlay"].keys()) == {
        "persona_summary",
        "style_directives",
        "do_not_do",
        "open_questions",
        "max_chars",
        "rendered_text",
    }
    assert set(overlay["effective_constraints"].keys()) == {
        "explicit_instruction_precedence",
        "narrowest_scope_wins",
        "cross_scope_fallback_requires_compatibility",
        "writes_allowed",
    }
    assert all(
        set(source.keys()) == {
            "source_id",
            "scope",
            "kind",
            "included",
            "priority",
            "confidence",
            "updated_at",
            "summary",
        }
        for source in overlay["sources"]
    )
    assert set(overlay["audit"].keys()) == {
        "included_source_ids",
        "excluded_sources",
        "conflicts_detected",
        "notes",
    }
    assert set(overlay["override_state"].keys()) == {
        "has_explicit_instruction",
        "override_applied",
        "instruction_source_id",
        "suppressed_source_ids",
        "effective_for_turn",
    }


def test_runtime_project_source_object_matches_contract_subset(monkeypatch, tmp_path):
    project_state = {
        "project_id": "project:dpm",
        "label": "DPM runtime seam",
        "summary": "Honor bounded project defaults only when they are compatible with the matched thread.",
        "updated_at": "2026-04-14T18:00:00Z",
        "compatible_thread_keys": ["demo-thread"],
    }
    _seed_runtime_files(tmp_path, monkeypatch, project_state=project_state)

    sources = continuity_recall.collect_runtime_read_sources(
        continuity_recall.load_dpm_config(),
        continuity_recall.load(continuity_recall.THREADS, {})["threads"]["thread:demo-thread"],
        "Demo thread checkpoint\nnext step here",
        continuity_recall.load_project_state(),
    )
    project_source = next(source for source in sources if source["scope"] == "project")

    assert set(project_source.keys()) == {
        "source_id",
        "scope",
        "kind",
        "label",
        "content",
        "priority",
        "confidence",
        "updated_at",
        "summary",
    }
    assert project_source["source_id"] == "project:dpm"
    assert project_source["scope"] == "project"
    assert project_source["kind"] == "project_summary"
    assert project_source["label"] == "DPM runtime seam"
    assert project_source["content"] == project_state["summary"]
    assert project_source["priority"] == 2
    assert project_source["confidence"] == 0.8
    assert project_source["updated_at"] == project_state["updated_at"]
    assert project_source["summary"] == "Project continuity compatible with Milestone 5 Bundle 2 smoke."


def test_helper_extraction_preserves_overlay_behavior(monkeypatch, tmp_path):
    project_state = {
        "project_id": "project:dpm",
        "label": "DPM runtime seam",
        "summary": "Honor bounded project defaults only when they are compatible with the matched thread.",
        "updated_at": "2026-04-14T18:00:00Z",
        "compatible_thread_keys": ["demo-thread"],
    }
    _seed_runtime_files(tmp_path, monkeypatch, project_state=project_state, max_overlay_chars=200)

    matched = continuity_recall.load(continuity_recall.THREADS, {})["threads"]["thread:demo-thread"]
    checkpoint_text = "Demo thread checkpoint\nnext step here"
    config = continuity_recall.load_dpm_config()
    loaded_project_state = continuity_recall.load_project_state()

    sources = continuity_recall.collect_runtime_read_sources(config, matched, checkpoint_text, loaded_project_state)
    overlay = continuity_recall.build_dpm_overlay(config, matched, checkpoint_text, loaded_project_state)

    expected_rendered_text = " ".join(
        f"{'Thread continuity' if source['scope'] == 'thread' else 'Project continuity'} for {source['label']}: {source['content']}"
        for source in sources
    )[: config["audit"]["max_overlay_chars"]]

    expected_thread_id = matched.get("thread_key") or matched.get("key")
    assert overlay["mode"] == config["mode"] == "active-read"
    assert overlay["overlay_id"] == f"overlay:thread:{expected_thread_id}:active-read"
    assert overlay["retrieval_order_applied"] == [source["scope"] for source in sources]
    assert [source["source_id"] for source in overlay["sources"]] == [source["source_id"] for source in sources]
    assert overlay["audit"]["included_source_ids"] == [source["source_id"] for source in sources]
    assert overlay["override_state"]["effective_for_turn"] == [source["source_id"] for source in sources]
    assert overlay["overlay"]["rendered_text"] == expected_rendered_text


def test_non_write_modes_remain_inert_for_runtime_payload(monkeypatch, tmp_path):
    for mode in ("disabled", "observe-only"):
        mode_tmp = tmp_path / mode
        mode_tmp.mkdir()
        _seed_runtime_files(mode_tmp, monkeypatch, config_mode=mode, max_overlay_chars=64)

        payload = continuity_recall.build_payload("demo-thread", None)

        assert payload["retrieval_status"] == "hit"
        assert payload["thread"]["id"] == "thread:demo-thread"
        assert payload["checkpoint_excerpt"] == "Demo thread checkpoint\nnext step here"
        assert payload["dpm_overlay"] is None
