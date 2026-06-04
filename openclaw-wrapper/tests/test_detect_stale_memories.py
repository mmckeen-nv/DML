import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "detect_stale_memories.py"
    spec = importlib.util.spec_from_file_location("detect_stale_memories", module_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["detect_stale_memories"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_stale_memory_report_flags_old_low_quality_items(tmp_path):
    mod = _load_module()
    state_file = tmp_path / "dml_state.jsonl"
    state_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "old-low-quality",
                        "text": "obsolete deployment note",
                        "timestamp": "2024-01-01T00:00:00Z",
                        "fidelity": 0.2,
                        "meta": {"quality_score": 0.2, "namespace": "ops"},
                    }
                ),
                json.dumps(
                    {
                        "id": "fresh-good",
                        "text": "current stable fact",
                        "timestamp": "2026-06-01T00:00:00Z",
                        "fidelity": 0.9,
                        "meta": {"summary": "current stable fact", "quality_score": 0.9},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    report = mod.build_report(
        state_file,
        stale_after_days=90,
        low_fidelity=0.55,
        limit=10,
        now=1_780_000_000.0,
    )

    assert report["schema_version"] == "dml.stale-memory-report.v1"
    assert report["totals"]["items_scanned"] == 2
    assert report["totals"]["candidates"] == 1
    candidate = report["candidates"][0]
    assert candidate["id"] == "old-low-quality"
    assert candidate["recommended_action"] in {"review", "suppress_candidate"}
    assert {"old_age", "low_fidelity", "low_quality_score"}.issubset(set(candidate["reasons"]))


def test_stale_memory_detector_cli_is_read_only_and_emits_json(tmp_path, capsys):
    mod = _load_module()
    state_file = tmp_path / "dml_state.jsonl"
    original = json.dumps({"id": "missing-summary", "timestamp": "2024-01-01T00:00:00Z", "meta": {}})
    state_file.write_text(original + "\n", encoding="utf-8")

    rc = mod.main(["--state-file", str(state_file), "--stale-after-days", "90", "--limit", "5"])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["status"] == "ok"
    assert payload["candidates"]
    assert state_file.read_text(encoding="utf-8") == original + "\n"
