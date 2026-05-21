import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "stress_harness.py"
    spec = importlib.util.spec_from_file_location("stress_harness", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["stress_harness"] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


class TestStressHarness(unittest.TestCase):
    def _args(self, *extra):
        return mod.build_parser().parse_args(
            [
                "--run-id",
                "unit",
                "--writes",
                "4",
                "--workers",
                "2",
                "--tenants",
                "2",
                "--sessions",
                "2",
                *extra,
            ]
        )

    def test_run_stress_passes_when_writes_and_isolation_pass(self):
        calls = []

        def fake_runner(argv):
            calls.append(argv)
            state_path = Path(tmp) / "store" / "dml_state.jsonl"
            if "ingest" in argv:
                marker = argv[argv.index("--text") + 1].split(":", 1)[0]
                state_path.parent.mkdir(parents=True, exist_ok=True)
                with state_path.open("a", encoding="utf-8") as handle:
                    handle.write(marker + "\n")
                return mod.CommandResult(argv, 0, 5.0, {"status": "ok", "chunks_ingested": 1}, "{}", "")
            if "retrieve" in argv:
                query = argv[argv.index("--query") + 1]
                return mod.CommandResult(argv, 0, 4.0, {"status": "ok", "items": [{"text": query}]}, "{}", "")
            if "verify" in argv:
                return mod.CommandResult(argv, 0, 3.0, {"status": "ok", "loaded_count": 4, "errors": []}, "{}", "")
            if "audit-tail" in argv:
                return mod.CommandResult(argv, 0, 2.0, {"status": "ok", "audit": {"event_count": 4}}, "{}", "")
            return mod.CommandResult(argv, 0, 1.0, {"status": "ok"}, "{}", "")

        with tempfile.TemporaryDirectory(prefix="stress-harness-test-") as tmp:
            args = self._args("--storage-dir", str(Path(tmp) / "store"))
            summary = mod.run_stress(args, runner=fake_runner)

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["write_count"], 4)
            self.assertTrue(summary["isolation_ok"])
            self.assertEqual(sum(1 for call in calls if "ingest" in call), 4)
            self.assertTrue(any("verify" in call for call in calls))

    def test_run_stress_fails_when_retrieval_leaks_forbidden_marker(self):
        def fake_runner(argv):
            if "ingest" in argv:
                return mod.CommandResult(argv, 0, 5.0, {"status": "ok", "chunks_ingested": 1}, "{}", "")
            if "retrieve" in argv:
                query = argv[argv.index("--query") + 1]
                return mod.CommandResult(
                    argv,
                    0,
                    4.0,
                    {"status": "ok", "items": [{"text": f"{query} unit-WRITER-001"}]},
                    "{}",
                    "",
                )
            if "verify" in argv:
                return mod.CommandResult(argv, 0, 3.0, {"status": "ok", "loaded_count": 4, "errors": []}, "{}", "")
            if "audit-tail" in argv:
                return mod.CommandResult(argv, 0, 2.0, {"status": "ok", "audit": {"event_count": 4}}, "{}", "")
            return mod.CommandResult(argv, 0, 1.0, {"status": "ok"}, "{}", "")

        summary = mod.run_stress(self._args(), runner=fake_runner)

        self.assertEqual(summary["status"], "fail")
        self.assertFalse(summary["isolation_ok"])
        self.assertTrue(summary["isolation"][0]["forbidden_hits"])

    def test_run_stress_fails_when_persisted_markers_are_missing(self):
        def fake_runner(argv):
            if "ingest" in argv:
                return mod.CommandResult(argv, 0, 5.0, {"status": "ok", "chunks_ingested": 1}, "{}", "")
            if "retrieve" in argv:
                query = argv[argv.index("--query") + 1]
                return mod.CommandResult(argv, 0, 4.0, {"status": "ok", "items": [{"text": query}]}, "{}", "")
            if "verify" in argv:
                return mod.CommandResult(argv, 0, 3.0, {"status": "ok", "loaded_count": 1, "errors": []}, "{}", "")
            if "audit-tail" in argv:
                return mod.CommandResult(argv, 0, 2.0, {"status": "ok", "audit": {"event_count": 4}}, "{}", "")
            return mod.CommandResult(argv, 0, 1.0, {"status": "ok"}, "{}", "")

        summary = mod.run_stress(self._args(), runner=fake_runner)

        self.assertEqual(summary["status"], "fail")
        self.assertFalse(summary["checks"]["markers"]["passed"])
        self.assertEqual(len(summary["checks"]["markers"]["missing"]), 4)


if __name__ == "__main__":
    unittest.main()
