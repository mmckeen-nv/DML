import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "beta_readiness.py"
    spec = importlib.util.spec_from_file_location("beta_readiness", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["beta_readiness"] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


class TestBetaReadiness(unittest.TestCase):
    def _args(self, *extra):
        return mod.build_parser().parse_args(
            [
                "--storage-dir",
                "/tmp/dml-readiness-test",
                "--tenant-id",
                "openclaw",
                "--recall-run-id",
                "unit",
                *extra,
            ]
        )

    def test_run_gate_passes_when_all_steps_pass(self):
        calls = []

        def fake_runner(argv):
            calls.append(argv)
            action = argv[-1]
            if "recall_eval.py" in str(argv[1]):
                return mod.CommandResult(argv, 0, 9.0, {"status": "pass"}, "{}", "")
            if action == "health":
                payload = {"status": "ok", "state": {"checksum_ok": True}}
            elif action == "verify":
                payload = {"status": "ok"}
            elif "conflicts" in argv:
                payload = {"status": "ok", "conflict_group_count": 0}
            elif "audit-tail" in argv:
                payload = {"status": "ok", "audit": {"event_count": 1}}
            else:
                payload = {"status": "ok"}
            return mod.CommandResult(argv, 0, 5.0, payload, "{}", "")

        with tempfile.TemporaryDirectory(prefix="beta-readiness-test-") as tmp:
            args = self._args("--output-dir", str(Path(tmp) / "reports"))
            summary = mod.run_gate(args, runner=fake_runner)

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["failed_steps"], [])
            self.assertTrue((Path(tmp) / "reports" / "beta_readiness_report.json").exists())
            self.assertTrue((Path(tmp) / "reports" / "beta_readiness_report.md").exists())
            self.assertTrue(any("recall_eval.py" in str(call[1]) for call in calls))

    def test_run_gate_fails_when_conflict_budget_exceeded(self):
        def fake_runner(argv):
            if "recall_eval.py" in str(argv[1]):
                return mod.CommandResult(argv, 0, 9.0, {"status": "pass"}, "{}", "")
            if "conflicts" in argv:
                return mod.CommandResult(argv, 0, 5.0, {"status": "ok", "conflict_group_count": 2}, "{}", "")
            return mod.CommandResult(argv, 0, 5.0, {"status": "ok"}, "{}", "")

        summary = mod.run_gate(self._args("--max-unresolved-conflicts", "0"), runner=fake_runner)

        self.assertEqual(summary["status"], "fail")
        self.assertIn("conflict_budget", summary["failed_steps"])

    def test_skip_recall_eval_omits_recall_step(self):
        calls = []

        def fake_runner(argv):
            calls.append(argv)
            payload = {"status": "ok", "conflict_group_count": 0}
            return mod.CommandResult(argv, 0, 5.0, payload, "{}", "")

        summary = mod.run_gate(self._args("--skip-recall-eval"), runner=fake_runner)

        self.assertEqual(summary["status"], "pass")
        self.assertFalse(any("recall_eval.py" in str(call[1]) for call in calls))
        self.assertEqual([step["name"] for step in summary["steps"]], ["health", "verify", "conflicts", "audit-tail"])


if __name__ == "__main__":
    unittest.main()
