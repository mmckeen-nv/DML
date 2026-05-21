import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "recall_eval.py"
    spec = importlib.util.spec_from_file_location("recall_eval", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["recall_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


class TestRecallEvalHarness(unittest.TestCase):
    def test_score_case_requires_expected_hits_and_forbidden_absence(self):
        case = mod.EvalCase(
            name="tenant_recall",
            action="retrieve",
            query="alpha plan",
            tenant_id="alpha",
            expected_markers=("ALPHA-MARKER",),
            forbidden_markers=("BETA-MARKER",),
        )
        result = mod.CommandResult(
            argv=["dml_memory.py", "retrieve"],
            returncode=0,
            elapsed_ms=12.5,
            payload={
                "status": "ok",
                "items": [{"text": "contains ALPHA-MARKER", "meta": {"tenant_id": "alpha"}}],
                "raw_context": "ALPHA-MARKER",
            },
            stdout="{}",
            stderr="",
        )

        scored = mod._score_case(case, result)

        self.assertTrue(scored["passed"])
        self.assertEqual(scored["expected_ranks"]["ALPHA-MARKER"], 1)
        self.assertFalse(scored["forbidden_hits"]["BETA-MARKER"])

    def test_score_case_fails_when_forbidden_marker_leaks(self):
        case = mod.EvalCase(
            name="tenant_recall",
            action="retrieve",
            query="alpha plan",
            tenant_id="alpha",
            expected_markers=("ALPHA-MARKER",),
            forbidden_markers=("BETA-MARKER",),
        )
        result = mod.CommandResult(
            argv=["dml_memory.py", "retrieve"],
            returncode=0,
            elapsed_ms=12.5,
            payload={
                "status": "ok",
                "items": [{"text": "ALPHA-MARKER and leaked BETA-MARKER"}],
            },
            stdout="{}",
            stderr="",
        )

        scored = mod._score_case(case, result)

        self.assertFalse(scored["passed"])
        self.assertTrue(scored["forbidden_hits"]["BETA-MARKER"])

    def test_run_eval_writes_json_and_markdown_reports(self):
        calls = []

        def fake_runner(argv):
            calls.append(argv)
            action = argv[argv.index("--lock-timeout-ms") + 2]
            if action == "ingest":
                return mod.CommandResult(argv, 0, 3.0, {"status": "ok", "chunks_ingested": 1}, "{}", "")
            if action == "resume":
                return mod.CommandResult(
                    argv,
                    0,
                    5.0,
                    {
                        "status": "ok",
                        "items": [{"text": "unit-CONTINUITY-NEXT-ACTION"}],
                        "raw_context": "unit-CONTINUITY-NEXT-ACTION",
                        "resume_total_latency_ms": 4.0,
                    },
                    "{}",
                    "",
                )
            return mod.CommandResult(
                argv,
                0,
                4.0,
                {
                    "status": "ok",
                    "items": [{"text": "unit-ALPHA-EXPORT-PLAN"}],
                    "raw_context": "unit-ALPHA-EXPORT-PLAN",
                    "retrieve_total_latency_ms": 3.5,
                },
                "{}",
                "",
            )

        with tempfile.TemporaryDirectory(prefix="recall-eval-test-") as tmp:
            output_dir = Path(tmp) / "reports"
            args = mod.build_parser().parse_args(
                [
                    "--run-id",
                    "unit",
                    "--storage-dir",
                    str(Path(tmp) / "store"),
                    "--output-dir",
                    str(output_dir),
                ]
            )
            summary = mod.run_eval(args, runner=fake_runner)

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["case_pass_count"], 3)
            self.assertTrue((output_dir / "recall_eval_report.json").exists())
            self.assertTrue((output_dir / "recall_eval_report.md").exists())
            self.assertEqual(sum(1 for call in calls if "ingest" in call), 4)
            self.assertEqual(sum(1 for call in calls if "retrieve" in call), 2)
            self.assertEqual(sum(1 for call in calls if "resume" in call), 1)


if __name__ == "__main__":
    unittest.main()
