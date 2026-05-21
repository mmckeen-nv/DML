import importlib.util
import sys
import unittest
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "resume_quality_smoke.py"
    spec = importlib.util.spec_from_file_location("resume_quality_smoke", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["resume_quality_smoke"] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


class TestResumeQualitySmoke(unittest.TestCase):
    def test_resume_quality_smoke_passes_when_latest_checkpoint_wins(self):
        def fake_runner(argv):
            if "handoff" in argv:
                return {"status": "ok", "_returncode": 0}
            return {
                "status": "ok",
                "_returncode": 0,
                "latest_checkpoint": {"next_action": "unit-NEXT-2"},
            }

        args = mod.build_parser().parse_args(["--run-id", "unit", "--sessions", "3"])
        summary = mod.run_smoke(args, runner=fake_runner)

        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["expected_next_action"], "unit-NEXT-2")

    def test_resume_quality_smoke_fails_when_resume_returns_old_checkpoint(self):
        def fake_runner(argv):
            if "handoff" in argv:
                return {"status": "ok", "_returncode": 0}
            return {
                "status": "ok",
                "_returncode": 0,
                "latest_checkpoint": {"next_action": "unit-NEXT-0"},
            }

        args = mod.build_parser().parse_args(["--run-id", "unit", "--sessions", "3"])
        summary = mod.run_smoke(args, runner=fake_runner)

        self.assertEqual(summary["status"], "fail")


if __name__ == "__main__":
    unittest.main()
