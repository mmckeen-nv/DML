import importlib.util
import os
import stat
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "dml_background_worker.py"
    spec = importlib.util.spec_from_file_location("dml_background_worker", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["dml_background_worker"] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


class TestBackgroundWorker(unittest.TestCase):
    def test_run_once_reports_queue_counts(self):
        with tempfile.TemporaryDirectory(prefix="dml-worker-test-") as tmp:
            root = Path(tmp)
            queue = root / "queue.jsonl"
            queue.write_text('{"status":"queued"}\n{"status":"done"}\n', encoding="utf-8")
            worker = root / "worker.sh"
            worker.write_text("#!/usr/bin/env bash\necho worker-ok\n", encoding="utf-8")
            worker.chmod(worker.stat().st_mode | stat.S_IXUSR)
            args = Namespace(
                queue_path=queue,
                openclaw_home=root,
                openclaw_workspace=root,
                storage_dir=root / "store",
                worker_script=worker,
                timeout_s=5,
            )

            payload = mod.run_once(args)

            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["queued_before"], 1)
            self.assertEqual(payload["queued_after"], 1)
            self.assertIn("worker-ok", payload["stdout"])


if __name__ == "__main__":
    unittest.main()
