import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestContinuityQueueWorker(unittest.TestCase):
    def test_process_queue_uses_portable_defaults_and_continuity_metadata(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "process_dml_ingest_queue.sh"

        with tempfile.TemporaryDirectory(prefix="dml-continuity-worker-") as tmp:
            tmp_path = Path(tmp)
            queue_path = tmp_path / "queue.jsonl"
            store_path = tmp_path / "store"
            calls_path = tmp_path / "calls.jsonl"
            fake_dml = tmp_path / "fake_dml.py"
            checkpoint = tmp_path / "20260521_main.md"

            fake_dml.write_text(
                "\n".join(
                    [
                        "import json, os, pathlib, sys",
                        "path = pathlib.Path(os.environ['CALLS_PATH'])",
                        "args = sys.argv[1:]",
                        "path.open('a', encoding='utf-8').write(json.dumps({'args': args}) + '\\n')",
                    ]
                ),
                encoding="utf-8",
            )
            checkpoint.write_text(
                "\n".join(
                    [
                        "- thread: main",
                        "- updated_at: 2026-05-21T12:00:00Z",
                        "state: executing",
                        "task: activate continuity loop",
                        "next_action: run smoke tests",
                    ]
                ),
                encoding="utf-8",
            )
            queue_path.write_text(
                json.dumps(
                    {
                        "checkpointPath": str(checkpoint),
                        "status": "queued",
                        "queuedAt": "2026-05-21T12:00:00Z",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            env = os.environ.copy()
            env.update(
                {
                    "DML_SCRIPT": str(fake_dml),
                    "PYTHON_BIN": sys.executable,
                    "DML_CONFIG_PATH": str(tmp_path / "config.yaml"),
                    "DML_REQUIRE_GPU": "0",
                    "DML_TENANT_ID": "tenant-worker",
                    "DML_CLIENT_ID": "client-worker",
                    "DML_SESSION_ID": "session-worker",
                    "DML_INSTANCE_ID": "instance-worker",
                    "CALLS_PATH": str(calls_path),
                }
            )

            subprocess.run(
                ["bash", str(script), str(queue_path), str(store_path)],
                check=True,
                env=env,
                cwd=str(root),
            )

            queue_line = json.loads(queue_path.read_text(encoding="utf-8").strip())
            self.assertEqual(queue_line["status"], "done")
            call = json.loads(calls_path.read_text(encoding="utf-8").strip())
            args = call["args"]
            self.assertIn("--no-require-gpu", args)
            self.assertIn("--kind", args)
            self.assertEqual(args[args.index("--kind") + 1], "plan")
            self.assertIn("--tenant-id", args)
            self.assertEqual(args[args.index("--tenant-id") + 1], "tenant-worker")
            self.assertIn("--client-id", args)
            self.assertEqual(args[args.index("--client-id") + 1], "client-worker")
            self.assertIn("--session-id", args)
            self.assertEqual(args[args.index("--session-id") + 1], "session-worker")
            self.assertIn("--instance-id", args)
            self.assertEqual(args[args.index("--instance-id") + 1], "instance-worker")
            meta = json.loads(args[args.index("--meta") + 1])
            self.assertEqual(meta["source"], "rolling_thread_checkpoint")
            self.assertEqual(meta["namespace"], "active_continuity")
            self.assertEqual(meta["memory_state"], "active")
            self.assertEqual(meta["tenant_id"], "tenant-worker")
            self.assertEqual(meta["client_id"], "client-worker")
            self.assertEqual(meta["session_id"], "session-worker")
            self.assertEqual(meta["instance_id"], "instance-worker")
            self.assertEqual(meta["checkpoint_path"], str(checkpoint))
            self.assertEqual(meta["thread"], "main")
            self.assertEqual(meta["state"], "executing")
            self.assertEqual(meta["task"], "activate continuity loop")
            self.assertEqual(meta["next_action"], "run smoke tests")
            self.assertEqual(meta["summary_source"], "deterministic")
            self.assertIn("task: activate continuity loop", meta["summary"])
            self.assertIn("next: run smoke tests", meta["summary"])


if __name__ == "__main__":
    unittest.main()
