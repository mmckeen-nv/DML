import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_openclaw_memory.py"
spec = importlib.util.spec_from_file_location("benchmark_openclaw_memory", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["benchmark_openclaw_memory"] = mod
spec.loader.exec_module(mod)


class TestBenchmarkMetrics(unittest.TestCase):
    def test_dcg_matches_log_discount_definition(self):
        scores = [1.0, 0.5, 0.25]
        expected = (
            ((2**1.0) - 1) / 1.0
            + ((2**0.5) - 1) / math.log2(3)
            + ((2**0.25) - 1) / math.log2(4)
        )
        self.assertAlmostEqual(mod._dcg(scores), expected, places=8)

    def test_p95_uses_nearest_rank(self):
        vals = [3.1, 8.2, 1.7, 4.0, 5.6, 2.4, 6.8, 9.9, 7.3, 10.5]
        # rank = ceil(0.95*10) = 10 -> highest value
        self.assertEqual(mod._p95(vals), 10.5)

    def test_p95_empty(self):
        self.assertEqual(mod._p95([]), 0.0)


class TestBenchmarkDelta(unittest.TestCase):
    def test_compute_delta_reports_expected_signals(self):
        current = {
            "avg_token_savings_pct": 91.5,
            "avg_latency_ms": 6.2,
            "avg_precision_at_k": 0.72,
            "avg_ndcg_at_k": 0.95,
            "avg_retrieval_noise_score": 0.11,
        }
        previous = {
            "status": "ok",
            "avg_token_savings_pct": 90.0,
            "avg_latency_ms": 7.0,
            "avg_precision_at_k": 0.70,
            "avg_ndcg_at_k": 0.94,
            "avg_retrieval_noise_score": 0.13,
        }
        delta = mod.compute_delta(current, previous)
        self.assertEqual(delta["from_status"], "ok")
        self.assertEqual(delta["token_savings_pct_delta"], 1.5)
        self.assertEqual(delta["latency_ms_delta"], -0.8)
        self.assertEqual(delta["precision_at_k_delta"], 0.02)
        self.assertEqual(delta["ndcg_at_k_delta"], 0.01)
        self.assertEqual(delta["retrieval_noise_delta"], -0.02)


class TestBenchmarkStorageReset(unittest.TestCase):
    def test_reset_storage_dir_recreates_target_under_workspace_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_workspace = mod.WORKSPACE
            try:
                mod.WORKSPACE = Path(tmp)
                target = Path(tmp) / "data" / "bench"
                target.mkdir(parents=True, exist_ok=True)
                (target / "stale.txt").write_text("old")

                out = Path(mod._reset_storage_dir(str(target)))
                self.assertEqual(out, target.resolve())
                self.assertTrue(out.exists())
                self.assertFalse((out / "stale.txt").exists())
            finally:
                mod.WORKSPACE = original_workspace

    def test_reset_storage_dir_rejects_paths_outside_workspace_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_workspace = mod.WORKSPACE
            try:
                mod.WORKSPACE = Path(tmp)
                outside = Path(tmp) / "elsewhere" / "bench"
                outside.mkdir(parents=True, exist_ok=True)
                with self.assertRaises(ValueError):
                    mod._reset_storage_dir(str(outside))
            finally:
                mod.WORKSPACE = original_workspace


if __name__ == "__main__":
    unittest.main()
