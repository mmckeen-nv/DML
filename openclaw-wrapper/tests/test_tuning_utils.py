import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "tuning_utils.py"
spec = importlib.util.spec_from_file_location("tuning_utils", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["tuning_utils"] = mod
spec.loader.exec_module(mod)


class TestTuningUtils(unittest.TestCase):
    def test_rewrite_query_expands_blocker_terms(self):
        q = "How do I export USD with fallback?"
        out = mod.rewrite_query(q)
        self.assertIn("expansion:", out)
        self.assertIn("usd_export_mode", out)
        self.assertIn("usd.fallback.json", out)

    def test_noise_filter_keeps_domain_chunks_drops_boilerplate(self):
        keep = "USD export mode uses native operator and fallback_glb_only manifest."
        drop = "fallback_trigger=none attempts=[{...}] heartbeat_summary PROJECT HEARTBEAT"
        self.assertTrue(mod.should_keep_chunk(keep))
        self.assertFalse(mod.should_keep_chunk(drop))

    def test_smart_chunking_splits_long_text(self):
        text = " ".join(["anti-blob chassis primitive stack" for _ in range(400)])
        chunks = mod.smart_chunks(text, chunk_chars=220, overlap=40)
        self.assertGreater(len(chunks), 5)
        self.assertTrue(all(len(c) <= 260 for c in chunks))


if __name__ == "__main__":
    unittest.main()
