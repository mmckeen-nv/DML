"""Unit tests for agentic memory infrastructure."""
import unittest
from unittest.mock import Mock, patch
import time

from daystrom_dml.agent_schema import (
    AgenticMemorySchema,
    MemoryKind,
    MemoryPhase,
    MemoryOutcome,
    make_agentic_memory,
)
from daystrom_dml.promotion_pipeline import (
    PromotionPipeline,
    MemoryEntry,
    ScratchStore,
    VerifiedStore,
    DurableStore,
)
from daystrom_dml.policy_router import (
    PolicyRouter,
    TaskType,
    SettingsOverride,
    RouterDecision,
)


class TestAgenticMemorySchema(unittest.TestCase):
    """Tests for memory schema and validation."""

    def setUp(self):
        self.schema = AgenticMemorySchema(strict=True)

    def test_valid_memory_kind(self):
        """Test valid memory kind enum values."""
        self.assertEqual(MemoryKind.ACTION, MemoryKind("action"))
        self.assertEqual(MemoryKind.OBSERVATION, MemoryKind("observation"))
        self.assertEqual(MemoryKind.PLAN, MemoryKind("plan"))

    def test_make_action(self):
        """Test action memory creation."""
        entry = self.schema.make_action("Deployed to production", {"phase": "execute"})
        self.assertEqual(entry["kind"], MemoryKind.ACTION.value)
        self.assertEqual(entry["text"], "Deployed to production")
        self.assertEqual(entry["phase"], "execute")

    def test_make_observation(self):
        """Test observation memory creation."""
        entry = self.schema.make_observation("Server CPU at 85%", {"tool": "ssh"})
        self.assertEqual(entry["kind"], MemoryKind.OBSERVATION.value)
        self.assertEqual(entry["tool"], "ssh")

    def test_make_error(self):
        """Test error memory creation."""
        entry = self.schema.make_error("Connection timeout", {"phase": "debug"})
        self.assertEqual(entry["kind"], MemoryKind.ERROR.value)
        self.assertEqual(entry["text"], "Connection timeout")

    def test_make_plan(self):
        """Test plan memory creation."""
        entry = self.schema.make_plan("Review PR before merge", {"phase": "plan"})
        self.assertEqual(entry["kind"], "plan")

    def test_invalid_kind_fails_strict(self):
        """Test strict mode rejects invalid kinds."""
        entry = {"kind": "invalid", "text": "test"}
        is_valid, errors = self.schema.validate(entry)
        self.assertFalse(is_valid)
        self.assertTrue(len(errors) > 0)

    def test_valid_entry_passes(self):
        """Test valid entry passes validation."""
        entry = {
            "kind": "action",
            "phase": "execute",
            "text": "Test",
            "provenance": {
                "task_id": "t1",
                "step_id": "s1",
                "episode_id": "e1",
                "timestamp": time.time(),
            }
        }
        is_valid, errors = self.schema.validate(entry)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)


class TestPromotionPipeline(unittest.TestCase):
    """Tests for memory promotion pipeline."""

    def setUp(self):
        self.pipeline = PromotionPipeline(
            commitment_threshold=0.75,
            allow_action_observation=True,
            strict_mode=True,
        )

    def test_scratch_store(self):
        """Test scratch store operations."""
        entry = MemoryEntry(
            text="Test",
            embedding=Mock(),
            timestamp=time.time(),
            kind="action",
            outcome="success",
        )
        self.pipeline.ingest_to_scratch(entry)
        self.assertEqual(len(self.pipeline.scratch.entries), 1)

    def test_verified_store(self):
        """Test verified store operations."""
        entry = MemoryEntry(
            text="Test",
            embedding=Mock(),
            timestamp=time.time(),
            kind="plan",
            outcome="success",
        )
        self.pipeline.verified.add(entry)
        self.assertEqual(len(self.pipeline.verified.entries), 1)

    def test_durable_store(self):
        """Test durable store operations."""
        entry = MemoryEntry(
            text="Test",
            embedding=Mock(),
            timestamp=time.time(),
            kind="action",
            outcome="success",
            fidelity=0.8,
            meta={"kind": "action", "outcome": "success"},
        )
        self.pipeline.promote_to_verified(entry)
        self.pipeline.promote_to_durable(entry)
        self.assertEqual(len(self.pipeline.durable.entries), 1)
        self.assertTrue(entry.is_durable)

    def test_promotion_requires_success(self):
        """Test that failures are not promoted."""
        entry = MemoryEntry(
            text="Test",
            embedding=Mock(),
            timestamp=time.time(),
            kind="action",
            outcome="fail",
            fidelity=0.5,
            meta={"kind": "action", "outcome": "fail"},
        )
        self.pipeline.promote_to_verified(entry)
        promoted = self.pipeline.promote_to_durable(entry)
        self.assertFalse(promoted)

    def test_partial_success_requires_threshold(self):
        """Test partial success requires high fidelity."""
        entry = MemoryEntry(
            text="Test",
            embedding=Mock(),
            timestamp=time.time(),
            kind="action",
            outcome="partial",
            fidelity=0.6,
            meta={"kind": "action", "outcome": "partial"},
        )
        self.pipeline.promote_to_verified(entry)
        promoted = self.pipeline.promote_to_durable(entry)
        self.assertFalse(promoted)  # Below threshold

    def test_partial_success_with_threshold(self):
        """Test partial success passes with high fidelity."""
        entry = MemoryEntry(
            text="Test",
            embedding=Mock(),
            timestamp=time.time(),
            kind="action",
            outcome="partial",
            fidelity=0.85,
            meta={"kind": "action", "outcome": "partial"},
        )
        self.pipeline.promote_to_verified(entry)
        promoted = self.pipeline.promote_to_durable(entry)
        self.assertTrue(promoted)

    def test_auto_promote_all(self):
        """Test automatic promotion of verified entries."""
        # Add entries to scratch
        for i in range(5):
            entry = MemoryEntry(
                text=f"Test {i}",
                embedding=Mock(),
                timestamp=time.time(),
                kind="action",
                outcome="success" if i > 0 else "fail",
                meta={"kind": "action", "outcome": "success" if i > 0 else "fail"},
            )
            self.pipeline.ingest_to_scratch(entry)

        # Promote verified ones
        verified_entry = MemoryEntry(
            text="Verified",
            embedding=Mock(),
            timestamp=time.time(),
            kind="action",
            outcome="success",
            meta={"kind": "action", "outcome": "success", "kind_in_meta": True},
        )
        self.pipeline.promote_to_verified(verified_entry)

        results = self.pipeline.auto_promote_all()
        self.assertGreater(results.get("promoted", 0), 0)


class TestPolicyRouter(unittest.TestCase):
    """Tests for policy router."""

    def setUp(self):
        self.router = PolicyRouter(enabled=True)

    def test_task_type_detection(self):
        """Test task type detection from metadata."""
        meta = {"tools": ["ssh", "docker"]}
        task_type = self.router.detect_task_type(meta)
        self.assertEqual(task_type, TaskType.DEVSOPS)

        meta = {"tools": ["git", "python"]}
        task_type = self.router.detect_task_type(meta)
        self.assertEqual(task_type, TaskType.CODING)

    def test_router_decision(self):
        """Test router makes decision."""
        decision = self.router.decide(meta={"tools": ["ssh"]}, phase=MemoryPhase.EXECUTE)

        self.assertIsNotNone(decision)
        self.assertEqual(decision.selected_profile, TaskType.DEVSOPS.value)
        self.assertEqual(decision.phase, MemoryPhase.EXECUTE)
        self.assertIsInstance(decision.overrides, SettingsOverride)

    def test_phase_modifier(self):
        """Test phase modifiers."""
        decision = self.router.decide(meta={"tools": ["ssh"]}, phase=MemoryPhase.EXECUTE)
        self.assertEqual(decision.overrides.top_k, 6)

        decision = self.router.decide(meta={"tools": ["ssh"]}, phase=MemoryPhase.DEBUG)
        self.assertEqual(decision.overrides.top_k, 4)

    def test_stuckness_adjustment(self):
        """Test stuckness detection."""
        decision = self.router.decide(
            meta={"tools": ["ssh"]},
            phase=MemoryPhase.EXECUTE,
            stuckness=3,
        )
        self.assertGreaterEqual(decision.overrides.similarity_threshold, 0.5)

    def test_token_pressure_adjustment(self):
        """Test token pressure adjustment."""
        decision = self.router.decide(
            meta={"tools": ["ssh"]},
            token_pressure=0.9,
        )
        self.assertLessEqual(decision.overrides.top_k, 4)

    def test_disabled_router_returns_none(self):
        """Test router disabled returns None."""
        router = PolicyRouter(enabled=False)
        decision = router.decide(meta={"tools": ["ssh"]})
        self.assertIsNone(decision)


class TestEndToEndSmokeTest(unittest.TestCase):
    """End-to-end smoke test for agentic workflow."""

    def test_complete_workflow(self):
        """Test complete workflow from ingest to retrieve."""
        from daystrom_dml.dml_adapter import DMLAdapter

        adapter = DMLAdapter(
            config_overrides={
                "storage_dir": "/tmp/test_agentic",
                "model_name": "gpt2",  # Real LLM model (tiny, fast)
                "embedding_model": "all-MiniLM-L6-v2",  # Embedding model for semantic search
                "dml.agentic_mode.enabled": True,
            },
            start_aging_loop=False,
        )

        try:
            # Ingest with agentic types
            adapter.ingest_agentic(
                text="Deployed to production successfully",
                kind=MemoryKind.ACTION,
                meta={
                    "phase": MemoryPhase.EXECUTE.value,
                    "tool": "docker",
                    "outcome": MemoryOutcome.SUCCESS.value,
                    "provenance": {
                        "task_id": "t1",
                        "step_id": "s1",
                        "episode_id": "e1",
                        "timestamp": time.time(),
                    }
                },
            )

            # Retrieve
            report = adapter.retrieve_context(
                prompt="What happened in deployment?",
            )

            # Verify
            self.assertIsNotNone(report["raw_context"])
            self.assertIn("production", report["raw_context"].lower())

            # Check metrics
            if adapter.metrics_enabled:
                self.assertGreater(report["context_tokens"], 0)

            print("✓ Smoke test passed")

        finally:
            adapter.close()


if __name__ == "__main__":
    unittest.main()