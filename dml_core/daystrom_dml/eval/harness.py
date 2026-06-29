"""Agentic evaluation harness for multi-objective scoring."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent_schema import MemoryPhase, MemoryOutcome

LOGGER = logging.getLogger(__name__)


@dataclass
class TaskSpec:
    """Specification of an evaluation task."""
    name: str
    description: str
    initial_context: str
    expected_success_signals: List[str]
    max_steps: int = 10
    tools: List[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Result of an evaluation run."""
    task_name: str
    success: bool
    steps_completed: int
    total_tokens: int
    final_state: Dict[str, Any]
    metrics: Dict[str, float]


class AgenticEvaluator:
    """Multi-objective evaluation harness for agent tasks."""

    def __init__(self, output_dir: str = "./eval/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[EvalResult] = []

    def create_task(self, name: str, description: str, initial_context: str, expected_success_signals: List[str], **kwargs) -> TaskSpec:
        """Create a task specification."""
        return TaskSpec(
            name=name,
            description=description,
            initial_context=initial_context,
            expected_success_signals=expected_success_signals,
            **kwargs,
        )

    def evaluate(
        self,
        task: TaskSpec,
        run_id: str = "latest",
    ) -> EvalResult:
        """Evaluate a single task."""
        LOGGER.info(f"Starting evaluation: {task.name}")

        start_time = time.time()
        steps = 0
        total_tokens = 0
        success = False

        # Simulate agent execution
        context = task.initial_context
        for step in range(task.max_steps):
            steps += 1

            # Simulate agent action
            context = self._simulate_agent_step(context, step, task)

            # Check for success signals
            if self._check_success(context, task.expected_success_signals):
                success = True
                break

        elapsed = time.time() - start_time

        # Simulate metrics
        result = EvalResult(
            task_name=task.name,
            success=success,
            steps_completed=steps,
            total_tokens=int(total_tokens),
            final_state={"context": context[:500] + "...", "steps": steps},
            metrics={
                "time_seconds": round(elapsed, 2),
                "steps_per_second": round(steps / elapsed, 2) if elapsed > 0 else 0,
                "token_efficiency": round(total_tokens / steps, 2) if steps > 0 else 0,
            },
        )

        self.results.append(result)
        LOGGER.info(f"Evaluation complete: {task.name} - Success: {success}, Steps: {steps}")

        return result

    def _simulate_agent_step(self, context: str, step: int, task: TaskSpec) -> str:
        """Simulate a single agent step."""
        context += f"\n\n[Step {step}] Agent action completed."
        return context

    def _check_success(self, context: str, success_signals: List[str]) -> bool:
        """Check if success signals are present in context."""
        context_lower = context.lower()
        for signal in success_signals:
            if signal.lower() in context_lower:
                return True
        return False

    def print_scoreboard(self) -> None:
        """Print evaluation scoreboard."""
        if not self.results:
            LOGGER.warning("No results to display")
            return

        print("\n" + "="*70)
        print("AGENTIC EVALUATION SCOREBOARD")
        print("="*70)

        total_tasks = len(self.results)
        successful_tasks = sum(1 for r in self.results if r.success)

        print(f"\nOVERALL:")
        print(f"  Total Tasks: {total_tasks}")
        print(f"  Successful: {successful_tasks}")
        print(f"  Success Rate: {successful_tasks/total_tasks*100:.1f}%")

        print(f"\n{'TASK':<40} {'SUCCESS':<10} {'STEPS':<8} {'TOKENS':<10} {'TIME'}")
        print("-"*78)

        for result in self.results:
            status = "✓" if result.success else "✗"
            print(f"{result.task_name:<40} {status:<10} {result.steps_completed:<8} {result.total_tokens:<10} {result.metrics['time_seconds']:.1f}s")

        print("="*70)

    def save_results(self) -> Path:
        """Save results to JSON file."""
        output_path = self.output_dir / f"{self.output_dir.name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tasks": len(self.results),
            "successful": sum(1 for r in self.results if r.success),
            "results": [
                {
                    "task_name": r.task_name,
                    "success": r.success,
                    "steps_completed": r.steps_completed,
                    "total_tokens": r.total_tokens,
                    "final_state": r.final_state,
                    "metrics": r.metrics,
                }
                for r in self.results
            ],
        }

        output_path.write_text(json.dumps(results_data, indent=2))
        LOGGER.info(f"Results saved to: {output_path}")
        return output_path

    def load_results(self, path: str) -> None:
        """Load results from JSON file."""
        result_path = Path(path)
        if not result_path.exists():
            LOGGER.error(f"Results file not found: {path}")
            return

        with result_path.open() as f:
            data = json.load(f)

        self.results = []
        for result_data in data.get("results", []):
            self.results.append(EvalResult(**result_data))

        LOGGER.info(f"Loaded {len(self.results)} results from {path}")


# Task templates
def create_development_tasks() -> List[TaskSpec]:
    """Create common development tasks."""
    return [
        TaskSpec(
            name="git_commit",
            description="Test agent's ability to commit code changes",
            initial_context="Modified files: main.py, utils.py\nChanges: Fixed bug",
            expected_success_signals=["commit", "git"],
        ),
        TaskSpec(
            name="docker_build",
            description="Test agent's ability to build Docker container",
            initial_context="Context: Dockerfile exists",
            expected_success_signals=["build", "docker"],
        ),
    ]


def create_coding_tasks() -> List[TaskSpec]:
    """Create common coding tasks."""
    return [
        TaskSpec(
            name="python_function",
            description="Test agent's ability to generate Python function",
            initial_context="Need to implement: def fibonacci(n):",
            expected_success_signals=["def", "fibonacci"],
        ),
        TaskSpec(
            name="code_review",
            description="Test agent's ability to review code",
            initial_context="Review this code:\nfor i in range(100):\n  print(i)",
            expected_success_signals=["review", "improve"],
        ),
    ]


def create_research_tasks() -> List[TaskSpec]:
    """Create common research tasks."""
    return [
        TaskSpec(
            name="summary_generation",
            description="Test agent's ability to summarize research",
            initial_context="Paper: Attention Is All You Need",
            expected_success_signals=["summary", "transformer"],
        ),
        TaskSpec(
            name="literature_search",
            description="Test agent's ability to search literature",
            initial_context="Topic: Quantum computing advances",
            expected_success_signals=["quantum", "search"],
        ),
    ]


if __name__ == "__main__":
    # Quick demo
    evaluator = AgenticEvaluator()

    tasks = create_development_tasks() + create_coding_tasks() + create_research_tasks()

    for task in tasks:
        evaluator.evaluate(task)

    evaluator.print_scoreboard()
    evaluator.save_results()