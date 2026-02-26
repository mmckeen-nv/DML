"""Example usage of DMLAgent for OpenClaw agents."""

from skills.daystrom_dml import DMLAgent


def example_basic_usage():
    """Basic example of using DMLAgent."""
    print("=== Basic DML Agent Usage ===\n")

    # Initialize agent
    with DMLAgent(
        model_name="gpt2",
        embedding_model="all-MiniLM-L6-v2",
        storage_dir="./data/dml",
    ) as dml:
        # Ingest some information
        dml.ingest(
            text="Deployed the application successfully to production",
            kind="action",
            meta={
                "phase": "execute",
                "tool": "git",
                "outcome": "success",
                "provenance": {
                    "task_id": "t1",
                    "step_id": "s1",
                }
            }
        )

        # Retrieve context
        report = dml.retrieve("deployment results")
        print(f"Context: {report.get('raw_context', '')}\n")

        # Get formatted context for LLM
        context = dml.get_context("deployment", max_tokens=500)
        print(f"Formatted context:\n{context}\n")


def example_agent_workflow():
    """Example of using DML in an agent workflow."""
    print("=== Agent Workflow Example ===\n")

    with DMLAgent() as dml:
        # Step 1: Store task information
        dml.ingest(
            text="Analyzing test results from the DML project",
            kind="planning",
            meta={"tool": "pytest", "phase": "analysis"}
        )

        # Step 2: Store execution results
        dml.ingest(
            text="57 tests passed, all critical functionality verified",
            kind="result",
            meta={
                "tool": "pytest",
                "outcome": "success",
                "tests_passed": 57,
                "tests_failed": 0
            }
        )

        # Step 3: Retrieve relevant context for decision making
        context = dml.get_context("test results")
        print(f"Retrieved context:\n{context}\n")

        # Step 4: Store decision
        dml.ingest(
            text="Project ready for deployment. All tests passing with GPU acceleration enabled.",
            kind="insight",
            meta={"decision": "deploy"}
        )


def example_quick_functions():
    """Example of using quick functions."""
    print("=== Quick Functions Example ===\n")

    # Using quick functions
    result = dml_ingest(
        text="Quick ingest test",
        kind="action",
        meta={"tool": "manual"}
    )
    print(f"Ingested: {result}\n")

    retrieval = dml_retrieve("quick ingest")
    print(f"Retrieved: {retrieval.get('raw_context', '')}\n")


def example_memory_management():
    """Example of memory management."""
    print("=== Memory Management Example ===\n")

    with DMLAgent() as dml:
        # Store various types of information
        dml.ingest("Project status: In development", kind="observation")
        dml.ingest("GPU acceleration implemented", kind="insight")
        dml.ingest("Tests passing: 57/57", kind="result")
        dml.ingest("Next step: Deploy to production", kind="planning")

        # Get summary of memory
        total = dml.memory_count()
        print(f"Total memories: {total}\n")

        # Retrieve specific types
        context = dml.get_context("status", max_tokens=300)
        print(f"Status context:\n{context}\n")


if __name__ == "__main__":
    example_basic_usage()
    example_agent_workflow()
    example_quick_functions()
    example_memory_management()