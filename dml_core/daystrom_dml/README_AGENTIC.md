# Agentic Mode

DML now supports **agentic mode** - an adaptive memory substrate for intelligent agents with policy-based routing and strict memory promotion.

## Quickstart

### Enable Agentic Mode

```yaml
# In your config.yaml
agentic_mode:
  enabled: true
router:
  enabled: true
  log_level: info
```

### Use in Code

```python
from daystrom_dml.dml_adapter import DMLAdapter

# Initialize with agentic mode enabled
adapter = DMLAdapter(
    config_overrides={
        "agentic_mode": {"enabled": True},
        "router": {"enabled": True},
    }
)

# Ingest memories with explicit types
adapter.ingest_agentic(
    text="Deployed container to production",
    kind="action",
    meta={
        "phase": "execute",
        "tool": "docker",
        "outcome": "success",
        "provenance": {
            "task_id": "task-123",
            "step_id": "step-1",
            "episode_id": "episode-1",
            "timestamp": time.time(),
        }
    }
)

# Retrieve with phase-aware filtering
report = adapter.retrieve_context(
    prompt="What happened in the deployment?",
    kinds=["action", "observation", "error"],
)
```

## Configuration

### Agentic Mode Settings

```yaml
agentic_mode:
  enabled: false                    # Enable agentic mode
router:
  enabled: true                     # Enable policy router; defaults on when agentic_mode is on
  profile: null                     # Force specific profile (debugging)
  log_level: info                   # Router log level
```

Compatibility note: current DML also accepts legacy flat override keys such as
`"dml.agentic_mode.enabled"` and the older documented `dml: agentic_mode:` YAML
shape. Prefer the top-level nested form above for portable harness profiles.

### Task Type Profiles

The router automatically selects settings based on detected task type:

| Task Type | Similarity Threshold | Top K | Token Budget | Kinds Allowed |
|-----------|---------------------|-------|--------------|---------------|
| **devops** | 0.4 | 6 | 400 | observation, action, plan |
| **coding** | 0.35 | 10 | 500 | action, observation, plan, artifact_ref |
| **research** | 0.5 | 10 | 600 | observation, plan, note |
| **chat** | 0.3 | 4 | 300 | note, plan |

### Phase Modifiers

Retrieval behavior changes based on execution phase:

| Phase | Similarity Threshold | Top K | Focus |
|-------|---------------------|-------|-------|
| **plan** | 0.25 (lower for exploration) | 12 | More context |
| **build** | 0.4 | 8 | Balanced |
| **execute** | 0.5 | 6 | Recent actions |
| **debug** | 0.6 (high precision) | 4 | Errors, observations |
| **reflect** | 0.35 | 8 | Plans, observations |

### Memory Promotion

```yaml
dml:
  agentic_promotion:
    commitment_threshold: 0.75      # Minimum fidelity for durable promotion
    allow_action_observation: true  # Allow action/observation in verified
    strict_mode: true               # Fail closed on invalid entries
```

## Memory Schema

Agentic memories have structured metadata:

```python
from daystrom_dml.agent_schema import MemoryKind, MemoryPhase, MemoryOutcome

# Memory kinds
MemoryKind.ACTION    # Actions taken
MemoryKind.OBSERVATION  # Observations made
MemoryKind.PLAN    # Plans and strategies
MemoryKind.ARTIFACT_REF  # Code/artifact references
MemoryKind.ERROR    # Errors encountered
MemoryKind.NOTE    # General notes

# Execution phases
MemoryPhase.PLAN
MemoryPhase.BUILD
MemoryPhase.EXECUTE
MemoryPhase.DEBUG
MemoryPhase.REFLECT

# Action outcomes
MemoryOutcome.SUCCESS
MemoryOutcome.FAIL
MemoryOutcome.PARTIAL
```

## Retrieval Report

`retrieve_context()` returns a detailed report:

```python
report = adapter.retrieve_context(
    prompt="What happened in deployment?",
    kinds=["action", "observation"],
)

print(report["raw_context"])  # Formatted context for model
print(report["context_tokens"])  # Token count
print(report["top_k"])  # Number of items retrieved
print(report["kinds"])  # Requested kinds
print(report["items"])  # List of retrieved items with metadata
```

## Evaluation Harness

Run multi-objective evaluation:

```bash
python -m daystrom_dml.eval.harness
```

Or create custom tasks:

```python
from daystrom_dml.eval.harness import AgenticEvaluator

evaluator = AgenticEvaluator()
task = evaluator.create_task(
    name="git_commit",
    description="Test git commit functionality",
    initial_context="Modified files: main.py",
    expected_success_signals=["commit", "git"],
)
result = evaluator.evaluate(task)
evaluator.print_scoreboard()
evaluator.save_results()
```

## Troubleshooting

### Router Not Applying Settings

1. Check `agentic_mode.enabled` is `true`.
2. If no `router` block is present, the router defaults to enabled when agentic mode is enabled.
3. To override it explicitly, set `router.enabled: true` and enable debug logging with `router.log_level: debug`.
4. Legacy `dml.agentic_mode.enabled` / `dml.router.enabled` flat override keys remain supported, but should not be used for new portable YAML profiles.

### Memory Rejected

1. Verify memory schema in strict mode
2. Check `commitment_threshold` in promotion pipeline
3. Ensure provenance fields are present

### Retrieval Not Phase-Aware

1. Check phase parameter in `retrieve_context()`
2. Verify kinds filtering is working
3. Check router decision logs

### Performance Issues

1. Reduce `top_k` in phase modifiers
2. Decrease `token_budget` for resource-constrained environments
3. Disable router for baseline performance

## Limitations

- Agentic mode requires structured metadata for full functionality
- Router decisions are deterministic but may not cover all edge cases
- Evaluation harness is a simulation - integrate with real agent for production
- Promotion pipeline uses in-memory stores - not persisted between sessions

## Next Steps

- Implement online autotuner (Step 7 of original plan)
- Add more task type profiles
- Enhance retrieval scoring with recency/fidelity weights
- Integrate with real agent workflows
- Add regression testing for agentic mode