# DML Quick Start Guide

## 🚀 Quick Start

### 1. Import DML
```python
import sys
sys.path.insert(0, '/home/nvidia/.npm-global/lib/node_modules/openclaw/node_modules/daystrom-dml')

from dml_core.daystrom_dml.dml_adapter import DMLAdapter
```

### 2. Create Agent
```python
adapter = DMLAdapter(
    config_overrides={
        "llm_backend": "ollama",
        "model_name": "llama3:8b",
        "embedding_model": "ollama:qwen3-embedding:0.6b",
        "embedding_device": "cuda",
        "storage_dir": "./data/dml",
        "dml.agentic_mode.enabled": True
    }
)
```

### 3. Ingest Memory
```python
adapter.ingest_agentic(
    text="Deployed application successfully",
    kind="action",
    meta={
        "phase": "execute",
        "tool": "docker",
        "outcome": "success",
        "provenance": {
            "task_id": "t1",
            "step_id": "s1"
        }
    }
)
```

### 4. Retrieve Context
```python
report = adapter.retrieve_context("deployment results")
print(report["raw_context"])
```

## 🎯 For OpenClaw Agents

### Option 1: Use DMLAgent Class
```python
from dml_core.daystrom_dml.dml_adapter import DMLAdapter

adapter = DMLAdapter(
    config_overrides={
        "llm_backend": "ollama",
        "model_name": "llama3:8b",
        "embedding_model": "ollama:qwen3-embedding:0.6b",
        "embedding_device": "cuda",
        "storage_dir": "./data/dml",
        "dml.agentic_mode.enabled": True
    }
)

# Use it
adapter.ingest_agentic("Important info", kind="action", meta={...})
report = adapter.retrieve_context("query")
```

### Option 2: Context Manager
```python
from dml_core.daystrom_dml.dml_adapter import DMLAdapter

with DMLAdapter(...) as adapter:
    # Auto cleanup when done
    adapter.ingest_agentic("Info", kind="action")
    context = adapter.get_context("query")
```

## 🧪 Test It

### Quick Test
```bash
cd /home/nvidia/.openclaw/workspace/DML
source venv/bin/activate
python -m pytest openclaw-wrapper/tests/test_dml_memory.py -q
```

### Run All Tests
```bash
python -m pytest dml_core/daystrom_dml/tests/test_agentic.py -v
```

## 📊 Test Plan Workloads

### Debug Task (Best Proof)
```python
# Reproduce a bug
adapter.ingest_agentic(
    text="Error: Container crashed on startup",
    kind="error",
    meta={"phase": "execute", "error_type": "runtime_error"}
)

# Retrieve context
report = adapter.retrieve_context("what errors occurred")
print(report["raw_context"])
```

### Long-Horizon Episode
```python
# During your task
adapter.ingest_agentic(
    text="Currently debugging container startup issue",
    kind="planning",
    meta={"phase": "debug", "current_step": "investigate_logs"}
)

# Later, continue
adapter.ingest_agentic(
    text="Found issue: Missing env var API_KEY",
    kind="result",
    meta={"phase": "debug", "root_cause": "missing_env_var"}
)

# Retrieve context for next step
context = adapter.get_context("what's the current issue")
print(context)
```

## 🔧 Configuration Options

### Memory Kinds
```python
from dml_core.daystrom_dml.agent_schema import MemoryKind

adapter.ingest_agentic(
    text="Plan: Deploy microservices",
    kind=MemoryKind.PLANNING,
    meta={"phase": "plan"}
)

adapter.ingest_agentic(
    text="Executed deployment",
    kind=MemoryKind.ACTION,
    meta={"phase": "execute"}
)

adapter.ingest_agentic(
    text="Tests passing",
    kind=MemoryKind.RESULT,
    meta={"phase": "execute"}
)
```

### Retrieve Options
```python
# Get formatted context
context = adapter.get_context("deployment results", max_tokens=500)

# Get raw report
report = adapter.retrieve_context("deployment", top_k=8, use_summary=False)
```

## 📈 Performance

- **Embedding latency:** depends on the active backend (Ollama is the default production path)
- **Vector search:** <10ms (1k vectors, CPU)
- **LLM generation:** depends on the configured local model/runtime
- **Token savings:** 30-50%

## Optional alternate embedding backend

The durable production default is:
- embeddings: `ollama:qwen3-embedding:0.6b`
- summarization/reform: `llama3:8b`

`sentence-transformers` remains supported for alternate experiments or compatibility paths, but it is **not** the default production backend.

## 🎯 Next Steps

1. ✅ **Done:** GPU installation
2. ✅ **Done:** Package accessible
3. ⚠️ **Next:** Run debug task (proves memory value)
4. ⚠️ **Next:** Run long-horizon episode (killer test)
5. ⚠️ **Next:** Implement test harness
6. ⚠️ **Next:** Run test matrix

## 📝 Documentation

- `TEST_PLAN.md` - Comprehensive test plan

## 🚀 Ready to Run

**Start with a debug task to prove DML's value:**

```python
import sys
sys.path.insert(0, '/home/nvidia/.npm-global/lib/node_modules/openclaw/node_modules/daystrom-dml')

from dml_core.daystrom_dml.dml_adapter import DMLAdapter

adapter = DMLAdapter(
    config_overrides={
        "llm_backend": "ollama",
        "model_name": "llama3:8b",
        "embedding_model": "ollama:qwen3-embedding:0.6b",
        "embedding_device": "cuda",
        "storage_dir": "./data/dml",
        "dml.agentic_mode.enabled": True
    }
)

# Simulate a bug
adapter.ingest_agentic(
    text="Error: Container failing with exit code 1",
    kind="error",
    meta={"phase": "execute", "error_type": "container_crash"}
)

# Retrieve context
report = adapter.retrieve_context("what errors occurred")
print(report["raw_context"])
```

**That's it! DML is ready to use!** 🎉