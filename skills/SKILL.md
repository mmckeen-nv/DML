# Daystrom DML Skill

Provides access to the Daystrom Memory Lattice (DML) for OpenClaw agents.

## Overview

The Daystrom Memory Lattice (DML) is a hierarchical memory system that compresses, abstracts, and retrieves large knowledge bases. It's designed for long-horizon assistants that need to remember, reason, and explain.

## Features

- **Semantic Memory Ingestion** - Store structured memories with metadata
- **Hierarchical Retrieval** - Retrieve verbatim fragments or abstracted summaries
- **GPU Acceleration** - Embeddings run on GPU for fast processing
- **Token Efficient** - 30-50% token savings vs raw text
- **Easy Integration** - OpenClaw-ready API with context managers

## Installation

```bash
# Install DML
pip install daystrom-dml[server,embeddings,faiss]

# Configure GPU (optional)
export DML_EMBEDDING_DEVICE=cuda
export DML_GPU_ACCELERATION=1
```

## Usage

### OpenClaw Agent Integration

Use the `DMLAgent` class for seamless integration:

```python
from skills.daystrom_dml import DMLAgent

# Initialize with context manager
with DMLAgent() as dml:
    # Ingest memory
    dml.ingest(
        text="Deployed the application successfully",
        kind="action",
        meta={"phase": "execute", "tool": "docker"}
    )

    # Retrieve context
    context = dml.get_context("deployment results")
    print(context)

# Automatic cleanup
```

### Quick Functions

For simple operations, use the quick functions:

```python
from skills.daystrom_dml import dml_ingest, dml_retrieve

# Quick ingest
dml_ingest("Project completed", kind="result")

# Quick retrieve
result = dml_retrieve("project status")
print(result.get('raw_context', ''))
```

### Basic Usage (DML Adapter)

For advanced usage, use the DML adapter directly:

```python
from daystrom_dml.dml_adapter import DMLAdapter

# Create adapter
adapter = DMLAdapter(
    model_name="gpt2",
    embedding_model="all-MiniLM-L6-v2",
    storage_dir="./data/dml",
    dml.agentic_mode.enabled=True
)

# Ingest
adapter.ingest_agentic(
    text="Deployed successfully",
    kind=MemoryKind.ACTION,
    meta={"phase": "execute"}
)

# Retrieve
report = adapter.retrieve_context(prompt="deployment")
print(report["raw_context"])

# Clean up
adapter.close()
```

## API Reference

### DMLAgent

**Methods:**

- `__init__(model_name, embedding_model, storage_dir, dml_agentic_mode)` - Initialize agent
- `ingest(text, kind, meta)` - Ingest new information
- `retrieve(query, top_k, use_summary)` - Retrieve context
- `get_context(query, max_tokens)` - Get formatted context string
- `memory_count()` - Get total memory count
- `shutdown()` - Clean up resources

**Context Manager:**
```python
with DMLAgent() as dml:
    # Use dml here
    pass  # Automatic cleanup
```

### dml_ingest

Quick ingest function:
```python
dml_ingest(text, kind="action", meta=None)
```

### dml_retrieve

Quick retrieve function:
```python
dml_retrieve(query, top_k=4)
```

## Memory Kinds

- `action` - Actions taken
- `observation` - Observations
- `insight` - Insights and learnings
- `planning` - Plans and decisions
- `execution` - Execution steps
- `result` - Results and outcomes

## Configuration

### Environment Variables
```bash
export DML_EMBEDDING_DEVICE=cuda  # GPU acceleration
export DML_GPU_ACCELERATION=1     # Enable GPU optimizations
```

### Default Settings
- **Model:** `gpt2` (LLM), `all-MiniLM-L6-v2` (embeddings)
- **Storage:** `./data/dml`
- **Agentic Mode:** Enabled
- **Top K:** 4 results per query
- **Max Tokens:** 1000

## Performance

- **Embedding latency:** <50ms (GPU)
- **Vector search:** <10ms (1k vectors, CPU)
- **LLM generation:** ~500ms (CPU) - faster with GPU
- **Token savings:** 30-50%

## Troubleshooting

### GPU Issues
```python
import torch
print("CUDA available:", torch.cuda.is_available())  # Should be True
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('all-MiniLM-L6-v2')
print("Device:", m.device)  # Should show cuda:0
```

### Test Issues
```bash
cd /path/to/DML
python -m pytest dml_core/daystrom_dml/tests/test_agentic.py -v
```

### Memory Issues
```python
# Clean up old memories
import os
import shutil
shutil.rmtree("./data/dml")  # Remove all memories
```

## Examples

See `example.py` for comprehensive examples:
- Basic usage
- Agent workflows
- Quick functions
- Memory management

## Project Status

- **Repository:** `staggeredsix/DML`
- **Branch:** `openclaw`
- **Tests:** 57/57 passing
- **GPU:** PyTorch CUDA 13.0 active
- **Status:** Ready for production use

## Architecture

- **Embeddings:** GPU-accelerated (all-MiniLM-L6-v2)
- **LLM:** Local transformers (GPT2 or Ollama)
- **Vector Search:** FAISS (CPU for now, GPU with conda)
- **Memory:** Hierarchical lattice (L0-Lk levels)

## Performance

- **Embedding latency:** <50ms (GPU)
- **Vector search:** <10ms (1k vectors, CPU)
- **LLM generation:** ~500ms (CPU) - faster with GPU
- **Token savings:** 30-50% vs raw text

## File Locations

- DML adapter: `dml_core/daystrom_dml/dml_adapter.py`
- Embeddings: `dml_core/daystrom_dml/embeddings.py`
- LLM runner: `dml_core/daystrom_dml/gpt_runner.py`
- Tests: `dml_core/daystrom_dml/tests/`

## Troubleshooting

### GPU Issues
- Check `nvidia-smi` for GPU availability
- Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Ensure SentenceTransformer loads to GPU: `python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('all-MiniLM-L6-v2'); print(m.device)"`

### Test Issues
- Run tests: `python -m pytest dml_core/daystrom_dml/tests/test_agentic.py -v`
- Check logs: `tail -f logs/dml.log`

### FAISS GPU
- Currently using CPU version
- GPU version requires conda: `conda install faiss-gpu`
- See `GPU_SETUP.md` for details

## References

- Project: `staggeredsix/DML`
- Branch: `openclaw`
- Status: 57/57 tests passing, GPU-accelerated embeddings