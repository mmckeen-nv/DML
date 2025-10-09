# Concept Memory Adapter (CMA)

CMA is a lightweight, lossy associative memory that can be attached to any large
language model (LLM). It maintains detailed memories for recent events while
progressively compressing older memories into higher-level concepts. The package
implements a configurable sliding-fidelity schedule, multi-level text and vector
compression, salience-aware retrieval, and reinforcement updates based on model
outputs.

## Features

* Multi-level compression pipeline (verbatim → summary → bullets → concept code)
* Sliding fidelity controlled by age and reinforcement weights
* Vector quantisation via KMeans with automatic fitting
* Recency and salience aware retrieval with deterministic or sampled selection
* Merge and eviction policies to control memory growth
* Command line interface and optional FastAPI server

## Installation

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[server,tokenizer,embeddings,faiss]
```

## Quickstart

```bash
cma init
cma ingest --text "We adopted a cat named Pixel. She loves jumping over the couch."
cma ingest --text "PCIe Gen5 lanes bottleneck observed at high queue depth; mitigation …"
cma query --prompt "What did we learn about the cat's behavior?"
```

The query command prints a memory preamble followed by your prompt. Feed this to
your LLM to provide contextual grounding.

## Testing

```bash
pytest
```

All tests are deterministic and require no network access. A random embedding
fallback is used automatically if sentence-transformers is not installed.
