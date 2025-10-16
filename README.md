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
pip install -e .[mcp]
```

## One-click deployment & testing

The repository ships with a helper script that provisions an isolated virtual
environment, installs all optional dependencies, and executes the full pytest
suite. Run it from the project root:

```bash
./scripts/one_click_deploy.sh
```

Use the ``--skip-tests`` flag if you only need the environment set up:

```bash
./scripts/one_click_deploy.sh --skip-tests
```

After the script finishes you can activate the environment with
``source .venv/bin/activate``.

## Web playground & API service

Install the ``server`` extra and launch the FastAPI application:

```bash
pip install -e .[server]
uvicorn daystrom_dml.server:app --host 0.0.0.0 --port 9000
```

Then open ``http://localhost:9000`` to access the bundled Daystrom Memory
Lattice playground. The interface allows you to upload PDF/Text documents for
ingestion, execute retrieval augmented generation (RAG) queries, compare base
model and DML-augmented responses, inspect token usage, and review retrieval
fidelity per memory entry.

The backend also exposes JSON endpoints:

* ``POST /upload`` – ingest PDF or plain text files.
* ``POST /rag/retrieve`` – return the retrieval report (context entries,
  average fidelity, token usage).
* ``POST /rag/compare`` – run both the base model and RAG pipeline, including
  token usage metadata when available from the underlying LLM runner.

## NVIDIA NIM integration

The DML server natively supports OpenAI-compatible backends such as NVIDIA NIM.
Set ``NIM_API_BASE`` (or ``OPENAI_API_BASE``) and ``NIM_API_KEY`` before
starting the server to route all generations and summaries to the remote model.

Step-by-step instructions for pairing the service with
``nvcr.io/nim/openai/gpt-oss-20b:latest`` are provided in ``nim/README.md`` and
include the official NGC container run command as well as Docker packaging for
the DML server itself.

## Quickstart

```bash
cma init
cma ingest --text "We adopted a cat named Pixel. She loves jumping over the couch."
cma ingest --text "PCIe Gen5 lanes bottleneck observed at high queue depth; mitigation …"
cma query --prompt "What did we learn about the cat's behavior?"
```

The query command prints a memory preamble followed by your prompt. Feed this to
your LLM to provide contextual grounding.

## MCP Server

Install the optional ``mcp`` extra to expose the Concept Memory Adapter via the
[Model Context Protocol](https://github.com/modelcontextprotocol). Run the
server with ``python -m cma.mcp_server`` or embed it in your application using
``from cma import create_mcp_server``.

## Testing

```bash
pytest
```

All tests are deterministic and require no network access. A random embedding
fallback is used automatically if sentence-transformers is not installed.
