# 🧠 Daystrom Memory Lattice (DML)
*A hierarchical, self-compressing memory architecture for intelligent data retrieval.*

---

## Overview
The **Daystrom Memory Lattice (DML)** is a GPU-accelerated memory substrate that compresses, abstracts, and retrieves large knowledge bases efficiently — including text, PDFs, code, and structured data.  
It allows LLMs and retrieval systems to *think less and know more* by serving pre-compressed, high-fidelity context windows instead of raw documents.

Unlike traditional Retrieval-Augmented Generation (RAG) pipelines that rely on brute-force vector search, DML builds an evolving *lattice of memory* — a hierarchy that automatically merges, summarizes, and decays information.

---

## 🚀 Key Features
- **Hierarchical memory lattice:** Multi-level structure (L0–Lk) for fine-grained to abstract knowledge.  
- **Dual retrieval modes:**  
  - *Semantic retrieval* → high-level reasoning and summarization.  
  - *Literal retrieval* → exact, surgical extractions (API calls, code lines, tabular entries).  
- **Adaptive routing:** Automatic query classifier chooses between semantic, literal, or hybrid retrieval.  
- **Mathematical decay and salience weighting:** Keeps relevant data fresh and merges redundancy.  
- **GPU acceleration:** Embedding, summarization, and vector math run on CUDA (Torch nightly cu130).  
- **Persistent research layer:** Queries external databases (not LLM memory) for exact and inferred answers.  
- **MCP + Docker ready:** One-click deployment for local or enterprise environments.  

---

## 🔬 Mathematical Foundation

### Memory Node
Each memory node is represented as:

```
M_i = (e_i, s_i, f_i, t_i)
```

where:
- **eᵢ** – embedding vector  
- **sᵢ** – salience  
- **fᵢ** – fidelity (quality / confidence)  
- **tᵢ** – timestamp  

---

### Retrieval Scoring
```
score_i = cos(e_i, q) + η * r_i + γ * s_i + κ * f_i
```
where:
- **rᵢ = 1 / (1 + ageᵢ)**  
- **η, γ, κ** control recency, salience, and fidelity weighting.  

This ensures fresher, higher-fidelity memories are prioritized even when embedding similarity is ambiguous.

---

### Decay and Fidelity
```
λ* = σ(β_r * r_i − β_a * age_i)
```
Older, less-reinforced data gradually lose fidelity and are abstracted upward into summarized forms.

---

### Merging and Abstraction
When two embeddings are similar:

```
if cos(e_a, e_b) >= θ_merge:
    e_m = (e_a + e_b) / 2
```

Their texts are summarized by the LLM summarizer, creating a higher-order abstraction node with improved fidelity.

---

### Token Budgeting
To fit an LLM’s context window:

```
while Σ(tokens(S_i)) < B,  S_i ∈ top_k
```

A greedy knapsack packs the highest information-density summaries within budget **B**.

---

## 🧩 Architecture

```
        ┌──────────────┐
        │ Data Sources │ ← PDFs, code, logs
        └──────┬───────┘
               │
      ┌────────▼────────┐
      │ Embedding Model │  → GPU (Sentence-Transformer)
      └────────┬────────┘
               │
     ┌─────────▼──────────┐
     │ Memory Lattice     │
     │  • Decay / Merge   │
     │  • Summarization   │
     └────────┬───────────┘
               │
      ┌────────▼──────────┐
      │ Retrieval Router  │  → literal / semantic / hybrid
      └────────┬──────────┘
               │
       ┌───────▼───────┐
       │  Query Engine │  → LLM / MCP
       └───────────────┘
```

---

## ⚙️ Usage

### Ingestion
```bash
python dml_ingest_dir.py /data/docs
```
Automatically parses PDFs, text, and code into the lattice.

### Query
```bash
curl "http://localhost:8000/query?prompt=show+API+call+fetchUserProfile"
```
Returns the minimal function snippet + surrounding context.

### Persistence & Checkpoints
- All ingests automatically persist to `data/` (configurable via `storage_dir`).
- Create an immediate snapshot via the CLI:
  ```bash
  cma checkpoint
  ```
- Continuous checkpoints can be enabled with `checkpoint_interval_seconds` in `daystrom_dml/config.yaml` or by setting the `DML_CHECKPOINT_INTERVAL_SECONDS` environment variable.

### Metrics & Observability
- Prometheus metrics are exposed at `GET /metrics` and include ingest counts, retrieval latency histograms, and active memory gauges.
- Metrics can be disabled with `DML_METRICS_ENABLED=false` when required.
- The `/visualizer/state` endpoint mirrors the latest prompt queued for the Streamlit live visualizer, enabling dashboards to stay in sync.

### Configuration & Secrets
- Runtime settings are driven by `daystrom_dml/config.yaml` and overridable via environment variables (e.g. `DML_MODEL_NAME`, `DML_STORAGE_DIR`).
- Optional `.env` files are loaded automatically for local development.
- GPU and NIM parameters can be tuned via `DML_GPU_ACCELERATION`, `DML_NIM_HEALTH_TIMEOUT`, and `DML_NIM_DEFAULT_ID`.

### Benchmarks & Load Tests
- A repeatable micro-benchmark is available:
  ```bash
  python scripts/benchmark.py --iterations 25
  ```
- The script reports average and p95 retrieval latencies to help tune deployments.

### Example
```
User:  "What were the average temperatures last year?"
→ semantic summarization (aggregate view)

User:  "Temperature on Jan 3 at 2 PM?"
→ literal retrieval (exact log entry)
```

---

## 🧠 Comparison — DML vs RAG

| Feature | Traditional RAG | Daystrom Memory Lattice |
|----------|-----------------|-------------------------|
| Retrieval granularity | Flat top-K chunks | Hierarchical (verbatim → summary → abstraction) |
| Context optimization | Fixed, redundant | Dynamic, token-efficient |
| Compression | Minimal | Continuous semantic + vector compression |
| Decay / Reinforcement | None | Mathematical fidelity decay + reinforcement |
| Exact lookup | Hard to control | Literal retriever for surgical precision |
| Compute cost | Linear scaling | Bounded, GPU-accelerated lattice |
| Output quality | Redundant, shallow | Dense, contextual, and traceable |

> **In essence:**  
> RAG *searches*, DML *remembers.*

---

## 🧰 Deployment

### Docker (single container)
Build the runtime image and start the API server directly with Docker:

```bash
docker build -t daystrom-dml .
docker run --gpus all \
  -p 9000:9000 \
  -e DML_PORT=9000 \
  -v "$(pwd)/data:/app/data" \
  daystrom-dml
```

- The bundled ``dml-server`` entrypoint automatically honours the
  ``DML_HOST`` and ``DML_PORT`` environment variables.
- Mounting ``./data`` keeps the lattice persistent between container restarts.
- Provide a custom configuration file by mounting it and setting
  ``DML_CONFIG_PATH`` (or ``DML_CONFIG``) to its location inside the container.

### Docker Compose
For repeatable deployments, a ``compose.yaml`` file is included:

```bash
docker compose up --build -d
```

The service exposes the API on ``${DML_PORT:-9000}`` and reserves a GPU when
available. Stop the stack with ``docker compose down``.

### Local execution
```bash
pip install .[server]
dml-server --host 0.0.0.0 --port 9000
```

The command accepts ``--reload`` for development and mirrors the Docker
environment variables described above.

### MCP server
```bash
pip install .[mcp]
dml-mcp-server --storage-path ./cma_store.json
```

Both ``--name`` and ``--storage-path`` can be supplied via CLI options or by
setting ``CMA_MCP_NAME`` / ``CMA_STORAGE_PATH`` environment variables.

### MCP Integration
```yaml
name: daystrom-dml
type: retrieval
entrypoint: http://localhost:9000/query
args: [prompt, mode]
```

Use ``dml-mcp-server`` with the configuration above to expose the lattice to
MCP-compatible clients.

## 🔌 Workflow Integration APIs

Interact with a running lattice using the ``DMLClient`` helper:

```python
from daystrom_dml import DMLClient

with DMLClient("http://localhost:9000") as client:
    client.ingest("Investigate the Daystrom memory lattice release notes.")
    result = client.query("Summarise the latest release.")
    print(result["response"])
```

The client wraps the public REST API (``/ingest``, ``/query``, ``/reinforce``,
``/stats`` and ``/knowledge``) with sensible defaults while remaining easy to
extend with custom ``requests.Session`` instances.

---

## ⚡ Performance (GPU Mode)

| Operation | Speedup vs CPU |
|------------|----------------|
| Embedding | 5–8× |
| Summarization | 3–5× |
| Vector math | 10–20× |
| Overall throughput | ~7× faster ingestion & query |

---

## 🌍 Position in the Data Stack
DML sits **between your database and your LLM.**

```
[Raw Data] → DML → [Context Window] → [LLM]
```

- Handles knowledge persistence, compression, and fidelity.  
- Supplies only relevant, compact context.  
- Reduces GPU cost while improving recall accuracy.

---

## 🧮 Research Layer Capabilities
DML lets you *research your own data*:
- Finds exact entities, timestamps, or API calls.  
- Expands retrieval iteratively to include relational context.  
- Returns dense, citation-ready summaries instead of massive token dumps.

This turns querying large datasets from an I/O problem into an **intelligence problem.**

---

## 📚 Summary
> **DML = Hierarchical Memory + Semantic Compression + GPU Efficiency**

DML is not a faster RAG; it’s a **self-organizing cognitive substrate** for persistent, scalable knowledge — designed for enterprises, research agents, and long-context LLMs.

---
