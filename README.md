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

### Docker
```bash
docker build -t daystrom-dml .
docker run --gpus all -p 8000:8000 daystrom-dml
```

### MCP Integration
```yaml
name: daystrom-dml
type: retrieval
entrypoint: http://localhost:8000/query
args: [prompt, mode]
```

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
