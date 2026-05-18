# DML Bug Report

**Date:** 2026-02-22
**Status:** Under Investigation
**Lab Machine:** 192.168.50.13 (nvidia/nvidia)
**Branch:** openclaw

---

## Issue #1: FastAPI Deprecation Warning

**Location:** `dml_core/daystrom_dml/server.py:274`

**Current Code:**
```python
@app.on_event("startup")
def _auto_launch_visualizer() -> None:
    """Ensure the visualizer is running when the service starts."""
    ...
```

**Error:**
```
DeprecationWarning: on_event is deprecated, use lifespan event handlers instead.
```

**Root Cause:**
FastAPI 0.100+ deprecated `@app.on_event("startup")` and `@app.on_event("shutdown")` in favor of lifespan context managers.

**Fix Required:**
Replace with lifespan context manager:
```python
@app.get("/visualizer/launch")
async def launch_visualizer():
    ...

@app.get("/visualizer/stop")
async def stop_visualizer():
    ...

async def lifespan(app: FastAPI):
    # Startup
    if VISUALIZER_URL:
        LOGGER.info("Visualizer configured for external deployment at %s", VISUALIZER_URL)
    else:
        try:
            _launch_visualizer_server()
            LOGGER.info("Visualizer startup complete on port %s", VISUALIZER_PORT)
        except HTTPException as exc:
            LOGGER.error("Visualizer failed to start during service startup: %s", exc.detail)

    yield

    # Shutdown
    if VISUALIZER_STATE.get("process") and not VISUALIZER_STATE["process"].poll():
        try:
            _stop_visualizer_server()
        except Exception:
            LOGGER.exception("Visualizer shutdown failed")

app = FastAPI(title="Daystrom Memory Lattice", lifespan=lifespan)
```

**Priority:** High (cosmetic, but should be fixed)

---

## Issue #2: Persistent RAG Store Initialization Failure

**Error:**
```
RuntimeError: faiss is required for the persistent RAG store
```

**Location:** `dml_core/daystrom_dml/dml_adapter.py:155-165`

**Current Code:**
```python
if rag_settings and getattr(rag_settings, "enable", False):
    try:
        self.persistent_rag_store = PersistentRAGStore(
            enable=True,
            index_path=index_path,
            meta_path=meta_path,
            dim=int(rag_settings.dim),
            backend=str(rag_settings.backend),
        )
    except Exception:
        LOGGER.exception("Failed to initialise persistent RAG store.")
        self.persistent_rag_store = None
```

**Root Cause:**
The `PersistentRAGStore.__init__()` checks for faiss module, but the check happens at import time, before faiss-cpu is installed.

**Fix Required:**
Add import guard in `rag_store.py`:
```python
try:
    import faiss  # type: ignore[import]
except Exception:
    faiss = None  # type: ignore[assignment]

class PersistentRAGStore:
    def __init__(self, ..., backend: str = "faiss"):
        if self.backend != "faiss":
            raise ValueError(f"Unsupported backend: {backend}")
        if self.enable and faiss is None:
            raise RuntimeError("faiss is required for the persistent RAG store")
```

**Priority:** High (prevents RAG store initialization)

---

## Issue #3: Embedding Model Loading Failure

**Error:**
```
Transformers backend unavailable: dummy is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
Using DummyGPT backend.
```

**Location:** `dml_core/daystrom_dml/embeddings.py`

**Current Code:**
```python
def create_embedder(model_name: str | None, *, device: str | None = None) -> Embedder:
    """Factory returning the best available embedder."""

    if model_name:
        normalised_device = (device or "").strip() or None
        return SentenceTransformerEmbedder(model_name, device=normalised_device)
    return RandomEmbedder()
```

**Root Cause:**
When `model_name="dummy"`, it's not a valid HuggingFace model identifier, causing `SentenceTransformer.__init__()` to fail and fall back to RandomEmbedder.

**Fix Required:**
Add special case for dummy model:
```python
def create_embedder(model_name: str | None, *, device: str | None = None) -> Embedder:
    """Factory returning the best available embedder."""

    if model_name and model_name == "dummy":
        return RandomEmbedder()

    if model_name:
        normalised_device = (device or "").strip() or None
        return SentenceTransformerEmbedder(model_name, device=normalised_device)
    return RandomEmbedder()
```

**Priority:** High (affects all embeddings)

---

## Issue #4: Retrieval Returns Zero Results

**Symptom:**
```
Retrieved 0 results
```

**Details:**
- Cosine similarities extremely low (-0.15 to 0.035) even for similar text
- Items stored successfully
- Vector index created (32KB)

**Root Cause:**
RandomEmbedder produces meaningless embeddings that don't capture semantic meaning.

**Fix Required:**
1. Ensure real embedding model is loaded (Issue #3 fix)
2. Add configurable similarity threshold
3. Investigate scoring logic in `memory_store._score_item()`

**Location:** `dml_core/daystrom_dml/memory_store.py:434-445`

```python
def _score_item(
    self,
    item: MemoryItem,
    query_embedding: np.ndarray,
    now: float,
    *,
    similarity: Optional[float] = None,
) -> float:
    if similarity is None:
        similarity = utils.cosine_similarity(item.embedding, query_embedding)
    age = utils.age_in_hours(item.timestamp, now)
    recency = 1.0 / (1.0 + age)
    return (
        similarity
        + self.eta * recency
        + self.gamma * item.salience
        + self.kappa * item.fidelity
    )
```

**Priority:** Critical (core functionality)

---

## Test Results Summary

### Tests Run
- ✅ 33 tests passed, 2 skipped
- ⚠️ 2 deprecation warnings
- ⚠️ pytest runtime: 4:49

### Functionality Tests
- ✅ Memory ingestion: Works (3 items verified)
- ✅ Vector index creation: Works (32KB)
- ✅ RAG store initialization: Works
- ✅ Storage persistence: Works
- ❌ Retrieval: Returns 0 results
- ❌ Cosine similarity: Very low values

### Benchmarks
- Using DummyGPT backend (no real LLM)
- No meaningful metrics yet

---

## Fixes Applied

### Fix #1: FastAPI Lifespan Events
**Status:** In Progress
**File:** `dml_core/daystrom_dml/server.py`

### Fix #2: Persistent RAG Store
**Status:** In Progress
**File:** `dml_core/daystrom_dml/rag_store.py`

### Fix #3: Embedding Model Loading
**Status:** In Progress
**File:** `dml_core/daystrom_dml/embeddings.py`

### Fix #4: Retrieval Scoring
**Status:** Pending
**File:** `dml_core/daystrom_dml/memory_store.py`

---

## Next Steps

1. Copy codebase to lab machine (192.168.50.13)
2. Install real dependencies (sentence-transformers, transformers, etc.)
3. Apply all fixes
4. Run tests to verify
5. Run real benchmarks with proper models
6. Document results

---

## Lab Machine Setup

**Hostname:** 192.168.50.13
**Username:** nvidia
**Password:** nvidia
**Purpose:** DML development and testing with real models