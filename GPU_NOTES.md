# GPU Configuration Status

## GPU Hardware
- NVIDIA GB10 GPU
- CUDA Driver: 13.0
- Ollama: ✅ Using GPU

## Components Status

### ✅ GPU (Confirmed Working)
1. **Ollama LLM** - Already running on GPU (31158 MiB used)
2. **TransformersBackend** - Can use GPU via `device_map="auto"` if pytorch-cuda installed

### ⚠️ GPU (Requires Setup)
3. **SentenceTransformer** - Need `device="cuda"` config
4. **FAISS** - GPU version requires CUDA 12.x runtime + conda
   - `faiss-cpu` (CPU) is currently installed
   - `faiss-gpu` needs: CUDA 12.x runtime, pytorch, nvidia channels via conda
   - Without conda, cannot install proper FAISS-GPU

### ❌ Not Available
- **CUDA Backend Extension** - Compiled `.so` exists but not installed
  - Needs: `nvcc`, `pybind11`, CUDA toolkit
  - Current: No nvcc found

## Recommended Approach

### Option 1: Use CPU FAISS (Quick)
- Keep `faiss-cpu` (already installed)
- Focus GPU on: SentenceTransformer + Transformers LLM
- Tradeoff: Vector search runs on CPU (but embeddings on GPU)

### Option 2: Set up Conda Environment (Proper)
- Install miniconda/micromamba
- Create GPU environment with:
  - python=3.12
  - pytorch pytorch-cuda=12.4
  - faiss-gpu
  - sentence-transformers (GPU support)
- Use this environment for DML

### Option 3: Docker (Cleanest)
- Use provided Dockerfile.cuda
- Everything runs on GPU including FAISS
- Isolated environment

## Priority Order for GPU
1. ✅ Ollama LLM - Already on GPU
2. ⚠️ SentenceTransformer - Easy fix (add `device="cuda"`)
3. ⚠️ Transformers LLM - Easy fix (add `device_map="auto"`)
4. ❌ FAISS - Requires conda/Docker (can run on CPU)

## Testing GPU
```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))
```

## Current Environment
- Python: 3.12
- torch: Available (from sentence-transformers)
- FAISS: CPU version installed
- No conda/mamba found
- No nvcc found

## Recommendation
Use Option 1 for now (CPU FAISS) while ensuring:
- SentenceTransformer uses GPU (`DML_EMBEDDING_DEVICE=cuda`)
- Transformers uses GPU (`device_map="auto"`)
- Document that FAISS is CPU-bound due to conda requirement