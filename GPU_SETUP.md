# GPU Configuration for DML

## Goal: All components on GPU (GB10)

### Components to Configure:
1. **FAISS** - Use `faiss-gpu` instead of `faiss-cpu`
2. **Transformers LLM (GPT2)** - Enable GPU via `device_map="auto"`
3. **SentenceTransformer (Embeddings)** - Set `device="cuda"`
4. **CUDA Backend** - Install compiled CUDA extension

### Steps:

1. Install GPU version of FAISS:
   ```bash
   pip uninstall faiss-cpu
   pip install faiss-gpu
   ```

2. Set environment variables:
   ```bash
   export DML_EMBEDDING_DEVICE=cuda
   export DML_GPU_ACCELERATION=1
   export TRANSFORMERS_OFFLINE=false
   ```

3. Rebuild CUDA extension:
   ```bash
   export DML_BUILD_CUDA=1
   pip install .[cuda]
   ```

4. Configure transformers to use GPU:
   - Pass `device_map="auto"` to TransformersBackend

5. Verify GPU usage with `nvidia-smi`

## Current GPU:
- NVIDIA GB10
- Ollama already using GPU
- Need to configure remaining components