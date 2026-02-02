# DML Chatbot + Telemetry + vLLM (gpt-oss:120B)

This deployment brings up three containers:

- **vLLM** serving a Hugging Face model with an OpenAI-compatible `/v1` API.
- **DML** configured for the default STM controller and pointed at vLLM.
- **Gradio UI** with chat and telemetry tabs.

## Quickstart

```bash
cd deployments/chatbot
cp .env.example .env
# Edit HF_MODEL_ID and HF_TOKEN if required

docker compose --env-file .env up -d --build
```

## URLs

- UI: <http://localhost:7860>
- DML API: <http://localhost:8000>
- vLLM OpenAI API: <http://localhost:8001/v1>

## Configuration Notes

- **HF_MODEL_ID** must be set to the Hugging Face repository for the model.
- **HF_TOKEN** is required if the model is gated.
- **SERVED_MODEL_NAME** controls the name exposed by vLLM and referenced by DML.
- **VLLM_TP_SIZE** should match the number of GPUs used for tensor parallelism.
- **VLLM_MAX_LEN** and **VLLM_DTYPE** tune context length and precision.
- **VLLM_GPU_UTIL** controls GPU memory utilization.

## DML Endpoints Used

The UI relies on these DML endpoints:

- `POST /query` with `{ "prompt": "...", "session_id": "..." }`
- `GET /stats`
- `GET /knowledge`
- `GET /health`
- `GET /metrics`

If your DML build does not expose these endpoints, add minimal handlers or enable the corresponding features in your existing server setup.
