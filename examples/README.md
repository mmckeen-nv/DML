# Examples

This directory contains end-to-end examples and demos built on top of DML.

## Contents
- `playground/` – Streamlit playground + benchmark suites
- `demos/` – FastAPI demo service and agent integrations
- `chatbot/` – Gradio chat UI + compose stack
- `bench/` – CLI benchmark script for DML vs RAG
- `visualizer/` – standalone Streamlit visualizer
- `nim/` – NVIDIA NIM setup notes

## Quick start (playground)

From the repository root:

```bash
./scripts/run_dml_playground.sh
```

Manual run:
```bash
PYTHONPATH=. streamlit run examples/playground/playground.py
```
