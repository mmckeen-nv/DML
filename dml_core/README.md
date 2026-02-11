# DML Core

This directory contains the core Daystrom Memory Lattice packages, supporting
scripts, and tests.

## Contents
- `daystrom_dml/` – main lattice implementation, API server, adapters, and web assets
- `cma/` – Concept Memory Adapter helpers
- `scripts/` – helper automation (bench runners, model downloads)
- `tests/` – unit and integration tests

## Quick start
```bash
pip install .[server]
dml-server --host 0.0.0.0 --port 8000
```

## Docker
```bash
docker build -f dml_core/Dockerfile -t daystrom-dml-core .
docker run -p 8000:8000 daystrom-dml-core
```

CUDA build:
```bash
docker build -f dml_core/Dockerfile.cuda -t daystrom-dml-cuda .
```
