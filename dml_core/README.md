# DML Core

This directory contains the core Daystrom Memory Lattice packages, supporting
scripts, and tests.

## Contents
- `daystrom_dml/` – main lattice implementation, API server, adapters, and web assets
- `cma/` – Concept Memory Adapter helpers
- `scripts/` – helper automation (bench runners, model downloads)
- `tests/` – unit and integration tests

## Quick start

### NGC credentials

Supply NGC credentials through your environment or deployment secret manager.
The server does not write configured credentials to disk by default and keeps
`NIM_API_KEY` distinct from `OPENAI_API_KEY`. If a legacy integration requires a
credential file, explicitly set `DML_PERSIST_NGC_KEY=true` and set `NGC_KEY_FILE`
to an absolute path outside the DML checkout (preferably a mounted secret
volume). The resulting file is written atomically with owner-only (`0600`)
permissions; repository paths and symbolic links are rejected.

```bash
pip install .[server]
dml-server --port 8000
```

The server binds to `127.0.0.1` by default. Set `DML_API_TOKEN` to require a
bearer token for API routes. You can additionally set `DML_ADMIN_TOKEN` to a
distinct token required by NIM and visualizer management routes. Health,
metrics, documentation, and static UI routes remain public. For example:

```bash
export DML_API_TOKEN="$(openssl rand -hex 32)"
export DML_ADMIN_TOKEN="$(openssl rand -hex 32)"
curl -H "Authorization: Bearer $DML_API_TOKEN" http://127.0.0.1:8000/knowledge
```

Only expose DML on a shared network behind an authenticated reverse proxy or
after configuring these tokens.

### Upload limits

`/upload` enforces request-scoped limits before accepting additional work. The
defaults cap each file at 10 MiB, the complete request at 25 MiB, expanded
archive data at 50 MiB, archive members at 5 MiB each, archive membership at
1,000 entries and nesting at three levels. Document, chunk, and estimated-token
budgets are also enforced. Deployments can lower these limits with the
`DML_MAX_UPLOAD_FILE_SIZE`, `DML_MAX_UPLOAD_BYTES`,
`DML_MAX_DECOMPRESSED_BYTES`, `DML_MAX_ARCHIVE_MEMBER_SIZE`,
`DML_MAX_ARCHIVE_MEMBERS`, `DML_MAX_ARCHIVE_DEPTH`,
`DML_MAX_INGEST_DOCUMENTS`, `DML_MAX_INGEST_CHUNKS`, and
`DML_MAX_INGEST_TOKENS` environment variables.

## Docker
```bash
docker build -f dml_core/Dockerfile -t daystrom-dml-core .
docker run -p 127.0.0.1:8000:8000 daystrom-dml-core
```

CUDA build:
```bash
docker build -f dml_core/Dockerfile.cuda -t daystrom-dml-cuda .
```
