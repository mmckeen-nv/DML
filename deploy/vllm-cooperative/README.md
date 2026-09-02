# Reproducible Daystrom cooperative vLLM deployment

This workflow deploys cooperative vLLM with deterministic operator actions:

- `init` creates or rewrites `.env` atomically, keeps key bytes unchanged unless `--rotate-key`, and supports commit refresh without rotating keys.
- `doctor` runs strict filesystem-only checks on any OS: env syntax, absolute paths, repo/worktree cleanliness, exact HEAD pin, key safety, and cache readiness.
- `pull` (or `preflight --pull`) fetches a missing image in one explicit command.
- `preflight` adds Linux/runtime checks (Docker, daemon, NVIDIA runtime, image presence, import compatibility) and validates Compose interpolation. Either the `docker compose` plugin or a current `docker-compose` executable is accepted.
- `deploy` records ordered rollback backups, waits for Docker health (not just running), verifies health/models/one-token completion, and prunes obsolete backups only after successful replacement.
- `verify`, `status`, `rollback`, and `logs` provide bounded, idempotent operations.

The connector and policy implementation under `dml_core/daystrom_dml/context/vllm_bridge` are not modified by this deployment wrapper.

## Quick path

```bash
cd deploy/vllm-cooperative
./deploy.sh init
./deploy.sh doctor
./deploy.sh pull
./deploy.sh preflight
./deploy.sh deploy
./deploy.sh verify
./deploy.sh status
```

`init` behaviors:

- `--force-env` rewrites `.env` while preserving any existing key file bytes.
- `--refresh-commit` updates only `DML_SOURCE_COMMIT` in-place (also preserves key bytes).
- `--rotate-key` is the only path that replaces a key; it prints a warning that live authorizations are invalidated.

## Recovery path

```bash
./deploy.sh status
./deploy.sh logs --tail 200
./deploy.sh rollback
# if a stale lock is provably orphaned:
./deploy.sh rollback --recover-stale-lock
./deploy.sh verify
```

Rollback is transactional: it verifies recorded backup metadata and image availability, stages the current container as recovery, starts/health-checks backup, and only then removes recovery + updates state.

## Security and boundaries

- Control key is file-based Compose secret only (`DAYSTROM_KV_SECRET_FILE`, mounted read-only).
- No key value is printed by `init` and no key is accepted through env var or argv.
- `.env` is parsed as strict `KEY=VALUE`; shell interpolation/command substitution is rejected.
- `DML_SOURCE_COMMIT` must be full 40-hex and match `HEAD` exactly; `dml_core` must be clean.
- Default bind is loopback (`127.0.0.1`). Public bind requires explicit `DAYSTROM_ALLOW_PUBLIC_BIND=yes` acknowledgement.
- Hardware KV lifecycle proof (save/reset/restore/purge canary) remains an operator-run boundary outside deploy/verify checks.

## Tunables in `.env`

- Daystrom: `DAYSTROM_CPU_KV_BYTES`, `DAYSTROM_MAX_TTL_SECONDS`, `DAYSTROM_MAX_RECORDS`
- Parallelism: `VLLM_TENSOR_PARALLEL_SIZE`, `VLLM_PIPELINE_PARALLEL_SIZE`, `VLLM_DATA_PARALLEL_SIZE`
- Precision: `VLLM_DTYPE`, `VLLM_KV_CACHE_DTYPE`, `VLLM_MAMBA_SSM_CACHE_DTYPE`
- Runtime: `VLLM_MAX_MODEL_LEN`, `VLLM_STARTUP_TIMEOUT_SECONDS`, `VLLM_VERIFY_TIMEOUT_SECONDS`
- Networking: `VLLM_BIND_ADDRESS`, `VLLM_PORT`

## Action reference

- `./deploy.sh init [--force-env] [--refresh-commit] [--rotate-key] [--env-file PATH] [--key-file PATH]`
- `./deploy.sh doctor`
- `./deploy.sh pull`
- `./deploy.sh preflight [--pull]`
- `./deploy.sh deploy`
- `./deploy.sh verify`
- `./deploy.sh status`
- `./deploy.sh rollback [--recover-stale-lock]`
- `./deploy.sh logs [--tail N]`
