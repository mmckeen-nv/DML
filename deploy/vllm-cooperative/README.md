# Reproducible Daystrom cooperative vLLM deployment

This deployment pins the DML source commit, mounts `dml_core` read-only, delivers the HMAC control key through a Compose secret, enables vLLM's hybrid KV manager and prefix caching, and keeps a stopped rollback container during migration from an unmanaged runtime.

It is intentionally specific to a Linux NVIDIA Docker host running vLLM 0.20.x. The connector and policy remain dependency-gated so ordinary DML installs on macOS and Windows do not import vLLM.

## Security and lifecycle boundaries

- Keep the control key outside the repository with owner-only permissions (`0600` or stricter).
- Do not place the key in `.env`, command arguments, container environment variables, logs, or artifacts.
- Set `DML_SOURCE_COMMIT` to the exact reviewed commit. Deployment fails if `dml_core` is dirty or the checkout is at another commit.
- Deployment refuses to replace a runtime with running or waiting requests.
- The prior unmanaged container is stopped and renamed, not deleted. Its name is saved under `DAYSTROM_DEPLOY_STATE_DIR` for explicit rollback.
- Checkpoints and CPU-offloaded KV remain process-local. A restart intentionally invalidates them.
- Selective purge is request-bound; generic KV erase/delete capabilities remain unsupported.

## Setup

```bash
cd deploy/vllm-cooperative
cp .env.example .env
chmod 600 .env
```

Edit `.env` with absolute target-host paths. Generate the control key without printing it:

```bash
umask 077
python3 -c 'import secrets; open("/secure/path/daystrom-kv-control.key", "w").write(secrets.token_hex(32) + "\n")'
```

Validate without touching the GPU service:

```bash
./deploy.sh preflight
```

The preflight verifies the source commit and cleanliness, key permissions, Compose rendering, dependency import against the pinned image, and zero active/waiting requests.

## Deploy and verify

```bash
./deploy.sh deploy
./deploy.sh status
```

The service uses `restart: unless-stopped` and a bounded health check. On startup failure, the script removes the failed Compose service and restores the retained container.

After deployment, run the digest-only save → local GPU reset → restore → selective purge → restore-denied probe. A healthy endpoint alone is not physical-zeroization evidence.

## Roll back

Rollback also refuses to interrupt active or waiting requests:

```bash
./deploy.sh rollback
```

After a fully verified soak, operators may manually remove old stopped rollback containers. The deployment script never deletes them automatically.
