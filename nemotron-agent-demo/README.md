# Nemotron Station Agent Demo

Agentic demo that drives a single LLM through supervisor/planner/coder/reviewer/ops/aggregator roles and visualizes progress live.

Default model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` via NVIDIA NIM.

**Quickstart (Docker Compose, recommended)**
1. `./ngc_login.sh`
2. `docker compose --env-file creds.env -f docker-compose.yml -f docker-compose.nemotron3-nim.yml up -d`
3. Open the UI at `http://localhost:7860`.

Containers are started with `restart: unless-stopped`, so they will auto-restart when the Docker daemon starts.

**What This Starts**
- `nemotron-nim` (NIM OpenAI-compatible API) on `http://localhost:8000/v1`
- `nemotron-ui` (Gradio UI) on `http://localhost:7860`
- `dml-service` (Daystrom Memory Lattice) on `http://localhost:9001/health`

**Compose Files**
- `docker-compose.yml`: UI + DML only (expects external vLLM at `VLLM_BASE_URL`).
- `docker-compose.nemotron3-nim.yml`: adds the Nemotron 3 Nano NIM service and points UI + DML to it.
- `docker-compose.nemotron3-nim-multi.yml`: runs separate NIM containers per agent role.
- `docker-compose.experimental.yml`: per-role vLLM containers with Hugging Face models (profile: `experimental`).

**Multi-NIM (Per-Role Agents)**
```bash
./ngc_login.sh
docker compose --env-file creds.env -f docker-compose.yml -f docker-compose.nemotron3-nim-multi.yml up -d
```

**Experimental vLLM (Per-Role Hugging Face Models)**
```bash
echo 'HUGGINGFACE_TOKEN=hf_xxx' > creds.env
```

Optional host pre-pull:
```bash
export HUGGINGFACE_TOKEN=hf_xxx
cd nemotron-agent-demo
./scripts/pull_hf_models.sh
```

Compose up:
```bash
docker compose --env-file creds.env \
  -f docker-compose.yml \
  -f docker-compose.experimental.yml \
  --profile experimental up -d
```

Verify:
```bash
docker compose ps
curl -s http://localhost:7860/
docker logs -f vllm-planner
```

Notes:
- Llama models are gated; your HF token must have access.
- First run downloads models into `nemotron-agent-demo/models/`.
- For reproducibility, pin the `vllm/vllm-openai:latest` digest after the first pull.

**Model Storage (Repo-Local)**
All models are stored in `nemotron-agent-demo/models/`.

First-time setup:
```bash
export HUGGINGFACE_TOKEN=hf_xxx
docker compose \
  -f docker-compose.yml \
  -f docker-compose.experimental.yml \
  --profile experimental up hf-prepull
```

Models are reused across all agent roles, no global cache is used, and deleting `nemotron-agent-demo/models/` forces a clean re-download.


**Environment Overrides**
- `NGC_API_KEY` for NGC registry auth (stored locally in `creds.env`).
- `HUGGINGFACE_TOKEN` for Hugging Face model access (stored locally in `creds.env`).
- `VLLM_BASE_URL` (default: `http://host.docker.internal:8000/v1`).
- `VLLM_MODEL_ID` (default: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`).
- `VLLM_TIMEOUT_S` (default: `120`).
- Per-role overrides: `ROLE_BASE_URL_<ROLE>` (e.g., `ROLE_BASE_URL_SUPERVISOR`, `ROLE_BASE_URL_PLANNER`, `ROLE_BASE_URL_CODER`).
- Experimental per-role model overrides: `EXPERIMENTAL_MODEL_<ROLE>` (e.g., `EXPERIMENTAL_MODEL_SUPERVISOR`).
- Experimental vLLM tuning: `EXPERIMENTAL_MAX_MODEL_LEN` (default: `8192`), `EXPERIMENTAL_GPU_UTIL` (default: `0.90`).
- `PLAYGROUND_SSH_DIR` (optional): host path to mount into `/root/.ssh` for SSH deployments.
- `PLAYGROUND_KUBECONFIG` (optional): host kubeconfig path mounted as `/root/.kube/config` for kubectl.
- `PLAYGROUND_DOCKER_SOCK` (optional, `1`/`0`): mount `/var/run/docker.sock` into playground for Docker CLI.
- `AGENT_MAX_ATTEMPTS` (default: `0` = unlimited retries until completion).

**Health Checks**
- NIM: `curl http://localhost:8000/v1/models`
- UI: `http://localhost:7860`
- DML: `curl http://localhost:9001/health`

**Infrastructure Tools (Playground)**
- `playground.expose_port` to publish container ports to the host.
- `ssh`, `scp`, `rsync` available inside playground when `PLAYGROUND_SSH_DIR` is mounted.
- `kubectl` available inside playground when `PLAYGROUND_KUBECONFIG` is mounted.
- `playground.docker` supports `docker compose` subcommands: `up`, `down`, `ps`, `logs`, `build`, `pull`, `restart`, `stop`, `start`.

**Remote Interactive Shell (tmux)**
Use tmux on the remote host for a persistent, automation-safe "interactive" session.

Examples:
```bash
export SSH_HOST=192.168.50.81 SSH_USER=nvidia SSH_PASSWORD=nvidia TMUX_SESSION=nemostation
./scripts/remote_tmux.sh start
marker=$(./scripts/remote_tmux.sh exec "uname -a")
./scripts/remote_tmux.sh wait "$marker"
```

Notes:
- `tmux` must be installed on the remote host.
- This is interactive-like (send-keys + capture-pane) but non-interactive for automation.
- Use `capture` or `wait` to retrieve output.

**Local Run Without Containers (Legacy)**
```bash
cd nemotron-agent-demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
VLLM_BASE_URL=http://localhost:8000/v1 ./run_ui.sh
```

**Build The Playground Image**
```bash
docker compose --profile playground build nemotron-playground-image
```

**Managing Prompts From The UI**
- Open the Prompts tab to manage goal presets and agent prompts.
- Goal Presets: edit, save, or create entries in `prompt_library/goal_presets.json`.
- Agent Prompts: save overrides to `prompt_library/agent_overrides/<agent>.txt`.

**Daystrom Memory Lattice (DML)**
- Toggle DML in the UI to enable persistent memory + retrieval reports.
- Storage: `../_dml` (repo root, persisted across runs).
- Reset: stop the UI and remove `../_dml`.

**Run The CLI Demo**
```bash
./run_demo_cli.sh "Build a resilient offline LLM demo" --scenario "Ship a resilient offline demo"
```

**Troubleshooting**
- GPU access: verify `nvidia-smi` works on the host and Docker can see GPUs.
- Port conflicts: adjust port mappings in the compose files.
