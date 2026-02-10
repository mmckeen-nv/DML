#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/creds.env}"

build_args=()
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      echo "Usage: $0 [--no-cache] [docker compose build args...]"
      exit 0
      ;;
    --no-cache)
      build_args+=(--no-cache)
      ;;
    *)
      build_args+=("$arg")
      ;;
  esac
done

compose_files=()
if [[ -n "${COMPOSE_FILES:-}" ]]; then
  read -r -a compose_files <<<"${COMPOSE_FILES}"
else
  compose_files+=("$ROOT_DIR/docker-compose.yml")
  if [[ "${NIM_MODE:-single}" == "multi" ]]; then
    compose_files+=("$ROOT_DIR/docker-compose.nemotron3-nim-multi.yml")
  else
    compose_files+=("$ROOT_DIR/docker-compose.nemotron3-nim.yml")
  fi
fi

compose_cmd=(docker compose)
if [[ -f "$ENV_FILE" ]]; then
  compose_cmd+=(--env-file "$ENV_FILE")
fi
for f in "${compose_files[@]}"; do
  compose_cmd+=( -f "$f" )
done

cd "$ROOT_DIR"

"${compose_cmd[@]}" build "${build_args[@]}"
"${compose_cmd[@]}" --profile playground build "${build_args[@]}" nemotron-playground-image

echo "Build complete."
