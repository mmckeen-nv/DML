#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/creds.env}"

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

set +e
"${compose_cmd[@]}" down --timeout 15 --remove-orphans
set -e

mapfile -t ids < <(
  docker ps -a --format '{{.ID}} {{.Names}}' | \
  awk '$2 ~ /^(nemotron-ui|dml-service|nemotron-nim(-.*)?|nemotron-playground-.*|nemotron-play-.*)$/ {print $1}'
)

if [[ ${#ids[@]} -gt 0 ]]; then
  set +e
  docker stop -t 15 "${ids[@]}"
  docker kill "${ids[@]}"
  docker rm -f "${ids[@]}"
  set -e
fi

echo "Services stopped."
