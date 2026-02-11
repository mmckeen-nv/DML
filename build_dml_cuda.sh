#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker build -f "${ROOT_DIR}/dml_core/Dockerfile.cuda" -t daystrom-dml-cuda "${ROOT_DIR}"
