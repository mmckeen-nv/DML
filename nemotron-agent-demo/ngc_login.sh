#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CREDS_FILE="${ROOT_DIR}/creds.env"

if [[ -z "${NGC_API_KEY:-}" && -f "${CREDS_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${CREDS_FILE}"
  set +a
fi

if [[ -z "${NGC_API_KEY:-}" ]]; then
  read -r -s -p "Enter NGC_API_KEY: " NGC_API_KEY
  echo
fi

umask 077
printf 'NGC_API_KEY=%s\n' "${NGC_API_KEY}" > "${CREDS_FILE}"
chmod 600 "${CREDS_FILE}"

printf '%s' "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin
