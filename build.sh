#!/usr/bin/env bash
set -euo pipefail

# --- Paths / globals ---
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_ENV_FILE=""

cleanup() {
  [[ -n "${TEMP_ENV_FILE}" ]] && rm -f "${TEMP_ENV_FILE}"
}
trap cleanup EXIT

require_command() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing command: $1"; exit 1; }
}

require_command gcloud
require_command docker
require_command python3

# --- Config (tune via env vars) ---
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
if [[ -z "${PROJECT_ID// }" ]]; then
  echo "PROJECT_ID is missing. Set it or configure gcloud default project."
  exit 1
fi

IMAGE_NAME="${IMAGE_NAME:-ragify}"
SERVICE_NAME="${SERVICE_NAME:-ragify}"
REGION="${REGION:-europe-west1}"
CLOUD_RUN_PORT="${CLOUD_RUN_PORT:-8000}"
CLOUD_RUN_MEMORY="${CLOUD_RUN_MEMORY:-2Gi}"
CLOUD_RUN_CPU="${CLOUD_RUN_CPU:-2}"
CLOUD_RUN_MIN_INSTANCES="${CLOUD_RUN_MIN_INSTANCES:-1}"

IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}"
BASE_IMAGE_NAME="${BASE_IMAGE_NAME:-ragify-backend-base}"
BASE_IMAGE_URI="${BASE_IMAGE_URI:-gcr.io/${PROJECT_ID}/${BASE_IMAGE_NAME}:latest}"

# Set BUILD_BASE_IMAGE=1 when you want to refresh the base image.
BUILD_BASE_IMAGE="${BUILD_BASE_IMAGE:-0}"

ENV_FILE_PATH="${ENV_FILE:-${ROOT_DIR}/.env}"
SQLITE_DB_PATH="${SQLITE_DB_PATH:-/tmp/ragify.db}"

if [[ "${SQLITE_DB_PATH}" == /* ]]; then
  DATABASE_URL_OVERRIDE="sqlite+aiosqlite:////${SQLITE_DB_PATH#/}"
else
  DATABASE_URL_OVERRIDE="sqlite+aiosqlite:///${SQLITE_DB_PATH}"
fi

if [[ -n "${ENV_FILE_PATH}" && ! "${ENV_FILE_PATH}" = /* ]]; then
  ENV_FILE_PATH="${ROOT_DIR}/${ENV_FILE_PATH}"
fi

# --- Env file generation (for Cloud Run) ---
generate_env_file() {
  TEMP_ENV_FILE="$(mktemp)"
  python3 - "$ENV_FILE_PATH" "$TEMP_ENV_FILE" "$DATABASE_URL_OVERRIDE" "$SQLITE_DB_PATH" <<'PY'
import json, os, sys

env_path, out_path, db_url, sqlite_path = sys.argv[1:5]
env_data = {}

if env_path and os.path.isfile(env_path):
    with open(env_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if not key:
                continue
            if value and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            env_data[key] = value

env_data["DATABASE_URL"] = db_url
env_data["SERVE_FRONTEND_BUILD"] = env_data.get("SERVE_FRONTEND_BUILD", "true") or "true"
env_data["SQLITE_DB_PATH"] = sqlite_path
env_data["REDIS_URL"] = ""

with open(out_path, "w", encoding="utf-8") as handle:
    for key in sorted(env_data):
        handle.write(f"{key}: {json.dumps(env_data[key])}\n")
PY
}

# --- Optional: build base image locally (rare) ---
if [[ "${BUILD_BASE_IMAGE}" != "0" ]]; then
  echo "Building base image (docker/backend-base.Dockerfile) ..."
  docker build \
    -f "${ROOT_DIR}/docker/backend-base.Dockerfile" \
    -t "${BASE_IMAGE_URI}" \
    "${ROOT_DIR}" >/dev/null
  echo "Pushing base image ..."
  docker push "${BASE_IMAGE_URI}" >/dev/null
fi

# --- Build app image locally ---
echo "Building app image ..."
docker build \
  -t "${IMAGE_URI}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE_URI}" \
  "${ROOT_DIR}" >/dev/null

echo "Pushing app image ..."
docker push "${IMAGE_URI}" >/dev/null

# --- Env vars for Cloud Run ---
generate_env_file

# --- Deploy to Cloud Run (quiet) ---
echo "Deploying to Cloud Run ..."
gcloud run deploy "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --allow-unauthenticated \
  --execution-environment "gen2" \
  --image "${IMAGE_URI}" \
  --port "${CLOUD_RUN_PORT}" \
  --memory "${CLOUD_RUN_MEMORY}" \
  --cpu "${CLOUD_RUN_CPU}" \
  --min-instances "${CLOUD_RUN_MIN_INSTANCES}" \
  --env-vars-file "${TEMP_ENV_FILE}" \
  --quiet \
  --no-user-output-enabled \
  >/dev/null

SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --format="value(status.url)")"

echo "${SERVICE_URL}"
echo "${SERVICE_URL}/health"
echo "${SERVICE_URL}/api/status"
