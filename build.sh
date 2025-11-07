#!/usr/bin/env bash

# ✨ RAGify Cloud Run Multi-Container Deployment - VICTORY LAP EDITION! ✨
# This script deploys your app AND a PostgreSQL container together! 🚀

set -euo pipefail

# --- Configuration Station 🚂 ---
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_ENV_FILE=""

# --- Cleanup Crew 🧹 ---
cleanup() {
  echo "🧹 Cleaning up temporary files..."
  [[ -n "${TEMP_ENV_FILE}" ]] && rm -f "${TEMP_ENV_FILE}"
}

trap 'cleanup; echo -e "\n\033[0;31m❌ Oh no! Something went wrong. Check the logs above.\033[0m"; exit 1' ERR
trap 'cleanup' EXIT

# --- Helper Functions 🛠️ ---
require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo -e "\033[0;31m❌ Missing a crucial tool: '$1'. Please install it!\033[0m"
    exit 1
  fi
}

ensure_api_enabled() {
  local api=$1
  if ! gcloud services list --enabled --filter="config.name=${api}" --format="value(config.name)" | grep -Fxq "${api}"; then
    echo "🔌 Powering up the ${api} API..."
    gcloud services enable "${api}"
  else
    echo "✅ API '${api}' is already sparkling!"
  fi
}

base_image_exists() {
  gcloud container images describe "${BASE_IMAGE_URI}" >/dev/null 2>&1
}

# --- Let's Do This! 🚀 ---

# 1️⃣  System Check & Variable Setup
echo "📋 Final pre-flight check..."
require_command gcloud
require_command python3

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
if [[ -z "${PROJECT_ID// }" ]]; then
  echo -e "\033[0;31m❌ Project ID missing! Set it: PROJECT_ID=my-gcp-project ./build.sh\033[0m"
  exit 1
fi

IMAGE_NAME="${IMAGE_NAME:-ragify}"
SERVICE_NAME="${SERVICE_NAME:-ragify}"
REGION="${REGION:-us-central1}"
CLOUD_RUN_PORT="${CLOUD_RUN_PORT:-8000}"
CLOUD_RUN_MEMORY="${CLOUD_RUN_MEMORY:-2Gi}"
CLOUD_RUN_CPU="${CLOUD_RUN_CPU:-1}"
IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}"
BASE_IMAGE_NAME="${BASE_IMAGE_NAME:-ragify-backend-base}"
BASE_IMAGE_URI="${BASE_IMAGE_URI:-gcr.io/${PROJECT_ID}/${BASE_IMAGE_NAME}:latest}"
BUILD_BASE_IMAGE="${BUILD_BASE_IMAGE:-auto}"
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

if [[ ! -f "${ENV_FILE_PATH}" ]]; then
  echo "ℹ️  ENV file '${ENV_FILE_PATH}' not found. Continuing with deployment overrides only."
fi

generate_env_file() {
  TEMP_ENV_FILE="$(mktemp)"
  python3 - "$ENV_FILE_PATH" "$TEMP_ENV_FILE" "$DATABASE_URL_OVERRIDE" "$SQLITE_DB_PATH" <<'PY'
import json
import os
import sys

env_path, out_path, db_url, sqlite_path = sys.argv[1:5]
env_data = {}

if env_path and os.path.isfile(env_path):
    with open(env_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
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
  echo "🧾 Environment variables written to ${TEMP_ENV_FILE}"
}

# 2️⃣  gcloud Authentication
echo "🔐 Verifying gcloud login..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
  echo -e "\033[0;31m❌ Not logged into gcloud! Please run 'gcloud auth login'.\033[0m"
  exit 1
fi
gcloud config set project "${PROJECT_ID}" >/dev/null
echo "✅ Project set to '${PROJECT_ID}'."

# 3️⃣  Enable APIs
echo "🔍 Checking Google Cloud APIs..."
ensure_api_enabled cloudbuild.googleapis.com
ensure_api_enabled run.googleapis.com
ensure_api_enabled artifactregistry.googleapis.com # Good to have

# 4️⃣  Build Base Image (if needed)
if [[ "${BUILD_BASE_IMAGE}" == "never" ]]; then
  echo "🏃‍♂️ Skipping base image build."
elif [[ "${BUILD_BASE_IMAGE}" != "always" ]] && base_image_exists; then
  echo "ℹ️  Reusing existing base image."
else
  echo "🧱 Building the base image... ☕️"
  gcloud builds submit "${ROOT_DIR}" --config <(cat <<EOF
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-f', 'docker/backend-base.Dockerfile', '-t', '${BASE_IMAGE_URI}', '.']
images: ['${BASE_IMAGE_URI}']
EOF
)
  echo "✅ Base image ready!"
fi

# 5️⃣  Build App Image
echo "🏗️  Building the application image..."
gcloud builds submit "${ROOT_DIR}" --config <(cat <<EOF
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', '\${_IMAGE_URI}', '--build-arg', 'BASE_IMAGE=\${_BASE_IMAGE_URI}', '.']
images: ['\${_IMAGE_URI}']
EOF
) --substitutions="_IMAGE_URI=${IMAGE_URI},_BASE_IMAGE_URI=${BASE_IMAGE_URI}"
echo "✅ App image built and pushed!"

generate_env_file

# 6️⃣  Deploy Both Containers to Cloud Run! 🚀🚀
echo "🚀 Deploying '${SERVICE_NAME}' using the embedded SQLite database (no Redis required)!"

gcloud run deploy "${SERVICE_NAME}" \
  --region "${REGION}" \
  --allow-unauthenticated \
  --execution-environment "gen2" \
  --image "${IMAGE_URI}" \
  --port "${CLOUD_RUN_PORT}" \
  --memory "${CLOUD_RUN_MEMORY}" \
  --cpu "${CLOUD_RUN_CPU}" \
  --env-vars-file "${TEMP_ENV_FILE}" \
  --quiet

# 7️⃣  The Grand Reveal! 🥳
SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format="value(status.url)")"

echo -e "\n\033[1;32m🎉🎉🎉 VICTORY! IT'S ALIVE! 🎉🎉🎉\033[0m"
echo -e "🌐 Your RAGify app is live at:  \033[1;34m${SERVICE_URL}\033[0m"
echo -e "📡 Health Check endpoint: \033[1;34m${SERVICE_URL}/health\033[0m"
echo -e "📊 API Status endpoint:   \033[1;34m${SERVICE_URL}/api/status\033[0m"
