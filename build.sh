#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_ENV_FILE=""

cleanup() {
  if [[ -n "${TEMP_ENV_FILE}" && -f "${TEMP_ENV_FILE}" ]]; then
    rm -f "${TEMP_ENV_FILE}"
  fi
}

trap 'cleanup; echo "❌ Build or deployment failed. Review the logs above."; exit 1' ERR
trap 'cleanup' EXIT

require_command() {
  local cmd=$1
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "❌ Missing dependency: $cmd"
    exit 1
  fi
}

echo "📋 Checking prerequisites..."
require_command gcloud

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
if [[ -z "${PROJECT_ID// }" ]]; then
  echo "❌ PROJECT_ID is not set and no gcloud default project was found."
  echo "   Set PROJECT_ID=my-gcp-project and rerun the script."
  exit 1
fi

IMAGE_NAME="${IMAGE_NAME:-ragify}"
SERVICE_NAME="${SERVICE_NAME:-ragify}"
REGION="${REGION:-us-central1}"
CLOUD_RUN_PORT="${CLOUD_RUN_PORT:-8000}"
CLOUD_RUN_MEMORY="${CLOUD_RUN_MEMORY:-2Gi}"
CLOUD_RUN_CPU="${CLOUD_RUN_CPU:-1}"
CLOUD_RUN_ENV_VARS="${CLOUD_RUN_ENV_VARS:-}"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/.env}"
IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}"
BASE_IMAGE_NAME="${BASE_IMAGE_NAME:-ragify-backend-base}"
BASE_IMAGE_URI="${BASE_IMAGE_URI:-gcr.io/${PROJECT_ID}/${BASE_IMAGE_NAME}:latest}"
BUILD_BASE_IMAGE="${BUILD_BASE_IMAGE:-auto}"

if [[ -z "$CLOUD_RUN_ENV_VARS" && -f "$ENV_FILE" ]]; then
  echo "📄 Loading environment variables from ${ENV_FILE}"
  TEMP_ENV_FILE="$(mktemp)"
  python <<'PY' "$ENV_FILE" "$TEMP_ENV_FILE"
import json, pathlib, sys
env_path = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])
data = {}
for raw_line in env_path.read_text().splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if key:
        data[key] = value
out_path.write_text(json.dumps(data))
PY
fi

if [[ -z "$CLOUD_RUN_ENV_VARS" && ( -z "${TEMP_ENV_FILE}" || ! -s "${TEMP_ENV_FILE}" ) ]]; then
  echo "❌ No Cloud Run environment variables found."
  echo "   Set CLOUD_RUN_ENV_VARS explicitly or provide an ENV_FILE (.env) with key=value pairs."
  exit 1
fi

ensure_env_var() {
  local key=$1
  local value=$2
  if [[ -n "${TEMP_ENV_FILE}" ]]; then
    python <<'PY' "${TEMP_ENV_FILE}" "${key}" "${value}"
import json, sys
path, key, value = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(path) as fh:
        data = json.load(fh)
except (FileNotFoundError, json.JSONDecodeError):
    data = {}
if key not in data:
    data[key] = value
with open(path, "w") as fh:
    json.dump(data, fh)
PY
  else
    if ! echo "$CLOUD_RUN_ENV_VARS" | tr ',' '\n' | grep -q "^${key}="; then
      if [[ -n "$CLOUD_RUN_ENV_VARS" ]]; then
        CLOUD_RUN_ENV_VARS+=","
      fi
      CLOUD_RUN_ENV_VARS+="${key}=${value}"
    fi
  fi
}

ensure_env_var "SERVE_FRONTEND_BUILD" "true"
ensure_env_var "FRONTEND_URL" "auto"

echo "🔐 Verifying gcloud authentication..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
  echo "❌ No active gcloud account found. Run 'gcloud auth login' and try again."
  exit 1
fi

echo "⚙️  Setting project to ${PROJECT_ID}..."
gcloud config set project "${PROJECT_ID}" >/dev/null

ensure_api_enabled() {
  local api=$1
  if ! gcloud services list --enabled --filter="config.name=${api}" --format="value(config.name)" | grep -Fxq "${api}"; then
    echo "🔌 Enabling ${api}..."
    gcloud services enable "${api}"
  else
    echo "ℹ️  ${api} already enabled."
  fi
}

echo "🔍 Ensuring Cloud Build and Cloud Run APIs are enabled..."
ensure_api_enabled cloudbuild.googleapis.com
ensure_api_enabled run.googleapis.com

base_image_exists() {
  gcloud container images describe "${BASE_IMAGE_URI}" >/dev/null 2>&1
}

maybe_build_base_image() {
  if [[ "${BUILD_BASE_IMAGE}" == "never" ]]; then
    echo "⚠️  Skipping base image build (BUILD_BASE_IMAGE=never)."
    return
  fi

  if [[ "${BUILD_BASE_IMAGE}" != "always" ]] && base_image_exists; then
    echo "ℹ️  Using existing base image ${BASE_IMAGE_URI}."
    return
  fi

  echo "🧱 Building base image ${BASE_IMAGE_URI}..."
  gcloud builds submit "${ROOT_DIR}" \
    --tag "${BASE_IMAGE_URI}" \
    --file "${ROOT_DIR}/docker/backend-base.Dockerfile"
}

maybe_build_base_image

echo "🏗️  Building and pushing image ${IMAGE_URI}..."
gcloud builds submit "${ROOT_DIR}" \
  --tag "${IMAGE_URI}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE_URI}"

echo "🚀 Deploying ${SERVICE_NAME} to Cloud Run (${REGION})..."
deploy_args=(
  "${SERVICE_NAME}"
  --image "${IMAGE_URI}"
  --platform managed
  --region "${REGION}"
  --allow-unauthenticated
  --port "${CLOUD_RUN_PORT}"
  --memory "${CLOUD_RUN_MEMORY}"
  --cpu "${CLOUD_RUN_CPU}"
)

if [[ -n "${TEMP_ENV_FILE}" ]]; then
  deploy_args+=(--env-vars-file "${TEMP_ENV_FILE}")
else
  deploy_args+=(--set-env-vars "${CLOUD_RUN_ENV_VARS}")
fi

gcloud run deploy "${deploy_args[@]}"

SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format="value(status.url)")"

echo "✅ Deployment complete."
echo "🌐 Service URL: ${SERVICE_URL}"
echo "📡 Health:      ${SERVICE_URL}/health"
echo "📊 Status:      ${SERVICE_URL}/api/status"
