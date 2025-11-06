#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

trap 'echo "❌ Build or deployment failed. Review the logs above."; exit 1' ERR

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

load_env_file() {
  local env_file=$1
  local env_pairs=()

  if [[ ! -f "$env_file" ]]; then
    return 0
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    # Trim whitespace
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "$line" =~ ^# ]] && continue

    if [[ "$line" == *"="* ]]; then
      key="${line%%=*}"
      value="${line#*=}"
      key="$(echo -n "$key" | xargs)"
      value="$(echo -n "$value" | xargs)"
      env_pairs+=("${key}=${value}")
    fi
  done < "$env_file"

  local joined=""
  for pair in "${env_pairs[@]}"; do
    if [[ -z "$joined" ]]; then
      joined="$pair"
    else
      joined="${joined},${pair}"
    fi
  done

  echo "$joined"
}

if [[ -z "$CLOUD_RUN_ENV_VARS" ]]; then
  if [[ -f "$ENV_FILE" ]]; then
    echo "📄 Loading environment variables from ${ENV_FILE}"
    CLOUD_RUN_ENV_VARS="$(load_env_file "$ENV_FILE")"
  fi
fi

if [[ -z "$CLOUD_RUN_ENV_VARS" ]]; then
  echo "❌ No Cloud Run environment variables found."
  echo "   Set CLOUD_RUN_ENV_VARS explicitly or provide an ENV_FILE (.env) with key=value pairs."
  exit 1
fi

ensure_env_var() {
  local key=$1
  local value=$2
  if ! echo "$CLOUD_RUN_ENV_VARS" | tr ',' '\n' | grep -q "^${key}="; then
    if [[ -n "$CLOUD_RUN_ENV_VARS" ]]; then
      CLOUD_RUN_ENV_VARS+=","
    fi
    CLOUD_RUN_ENV_VARS+="${key}=${value}"
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

echo "🏗️  Building and pushing image ${IMAGE_URI}..."
gcloud builds submit "${ROOT_DIR}" --tag "${IMAGE_URI}"

echo "🚀 Deploying ${SERVICE_NAME} to Cloud Run (${REGION})..."
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE_URI}" \
  --platform managed \
  --region "${REGION}" \
  --allow-unauthenticated \
  --port "${CLOUD_RUN_PORT}" \
  --memory "${CLOUD_RUN_MEMORY}" \
  --cpu "${CLOUD_RUN_CPU}" \
  --set-env-vars "${CLOUD_RUN_ENV_VARS}"

SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format="value(status.url)")"

echo "✅ Deployment complete."
echo "🌐 Service URL: ${SERVICE_URL}"
echo "📡 Health:      ${SERVICE_URL}/health"
echo "📊 Status:      ${SERVICE_URL}/api/status"
