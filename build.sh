#!/usr/bin/env bash

# ✨ RAGify Cloud Run Deployment Script ✨
# This script builds and deploys your awesome project to Google Cloud Run!

set -euo pipefail

# --- Configuration Station 🚂 ---
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_ENV_FILE=""
TEMP_BUILD_CONFIG=""

# --- Cleanup Crew 🧹 ---
# Ensures we leave no mess behind!
cleanup() {
  echo "🧹 Cleaning up temporary files..."
  rm -f "${TEMP_ENV_FILE}" "${TEMP_BUILD_CONFIG}"
}

trap 'cleanup; echo -e "\n\033[0;31m❌ Oh no! Something went wrong. Check the logs above.\033[0m"; exit 1' ERR
trap 'cleanup' EXIT

# --- Helper Functions 🛠️ ---
require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo -e "\033[0;31m❌ Missing a crucial tool: '$1'. Please install it and ensure it's in your PATH.\033[0m"
    exit 1
  fi
}

ensure_api_enabled() {
  local api=$1
  if ! gcloud services list --enabled --filter="config.name=${api}" --format="value(config.name)" | grep -Fxq "${api}"; then
    echo "🔌 Powering up the ${api} API... (This might take a moment)"
    gcloud services enable "${api}"
  else
    echo "✅ API '${api}' is already sparkling!"
  fi
}

base_image_exists() {
  gcloud container images describe "${BASE_IMAGE_URI}" >/dev/null 2>&1
}

# --- Let the Adventure Begin! 🚀 ---

# 1️⃣  System Check & Variable Setup
echo "📋 Checking your setup..."
require_command gcloud

# Get your Google Cloud Project ID
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
if [[ -z "${PROJECT_ID// }" ]]; then
  echo -e "\033[0;31m❌ We need a Project ID! Can't find a default.\033[0m"
  echo "   Please set it like this: PROJECT_ID=my-gcp-project ./build.sh"
  exit 1
fi

# Fun default names and settings!
IMAGE_NAME="${IMAGE_NAME:-ragify}"
SERVICE_NAME="${SERVICE_NAME:-ragify}"
REGION="${REGION:-us-central1}"
CLOUD_RUN_PORT="${CLOUD_RUN_PORT:-8000}" # This is the port INSIDE the container to target
CLOUD_RUN_MEMORY="${CLOUD_RUN_MEMORY:-2Gi}"
CLOUD_RUN_CPU="${CLOUD_RUN_CPU:-1}"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/.env}"
IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}"
BASE_IMAGE_NAME="${BASE_IMAGE_NAME:-ragify-backend-base}"
BASE_IMAGE_URI="${BASE_IMAGE_URI:-gcr.io/${PROJECT_ID}/${BASE_IMAGE_NAME}:latest}"
BUILD_BASE_IMAGE="${BUILD_BASE_IMAGE:-auto}"

# 2️⃣  Magically Load Environment Variables from `.env` file
if [[ -f "$ENV_FILE" ]]; then
  echo "📄 Found your .env file! Reading secrets and settings..."
  TEMP_ENV_FILE="$(mktemp)"
  # This Python wizardry securely converts your .env into JSON for gcloud.
  # We are NOT adding the PORT variable here anymore! 🧙‍♂️
  python - "$ENV_FILE" "$TEMP_ENV_FILE" <<'PY'
import json, pathlib, sys
env_path, out_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
data = {}
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            # IMPORTANT: We skip the PORT variable
            if key.strip() and key.strip() != "PORT":
                data[key.strip()] = value.strip()
out_path.write_text(json.dumps(data))
PY
else
  echo -e "\033[1;33m⚠️  No .env file found. I hope you've set your secrets in Cloud Run already!\033[0m"
fi

# 3️⃣  gcloud Authentication and Configuration
echo "🔐 Checking your gcloud credentials and setting project..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
  echo -e "\033[0;31m❌ You're not logged into gcloud! Please run 'gcloud auth login'.\033[0m"
  exit 1
fi
gcloud config set project "${PROJECT_ID}" >/dev/null
echo "✅ Project set to '${PROJECT_ID}'."

# 4️⃣  Turn on the Lights! (Enable GCP APIs)
echo "🔍 Checking if essential Google Cloud APIs are enabled..."
ensure_api_enabled cloudbuild.googleapis.com
ensure_api_enabled run.googleapis.com

# 5️⃣  Build the Heavy-Lifter Base Image (if it's missing or told to)
if [[ "${BUILD_BASE_IMAGE}" == "never" ]]; then
  echo "🏃‍♂️ Skipping base image build as requested."
elif [[ "${BUILD_BASE_IMAGE}" != "always" ]] && base_image_exists; then
  echo "ℹ️  Found your hefty base image! Reusing it to save time."
else
  echo "🧱 Building the heavy base image. This can take a few minutes, time for a ☕️..."
  TEMP_BUILD_CONFIG="$(mktemp)"
  cat > "${TEMP_BUILD_CONFIG}" <<EOF
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-f', 'docker/backend-base.Dockerfile', '-t', '${BASE_IMAGE_URI}', '.']
images:
- '${BASE_IMAGE_URI}'
EOF
  gcloud builds submit "${ROOT_DIR}" --config "${TEMP_BUILD_CONFIG}"
  echo "✅ Base image is ready for action!"
fi

# 6️⃣  Build the Lightweight Application Image
echo "🏗️  Now building your lightweight application image..."
TEMP_BUILD_CONFIG="$(mktemp)"
cat > "${TEMP_BUILD_CONFIG}" <<EOF
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: [
    'build',
    '-t', '\${_IMAGE_URI}',
    '--build-arg', 'BASE_IMAGE=\${_BASE_IMAGE_URI}',
    '.'
  ]
images:
- '\${_IMAGE_URI}'
EOF

gcloud builds submit "${ROOT_DIR}" \
  --config "${TEMP_BUILD_CONFIG}" \
  --substitutions="_IMAGE_URI=${IMAGE_URI},_BASE_IMAGE_URI=${BASE_IMAGE_URI}"
echo "✅ Application image is built and pushed to the container registry!"

# 7️⃣  Deploy to the Cloud! 🚀 (With the corrected arguments!)
echo "🚀 Deploying '${SERVICE_NAME}' to Cloud Run in '${REGION}'! This is the one..."
deploy_args=(
  "${SERVICE_NAME}"
  --image "${IMAGE_URI}"
  --platform "managed"
  --region "${REGION}"
  --allow-unauthenticated
  --port "${CLOUD_RUN_PORT}" # Tells Cloud Run which container port to send traffic to
  --memory "${CLOUD_RUN_MEMORY}"
  --cpu "${CLOUD_RUN_CPU}"
)

# Attach the environment variables file if it was created
if [[ -n "${TEMP_ENV_FILE}" && -s "${TEMP_ENV_FILE}" ]]; then
  deploy_args+=(--env-vars-file "${TEMP_ENV_FILE}")
fi

gcloud run deploy "${deploy_args[@]}" --quiet

# 8️⃣  The Grand Reveal! 🥳
SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format="value(status.url)")"

echo -e "\n\033[1;32m🎉 CONGRATULATIONS! VICTORY! Your RAGify app is LIVE!\033[0m"
echo -e "🌐 Service URL:  \033[1;34m${SERVICE_URL}\033[0m"
echo -e "📡 Health Check: \033[1;34m${SERVICE_URL}/health\033[0m"
echo -e "📊 API Status:   \033[1;34m${SERVICE_URL}/api/status\033[0m"