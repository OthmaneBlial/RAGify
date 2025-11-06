#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="${DOCKERFILE:-${ROOT_DIR}/Dockerfile}"
BUILD_CONTEXT="${BUILD_CONTEXT:-${ROOT_DIR}}"
IMAGE_NAME="${IMAGE_NAME:-ragify-app}"

if [[ ! -f "$DOCKERFILE" ]]; then
  echo "❌ Dockerfile not found at ${DOCKERFILE}"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "❌ Docker is not installed or not available on PATH."
  exit 1
fi

echo "🐳 Building image '${IMAGE_NAME}' from ${DOCKERFILE}..."
docker build -t "${IMAGE_NAME}" -f "${DOCKERFILE}" "${BUILD_CONTEXT}"

echo ""
echo "✅ Image build complete."
echo "   Run it locally with:"
echo "     docker run --rm -p 8000:8000 ${IMAGE_NAME}"
