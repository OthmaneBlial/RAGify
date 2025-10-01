#!/bin/bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--build] [--down] [--logs]

Options:
  --build   Rebuild images before starting containers
  --down    Stop and remove containers
  --logs    Follow logs after starting (combine with --build if needed)
  --help    Show this message
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
ENV_FILE="$SCRIPT_DIR/.env"
BASE_IMAGE="ragify-backend-base:latest"
BASE_DOCKERFILE="$SCRIPT_DIR/docker/backend-base.Dockerfile"

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "docker-compose.yml not found at $COMPOSE_FILE" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or not on PATH." >&2
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  compose_cmd=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  compose_cmd=(docker-compose)
else
  echo "Neither 'docker compose' nor 'docker-compose' is available." >&2
  exit 1
fi

ACTION="up"
DO_BUILD=0
FOLLOW_LOGS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)
      DO_BUILD=1
      ;;
    --down)
      ACTION="down"
      ;;
    --logs)
      FOLLOW_LOGS=1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

if [[ "$ACTION" == "down" ]]; then
  echo "Stopping Docker services..."
  "${compose_cmd[@]}" -f "$COMPOSE_FILE" down
  exit 0
fi

build_base_image() {
  if [[ ! -f "$BASE_DOCKERFILE" ]]; then
    echo "Base Dockerfile not found at $BASE_DOCKERFILE" >&2
    exit 1
  fi

  if [[ $DO_BUILD -eq 1 ]] || ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
    echo "Building backend base image ($BASE_IMAGE)..."
    docker build -f "$BASE_DOCKERFILE" -t "$BASE_IMAGE" "$SCRIPT_DIR"
  else
    echo "Using existing backend base image ($BASE_IMAGE)."
  fi
}

build_base_image

args=(-f "$COMPOSE_FILE" up -d)
if [[ $DO_BUILD -eq 1 ]]; then
  args+=('--build')
fi

echo "Starting Docker services..."
"${compose_cmd[@]}" "${args[@]}"

if [[ -f "$ENV_FILE" ]]; then
  echo "Loaded environment variables from $ENV_FILE"
else
  echo "Warning: $ENV_FILE not found. Containers are using built-in defaults."
fi

echo "Backend:  http://localhost:18000"
echo "Frontend: http://localhost:15173"
echo "PostgreSQL: localhost:15432"
echo "Redis:      localhost:16379"

echo "Use $0 --down to stop containers."

if [[ $FOLLOW_LOGS -eq 1 ]]; then
  echo "\nStreaming logs (Ctrl+C to stop)..."
  "${compose_cmd[@]}" -f "$COMPOSE_FILE" logs -f
fi
