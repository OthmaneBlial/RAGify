#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/.env}"
APP_IMAGE="${APP_IMAGE:-ragify-app}"
POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-ragify-postgres-local}"
POSTGRES_IMAGE="${POSTGRES_IMAGE:-pgvector/pgvector:pg16}"
POSTGRES_PORT="${POSTGRES_PORT:-15432}"
REDIS_CONTAINER="${REDIS_CONTAINER:-ragify-redis-local}"
REDIS_IMAGE="${REDIS_IMAGE:-redis:7-alpine}"
REDIS_PORT="${REDIS_PORT:-16379}"
BUILD_IMAGE="${BUILD_IMAGE:-1}"
CLEAN_START="${CLEAN_START:-1}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "❌ Missing dependency: $1"
    exit 1
  fi
}

require_cmd docker

if [[ ! -f "$ENV_FILE" ]]; then
  echo "❌ Env file not found at ${ENV_FILE}"
  exit 1
fi

remove_container_if_exists() {
  local name=$1
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "🧹 Removing existing container ${name}..."
    docker rm -f "${name}" >/dev/null 2>&1 || true
  fi
}

start_postgres() {
  if [[ "${CLEAN_START}" == "1" ]]; then
    remove_container_if_exists "${POSTGRES_CONTAINER}"
  elif docker ps --format '{{.Names}}' | grep -Fxq "${POSTGRES_CONTAINER}"; then
    echo "ℹ️  Using existing Postgres container ${POSTGRES_CONTAINER}."
    return
  fi

  echo "🐘 Launching local Postgres (${POSTGRES_CONTAINER})..."
  docker run -d \
    --name "${POSTGRES_CONTAINER}" \
    -e POSTGRES_DB=ragify \
    -e POSTGRES_USER=ragify \
    -e POSTGRES_PASSWORD=RagifyStrongPass2023 \
    -p "${POSTGRES_PORT}:5432" \
    "${POSTGRES_IMAGE}" \
    >/dev/null || {
      echo "❌ Failed to start Postgres container. Is port ${POSTGRES_PORT} already in use?"
      exit 1
    }
}

start_redis() {
  if [[ "${CLEAN_START}" == "1" ]]; then
    remove_container_if_exists "${REDIS_CONTAINER}"
  elif docker ps --format '{{.Names}}' | grep -Fxq "${REDIS_CONTAINER}"; then
    echo "ℹ️  Using existing Redis container ${REDIS_CONTAINER}."
    return
  fi

  echo "🧠 Launching local Redis (${REDIS_CONTAINER})..."
  docker run -d \
    --name "${REDIS_CONTAINER}" \
    -p "${REDIS_PORT}:6379" \
    "${REDIS_IMAGE}" \
    >/dev/null || {
      echo "❌ Failed to start Redis container. Is port ${REDIS_PORT} already in use?"
      exit 1
    }
}

wait_for_postgres() {
  echo -n "⏳ Waiting for Postgres..."
  for _ in {1..30}; do
    if docker exec "${POSTGRES_CONTAINER}" pg_isready -U ragify -d ragify >/dev/null 2>&1; then
      echo " ready"
      return
    fi
    sleep 1
    echo -n "."
  done
  echo ""
  echo "❌ Postgres did not become ready."
  exit 1
}

wait_for_redis() {
  echo -n "⏳ Waiting for Redis..."
  for _ in {1..30}; do
    if docker exec "${REDIS_CONTAINER}" redis-cli ping >/dev/null 2>&1; then
      echo " ready"
      return
    fi
    sleep 1
    echo -n "."
  done
  echo ""
  echo "❌ Redis did not become ready."
  exit 1
}

if [[ "${BUILD_IMAGE}" == "1" ]]; then
  echo "🔨 Building application image (${APP_IMAGE})..."
  "${ROOT_DIR}/startupdocker.sh"
else
  echo "⚠️  Skipping image build (BUILD_IMAGE=${BUILD_IMAGE})."
fi

start_postgres
start_redis
wait_for_postgres
wait_for_redis

echo "🧩 Ensuring pgvector extension..."
docker exec "${POSTGRES_CONTAINER}" psql -U ragify -d ragify -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null

echo ""
echo "🔗 Local services:"
echo "   Postgres → host.docker.internal:${POSTGRES_PORT}"
echo "   Redis    → host.docker.internal:${REDIS_PORT}"
echo ""
echo "🚀 Starting ragify container..."
docker run --rm -p 8000:8000 \
  --env-file "${ENV_FILE}" \
  --add-host host.docker.internal:host-gateway \
  -e DATABASE_URL="postgresql+asyncpg://ragify:RagifyStrongPass2023@host.docker.internal:${POSTGRES_PORT}/ragify" \
  -e REDIS_URL="redis://host.docker.internal:${REDIS_PORT}/0" \
  "${APP_IMAGE}"
