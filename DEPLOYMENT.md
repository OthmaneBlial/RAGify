# Deployment Guide

This document explains how to build the optimized RAGify container image and deploy it to Google Cloud Run. The flow mirrors the simple CRUD sample you provided, but is tailored to this repository.

## Prerequisites

- Docker (for local image builds/test runs)
- Google Cloud CLI (`gcloud`) authenticated against your project
- A Google Cloud project with billing enabled

Environment variables used during deployment:

| Variable | Purpose | Default |
| --- | --- | --- |
| `PROJECT_ID` | Google Cloud project to use | active gcloud config |
| `IMAGE_NAME` | Container image name (without registry) | `ragify` |
| `SERVICE_NAME` | Cloud Run service name | `ragify` |
| `REGION` | Cloud Run region | `us-central1` |
| `CLOUD_RUN_PORT` | Container port exposed to Cloud Run | `8000` |
| `CLOUD_RUN_MEMORY` | Cloud Run memory limit | `2Gi` |
| `CLOUD_RUN_CPU` | Cloud Run CPU allocation | `1` |
| `CLOUD_RUN_ENV_VARS` | Comma-separated env vars passed to the service | Reads from `.env` if unset |

Add any runtime-specific secrets (e.g., `DATABASE_URL`, `REDIS_URL`, `OPENROUTER_API_KEY`, `SECRET_KEY`) by extending `CLOUD_RUN_ENV_VARS`.

## Local Build & Test

```bash
# Build the multi-stage image
docker build -t ragify-local .

# Run it locally
docker run --rm -p 8000:8000 \
  -e DATABASE_URL="postgresql+asyncpg://..." \
  -e REDIS_URL="redis://..." \
  -e OPENROUTER_API_KEY="..." \
  ragify-local
```

The container serves the FastAPI backend and (when `SERVE_FRONTEND_BUILD=true`) the static frontend bundle under `/`.

## Deploy to Cloud Run

`build.sh` automates Cloud Build submission and Cloud Run deployment. It automatically loads key/value pairs from `.env` (or `ENV_FILE=path/to/file`) when `CLOUD_RUN_ENV_VARS` is not explicitly supplied:

```bash
PROJECT_ID=my-project \
REGION=us-central1 \
./build.sh
```

What the script does:

1. Verifies `gcloud` availability and authentication.
2. Ensures `cloudbuild.googleapis.com` and `run.googleapis.com` are enabled.
3. Builds the Docker image with `gcloud builds submit` and pushes it to `gcr.io/${PROJECT_ID}/${IMAGE_NAME}`.
4. Deploys the image to Cloud Run with the provided runtime configuration.
5. Prints the public service URL plus health & status endpoints for quick validation.

## Verification Checklist

- `curl https://SERVICE_URL/health` returns `{"status": "ok", ...}`.
- `curl https://SERVICE_URL/api/status` includes `"frontend": "https://SERVICE_URL"`.
- Visiting `https://SERVICE_URL/` loads the bundled UI when `SERVE_FRONTEND_BUILD=true`.
- Logs (`gcloud run logs read --service SERVICE_NAME --region REGION`) show background tasks starting without errors.
