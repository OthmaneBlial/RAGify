# syntax=docker/dockerfile:1.6

FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements-docker.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
 && pip wheel --wheel-dir /wheels -r requirements-docker.txt

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --upgrade pip \
 && pip install --no-index --find-links=/wheels -r /tmp/requirements-docker.txt \
 && rm -rf /wheels /tmp/requirements-docker.txt

# Create non-root user for downstream images
RUN useradd --create-home --shell /bin/bash ragify && \
    chown -R ragify:ragify /app

USER ragify

ENV DATABASE_URL="postgresql+asyncpg://ragify:RagifyStrongPass2023@postgres:5432/ragify" \
    REDIS_URL="redis://redis:6379/0" \
    SECRET_KEY="ragify-local-secret" \
    OPENROUTER_API_KEY=""

CMD ["python", "-c", "print('Base image for RAGify backend')"]
