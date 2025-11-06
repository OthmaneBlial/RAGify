# syntax=docker/dockerfile:1.6

ARG PYTHON_VERSION=3.12-slim

################################################################################
# Stage 1: Install Python dependencies into a reusable virtual environment
################################################################################
FROM python:${PYTHON_VERSION} AS python-deps

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PIP_NO_CACHE_DIR=0

# Create virtual environment ahead of time so we can copy it into the final image
RUN python -m venv "${VIRTUAL_ENV}"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
COPY requirements-docker.txt .

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements-docker.txt

################################################################################
# Stage 2: Runtime image with only what we need to serve traffic
################################################################################
FROM python:${PYTHON_VERSION} AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=python-deps /opt/venv /opt/venv

WORKDIR /app

# Copy application code (frontend/node_modules is excluded via .dockerignore)
COPY backend ./backend
COPY shared ./shared
COPY frontend ./frontend
COPY pyproject.toml README.md ./

RUN adduser --disabled-password --gecos "" ragify \
    && chown -R ragify:ragify /app
USER ragify

ENV PORT=8000 \
    SERVE_FRONTEND_BUILD=true \
    FRONTEND_URL=auto

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f "http://127.0.0.1:${PORT:-8000}/health" || exit 1

CMD ["/bin/sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
