# syntax=docker/dockerfile:1.6

ARG BASE_IMAGE=ragify-backend-base:latest

FROM ${BASE_IMAGE} AS runtime

USER root
WORKDIR /app

COPY pyproject.toml README.md ./
COPY backend ./backend
COPY shared ./shared
COPY frontend ./frontend

RUN pip install --no-cache-dir --no-deps .

RUN chown -R ragify:ragify /app
USER ragify

ENV PORT=8000 \
    SERVE_FRONTEND_BUILD=true \
    FRONTEND_URL=auto

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f "http://127.0.0.1:${PORT:-8000}/health" || exit 1

CMD ["/bin/sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
