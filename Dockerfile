# syntax=docker/dockerfile:1.6

ARG BASE_IMAGE=ragify-backend-base:latest
FROM ${BASE_IMAGE} AS runtime

USER root
WORKDIR /app

# Copy application source code
COPY pyproject.toml README.md ./
COPY backend ./backend
COPY shared ./shared
COPY frontend ./frontend

# Install the project using existing dependencies from the base image
RUN pip install --no-cache-dir --no-deps .

RUN chown -R ragify:ragify /app
USER ragify

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
