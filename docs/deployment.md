# Deployment Guide

This guide covers deploying RAGify to production environments, including Docker deployment, scaling strategies, monitoring, and security considerations.

## Quick Deployment Options

### Docker Compose (Recommended)

The fastest way to deploy RAGify is using Docker Compose:

```yaml
# docker-compose.yml
version: "3.8"

services:
  ragify-backend:
    build: .
    ports:
      - "8000:8000"

  ragify-frontend:
    build: ./frontend
    ports:
      - "5173:5173"
    environment:
      - DATABASE_URL=postgresql://user:password@postgres/ragify
      - REDIS_URL=redis://redis:6379
      - SECRET_KEY=your-production-secret-key
    depends_on:
      - postgres
      - redis
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=ragify
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

```bash
# Deploy
docker-compose up -d

# View logs
docker-compose logs -f

# Scale services
docker-compose up -d --scale ragify=3
```

### Single Docker Container

For simpler deployments:

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Production Configuration

### Environment Variables

```bash
# Application Settings
DEBUG=false
LOG_LEVEL=WARNING
SECRET_KEY=your-256-bit-secret-key-here

# Database
DATABASE_URL=postgresql://user:password@host:5432/db
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30

# Redis
REDIS_URL=redis://host:6379
REDIS_DB=0

# Security
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
SESSION_COOKIE_SECURE=true
SESSION_COOKIE_HTTPONLY=true

# File Upload
MAX_UPLOAD_SIZE=100MB
UPLOAD_PATH=/app/uploads

# OpenRouter API Key (Primary AI Provider)
OPENROUTER_API_KEY=sk-or-v1-your-production-openrouter-key

# Rate Limiting
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=3600

# Monitoring
SENTRY_DSN=https://your-sentry-dsn
PROMETHEUS_ENABLED=true
```

### SSL/TLS Configuration

#### Using Nginx as Reverse Proxy

```nginx
# nginx.conf
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    # SSL Configuration
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    # API endpoints
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Frontend (adjust the upstream port if running the Docker stack)
    location / {
        proxy_pass http://localhost:5173;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### Using Caddy (Simpler)

```caddyfile
# Caddyfile
yourdomain.com {
    reverse_proxy localhost:8000
    # Replace 5173 with 15173 if you expose the frontend via Docker
    reverse_proxy localhost:5173

    # Automatic HTTPS
    tls your-email@example.com

    # Security headers
    header {
        X-Frame-Options DENY
        X-Content-Type-Options nosniff
        X-XSS-Protection "1; mode=block"
        Strict-Transport-Security "max-age=31536000; includeSubDomains"
    }
}
```

## Database Setup

### PostgreSQL Production Configuration

```sql
-- Create production database
CREATE DATABASE ragify_prod;
CREATE USER ragify_prod_user WITH PASSWORD 'strong_password';
GRANT ALL PRIVILEGES ON DATABASE ragify_prod TO ragify_prod_user;

-- Enable extensions
\c ragify_prod
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "vector";  -- For pgvector if used
```

### Connection Pooling

```python
# backend/core/database.py
from sqlalchemy.pool import QueuePool

engine = create_async_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,          # Number of connections to keep open
    max_overflow=30,       # Additional connections allowed
    pool_timeout=30,       # Timeout for getting connection from pool
    pool_recycle=3600,     # Recycle connections after 1 hour
    pool_pre_ping=True     # Check connection health
)
```

### Database Backup Strategy

```bash
# Daily backup script
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/ragify_$DATE.sql"

# Create backup
pg_dump -h localhost -U ragify_prod_user -d ragify_prod > "$BACKUP_FILE"

# Compress
gzip "$BACKUP_FILE"

# Keep only last 7 days
find "$BACKUP_DIR" -name "*.sql.gz" -mtime +7 -delete

# Upload to cloud storage (optional)
# aws s3 cp "$BACKUP_FILE.gz" s3://your-backup-bucket/
```

## Scaling Strategies

### Horizontal Scaling

#### Load Balancer Configuration

```nginx
# nginx.conf (load balancer)
upstream ragify_backend {
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}

upstream ragify_frontend {
    server frontend1:5173;
    server frontend2:5173;
}

server {
    listen 80;
    server_name yourdomain.com;

    location /api/ {
        proxy_pass http://ragify_backend;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location / {
        proxy_pass http://ragify_frontend;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

#### Database Scaling

```yaml
# docker-compose.yml (scaled database)
services:
  postgres:
    image: postgres:15
    deploy:
      replicas: 1
      resources:
        limits:
          cpus: "2.0"
          memory: 4G
    environment:
      - POSTGRES_DB=ragify
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password

  pgpool:
    image: pgpool/pgpool:4.3
    depends_on:
      - postgres
    ports:
      - "5432:5432"
    environment:
      - PGPOOL_BACKEND_NODES=0:postgres:5432
      - PGPOOL_PASSWORD=password
```

### Vertical Scaling

```yaml
# docker-compose.yml (resource allocation)
services:
  ragify:
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4G
        reservations:
          cpus: "1.0"
          memory: 2G
```

## Monitoring and Observability

### Application Metrics

```python
# backend/core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Request metrics
REQUEST_COUNT = Counter('ragify_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('ragify_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('ragify_active_connections', 'Active connections')

# Business metrics
DOCUMENTS_PROCESSED = Counter('ragify_documents_processed_total', 'Documents processed')
CHAT_MESSAGES = Counter('ragify_chat_messages_total', 'Chat messages sent')
VECTOR_SEARCHES = Counter('ragify_vector_searches_total', 'Vector searches performed')
```

### Health Checks

```python
# backend/api/v1/endpoints/health.py
from fastapi import APIRouter
import psutil
import time

router = APIRouter()

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "uptime": time.time() - psutil.boot_time(),
        "memory": {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "percent": psutil.virtual_memory().percent
        },
        "cpu": {
            "percent": psutil.cpu_percent(interval=1)
        }
    }

@router.get("/health/detailed")
async def detailed_health_check():
    # Include database connectivity, cache status, etc.
    pass
```

### Logging Configuration

```python
# backend/core/logging.py
import logging
import logging.config
from pythonjsonlogger import jsonlogger

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'json',
            'level': 'INFO'
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': '/var/log/ragify/app.log',
            'formatter': 'json',
            'level': 'WARNING'
        }
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO'
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
```

### Monitoring Stack

```yaml
# docker-compose.monitoring.yml
version: "3.8"

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    volumes:
      - loki_data:/loki

  promtail:
    image: grafana/promtail:latest
    volumes:
      - /var/log:/var/log
      - ./monitoring/promtail.yml:/etc/promtail/config.yml
    command:
      - "--config.file=/etc/promtail/config.yml"

volumes:
  grafana_data:
  loki_data:
```

## Security Hardening

### Container Security

```dockerfile
# Dockerfile (security hardened)
FROM python:3.9-slim

# Install only necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory with correct permissions
WORKDIR /app
RUN chown -R appuser:appuser /app

# Copy and install with user
COPY --chown=appuser:appuser requirements.txt .
USER appuser

RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application
COPY --chown=appuser:appuser . .

# Remove unnecessary files
RUN find . -name "*.pyc" -delete && find . -name "__pycache__" -type d -exec rm -rf {} +

EXPOSE 8000

USER appuser
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Network Security

```bash
# Firewall configuration (Ubuntu)
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw --force enable
```

### Secrets Management

```python
# backend/core/secrets.py
import os
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

class SecretsManager:
    def __init__(self):
        self.credential = DefaultAzureCredential()
        self.client = SecretClient(
            vault_url="https://your-keyvault.vault.azure.net/",
            credential=self.credential
        )

    async def get_secret(self, name: str) -> str:
        secret = await self.client.get_secret(name)
        return secret.value
```

## Backup and Recovery

### Automated Backups

```bash
# /etc/cron.daily/ragify-backup
#!/bin/bash

BACKUP_DIR="/backups/ragify"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="ragify_$DATE"

# Database backup
docker exec ragify_postgres_1 pg_dump -U user ragify > "$BACKUP_DIR/$BACKUP_NAME.sql"

# Application data backup
tar -czf "$BACKUP_DIR/${BACKUP_NAME}_data.tar.gz" /app/data/

# Upload to cloud
aws s3 cp "$BACKUP_DIR/$BACKUP_NAME.sql" s3://your-backup-bucket/database/
aws s3 cp "$BACKUP_DIR/${BACKUP_NAME}_data.tar.gz" s3://your-backup-bucket/data/

# Cleanup old backups
find "$BACKUP_DIR" -name "*.sql" -mtime +7 -delete
find "$BACKUP_DIR" -name "*_data.tar.gz" -mtime +7 -delete
```

### Disaster Recovery

```bash
# Recovery script
#!/bin/bash

BACKUP_DATE="20240115_120000"

# Stop services
docker-compose down

# Restore database
docker-compose up -d postgres
sleep 30
docker exec -i ragify_postgres_1 psql -U user -d ragify < "/backups/ragify/ragify_$BACKUP_DATE.sql"

# Restore data
tar -xzf "/backups/ragify/ragify_${BACKUP_DATE}_data.tar.gz" -C /

# Start all services
docker-compose up -d
```

## Performance Optimization

### Caching Strategy

```python
# backend/core/cache.py
from redis.asyncio import Redis
import json
import hashlib

class CacheManager:
    def __init__(self, redis_url: str):
        self.redis = Redis.from_url(redis_url)

    async def get(self, key: str):
        data = await self.redis.get(key)
        return json.loads(data) if data else None

    async def set(self, key: str, value: any, ttl: int = 3600):
        await self.redis.set(key, json.dumps(value), ex=ttl)

    def _make_key(self, *args) -> str:
        key_string = "_".join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()
```

### Database Optimization

```sql
-- Performance indexes
CREATE INDEX CONCURRENTLY idx_documents_kb_id ON documents(knowledge_base_id);
CREATE INDEX CONCURRENTLY idx_documents_created_at ON documents(created_at DESC);
CREATE INDEX CONCURRENTLY idx_chat_messages_application_id ON chat_messages(application_id);
CREATE INDEX CONCURRENTLY idx_chat_messages_created_at ON chat_messages(created_at DESC);

-- Partitioning for large tables (if needed)
CREATE TABLE chat_messages_y2024m01 PARTITION OF chat_messages
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

### CDN Integration

```python
# backend/core/cdn.py
import boto3
from botocore.config import Config

class CDNManager:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            config=Config(region_name='us-east-1')
        )
        self.bucket = 'your-cdn-bucket'

    async def upload_file(self, file_path: str, key: str):
        await self.s3.upload_file(file_path, self.bucket, key)
        return f"https://{self.bucket}.s3.amazonaws.com/{key}"
```

## Troubleshooting Production Issues

### Common Issues

#### High Memory Usage

```bash
# Check memory usage
docker stats

# Check application memory
ps aux --sort=-%mem | head

# Restart with memory limits
docker-compose up -d --scale ragify=1
```

#### Database Connection Pool Exhaustion

```python
# Monitor connection pool
from sqlalchemy import text

async def check_pool_status(db):
    result = await db.execute(text("SELECT count(*) FROM pg_stat_activity"))
    return result.scalar()
```

#### Slow API Responses

```bash
# Enable query logging
# In postgresql.conf
log_statement = 'all'
log_duration = on

# Check slow queries
SELECT query, total_time, calls
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;
```

### Log Analysis

```bash
# Search for errors
grep "ERROR" /var/log/ragify/app.log | tail -20

# Monitor response times
grep "process_time" /var/log/ragify/app.log | awk '{print $NF}' | sort -n

# Check for failed requests
grep "status_code.*[45][0-9][0-9]" /var/log/ragify/app.log
```

## Cost Optimization

### Resource Optimization

```yaml
# docker-compose.yml (cost optimized)
services:
  ragify:
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 2G
        reservations:
          cpus: "0.5"
          memory: 1G
    environment:
      - MAX_WORKERS=2
      - DB_POOL_SIZE=10
```

### Auto-scaling

```yaml
# docker-compose.yml (auto-scaling)
services:
  ragify:
    deploy:
      replicas: 1
      resources:
        limits:
          cpus: "1.0"
          memory: 2G
    # Scale based on CPU usage
    # Use tools like Docker Swarm or Kubernetes for auto-scaling
```

---

**Next Steps**: Review the [setup guide](setup.md) for initial configuration or [API documentation](api.md) for integration details.
