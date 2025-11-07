# Setup Guide

This guide provides detailed instructions for setting up RAGify from scratch, including prerequisites, environment configuration, database setup, and troubleshooting common issues.

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows 10+
- **Python**: Version 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: At least 2GB free space for models and data
- **Network**: Internet connection for downloading dependencies

### Required Software

- **Python 3.8+** – main runtime
- **SQLite** – ships with Python, used by default
- **Optional**: PostgreSQL 12+ (if you need pgvector/permanent DB)
- **Optional**: Redis 6+ (for distributed caching; the app auto-falls back to an in-memory cache if Redis is unavailable)

## Environment Configuration

### 1. Clone the Repository

```bash
git clone <repository-url>
cd RAGify
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install the package in development mode
pip install -e .

# Or install dependencies manually
pip install -r requirements.txt
```

### 4. Environment Variables Setup

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit the `.env` file with your configuration:

```env
# Database Configuration
# Default: local SQLite file (no external service required)
DATABASE_URL=sqlite+aiosqlite:///./ragify.db

# To use PostgreSQL instead, replace the line above with:
# DATABASE_URL=postgresql+asyncpg://username:password@localhost:5432/ragify

# Security Settings
SECRET_KEY=your-super-secret-key-change-this-in-production
DEBUG=true

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379

# OpenRouter API Key (Primary AI Provider)
OPENROUTER_API_KEY=sk-or-v1-your-openrouter-api-key-here

# Default Model Settings
DEFAULT_MODEL=openai/gpt-4o-mini
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=4096

# Application Settings
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE=100MB
EMBEDDING_MODEL=all-MiniLM-L6-v2

# CORS Settings
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:15173,http://localhost:3000

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

## Database Setup

SQLite works out of the box—no setup required. Just run `./startup.sh` and pick option 1.

If you prefer PostgreSQL + pgvector:

1. **Create database and user**
    ```sql
    sudo -u postgres psql
    CREATE DATABASE ragify;
    CREATE USER ragify_user WITH PASSWORD 'your_secure_password';
    GRANT ALL PRIVILEGES ON DATABASE ragify TO ragify_user;
    \q
    ```
2. **Verify connection**
    ```bash
    psql -h localhost -U ragify_user -d ragify
    ```
3. **Enable pgvector**
    ```sql
    psql -d ragify -c "CREATE EXTENSION IF NOT EXISTS vector;"
    ```
4. Update `DATABASE_URL` in `.env`, then run `./startup.sh` and choose option 2 (PostgreSQL).

Tables are created automatically on startup; Alembic migrations remain available if you need manual control.

## OpenRouter API Configuration

RAGify uses OpenRouter as the primary AI provider, giving you access to 100+ models through a single API.

### Get Your OpenRouter API Key

1. Visit [OpenRouter](https://openrouter.ai/)
2. Create an account or sign up
3. Navigate to the API Keys section
4. Create a new API key
5. Add to your `.env` file: `OPENROUTER_API_KEY=sk-or-v1-your-key-here`

### Available Models

OpenRouter provides access to models from:
- **OpenAI**: GPT-4, GPT-3.5 Turbo, GPT-4o, etc.
- **Anthropic**: Claude-3, Claude-2, etc.
- **Google**: Gemini models
- **Meta**: Llama models
- **And many more providers**

You can specify any supported model in your application configuration using the model identifier (e.g., `openai/gpt-4o-mini`, `anthropic/claude-3-haiku`).

## Running the Application

### Docker Quick Start (Optional)

If you prefer containers, the project ships with a helper script that builds a reusable base image and launches the full stack (backend, frontend, PostgreSQL with pgvector, and Redis).

```bash
# Build or refresh the dependency base image and start all services
./startupdocker.sh --build

# Follow logs during startup (optional)
./startupdocker.sh --build --logs

# Stop the containers when you're done
./startupdocker.sh --down
```

The Docker stack publishes the following ports on the host:

- Backend API: `http://localhost:18000`
- Frontend UI: `http://localhost:15173`
- PostgreSQL: `localhost:15432` (user/password `ragify` / `RagifyStrongPass2023`)
- Redis: `localhost:16379`

The first `--build` run creates the `ragify-backend-base` image that caches all heavy Python dependencies such as PyTorch and transformers. Subsequent builds reuse that base layer, so only application code changes trigger rebuilds. Environment variables are loaded from the project’s `.env` file inside the containers.

### Development Mode

```bash
# Start backend server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, start frontend
cd frontend && npm run dev
```

### Production Mode

```bash
# Using uvicorn with workers
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4

# Or using gunicorn
gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Verification

### 1. Health Check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "timestamp": 1234567890.123,
  "database": {...},
  "cache": {...},
  "task_queue": {...}
}
```

### 2. API Documentation

Visit: http://localhost:8000/docs

### 3. Frontend Interface

Visit: `http://localhost:5173` (local dev) or `http://localhost:15173` when using the Docker stack.

## Troubleshooting

### Common Issues

#### 1. Database Connection Failed

**Error**: `psycopg2.OperationalError: could not connect to server`

**Solutions**:

- Ensure PostgreSQL is running: `sudo systemctl status postgresql`
- Check database credentials in `.env`
- Verify database exists: `psql -l`
- Check PostgreSQL logs: `sudo tail -f /var/log/postgresql/postgresql-*.log`

#### 2. Import Errors

**Error**: `ModuleNotFoundError: No module named 'sentence_transformers'`

**Solutions**:

- Install dependencies: `pip install -e .`
- Activate virtual environment: `source venv/bin/activate`
- Check Python path: `python -c "import sys; print(sys.path)"`

#### 3. Port Already in Use

**Error**: `[Errno 48] Address already in use`

**Solutions**:

- Kill process using port: `lsof -ti:8000 | xargs kill -9`
- Change port in command: `--port 8001`
- Check what's using the port: `lsof -i :8000`

#### 4. Embedding Model Download Failed

**Error**: `ConnectionError: Couldn't reach server`

**Solutions**:

- Check internet connection
- Use different model: Set `EMBEDDING_MODEL` to a local model
- Download manually and place in cache directory

#### 5. Redis Connection Failed

**Error**: `redis.ConnectionError: Error 61 connecting to localhost:6379`

**Solutions**:

- Start Redis: `sudo systemctl start redis-server`
- Check Redis status: `redis-cli ping`
- Disable Redis in `.env`: Comment out `REDIS_URL`
- Install Redis if not present

### Performance Issues

#### High Memory Usage

- Reduce batch size in embedding configuration
- Use smaller embedding models
- Implement document chunking for large files

#### Slow Response Times

- Enable Redis caching
- Optimize database queries
- Use connection pooling
- Implement rate limiting

### Logs and Debugging

#### Enable Debug Logging

```env
LOG_LEVEL=DEBUG
DEBUG=true
```

#### View Application Logs

```bash
# Backend logs (when running with uvicorn)
uvicorn backend.main:app --log-level debug

# Frontend logs
streamlit run app.py --logger.level=debug
```

#### Database Query Logging

```python
# In backend/core/database.py
import logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

## Advanced Configuration

### Custom Embedding Models

```python
# In backend/modules/rag/embedding.py
from sentence_transformers import SentenceTransformer

# Load custom model
model = SentenceTransformer('path/to/your/model')
```

### Database Connection Pooling

```python
# In backend/core/database.py
from sqlalchemy.pool import QueuePool

engine = create_async_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)
```

### SSL Configuration

```bash
# Generate SSL certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

# Run with SSL
uvicorn backend.main:app --ssl-keyfile key.pem --ssl-certfile cert.pem
```

## Next Steps

After successful setup:

1. **Create your first knowledge base** in the web interface
2. **Upload documents** (PDF, DOCX, TXT)
3. **Build an application** and associate knowledge bases
4. **Test the chat functionality**
5. **Review the API documentation** for integration options

For production deployment, see the [deployment guide](deployment.md).
