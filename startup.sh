#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
  source "$SCRIPT_DIR/venv/bin/activate"
fi

# Kill processes already using ports 8000 and 5173
for port in 8000 5173; do
  pid=$(lsof -t -i :$port || true)
  if [ -n "$pid" ]; then
    echo "Port $port already in use by PID $pid. Killing..."
    kill -9 $pid
  fi
done

# Database selection
if [ -z "${DATABASE_URL:-}" ]; then
    echo "Choose database type:"
    echo "1) SQLite (file-based) - Fast, no setup needed (recommended for testing/Cloud Run)"
    echo "2) PostgreSQL - Full-featured, persistent (recommended for production)"
    read -p "Enter choice (1 or 2): " db_choice

    case $db_choice in
        1)
            export DATABASE_URL="sqlite+aiosqlite:///./ragify.db"
            echo "Using SQLite (file-based) database at ./ragify.db"
            if [ -f "$SCRIPT_DIR/ragify.db" ]; then
                echo "Removing stale SQLite database file..."
                rm -f "$SCRIPT_DIR/ragify.db"
            fi
            ;;
        2)
            export DATABASE_URL="postgresql+asyncpg://ragify:RagifyStrongPass2023@localhost/ragify"
            echo "Using PostgreSQL database"
            ;;
        *)
            echo "Invalid choice. Using SQLite (file-based) as default."
            export DATABASE_URL="sqlite+aiosqlite:///./ragify.db"
            ;;
    esac
else
    echo "Using DATABASE_URL from environment: $DATABASE_URL"
fi

is_sqlite=0
if [[ "$DATABASE_URL" == sqlite+aiosqlite* ]]; then
    is_sqlite=1
fi

if (( is_sqlite )); then
    export REDIS_URL=""
    echo "SQLite selected; Redis caching disabled (REDIS_URL cleared)."
else
    if [ -z "${REDIS_URL:-}" ]; then
        if python3 - <<'PY' >/dev/null 2>&1
import socket, sys
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.5)
try:
    sock.connect(("127.0.0.1", 6379))
    sock.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
        then
            export REDIS_URL="redis://localhost:6379"
        else
            export REDIS_URL=""
            echo "Redis not reachable; disabling cache (REDIS_URL set to empty)"
        fi
    else
        echo "Using REDIS_URL from environment: $REDIS_URL"
    fi
fi

echo "Starting RAGify..."

echo "Ensuring database schema and default application exist..."
PYTHONPATH=/home/othmane/projects/RAGify python3 -c "
import asyncio, sys
sys.path.insert(0, '/home/othmane/projects/RAGify')

from backend.core.database import engine, get_db_session
from backend.modules.knowledge.models import Base
from backend.modules.applications.crud import create_application, list_applications

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        print('Tables created for current database')

    async with get_db_session() as db:
        apps = await list_applications(db)
        if not apps:
            await create_application(
                db=db,
                name='Default Chat Application',
                description='Default application for chat functionality',
                config={'provider': 'openrouter', 'model': 'openai/gpt-5-nano'},
                knowledge_base_ids=[]
            )
            print('Default application created')
        else:
            print('Default application already exists')

asyncio.run(init_db())
"

if [[ "$DATABASE_URL" == postgresql+asyncpg* ]]; then
    echo "Ensuring PostgreSQL vector schema/extension..."
    python3 - <<'PY'
import asyncio, os
import asyncpg

dsn = os.environ["DATABASE_URL"].replace("postgresql+asyncpg", "postgresql")

async def prepare_vector():
    conn = await asyncpg.connect(dsn=dsn)
    try:
        await conn.execute("CREATE SCHEMA IF NOT EXISTS text")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA text")
        print("Vector extension ensured in schema 'text'")
    finally:
        await conn.close()

asyncio.run(prepare_vector())
PY

fi

# Start backend
echo "Starting backend (FastAPI) on port 8000..."
SERVE_FRONTEND_BUILD=true PYTHONPATH=/home/othmane/projects/RAGify uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "Backend started with PID $BACKEND_PID"

# Start frontend (simple HTTP server for static files)
echo "Starting frontend (HTTP server) on port 5173..."
cd frontend && python3 -m http.server 5173 &
FRONTEND_PID=$!
cd ..
echo "Frontend started with PID $FRONTEND_PID"

# Function to stop services
stop_services() {
    echo "Stopping services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo "Services stopped."
    exit 0
}

trap stop_services SIGINT

echo "Both services are running. Press Ctrl+C to stop."

wait $BACKEND_PID
wait $FRONTEND_PID
