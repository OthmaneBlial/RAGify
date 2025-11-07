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
    echo "1) SQLite (in-memory) - Fast, no setup needed (recommended for testing/Cloud Run)"
    echo "2) PostgreSQL - Full-featured, persistent (recommended for production)"
    read -p "Enter choice (1 or 2): " db_choice

    case $db_choice in
        1)
            export DATABASE_URL="sqlite+aiosqlite:///:memory:"
            echo "Using SQLite (in-memory) database"
            ;;
        2)
            export DATABASE_URL="postgresql+asyncpg://ragify:RagifyStrongPass2023@localhost/ragify"
            echo "Using PostgreSQL database"
            ;;
        *)
            echo "Invalid choice. Using SQLite (in-memory) as default."
            export DATABASE_URL="sqlite+aiosqlite:///:memory:"
            ;;
    esac
else
    echo "Using DATABASE_URL from environment: $DATABASE_URL"
fi

echo "Starting RAGify..."

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
