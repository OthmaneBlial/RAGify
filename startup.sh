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

echo "Starting RAGify..."

# Start backend
echo "Starting backend (FastAPI) on port 8000..."
PYTHONPATH=/home/othmane/projects/RAGify uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
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
