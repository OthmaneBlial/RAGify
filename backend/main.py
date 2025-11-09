import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from .core.database import engine, warmup_connections, is_postgres
from .core.async_tasks import task_manager
from .modules.knowledge.models import Base
from .modules.applications.models import ChatMessage
from .modules.models.model_manager import model_manager
from .api.v1 import api_router

logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

BASE_DIR = Path(__file__).resolve().parent
POSSIBLE_FRONTEND_DIRS = [
    BASE_DIR.parent / "frontend" / "dist",
    BASE_DIR.parent / "frontend",
    BASE_DIR / "frontend_dist",
    Path("/app/frontend_dist"),
    Path("/app/frontend"),
]


def resolve_frontend_build_dir() -> Optional[Path]:
    for candidate in POSSIBLE_FRONTEND_DIRS:
        if candidate.is_dir():
            return candidate
    return None


FRONTEND_BUILD_DIR = resolve_frontend_build_dir()
FRONTEND_STATIC_DIR: Optional[Path] = None
if FRONTEND_BUILD_DIR:
    for static_candidate in ("assets", "static"):
        candidate = FRONTEND_BUILD_DIR / static_candidate
        if candidate.is_dir():
            FRONTEND_STATIC_DIR = candidate
            break
    if FRONTEND_STATIC_DIR is None:
        FRONTEND_STATIC_DIR = FRONTEND_BUILD_DIR


def str_to_bool(value: str) -> bool:
    return value.lower() not in {"0", "false", "no", "off"}


SERVE_FRONTEND_BUILD = str_to_bool(os.environ.get("SERVE_FRONTEND_BUILD", "false"))
FRONTEND_URL = os.environ.get("FRONTEND_URL", "auto")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create tables and warmup connections
    logger.info("Starting RAGify application...")
    async with engine.begin() as conn:
        # For PostgreSQL, create vector extension if needed
        if is_postgres:
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                logger.info("PostgreSQL vector extension ensured")
            except Exception as e:
                logger.warning(f"Could not create vector extension: {e}")

        await conn.run_sync(Base.metadata.create_all)
        # Create ChatMessage table
        await conn.run_sync(ChatMessage.__table__.create, checkfirst=True)

    # Warm up database connections
    await warmup_connections()

    # Create default application if none exist
    from .core.database import get_db_session
    from .modules.applications.crud import list_applications, create_application

    async with get_db_session() as db:
        applications = await list_applications(db)
        if not applications:
            logger.info("Creating default application...")
            await create_application(
                db=db,
                name="Default Chat Application",
                description="Default application for chat functionality",
                config={"provider": "openrouter", "model": "openai/gpt-5-nano"},
                knowledge_base_ids=[],
            )
            logger.info("Default application created")

    # Start async task manager
    await task_manager.start()

    # Initialize model manager
    await model_manager.initialize()

    logger.info("Application startup complete")
    yield

    # Shutdown: Clean up resources
    logger.info("Shutting down RAGify application...")
    await task_manager.stop()
    await engine.dispose()
    logger.info("Application shutdown complete")


app = FastAPI(title="RAGify", version="0.1.0", lifespan=lifespan)

if SERVE_FRONTEND_BUILD and FRONTEND_STATIC_DIR:
    # Expose hashed assets (JS/CSS) for the static frontend build
    app.mount("/static", StaticFiles(directory=str(FRONTEND_STATIC_DIR)), name="static")

# Performance middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)  # Compress responses > 1KB

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["*"]
)  # Configure for production

# Optimized CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:15173",
        "http://127.0.0.1:15173",
    ],  # Include new frontend port
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,  # Cache preflight for 24 hours
)

# Rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


# Request timing middleware
@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    start_time = time.time()

    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")

    response = await call_next(request)

    # Calculate and log processing time
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    logger.info(f"Processed in {process_time:.2f}s")

    return response


app.include_router(api_router)


def status_payload(request: Optional[Request]) -> Dict[str, Any]:
    if FRONTEND_URL and FRONTEND_URL.lower() != "auto":
        frontend_location = FRONTEND_URL
    else:
        base_url = str(request.base_url) if request else ""
        frontend_location = base_url.rstrip("/") if base_url else ""

    return {
        "message": "Welcome to RAGify",
        "status": "ok",
        "frontend": frontend_location,
    }


def serve_frontend(path: str, request: Request):
    if not (SERVE_FRONTEND_BUILD and FRONTEND_BUILD_DIR):
        if path:
            raise HTTPException(status_code=404, detail="Resource not found")
        return JSONResponse(status_payload(request))

    relative_path = Path(path or "index.html")
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise HTTPException(status_code=404, detail="Resource not found")

    candidate = FRONTEND_BUILD_DIR / relative_path
    if candidate.is_file():
        return FileResponse(candidate)

    index_path = FRONTEND_BUILD_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)

    return JSONResponse(status_payload(request))


@app.get("/", include_in_schema=False)
@limiter.limit("100/minute")  # Rate limit root endpoint
async def root(request: Request):
    if SERVE_FRONTEND_BUILD and FRONTEND_BUILD_DIR:
        return serve_frontend("", request)
    return status_payload(request)


@app.get("/api/status", include_in_schema=False)
async def api_status(request: Request):
    return JSONResponse(status_payload(request))


@app.get("/health")
@limiter.limit("60/minute")  # Rate limit health checks
async def health(request: Request):
    from .core.database import get_connection_stats
    from .core.cache import cache_manager

    # Get system health info
    db_stats = await get_connection_stats()
    cache_stats = await cache_manager.get_stats()

    return {
        "status": "ok",
        "timestamp": time.time(),
        "database": db_stats,
        "cache": cache_stats,
        "task_queue": {
            "active_tasks": len(task_manager.tasks),
            "queue_size": task_manager.get_queue_size(),
        },
    }


if SERVE_FRONTEND_BUILD:

    RESERVED_FRONTEND_PREFIXES = (
        "api",
        "docs",
        "redoc",
        "openapi.json",
        "health",
        "static",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    async def frontend_catch_all(full_path: str, request: Request):
        normalized = full_path.strip("/")
        for reserved in RESERVED_FRONTEND_PREFIXES:
            if normalized == reserved or normalized.startswith(f"{reserved}/"):
                raise HTTPException(status_code=404, detail="Resource not found")
        return serve_frontend(normalized, request)
