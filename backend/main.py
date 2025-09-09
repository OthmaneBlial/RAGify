from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import time
import logging
from contextlib import asynccontextmanager
from .core.database import engine, warmup_connections
from .core.async_tasks import task_manager
from .modules.knowledge.models import Base
from .modules.applications.models import ChatMessage
from .modules.models.model_manager import model_manager
from .api.v1 import api_router

logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create tables and warmup connections
    logger.info("Starting RAGify application...")
    async with engine.begin() as conn:
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


@app.get("/")
@limiter.limit("100/minute")  # Rate limit root endpoint
async def root(request: Request):
    return {"message": "Welcome to RAGify"}


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
