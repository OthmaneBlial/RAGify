from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import AsyncAdaptedQueuePool
from typing import AsyncGenerator, Any, Dict
from contextlib import asynccontextmanager
from .config import settings
from sqlalchemy import text

# Assuming pgvector is installed
try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    Vector = None

if Vector is None:
    raise ImportError("pgvector is required. Install pgvector and enable extension.")

# Optimized async engine with connection pooling
# Check if we're using PostgreSQL or SQLite
is_postgres = "postgresql" in settings.database_url.lower()

connect_args = {}
if is_postgres:
    connect_args = {
        "server_settings": {
            "jit": "off",  # Disable JIT for better performance with embeddings
            "work_mem": "64MB",  # Increase work memory for complex queries
            "maintenance_work_mem": "256MB",  # Increase maintenance work memory
        }
    }

engine = create_async_engine(
    settings.database_url,
    echo=False,  # Disable for production performance
    future=True,
    poolclass=AsyncAdaptedQueuePool,
    pool_pre_ping=True,  # Verify connections before use
    pool_size=10,  # Base pool size
    max_overflow=20,  # Maximum overflow connections
    pool_timeout=30,  # Connection timeout
    pool_recycle=3600,  # Recycle connections after 1 hour
    connect_args=connect_args,
)

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session with optimized connection handling."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


@asynccontextmanager
async def get_db_session():
    """Context manager for database sessions with transaction support."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def bulk_insert_embeddings(session: AsyncSession, embeddings_data):
    if not embeddings_data:
        return 0
    from sqlalchemy import text

    # Convert vectors to PostgreSQL vector string format
    vectors = [
        "[" + ",".join(str(x) for x in e["vector"]) + "]" for e in embeddings_data
    ]
    para_ids = [e["paragraph_id"] for e in embeddings_data]
    q = text(
        """
      WITH data AS (
        SELECT unnest(:vectors)::vector, unnest(:pids)::uuid
      )
      INSERT INTO embeddings (vector, paragraph_id)
      SELECT * FROM data
      ON CONFLICT (paragraph_id) DO NOTHING
    """
    )
    res = await session.execute(q, {"vectors": vectors, "pids": para_ids})
    return res.rowcount or 0


async def execute_optimized_query(
    session: AsyncSession, query: str, params: Dict[str, Any] = None
) -> Any:
    """
    Execute query with optimization hints.

    Args:
        session: Database session
        query: SQL query string
        params: Query parameters

    Returns:
        Query result
    """
    from sqlalchemy import text

    # Add optimization hints for embedding queries
    if "embeddings" in query.lower() and "<->" in query:
        # Add index hint for vector similarity search
        optimized_query = f"SET LOCAL enable_seqscan = off; {query}"
    else:
        optimized_query = query

    result = await session.execute(text(optimized_query), params or {})
    return result


async def get_connection_stats() -> Dict[str, Any]:
    """Get database connection pool statistics."""
    try:
        pool = engine.pool
        return {
            "pool_size": pool.size(),
            "checkedin": pool.checkedin(),
            "checkedout": pool.checkedout(),
            "overflow": pool.overflow(),
            "total_connections": pool.size() + pool.overflow(),
        }
    except AttributeError:
        # Handle case where engine might be an async generator (test environment)
        return {
            "pool_size": 0,
            "checkedin": 0,
            "checkedout": 0,
            "overflow": 0,
            "total_connections": 0,
        }


async def warmup_connections():
    """Warm up database connections for better initial performance."""
    connections = []
    for _ in range(min(5, engine.pool.size())):
        conn = await engine.connect()
        connections.append(conn)

    # Execute a simple query to establish connections
    for conn in connections:
        await conn.execute(text("SELECT 1"))

    # Close connections
    for conn in connections:
        await conn.close()
