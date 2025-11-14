import json
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import AsyncAdaptedQueuePool
from typing import AsyncGenerator, Any, Dict
from contextlib import asynccontextmanager
from uuid import UUID
from .config import settings
from sqlalchemy import text

# Database type detection and vector support
is_sqlite = "sqlite" in settings.database_url.lower()
is_postgres = "postgresql" in settings.database_url.lower()

# Vector support - only required for PostgreSQL
Vector = None
if is_postgres:
    try:
        from pgvector.sqlalchemy import Vector
    except ImportError:
        Vector = None

    if Vector is None:
        raise ImportError("pgvector is required for PostgreSQL. Install pgvector and enable extension.")
elif is_sqlite:
    # For SQLite, we'll use a simple list-based vector storage
    pass


def normalize_uuid(value):
    """
    Coerce UUID/string identifiers into the database-friendly representation.

    SQLite stores UUID columns as strings, while PostgreSQL expects UUID objects.
    """
    if value is None:
        return None

    if is_sqlite:
        if isinstance(value, UUID):
            return str(value)
        return str(value)

    if isinstance(value, UUID):
        return value

    return UUID(str(value))

# Optimized async engine with connection pooling
connect_args = {}
poolclass = AsyncAdaptedQueuePool

if is_postgres:
    connect_args = {
        "server_settings": {
            "jit": "off",  # Disable JIT for better performance with embeddings
            "work_mem": "64MB",  # Increase work memory for complex queries
            "maintenance_work_mem": "256MB",  # Increase maintenance work memory
        }
    }
elif is_sqlite:
    # SQLite-specific configuration
    connect_args = {
        "check_same_thread": False,  # Allow multi-threaded access
    }
    # Use StaticPool for SQLite to avoid connection pool issues
    from sqlalchemy.pool import StaticPool
    poolclass = StaticPool

# Event listener to enable foreign keys for SQLite
from sqlalchemy import event

if is_sqlite:
    engine = create_async_engine(
        settings.database_url,
        echo=False,  # Disable for production performance
        future=True,
        poolclass=poolclass,
        connect_args=connect_args,
    )
else:
    engine = create_async_engine(
        settings.database_url,
        echo=False,  # Disable for production performance
        future=True,
        poolclass=poolclass,
        pool_pre_ping=True,  # Verify connections before use
        pool_size=10,  # Base pool size
        max_overflow=20,  # Maximum overflow connections
        pool_timeout=30,  # Connection timeout
        pool_recycle=3600,  # Recycle connections after 1 hour
        connect_args=connect_args,
    )

# Register event listener after engine is created
@event.listens_for(engine.sync_engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """Enable foreign key constraints and performance optimizations for SQLite."""
    if is_sqlite:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for better concurrency
        cursor.execute("PRAGMA synchronous=NORMAL")  # Balance between speed and safety
        cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
        cursor.execute("PRAGMA temp_store=MEMORY")  # Store temp tables in memory
        cursor.close()

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

    if is_postgres:
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
    elif is_sqlite:
        # For SQLite, insert one by one since it doesn't have unnest or vector types
        inserted_count = 0
        for embedding in embeddings_data:
            vector_str = json.dumps(embedding["vector"])
            q = text(
                """
                INSERT OR IGNORE INTO embeddings (vector, paragraph_id)
                VALUES (:vector, :paragraph_id)
                """
            )
            res = await session.execute(q, {
                "vector": vector_str,
                "paragraph_id": embedding["paragraph_id"]
            })
            if res.rowcount and res.rowcount > 0:
                inserted_count += res.rowcount
        return inserted_count
    else:
        raise ValueError("Unsupported database type for bulk embedding insertion")


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
    if is_postgres and "embeddings" in query.lower() and "<->" in query:
        # Add index hint for vector similarity search (PostgreSQL only)
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

    # Handle different pool types
    if hasattr(engine.pool, 'size'):
        # AsyncAdaptedQueuePool (PostgreSQL)
        pool_size = engine.pool.size()
    else:
        # StaticPool (SQLite) - use a fixed small number
        pool_size = 1

    for _ in range(min(5, pool_size)):
        conn = await engine.connect()
        connections.append(conn)

    # Execute a simple query to establish connections
    for conn in connections:
        await conn.execute(text("SELECT 1"))

    # Close connections
    for conn in connections:
        await conn.close()
