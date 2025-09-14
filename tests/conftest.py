"""
Test configuration and fixtures for RAGify tests.
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, MagicMock
import sys
import os
from typing import AsyncGenerator
from uuid import uuid4

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import StaticPool

# Import models and database setup
from backend.modules.knowledge.models import Base
from backend.modules.rag.embedding import EmbeddingModel
from backend.modules.rag.retrieval import RetrievalService
from backend.modules.rag.rag_pipeline import RAGPipeline


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_engine():
    """Create a test database engine."""
    # Use SQLite for testing
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

    # Create all tables synchronously
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(create_tables(engine))
    finally:
        loop.close()

    yield engine
    # Dispose synchronously
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(engine.dispose())
    finally:
        loop.close()


async def create_tables(engine):
    """Create all database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    async_session = async_sessionmaker(test_engine, expire_on_commit=False)

    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for testing."""
    mock_model = AsyncMock()
    mock_model.encode_single.return_value = [0.1, 0.2, 0.3] * 128  # 384-dim vector
    mock_model.encode_batch.return_value = [[0.1, 0.2, 0.3] * 128] * 3
    mock_model.get_similarity.return_value = 0.8
    mock_model.dimension = 384
    return mock_model


@pytest.fixture
def sample_knowledge_base_data():
    """Sample knowledge base data for testing."""
    return {
        "id": str(uuid4()),
        "name": "Test Knowledge Base",
        "description": "A test knowledge base for unit tests",
    }


@pytest.fixture
def sample_document_data():
    """Sample document data for testing."""
    return {
        "id": str(uuid4()),
        "title": "Test Document",
        "content": "This is a test document with some content for testing purposes.",
        "knowledge_base_id": str(uuid4()),
    }


@pytest.fixture
def sample_paragraph_data():
    """Sample paragraph data for testing."""
    return {
        "id": str(uuid4()),
        "content": "This is a test paragraph.",
        "document_id": str(uuid4()),
    }


@pytest.fixture
def sample_embedding_data():
    """Sample embedding data for testing."""
    return {
        "id": str(uuid4()),
        "vector": [0.1, 0.2, 0.3] * 128,  # 384-dim vector
        "paragraph_id": str(uuid4()),
    }


@pytest.fixture
def sample_application_data():
    """Sample application data for testing."""
    return {
        "id": str(uuid4()),
        "name": "Test Application",
        "description": "A test application",
        "config": {"model": "gpt-3.5-turbo"},
        "knowledge_base_ids": [str(uuid4())],
    }


@pytest.fixture
def mock_db_session():
    """Mock database session for testing."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.rollback = AsyncMock()
    session.execute = AsyncMock()
    session.scalar_one_or_none = AsyncMock()
    session.scalars = AsyncMock()
    session.add = AsyncMock()
    return session


@pytest.fixture
def mock_retrieval_service():
    """Mock retrieval service for testing."""
    service = AsyncMock()
    service.semantic_search.return_value = []
    service.keyword_search.return_value = []
    service.hybrid_search.return_value = []
    return service


@pytest.fixture
def test_client(test_engine):
    """FastAPI test client fixture."""
    from fastapi.testclient import TestClient
    from backend.main import app
    import backend.core.database as db_module

    # Override the database engine for testing
    original_engine = db_module.engine
    db_module.engine = test_engine

    # Also override the session makers
    original_session_local = db_module.AsyncSessionLocal
    db_module.AsyncSessionLocal = db_module.sessionmaker(
        test_engine,
        class_=db_module.AsyncSession,
        expire_on_commit=False,
    )

    try:
        yield TestClient(app)
    finally:
        # Restore original engine and session maker
        db_module.engine = original_engine
        db_module.AsyncSessionLocal = original_session_local


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_sentence_transformers():
    """Mock sentence transformers for testing."""
    with pytest.mock.patch("sentence_transformers.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3] * 128
        mock_st.return_value = mock_model
        yield mock_st


@pytest.fixture
async def async_test_client():
    """Async test client for FastAPI."""
    from httpx import AsyncClient
    from backend.main import app

    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client


# Utility fixtures for common test data
@pytest.fixture
def test_query():
    """Sample test query."""
    return "What is machine learning?"


@pytest.fixture
def test_documents():
    """Sample test documents."""
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand text.",
    ]


@pytest.fixture
def test_paragraphs():
    """Sample test paragraphs."""
    return [
        {
            "content": "Machine learning algorithms learn from data.",
            "document_id": str(uuid4()),
        },
        {
            "content": "Neural networks are inspired by the human brain.",
            "document_id": str(uuid4()),
        },
        {
            "content": "Data preprocessing is crucial for model performance.",
            "document_id": str(uuid4()),
        },
    ]


@pytest.fixture
async def embedding_model():
    """Real embedding model fixture."""
    model = EmbeddingModel()
    yield model
    # Clean up cache after test
    model.clear_cache()


@pytest.fixture
def retrieval_service():
    """Retrieval service fixture."""
    return RetrievalService(max_results=5, similarity_threshold=0.1)


@pytest.fixture
def rag_pipeline():
    """RAG pipeline fixture."""
    return RAGPipeline()
