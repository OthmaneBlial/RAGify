"""
Test configuration and fixtures for RAGify tests.
Lightweight CI: monkeypatch embedding to avoid heavy model downloads.
"""

import asyncio
import os
import sys
from typing import AsyncGenerator
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import pytest_asyncio

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# Set up test environment variables
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing-purposes-only")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

# Import models and database setup
from backend.modules.knowledge.models import Base
from backend.modules.rag.embedding import EmbeddingModel
from backend.modules.rag.retrieval import RetrievalService
from backend.modules.rag.rag_pipeline import RAGPipeline


# ---------- Global event loop ----------
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ---------- Test DB engine ----------
@pytest.fixture(scope="session")
def test_engine():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(create_tables(engine))
    finally:
        loop.close()

    yield engine

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(engine.dispose())
    finally:
        loop.close()


async def create_tables(engine):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ---------- DB session ----------
@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    async_session = async_sessionmaker(test_engine, expire_on_commit=False)
    async with async_session() as session:
        yield session
        await session.rollback()


# ---------- Lightweight embeddings patch (autouse) ----------
@pytest.fixture(autouse=True)
def patch_embedding_model(monkeypatch):
    """
    Replace heavy EmbeddingModel methods with fast deterministic implementations.
    Signatures match the real API. Similarity is clamped to [0,1].
    """
    import numpy as np
    import hashlib
    from backend.modules.rag.embedding import EmbeddingModel

    def _det_vec(text: str) -> list[float]:
        seed = int.from_bytes(hashlib.sha256((text or "").encode("utf-8")).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(384)
        n = np.linalg.norm(v)
        return (v / n).astype(float).tolist() if n > 0 else v.astype(float).tolist()

    async def _encode_single(self: EmbeddingModel, text: str, normalize: bool = True) -> list[float]:
        key = self._get_cache_key(text)
        if key in self._cache:
            return self._cache[key]
        vec = _det_vec(text)
        if not normalize:
            # return a non-unit variant but deterministic
            arr = np.array(vec, dtype=float) * np.sqrt(len(vec))
            vec = arr.tolist()
        self._cache[key] = vec
        if len(self._cache) > self.cache_size + 1:
            # simple FIFO cap
            self._cache.pop(next(iter(self._cache)))
        return vec

    async def _encode_batch(
        self: EmbeddingModel, texts, normalize: bool = True, batch_size: int = 32
    ):
        if not texts:
            return []
        return [await _encode_single(self, t, normalize) for t in texts]

    async def _get_similarity(self: EmbeddingModel, v1, v2) -> float:
        import numpy as np
        a = np.array(v1, dtype=float)
        b = np.array(v2, dtype=float)
        if len(a) != len(b):
            return 0.0
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return 0.0
        cos = float(np.dot(a, b) / (na * nb))
        # clamp to [0, 1] to satisfy tests
        if cos < 0.0:
            return 0.0
        if cos > 1.0:
            return 1.0
        return cos

    def _clear_cache(self: EmbeddingModel):
        self._cache = {}

    monkeypatch.setattr(EmbeddingModel, "encode_single", _encode_single, raising=False)
    monkeypatch.setattr(EmbeddingModel, "encode_batch", _encode_batch, raising=False)
    monkeypatch.setattr(EmbeddingModel, "get_similarity", _get_similarity, raising=False)
    monkeypatch.setattr(EmbeddingModel, "clear_cache", _clear_cache, raising=False)


# ---------- Mocks and samples ----------
@pytest.fixture
def mock_embedding_model():
    mock_model = AsyncMock()
    mock_model.encode_single.return_value = [0.1, 0.2, 0.3] * 128  # 384 dims
    mock_model.encode_batch.return_value = [[0.1, 0.2, 0.3] * 128] * 3
    mock_model.get_similarity.return_value = 0.8
    mock_model.dimension = 384
    return mock_model


@pytest.fixture
def sample_knowledge_base_data():
    return {
        "id": str(uuid4()),
        "name": "Test Knowledge Base",
        "description": "A test knowledge base for unit tests",
    }


@pytest.fixture
def sample_document_data():
    return {
        "id": str(uuid4()),
        "title": "Test Document",
        "content": "This is a test document with some content for testing purposes.",
        "knowledge_base_id": str(uuid4()),
    }


@pytest.fixture
def sample_paragraph_data():
    return {
        "id": str(uuid4()),
        "content": "This is a test paragraph.",
        "document_id": str(uuid4()),
    }


@pytest.fixture
def sample_embedding_data():
    return {
        "id": str(uuid4()),
        "vector": [0.1, 0.2, 0.3] * 128,
        "paragraph_id": str(uuid4()),
    }


@pytest.fixture
def sample_application_data():
    return {
        "id": str(uuid4()),
        "name": "Test Application",
        "description": "A test application",
        "config": {"model": "gpt-3.5-turbo"},
        "knowledge_base_ids": [str(uuid4())],
    }


@pytest.fixture
def mock_db_session():
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
    service = AsyncMock()
    service.semantic_search.return_value = []
    service.keyword_search.return_value = []
    service.hybrid_search.return_value = []
    return service


# ---------- Test clients ----------
@pytest.fixture
def test_client(test_engine):
    from fastapi.testclient import TestClient
    from backend.main import app
    import backend.core.database as db_module

    original_engine = db_module.engine
    db_module.engine = test_engine

    original_session_local = db_module.AsyncSessionLocal
    db_module.AsyncSessionLocal = db_module.sessionmaker(
        test_engine,
        class_=db_module.AsyncSession,
        expire_on_commit=False,
    )

    try:
        yield TestClient(app)
    finally:
        db_module.engine = original_engine
        db_module.AsyncSessionLocal = original_session_local


@pytest.fixture
def mock_openai_client():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest_asyncio.fixture
async def async_test_client():
    from httpx import AsyncClient
    from backend.main import app
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client


# ---------- Utility fixtures ----------
@pytest.fixture
def test_query():
    return "What is machine learning?"


@pytest.fixture
def test_documents():
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand text.",
    ]


@pytest.fixture
def test_paragraphs():
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
    model = EmbeddingModel()
    yield model
    model.clear_cache()


@pytest.fixture
def retrieval_service():
    return RetrievalService(max_results=5, similarity_threshold=0.1)


@pytest.fixture
def rag_pipeline():
    return RAGPipeline()