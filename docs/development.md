# Development Guide

This guide provides information for developers who want to contribute to RAGify, add new features, or understand the codebase architecture.

## Project Structure

```
RAGify/
├── backend/                    # FastAPI backend
│   ├── api/                    # API layer
│   │   ├── v1/
│   │   │   ├── endpoints/      # API endpoints
│   │   │   │   ├── chat.py     # Chat endpoints
│   │   │   │   ├── knowledge.py # Knowledge base endpoints
│   │   │   │   └── applications.py # Application endpoints
│   │   │   └── __init__.py
│   │   └── docs/               # API documentation
│   ├── core/                   # Core functionality
│   │   ├── database.py         # Database connection
│   │   ├── security.py         # Security utilities
│   │   └── async_tasks.py      # Background task management
│   ├── modules/                # Business logic modules
│   │   ├── applications/       # Application management
│   │   │   ├── crud.py         # Database operations
│   │   │   ├── models.py       # SQLAlchemy models
│   │   │   └── service.py      # Business logic
│   │   ├── knowledge/          # Knowledge base management
│   │   │   ├── crud.py
│   │   │   ├── models.py
│   │   │   └── processing.py   # Document processing
│   │   └── rag/                # RAG pipeline
│   │       ├── embedding.py    # Vector embeddings
│   │       ├── retrieval.py    # Document retrieval
│   │       └── rag_pipeline.py # Main RAG logic
│   └── main.py                 # FastAPI application
├── frontend/                   # Vite + HTML/JS frontend
│   ├── src/                    # Source code
│   │   ├── components/         # React/Vue components
│   │   ├── pages/              # Page components
│   │   ├── utils/              # Frontend utilities
│   │   └── api/                # API communication
│   ├── public/                 # Static assets
│   ├── package.json            # Dependencies
│   └── vite.config.js          # Vite configuration
├── shared/                     # Shared code
│   ├── models/                 # Pydantic models
│   │   ├── KnowledgeBase.py
│   │   ├── Document.py
│   │   └── ChatMessage.py
│   └── utils/                  # Shared utilities
│       └── text_processing.py
├── tests/                      # Test suite
│   ├── test_basic_structure.py
│   └── ...                     # Additional tests
├── docs/                       # Documentation
└── pyproject.toml              # Project configuration
```

## Architecture Overview

### Backend Architecture

RAGify follows a modular architecture with clear separation of concerns:

#### API Layer (`backend/api/`)

- **FastAPI routers** for REST endpoints
- **Pydantic models** for request/response validation
- **Dependency injection** for database sessions and authentication

#### Core Layer (`backend/core/`)

- **Database connection** management with SQLAlchemy
- **Security utilities** for authentication and authorization
- **Async task management** for background processing

#### Modules (`backend/modules/`)

- **Applications**: Chat application management
- **Knowledge**: Document and knowledge base operations
- **RAG**: Retrieval-Augmented Generation pipeline

### Frontend Architecture

- **Modern web frontend** built with Vite and vanilla JavaScript
- **SPA (Single Page Application)** with client-side routing
- **RESTful API integration** with the FastAPI backend
- **Responsive design** with modern CSS and JavaScript
- **Component-based architecture** for maintainable code

### Data Flow

```
User Request → Frontend → API Endpoint → Business Logic → Database/Cache → Response
```

## Development Environment Setup

### Prerequisites

1. **Python 3.8+** with virtual environment support
2. **PostgreSQL** for primary database
3. **Redis** for caching (optional)
4. **Git** for version control

### Local Development Setup

```bash
# Clone repository
git clone <repository-url>
cd RAGify

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Setup database
createdb ragify
# Run migrations if available

# Start development servers
# Terminal 1: Backend
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend && npm run dev
```

## Adding New Features

### 1. Planning

Before implementing a new feature:

1. **Define requirements** clearly
2. **Check existing functionality** to avoid duplication
3. **Design API endpoints** and data models
4. **Plan database schema** changes if needed
5. **Consider UI/UX implications**

### 2. Backend Development

#### Adding a New API Endpoint

```python
# backend/api/v1/endpoints/example.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from ...core.database import get_db

router = APIRouter()

@router.get("/example")
async def get_example(db: AsyncSession = Depends(get_db)):
    # Implementation
    return {"message": "Example endpoint"}
```

#### Adding Business Logic

```python
# backend/modules/example/service.py
from ...core.database import get_db
from .models import ExampleModel

class ExampleService:
    def __init__(self, db):
        self.db = db

    async def create_example(self, data):
        # Business logic implementation
        pass
```

#### Database Models

```python
# backend/modules/example/models.py
from sqlalchemy import Column, Integer, String
from ...core.database import Base

class ExampleModel(Base):
    __tablename__ = "examples"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
```

### 3. Frontend Development

#### Adding a New Page

```javascript
// frontend/src/pages/ExamplePage.js
import { apiClient } from '../utils/api.js';

export class ExamplePage {
    constructor() {
        this.container = document.createElement('div');
        this.init();
    }

    async init() {
        this.render();
        this.attachEventListeners();
    }

    render() {
        this.container.innerHTML = `
            <div class="example-page">
                <h1>Example Page</h1>
                <button id="load-data-btn">Load Data</button>
                <div id="data-container"></div>
            </div>
        `;
    }

    attachEventListeners() {
        const loadBtn = this.container.querySelector('#load-data-btn');
        loadBtn.addEventListener('click', () => this.loadData());
    }

    async loadData() {
        try {
            const data = await apiClient.get('/example');
            const container = this.container.querySelector('#data-container');
            container.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
        } catch (error) {
            console.error('Error loading data:', error);
        }
    }
}
```

#### Adding UI Components

```javascript
// frontend/src/components/ExampleComponent.js
export class ExampleComponent {
    constructor(data) {
        this.data = data;
        this.element = this.createElement();
    }

    createElement() {
        const div = document.createElement('div');
        div.className = 'example-component';
        div.innerHTML = `
            <div class="component-container">
                <h3>Example Component</h3>
                <div class="data-display">
                    ${this.formatData(this.data)}
                </div>
            </div>
        `;
        return div;
    }

    formatData(data) {
        return `<pre>${JSON.stringify(data, null, 2)}</pre>`;
    }
}
```

### 4. Testing

#### Unit Tests

```python
# tests/test_example.py
import pytest
from backend.modules.example.service import ExampleService

@pytest.mark.asyncio
async def test_create_example():
    # Test implementation
    pass
```

#### Integration Tests

```python
# tests/test_example_integration.py
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_example_endpoint():
    response = client.get("/api/v1/example")
    assert response.status_code == 200
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/              # Unit tests
├── integration/       # Integration tests
├── e2e/              # End-to-end tests
└── fixtures/         # Test data and fixtures
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend --cov=frontend --cov-report=html

# Run specific test file
pytest tests/test_example.py

# Run tests in verbose mode
pytest -v

# Run tests with specific marker
pytest -m "unit"
```

### Test Best Practices

1. **Use descriptive test names** that explain what they're testing
2. **Test one thing per test function**
3. **Use fixtures** for common test setup
4. **Mock external dependencies**
5. **Test error conditions** as well as success cases
6. **Keep tests fast** and independent

### Example Test

```python
@pytest.mark.asyncio
async def test_create_knowledge_base(db_session):
    """Test creating a knowledge base."""
    from backend.modules.knowledge.crud import create_knowledge_base

    kb = await create_knowledge_base(db_session, "Test KB", "Description")

    assert kb.name == "Test KB"
    assert kb.description == "Description"
    assert kb.id is not None
```

## Code Style and Conventions

### Python Style Guide

RAGify follows PEP 8 with some additional conventions:

#### Naming Conventions

```python
# Classes: PascalCase
class KnowledgeBaseService:
    pass

# Functions: snake_case
def create_knowledge_base():
    pass

# Constants: UPPER_CASE
MAX_FILE_SIZE = 100 * 1024 * 1024

# Private methods: _leading_underscore
def _process_document(self):
    pass
```

#### Import Organization

```python
# Standard library imports
import os
import sys
from typing import List, Optional

# Third-party imports
import fastapi
import sqlalchemy

# Local imports
from ...core.database import get_db
from ..models import KnowledgeBase
```

#### Type Hints

```python
from typing import List, Dict, Optional, AsyncGenerator

async def get_knowledge_bases(db: AsyncSession) -> List[KnowledgeBase]:
    pass

def process_text(text: str, options: Optional[Dict[str, Any]] = None) -> str:
    pass
```

### Code Quality Tools

#### Linting

```bash
# Backend: Check Python code style
flake8 backend/

# Backend: Auto-format Python code
black backend/
isort backend/

# Frontend: Check JavaScript code
cd frontend && npm run lint

# Frontend: Format JavaScript code
cd frontend && npm run format
```

#### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
```

### Documentation

#### Docstrings

```python
def create_knowledge_base(db: AsyncSession, name: str, description: str = None) -> KnowledgeBase:
    """
    Create a new knowledge base.

    Args:
        db: Database session
        name: Knowledge base name
        description: Optional description

    Returns:
        Created knowledge base instance

    Raises:
        ValueError: If name is empty
    """
    pass
```

#### Comments

```python
# Good comment explains why, not what
# Bad: Increment counter
counter += 1

# Good: Track API usage for rate limiting
counter += 1
```

## Database Development

### Schema Changes

1. **Create migration scripts** for schema changes
2. **Test migrations** on development data
3. **Document breaking changes**
4. **Update models** after migrations

### Query Optimization

```python
# Good: Use selectinload for relationships
query = select(KnowledgeBase).options(selectinload(KnowledgeBase.documents))

# Good: Use indexed columns in WHERE clauses
query = select(Document).where(Document.knowledge_base_id == kb_id)

# Avoid: N+1 queries
# Bad: Loop with individual queries
for kb in knowledge_bases:
    documents = await get_documents_for_kb(db, kb.id)  # N queries
```

### Connection Management

```python
# Use context managers for transactions
async with AsyncSessionLocal() as session:
    async with session.begin():
        # Database operations
        pass
```

## API Development

### Endpoint Design

```python
@router.post("/knowledge-bases/", response_model=KnowledgeBase)
async def create_kb(
    kb: KnowledgeBaseCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a knowledge base."""
    return await create_knowledge_base(db, kb.name, kb.description)
```

### Error Handling

```python
@router.get("/knowledge-bases/{kb_id}")
async def get_kb(kb_id: UUID, db: AsyncSession = Depends(get_db)):
    kb = await get_knowledge_base(db, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb
```

### Validation

```python
from pydantic import BaseModel, Field

class KnowledgeBaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
```

## Performance Optimization

### Async/Await Best Practices

```python
# Good: Use async database operations
async def get_kb_with_documents(db: AsyncSession, kb_id: UUID):
    kb = await db.get(KnowledgeBase, kb_id)
    await db.refresh(kb, ["documents"])  # Load relationship
    return kb

# Avoid: Blocking operations in async functions
# Bad: Synchronous file I/O
def read_file_sync(path):
    with open(path, 'r') as f:
        return f.read()
```

### Caching Strategy

```python
from ...core.cache import cache_manager

@cache_manager.cached(ttl=300)  # Cache for 5 minutes
async def get_frequent_data():
    # Expensive operation
    return await expensive_db_query()
```

### Memory Management

```python
# Process large files in chunks
async def process_large_file(file_path: str):
    chunk_size = 8192
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            await process_chunk(chunk)
```

## Security Considerations

### Input Validation

```python
# Sanitize file uploads
allowed_extensions = {'.pdf', '.docx', '.txt'}

def validate_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in allowed_extensions
```

### SQL Injection Prevention

```python
# Good: Use parameterized queries
query = select(User).where(User.email == email)

# Bad: String formatting
# query = f"SELECT * FROM users WHERE email = '{email}'"
```

### Authentication (Future)

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Verify token and return user
    pass
```

## Deployment and CI/CD

### Docker Development

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -e .
      - name: Run tests
        run: pytest --cov=backend --cov=frontend
```

## Contributing Guidelines

### Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes with tests
4. **Run** the test suite
5. **Update** documentation if needed
6. **Submit** a pull request

### Commit Messages

```bash
# Good commit message format
feat: add document upload validation
fix: resolve memory leak in embedding service
docs: update API documentation
test: add integration tests for chat endpoints
```

### Code Review Checklist

- [ ] Tests pass
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No security vulnerabilities
- [ ] Performance considerations addressed
- [ ] Database migrations included if needed

## Troubleshooting Development Issues

### Common Issues

#### Import Errors

```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall package
pip install -e .
```

#### Database Connection Issues

```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
psql -h localhost -U user -d ragify
```

#### Frontend Hot Reload Issues

```bash
# Clear Streamlit cache
rm -rf ~/.streamlit

# Restart with clean cache
streamlit run app.py --server.headless true
```

## Resources

### Learning Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy Documentation](https://sqlalchemy.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)

### Development Tools

- **VS Code** with Python extension
- **PostgreSQL client** (pgAdmin, DBeaver)
- **Redis client** (RedisInsight)
- **API testing** (Postman, Insomnia)

---

**Next Steps**: Check the [deployment guide](deployment.md) for production setup or [API documentation](api.md) for endpoint details.
