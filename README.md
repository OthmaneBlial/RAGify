# RAGify

A modern, modular Retrieval-Augmented Generation (RAG) chat application built with FastAPI and a responsive web frontend. RAGify enables you to create knowledge bases from documents, build intelligent chat applications, and interact with 100+ AI models through OpenRouter's unified API.

## 🚀 Features

- **OpenRouter AI Integration**: Access to 100+ AI models through OpenRouter's unified API
- **Knowledge Base Management**: Create and organize knowledge bases with document uploads
- **Document Processing**: Support for PDF, DOCX, and text files with automatic text extraction
- **Advanced RAG Pipeline**: Semantic search using sentence transformers and vector embeddings
- **Chat Applications**: Build custom chat interfaces with associated knowledge bases
- **Real-time Streaming**: Streaming chat responses for better user experience
- **Modern Web Frontend**: Responsive HTML/JS interface built with Vite
- **RESTful API**: Comprehensive FastAPI backend with automatic documentation
- **Async Architecture**: High-performance async processing with SQLAlchemy and PostgreSQL
- **Vector Search**: pgvector-powered semantic search for accurate document retrieval
- **Rate Limiting**: Built-in rate limiting and security middleware
- **Background Processing**: Async task management for document processing
- **Caching**: Redis integration for performance optimization

## 🏗️ Architecture

```
RAGify Architecture
├── Frontend (HTML/JS + Vite)
│   ├── Chat Interface
│   ├── Knowledge Base Management
│   ├── Application Builder
│   └── Settings Panel
├── Backend (FastAPI)
│   ├── API Layer (REST endpoints)
│   │   ├── Knowledge Base APIs
│   │   ├── Application APIs
│   │   ├── Chat APIs
│   │   └── Model Management APIs
│   ├── RAG Pipeline
│   │   ├── Document Processing
│   │   ├── Embedding Service (Sentence Transformers)
│   │   ├── Vector Search (pgvector)
│   │   └── Response Generation
│   ├── OpenRouter Integration
│   │   ├── Unified API Access
│   │   ├── 100+ Model Support
│   │   ├── Load Balancing
│   │   └── Fallback Handling
│   ├── Core Services
│   │   ├── Database (PostgreSQL + pgvector)
│   │   ├── Cache (Redis)
│   │   ├── Security & Rate Limiting
│   │   └── Async Task Management
│   └── Modules
│       ├── Applications Management
│       ├── Knowledge Base Management
│       └── RAG Pipeline Components
└── Shared Components
    ├── Pydantic Models
    └── Utility Functions
```

## 📋 Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows 10+
- **Python**: Version 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended for document processing)
- **Storage**: At least 2GB free space for models and data
- **Network**: Internet connection for downloading dependencies and AI model access

### Required Software

#### Python 3.8+

```bash
# Check Python version
python --version

# If not installed, download from python.org
# Or use your system package manager
```

#### PostgreSQL 12+ with pgvector

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib

# macOS with Homebrew
brew install postgresql
brew services start postgresql

# Windows: Download from postgresql.org

# Install pgvector extension
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
make install
```

#### Redis 6+ (Optional but recommended)

```bash
# Ubuntu/Debian
sudo apt install redis-server
sudo systemctl start redis-server

# macOS with Homebrew
brew install redis
brew services start redis

# Windows: Download from redis.io
```

## ⚡ Quick Start

1. **Clone and setup**:

   ```bash
   git clone <repository-url>
   cd RAGify
   ```

2. **Install dependencies**:

   ```bash
   pip install -e .
   ```

3. **Configure environment**:

   ```bash
   cp .env.example .env
   # Edit .env with your database URL and API keys
   ```

4. **Setup database**:

   ```bash
   # Create PostgreSQL database
   createdb ragify

   # Enable pgvector extension
   psql -d ragify -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```

5. **Start the application**:

   ```bash
   # Start backend
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

   # In another terminal, start frontend
   cd frontend && npm run dev
   ```

6. **Access the application**:
   - Frontend: http://localhost:5173 (Vite dev server)
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## 📦 Installation

### Backend Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Frontend Installation

```bash
cd frontend
npm install
npm run dev
```

### Database Setup

1. **Create database and user**:

   ```sql
   -- Connect to PostgreSQL as superuser
   sudo -u postgres psql

   -- Create database
   CREATE DATABASE ragify;

   -- Create user
   CREATE USER ragify_user WITH PASSWORD 'your_secure_password';

   -- Grant privileges
   GRANT ALL PRIVILEGES ON DATABASE ragify TO ragify_user;

   -- Enable pgvector
   \c ragify
   CREATE EXTENSION IF NOT EXISTS vector;

   -- Exit PostgreSQL
   \q
   ```

2. **Verify database connection**:

   ```bash
   PGPASSWORD='your_password' psql -U ragify_user -d ragify -c "SELECT version();"
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```env
# Database
DATABASE_URL=postgresql+asyncpg://ragify_user:password@localhost/ragify

# Security
SECRET_KEY=your-super-secret-key-change-this-in-production
DEBUG=true
LOG_LEVEL=INFO

# Redis (optional)
REDIS_URL=redis://localhost:6379

# OpenRouter API Key (Primary AI Provider)
OPENROUTER_API_KEY=sk-or-v1-your-openrouter-api-key-here

# Default Model Settings
DEFAULT_PROVIDER=openrouter
DEFAULT_MODEL=openai/gpt-4o-mini
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=4096

# Application Settings
MAX_UPLOAD_SIZE=104857600  # 100MB in bytes
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_DIMENSION=384

# CORS Settings
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

### OpenRouter Configuration

RAGify uses OpenRouter as the primary AI provider, giving you access to 100+ models from different providers through a single API.

#### Get Your OpenRouter API Key

1. Visit [OpenRouter](https://openrouter.ai/)
2. Sign up for an account
3. Navigate to API Keys section
4. Create a new API key
5. Add to your `.env` file:

```env
OPENROUTER_API_KEY=sk-or-v1-your-openrouter-api-key-here
```

#### Available Models

OpenRouter provides access to models from:
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude-3, Claude-2, etc.)
- Google (Gemini, etc.)
- Meta (Llama models)
- And many more...

You can specify any supported model in your application configuration.

## 🎯 Usage

### Creating Knowledge Bases

Knowledge bases are containers for your documents and serve as the foundation for AI applications.

1. **Access the web interface** at http://localhost:5173
2. **Navigate to Knowledge Bases** section
3. **Create a new knowledge base**:
   - Name: Descriptive name (e.g., "Company Policies")
   - Description: Optional details about the content
4. **Upload documents**:
   - Supported formats: PDF, DOCX, TXT
   - Maximum size: 100MB per file
   - Multiple files supported

### Building Chat Applications

Applications are chat interfaces powered by your knowledge bases.

1. **Go to Applications** section
2. **Create a new application**:
   - Name: Application display name
   - Description: Purpose of the application
   - Model Selection: Choose from 100+ models via OpenRouter
   - Knowledge Bases: Select relevant knowledge bases
3. **Configure settings**:
   - Temperature: Response creativity (0.0-1.0)
   - Max tokens: Response length limit
   - System prompt: Custom instructions

### Chat Interface

1. **Select an application** from the dropdown
2. **Start chatting**:
   - Type your message
   - Get AI-powered responses based on your knowledge bases
   - Enjoy real-time streaming responses
3. **View conversation history** and manage past interactions

## 📚 API Documentation

The API documentation is automatically generated and available at `http://localhost:8000/docs` when the backend is running.

### Key Endpoints

#### Knowledge Bases
- `GET /api/v1/knowledge-bases/` - List all knowledge bases
- `POST /api/v1/knowledge-bases/` - Create knowledge base
- `GET /api/v1/knowledge-bases/{id}` - Get knowledge base details
- `PUT /api/v1/knowledge-bases/{id}` - Update knowledge base
- `DELETE /api/v1/knowledge-bases/{id}` - Delete knowledge base

#### Documents
- `POST /api/v1/knowledge-bases/{id}/documents/` - Upload document
- `GET /api/v1/knowledge-bases/{id}/documents/` - List documents
- `GET /api/v1/documents/{id}/status` - Get processing status

#### Applications
- `GET /api/v1/applications/` - List applications
- `POST /api/v1/applications/` - Create application
- `GET /api/v1/applications/{id}` - Get application details
- `PUT /api/v1/applications/{id}` - Update application
- `DELETE /api/v1/applications/{id}` - Delete application

#### Chat
- `POST /api/v1/applications/{id}/chat` - Send chat message
- `GET /api/v1/applications/{id}/chat/history` - Get chat history
- `DELETE /api/v1/applications/{id}/chat/history` - Clear history

#### Search
- `POST /api/v1/search/` - Semantic search across knowledge bases

### Example API Usage

```python
import requests

# Create knowledge base
kb_response = requests.post("http://localhost:8000/api/v1/knowledge-bases/",
    json={"name": "Documentation", "description": "Product docs"}
)
kb_id = kb_response.json()["id"]

# Upload document
with open("manual.pdf", "rb") as f:
    files = {"file": ("manual.pdf", f, "application/pdf")}
    requests.post(f"http://localhost:8000/api/v1/knowledge-bases/{kb_id}/documents/", files=files)

# Create application
app_response = requests.post("http://localhost:8000/api/v1/applications/",
    json={
        "name": "Support Bot",
        "description": "Customer support assistant",
        "model_config": {"model": "openai/gpt-4o-mini"},  # OpenRouter handles the provider
        "knowledge_base_ids": [kb_id]
    }
)
app_id = app_response.json()["id"]

# Chat with application
chat_response = requests.post(f"http://localhost:8000/api/v1/applications/{app_id}/chat",
    json={"message": "How do I reset my password?"}
)
print(chat_response.json()["response"])
```

## 🔧 Development

### Project Structure

```
RAGify/
├── backend/
│   ├── api/v1/endpoints/     # API endpoints
│   ├── core/                 # Core functionality
│   │   ├── database.py       # Database connection
│   │   ├── config.py         # Configuration management
│   │   ├── security.py       # Security utilities
│   │   └── async_tasks.py    # Background task processing
│   ├── modules/              # Business logic modules
│   │   ├── applications/     # Application management
│   │   ├── knowledge/        # Knowledge base operations
│   │   ├── rag/              # RAG pipeline components
│   │   └── models/           # AI model providers
│   └── main.py               # FastAPI application entry point
├── frontend/
│   ├── src/                  # Source code
│   ├── public/               # Static assets
│   ├── package.json          # Dependencies
│   └── vite.config.js        # Vite configuration
├── shared/
│   ├── models/               # Pydantic models
│   └── utils/                # Shared utilities
├── tests/                    # Test suite
├── docs/                     # Documentation
└── pyproject.toml            # Python project configuration
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend --cov-report=html

# Run specific test file
pytest tests/test_api_endpoints.py
```

### Code Quality

```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8 .
```

## 🚀 Deployment

### Docker Deployment

```yaml
# docker-compose.yml
version: "3.8"

services:
  ragify-backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@postgres/ragify
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis

  ragify-frontend:
    build: ./frontend
    ports:
      - "5173:5173"
    depends_on:
      - ragify-backend

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=ragify
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

```bash
# Deploy
docker-compose up -d

# View logs
docker-compose logs -f
```

### Production Configuration

- Set `DEBUG=false` in environment
- Use production database with proper backups
- Configure SSL/TLS certificates
- Set up monitoring and logging
- Configure proper CORS origins
- Use environment-specific configuration files

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest`
5. Commit your changes: `git commit -am 'Add new feature'`
6. Push to the branch: `git push origin feature/your-feature`
7. Submit a pull request

### Development Guidelines

- Write comprehensive tests for new features
- Update documentation for API changes
- Follow the existing code style and patterns
- Add type hints for new functions
- Keep commit messages clear and descriptive
- Test your changes across different environments

### Code Style

This project follows PEP 8 standards with some additional conventions:

- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions small and focused on single responsibilities
- Use async/await for I/O operations
- Handle errors gracefully with appropriate logging

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) for the backend API
- Frontend powered by [Vite](https://vitejs.dev/) and modern JavaScript
- Vector search enabled by [pgvector](https://github.com/pgvector/pgvector)
- Embeddings generated using [Sentence Transformers](https://www.sbert.net/)
- Database operations handled by [SQLAlchemy](https://www.sqlalchemy.org/)
- UI components styled with modern CSS and JavaScript frameworks

## 📞 Support

For questions and support:

- Check the [API Documentation](http://localhost:8000/docs)
- Review the [Setup Guide](docs/setup.md)
- Read the [Usage Guide](docs/usage.md)
- Open an issue on GitHub
- Check the documentation in the `docs/` directory

---

**RAGify** - Making RAG applications simple, powerful, and accessible.
