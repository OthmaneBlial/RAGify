# RAGify

**Retrieval-Augmented Generation that actually works.** RAGify is a modern chat application that provides accurate, hallucination-free answers by grounding responses in your documents. No more made-up information - if the answer isn't in your knowledge base, RAGify tells you so.

## ✨ Key Features

- **🎯 Grounded Answers**: Uses retrieved document context; if evidence is missing the assistant clearly says it cannot answer
- **📚 Multiple Knowledge Bases**: Organize documents by topic, department, or project
- **🤖 100+ AI Models**: Access OpenAI, Anthropic, Google, and more through OpenRouter
- **⚡ Real-time Streaming**: Get responses as they're generated
- **🔍 Semantic Search**: Finds relevant information using advanced vector embeddings
- **📱 Modern Web Interface**: Clean, responsive chat interface
- **🚀 Production Ready**: Built with FastAPI, PostgreSQL, and Redis

## 🎬 Demo Video

*Coming soon - watch RAGify in action!*

## 📖 Example Scenarios

RAGify comes with pre-configured examples to help you get started quickly:

### Business Applications
- **[API Documentation Assistant](examples/README.md#api-documentation-assistant)** - Help developers understand and implement APIs
- **[Customer Support Bot](examples/README.md#customer-support-assistant)** - Instant answers to common customer inquiries
- **[HR Assistant](examples/README.md#hr-assistant)** - Employee policy and benefits questions
- **[Legal Document Analysis](examples/README.md#legal-analysis-assistant)** - Compliance and contract analysis
- **[Sales Enablement](examples/README.md#sales-enablement-assistant)** - Product information and competitive intelligence

### Education & Research
- **[Research Paper Assistant](examples/README.md#research-paper-assistant)** - Analyze academic papers across disciplines
- **[Interactive Study Guide](examples/README.md#interactive-study-guide)** - Personalized learning for students

Each example includes sample documents, configurations, and suggested questions. Load them with:

```bash
python examples/scripts/load_examples.py
```

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone <repository-url>
cd RAGify
pip install -e .

# 2. Setup database (PostgreSQL with pgvector required)
createdb ragify
psql -d ragify -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 3. Configure environment
cp .env.example .env
# Add your OpenRouter API key and database URL

# 4. Load examples
python examples/scripts/load_examples.py

# 5. Start the app
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
# Frontend: http://localhost:5173 (after cd frontend && npm run dev)
```

## 🐳 Docker Quick Start

Prefer containers? The repository now ships with a full Docker setup for the API, frontend, PostgreSQL (with pgvector), and Redis.

```bash
# Build (or rebuild) the base dependency image and start everything
./startupdocker.sh --build

# Stream logs if you want to watch startup
./startupdocker.sh --build --logs

# Stop the stack when you're done
./startupdocker.sh --down
```

Host ports for the Docker stack:

- Backend API: `http://localhost:18000`
- Frontend UI: `http://localhost:15173`
- PostgreSQL: `localhost:15432` (user/password: `ragify` / `RagifyStrongPass2023`)
- Redis: `localhost:16379`

The script automatically maintains a `ragify-backend-base` image that contains all heavy Python dependencies (Torch, transformers, etc.). The first `--build` run prepares that base; subsequent builds reuse it so only your application code is rebuilt unless `requirements-docker.txt` changes.

Environment variables are loaded from the project’s `.env` file, so edit that file if you need to change keys, database credentials, or model configuration before running `./startupdocker.sh`.

## 🎯 Why RAGify?

**Traditional AI chatbots often hallucinate** - they make up information that sounds plausible but isn't accurate. RAGify eliminates this by:

- **Grounding responses in your documents** - Every answer is based on actual content you've uploaded
- **Clear "I don't know" responses** - When information isn't available, RAGify admits it rather than guessing
- **Source attribution** - See exactly which documents and sections informed each response
- **Configurable knowledge boundaries** - Control what information each application can access

**Result**: Reliable, trustworthy AI that you can depend on for accurate information.

## 📋 Requirements

- **Python 3.8+**
- **PostgreSQL 12+** with pgvector extension
- **OpenRouter API key** (for AI models)
- **Redis** (optional, for caching)

## ⚙️ Configuration

1. **Get OpenRouter API key** from [openrouter.ai](https://openrouter.ai/)
2. **Copy environment file**: `cp .env.example .env`
3. **Edit `.env`** with your database URL and API key
4. **Setup database**: Create PostgreSQL database with pgvector enabled

## 🎯 How It Works

1. **Upload Documents** - Create knowledge bases and add your documents (PDF, DOCX, TXT)
2. **Build Applications** - Configure chat interfaces with specific knowledge bases and AI models
3. **Start Chatting** - Ask questions and get accurate answers grounded in your documents

## 📚 Documentation

- **API Docs**: `http://localhost:8000/docs` (when running)
- **Examples**: See `examples/README.md` for detailed scenarios
- **Setup Guide**: See `docs/setup.md` for detailed installation

## 🛠️ Development

```bash
# Run tests
pytest

# Format code
black . && isort .

# Start development server
uvicorn backend.main:app --reload
```
