# RAGify

Retrieval-Augmented Generation that actually works. RAGify is a modern chat experience that answers questions from your documents, cites its sources, and admits when it does not know. Try it live: **https://ragify-sqhu4om4aq-ew.a.run.app/**

## 🔥 Highlights

- **Grounded answers** with inline citations and “I don’t know” fallbacks
- **Multiple assistants** tied to different knowledge bases and model presets
- **100+ AI models** via OpenRouter (OpenAI, Anthropic, Google, Meta, etc.)
- **Hybrid retrieval** (semantic vectors + keyword search) running locally on SQLite or remotely on PostgreSQL
- **Modern UI** built with Tailwind and vanilla JS
- **Cloud-ready**: single-container Cloud Run deploy with the embedded SQLite database—no external services required

## 🌐 Live Demo

The public Cloud Run deployment runs the same configuration described below. Feel free to explore it, upload a PDF, and ask questions:

- Chat + Knowledge Base UI: `https://ragify-sqhu4om4aq-ew.a.run.app/`

> Uploads made to the demo instance are purged periodically. Bring your own OpenRouter key for local/private deployments.

## 🧱 Default Stack

| Layer | Default | Optional |
| --- | --- | --- |
| API | FastAPI + SQLAlchemy | – |
| Database | SQLite (`ragify.db`) via SQLAlchemy Async + StaticPool | PostgreSQL + pgvector |
| Cache | In-memory LRU | Redis (auto-disabled when unavailable) |
| Models | OpenRouter (configurable in UI or `.env`) | Other providers via `shared/models` |
| Frontend | Static HTML/CSS/JS (Tailwind) served by FastAPI or Vite dev server | – |

SQLite is now the recommended default for local development **and** Cloud Run. PostgreSQL/Redis remain fully supported—flip the switch at runtime if you need them.

## 🚀 Quick Start (SQLite, zero external deps)

```bash
git clone https://github.com/OthmaneBlial/RAGify.git
cd RAGify
cp .env.example .env         # add your OPENROUTER_API_KEY
python -m venv venv && source venv/bin/activate
pip install -e .

./startup.sh                 # choose option 1 for SQLite when prompted
# Backend → http://localhost:8000, Frontend → http://localhost:5173
```

What the helper does:
1. Lets you choose SQLite or PostgreSQL at runtime.
2. Creates/updates `ragify.db` (SQLite) or warms up the Postgres pool.
3. Boots FastAPI on `:8000` and serves the static frontend on `:5173`.

Stop it with `Ctrl+C`. Delete `ragify.db` anytime for a clean slate.

## 🐳 Docker / Local Stack

Still prefer containers? The repo keeps the base ML dependencies in a reusable image so rebuilds stay quick:

```bash
./startupdocker.sh --build     # builds base + app images, starts API, frontend, Postgres, Redis
./startupdocker.sh --down      # stop the stack
```

Host ports:
- Backend API: `http://localhost:18000`
- Frontend: `http://localhost:15173`
- PostgreSQL (pgvector): `localhost:15432`
- Redis: `localhost:16379`

The Docker option mirrors the legacy Postgres/Redis architecture. For most workflows, the lighter SQLite mode above is enough.

## ☁️ Cloud Run Deployment (SQLite-only image)

Deploy the exact image that powers the public demo:

```bash
PROJECT_ID=my-gcp-project \
REGION=us-central1 \
OPENROUTER_API_KEY=... (inside .env) \
./build.sh
```

`build.sh` now:
1. Builds/pushes the shared dependency base image (if needed).
2. Builds the FastAPI+frontend image.
3. Generates a Cloud Run–friendly env file from `.env`, overrides `DATABASE_URL` with `sqlite+aiosqlite:///tmp/ragify.db`, and disables Redis.
4. Deploys a **single** container (no sidecars) with the new env file and prints the service URL.

See [DEPLOYMENT.md](DEPLOYMENT.md) for the full breakdown or to customize memory/CPU.

## 🧪 Example Applications

`examples/` ships with seeds and documents you can import:

- API Documentation Assistant
- Customer Support Bot
- HR & Onboarding Assistant
- Legal Review Assistant
- Research/Study Guides

Load them with:

```bash
python examples/scripts/load_examples.py
```

Each application binds to specific knowledge bases and model presets so you can see how per-assistant configuration works.

## 📦 Requirements

Minimum setup (SQLite mode):
- Python 3.8+
- OpenRouter API key (or other provider credentials)

Optional services:
- PostgreSQL 12+ with `pgvector` for large-scale or shared deployments
- Redis 6+ if you want distributed caching (RAGify automatically falls back to the in-memory cache when Redis is absent)

## 🔧 Configuration Tips

1. Copy `.env.example` → `.env`.
2. Add your `OPENROUTER_API_KEY`, `SECRET_KEY`, etc.
3. Override `DEFAULT_MODEL`, `DEFAULT_TEMPERATURE`, or other knobs as needed.
4. For PostgreSQL: set `DATABASE_URL=postgresql+asyncpg://...` before running `startup.sh` (choose option 2).
5. Redis is optional—set `REDIS_URL=` (empty) to disable it entirely.

## 📚 Documentation & Tooling

- API docs: `http://localhost:8000/docs`
- Setup guide: [docs/setup.md](docs/setup.md)
- Deployment details: [DEPLOYMENT.md](DEPLOYMENT.md)
- Frontend pages: `/chat.html`, `/knowledge.html`, `/settings.html`

## 🛠️ Development

```bash
pytest                         # run tests
black . && isort .             # formatting
uvicorn backend.main:app --reload -p 8000
```

Happy hacking! Upload a PDF, ask a question, and watch RAGify stay grounded.
