# Campaign Intelligence Assistant

> AI-powered campaign analytics and reporting tool for adtech teams.

An internal tool that automates campaign report generation and enables natural-language querying of campaign performance data. Built with FastAPI, LangGraph agents, pgvector RAG retrieval, and a Next.js frontend. Deployable to Vercel.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Next.js Chat UI (web/)                    │
│  SSE streaming  ·  dark sidebar  ·  example chips           │
└──────────────────────────┬──────────────────────────────────┘
                           │  HTTP / SSE
┌──────────────────────────▼──────────────────────────────────┐
│                   FastAPI Backend                            │
│  /api/chat (SSE)  ·  /api/campaigns  ·  /api/reports        │
│  /api/audience/recommend  ·  /api/health                    │
└────┬─────────────┬─────────────┬────────────────────────────┘
     │             │             │
┌────▼────┐  ┌─────▼─────┐  ┌───▼────────────┐
│LangGraph│  │  Report   │  │  RAG Service   │
│  Agent  │  │ Generator │  │  (pgvector)    │
│         │  │  (LLM +   │  │  embed/search  │
│  tools: │  │   FPDF2)  │  │  campaign data │
│  query  │  └─────┬─────┘  └───┬────────────┘
│  search │        │            │
│  compare│  ┌─────▼────────────▼─────────────┐
│  report │  │   LLM Client (Google Gemini)   │
│  reco   │  │  structured output · streaming │
└────┬────┘  │  token counting · cost tracking│
     │       └─────┬──────────────────────────┘
     │             │
┌────▼─────────────▼──────────────────────────┐
│     PostgreSQL + pgvector (async)            │
│  Campaign · Metrics · Audience · Embeddings │
└─────────────────────────────────────────────┘
```

## Features

- **Natural Language Queries** — Ask questions about campaign performance in plain English via a LangGraph conversational agent with 5 specialized tools.
- **SSE Streaming** — Real-time streamed responses via Server-Sent Events for instant feedback.
- **Automated Report Generation** — Generate formatted Markdown, PDF, or Slack-ready campaign reports combining LLM analysis with live metrics.
- **Campaign Comparison** — Side-by-side comparison of two campaigns with metric-level winner highlighting.
- **RAG-Powered Retrieval** — Semantic search over campaign data using pgvector embeddings for context-aware answers.
- **Structured LLM Output** — Type-safe responses using Gemini structured JSON output and Pydantic schemas.
- **Audience Recommendations** — AI-driven audience segment suggestions based on campaign history and semantic similarity.
- **Next.js Chat Interface** — Modern React UI with dark sidebar, streaming chat, tool badges, and example queries.
- **Vercel-Ready** — Deployable to Vercel with Neon Postgres for fully serverless operation.

## Tech Stack

| Layer        | Technology                              |
|--------------|-----------------------------------------|
| API          | FastAPI, Uvicorn                        |
| Agent        | LangGraph, LangChain Core              |
| LLM          | Google Gemini 2.0 Flash (structured output) |
| Embeddings   | Google text-embedding-004              |
| Vector Store | pgvector (PostgreSQL extension)         |
| Database     | PostgreSQL 16, SQLAlchemy (async)       |
| Frontend     | Next.js 14, React, Tailwind CSS         |
| Reports      | FPDF2 (PDF), Markdown, Slack           |
| Testing      | pytest, pytest-asyncio                  |
| Infra        | Docker Compose, Vercel                  |

## Quickstart

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- Google Gemini API key

### 1. Clone & configure

```bash
cp .env.example .env
# Edit .env with your GOOGLE_API_KEY
```

### 2. Start infrastructure

```bash
docker compose up -d postgres
```

### 3. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
make setup
make web-setup
```

### 4. Seed the database

```bash
make seed
# Creates tables, loads 18 mock campaigns, and generates pgvector embeddings
```

### 5. Start the API & frontend

```bash
# Terminal 1: API server
make run
# API available at http://localhost:8080

# Terminal 2: Next.js frontend
make web-dev
# Frontend at http://localhost:3000
```

### Docker (all-in-one)

```bash
make docker-up
# API: http://localhost:8080
```

## Deploy to Vercel

1. Push your repo to GitHub
2. Import project in [Vercel](https://vercel.com/new)
3. Set up a [Neon Postgres](https://neon.tech) database with pgvector (see [docs/setup_neon.md](docs/setup_neon.md))
4. Add environment variables in Vercel:
   - `DATABASE_URL` — Neon connection string (use `postgresql+asyncpg://...`)
   - `GOOGLE_API_KEY` — Your Google Gemini API key
5. Deploy!

## API Endpoints

| Method | Endpoint                  | Description                                |
|--------|---------------------------|--------------------------------------------|
| GET    | `/`                       | API info and available endpoints           |
| GET    | `/api/health`             | Health check (DB, pgvector, LLM status)    |
| POST   | `/api/chat`               | Chat via LangGraph agent (supports SSE)    |
| GET    | `/api/campaigns`          | List campaigns (paginated, filterable)     |
| GET    | `/api/campaigns/{id}`     | Get campaign details with metrics          |
| POST   | `/api/reports/generate`   | Generate report (markdown/pdf/slack)       |
| POST   | `/api/reports/compare`    | Compare two campaigns side-by-side         |
| POST   | `/api/audience/recommend` | Get AI audience segment recommendations   |

### SSE Streaming

```bash
curl -N -X POST 'http://localhost:8080/api/chat?stream=true' \
  -H 'Content-Type: application/json' \
  -d '{"message": "What are the top QSR campaigns?"}'
```

## Project Structure

```
app/
  main.py            -> FastAPI application with lifespan, middleware
  config.py          -> Environment configuration (Pydantic Settings)
  database.py        -> Async SQLAlchemy engine & session factory
  models/
    campaign.py      -> ORM models (Campaign, Metrics, Audience, Embedding)
    schemas.py       -> Pydantic schemas (API + LLM structured output)
  services/
    llm_client.py    -> Google Gemini wrapper (chat, streaming, structured output, embeddings)
    rag.py           -> pgvector RAG service (embed, retrieve, hybrid search)
    report_gen.py    -> Report generator (Markdown, PDF, Slack, comparison)
  agents/
    campaign_agent.py -> LangGraph StateGraph (router -> tools -> synthesizer)
    tools.py         -> 5 agent tools (query, search, compare, report, audience)
  api/
    routes.py        -> FastAPI route handlers with SSE streaming
api/
  index.py           -> Vercel serverless entry point
web/
  app/               -> Next.js App Router pages
  components/        -> React components (ChatInterface, Sidebar, etc.)
  package.json       -> Node.js dependencies
data/
  mock_campaigns.json -> 18 realistic campaigns across 5 verticals
  seed.py            -> Database + pgvector seed pipeline
docs/
  setup_neon.md      -> Neon Postgres setup guide
tests/
  conftest.py        -> Shared fixtures (mock LLM, mock RAG, sample data)
  test_agent.py      -> Agent routing, tool, and graph tests
  test_report.py     -> Report generation tests
  test_api.py        -> API endpoint tests
```

## Make Commands

| Command          | Description                          |
|------------------|--------------------------------------|
| `make setup`     | Install Python dependencies          |
| `make seed`      | Seed database with mock campaigns    |
| `make run`       | Start FastAPI dev server             |
| `make web-setup` | Install Next.js dependencies         |
| `make web-dev`   | Start Next.js dev server             |
| `make web-build` | Build Next.js for production         |
| `make test`      | Run pytest suite                     |
| `make lint`      | Run ruff linter                      |
| `make format`    | Auto-format with ruff                |
| `make docker-up` | Start all services via Docker        |
| `make clean`     | Remove __pycache__ and caches        |

## Running Tests

```bash
make test
```

---

## Built for InMarket AI Builder Role

This project demonstrates the architecture and engineering patterns needed for an AI-powered internal tool at an adtech company like InMarket:

- **Campaign Operations teams** spend significant time manually pulling metrics, formatting reports, and answering ad-hoc performance questions from stakeholders.
- **This tool replaces that workflow** with an AI agent that can query the campaign database, retrieve semantically relevant context, generate formatted reports, and recommend audience segments — all through a natural-language chat interface.
- **The architecture is production-oriented**: async database access, SSE streaming, pgvector for vector search (no separate service), structured LLM outputs for reliability, and a modular agent design that can be extended with new tools as business needs evolve.
- **Vercel-deployable**: serverless Python API + Next.js frontend, backed by Neon Postgres with pgvector.

---

*Campaign Intelligence Assistant — internal tooling for smarter campaign operations.*
