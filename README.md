# Campaign Intelligence Assistant

> AI-powered campaign analytics and reporting tool for adtech teams.

An internal tool that automates campaign report generation and enables natural-language querying of campaign performance data. Built with FastAPI, LangGraph agents, and RAG-powered retrieval over campaign metrics.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Chat UI                        │
└──────────────────────────┬──────────────────────────────────┘
                           │  HTTP
┌──────────────────────────▼──────────────────────────────────┐
│                   FastAPI Backend                           │
│  /api/chat  ·  /api/campaigns  ·  /api/reports  ·  /api/   │
│  /api/audience/recommend  ·  /api/health                    │
└────┬─────────────┬─────────────┬────────────────────────────┘
     │             │             │
┌────▼────┐  ┌─────▼─────┐  ┌───▼────────────┐
│LangGraph│  │  Report   │  │  RAG Service   │
│  Agent  │  │ Generator │  │  (ChromaDB)    │
│         │  │  (LLM +   │  │  embed/search  │
│  tools: │  │   FPDF2)  │  │  campaign data │
│  query  │  └─────┬─────┘  └───┬────────────┘
│  search │        │            │
│  compare│  ┌─────▼────────────▼─────────────┐
│  report │  │     LLM Client (OpenAI)        │
│  reco   │  │  structured output · retries   │
└────┬────┘  │  token counting · cost tracking│
     │       └─────┬──────────────────────────┘
     │             │
┌────▼─────────────▼──────────────────────────┐
│          PostgreSQL (async)                  │
│  Campaign · CampaignMetrics · Audience      │
└─────────────────────────────────────────────┘
```

## Features

- **Natural Language Queries** — Ask questions about campaign performance in plain English via a LangGraph conversational agent with 5 specialized tools.
- **Automated Report Generation** — Generate formatted Markdown, PDF, or Slack-ready campaign reports combining LLM analysis with live metrics.
- **Campaign Comparison** — Side-by-side comparison of two campaigns with metric-level winner highlighting.
- **RAG-Powered Retrieval** — Semantic search over campaign data using ChromaDB embeddings for context-aware answers.
- **Structured LLM Output** — Type-safe responses using OpenAI structured output and Pydantic schemas.
- **Audience Recommendations** — AI-driven audience segment suggestions based on campaign history and semantic similarity.
- **Streamlit Chat Interface** — Professional internal UI with sidebar actions, system status, and example queries.

## Tech Stack

| Layer        | Technology                              |
|--------------|-----------------------------------------|
| API          | FastAPI, Uvicorn                        |
| Agent        | LangGraph, LangChain Core              |
| LLM          | OpenAI GPT-4o (structured output)      |
| Embeddings   | OpenAI text-embedding-3-small          |
| Vector Store | ChromaDB                                |
| Database     | PostgreSQL 16, SQLAlchemy (async)       |
| Migrations   | Alembic                                 |
| Reports      | FPDF2 (PDF), Markdown, Slack           |
| UI           | Streamlit                               |
| Testing      | pytest, pytest-asyncio                  |
| Infra        | Docker Compose                          |

## Quickstart

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenAI API key

### 1. Clone & configure

```bash
cp .env.example .env
# Edit .env with your OPENAI_API_KEY
```

### 2. Start infrastructure

```bash
docker compose up -d postgres chromadb
```

### 3. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
make setup
```

### 4. Run migrations & seed data

```bash
alembic upgrade head
make seed
```

### 5. Start the API

```bash
make run
# API available at http://localhost:8080
# Swagger docs at http://localhost:8080/docs
```

### 6. Launch the UI

```bash
make ui
# Streamlit UI at http://localhost:8501
```

### Docker (all-in-one)

```bash
make docker-up
# API: http://localhost:8080  |  UI: http://localhost:8501
```

## API Endpoints

| Method | Endpoint                  | Description                                |
|--------|---------------------------|--------------------------------------------|
| GET    | `/`                       | API info and available endpoints           |
| GET    | `/api/health`             | Health check (DB, ChromaDB, LLM status)    |
| POST   | `/api/chat`               | Natural language query via LangGraph agent |
| GET    | `/api/campaigns`          | List campaigns (paginated, filterable)     |
| GET    | `/api/campaigns/{id}`     | Get campaign details with metrics          |
| POST   | `/api/reports/generate`   | Generate report (markdown/pdf/slack)       |
| POST   | `/api/reports/compare`    | Compare two campaigns side-by-side         |
| POST   | `/api/audience/recommend` | Get AI audience segment recommendations   |

## Project Structure

```
app/
  main.py            → FastAPI application with lifespan, middleware
  config.py          → Environment configuration (Pydantic Settings)
  database.py        → Async SQLAlchemy engine & session factory
  models/
    campaign.py      → SQLAlchemy ORM models (Campaign, Metrics, Audience)
    schemas.py       → Pydantic schemas (API + LLM structured output)
  services/
    llm_client.py    → OpenAI wrapper (chat, structured output, embeddings)
    rag.py           → ChromaDB RAG service (embed, retrieve, hybrid search)
    report_gen.py    → Report generator (Markdown, PDF, Slack, comparison)
  agents/
    campaign_agent.py → LangGraph StateGraph (router → tools → synthesizer)
    tools.py         → 5 agent tools (query, search, compare, report, audience)
  api/
    routes.py        → FastAPI route handlers
  ui/
    streamlit_app.py → Streamlit chat interface
data/
  mock_campaigns.json → 18 realistic campaigns across 5 verticals
  seed.py            → Database + ChromaDB seed pipeline
migrations/          → Alembic migration versions
tests/
  conftest.py        → Shared fixtures (mock LLM, mock RAG, sample data)
  test_agent.py      → Agent routing, tool, and graph tests (20 tests)
  test_report.py     → Report generation tests (30 tests)
  test_api.py        → API endpoint tests (13 tests)
```

## Running Tests

```bash
make test
# 59 passing, 3 skipped (DB-dependent integration tests)
```

## Make Commands

| Command          | Description                          |
|------------------|--------------------------------------|
| `make setup`     | Install Python dependencies          |
| `make seed`      | Seed database with mock campaigns    |
| `make run`       | Start FastAPI dev server             |
| `make ui`        | Start Streamlit UI                   |
| `make test`      | Run pytest suite                     |
| `make lint`      | Run ruff linter                      |
| `make format`    | Auto-format with ruff                |
| `make docker-up` | Start all services via Docker        |
| `make clean`     | Remove __pycache__ and caches        |

---

## Built for InMarket AI Builder Role

This project demonstrates the architecture and engineering patterns needed for an AI-powered internal tool at an adtech company like InMarket:

- **Campaign Operations teams** spend significant time manually pulling metrics, formatting reports, and answering ad-hoc performance questions from stakeholders.
- **This tool replaces that workflow** with an AI agent that can query the campaign database, retrieve semantically relevant context, generate formatted reports, and recommend audience segments — all through a natural-language chat interface.
- **The architecture is production-oriented**: async database access, structured LLM outputs for reliability, vector-based retrieval for grounding, and a modular agent design that can be extended with new tools as business needs evolve.
- **Key adtech concepts modeled**: campaign lifecycle (draft → active → paused → completed), standard performance metrics (impressions, visit lift, sales lift, ROAS), audience segmentation, and multi-market analysis.

---

*Campaign Intelligence Assistant — internal tooling for smarter campaign operations.*
