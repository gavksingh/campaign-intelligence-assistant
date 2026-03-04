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
│  /chat  ·  /report/{id}  ·  /health                        │
└────┬─────────────┬─────────────┬────────────────────────────┘
     │             │             │
┌────▼────┐  ┌─────▼─────┐  ┌───▼────────────┐
│LangGraph│  │  Report   │  │  RAG Service   │
│  Agent  │  │ Generator │  │  (ChromaDB)    │
│         │  │  (LLM +   │  │  embed/search  │
│  tools: │  │   FPDF2)  │  │  campaign data │
│  query  │  └─────┬─────┘  └───┬────────────┘
│  search │        │            │
│  report │  ┌─────▼────────────▼─────────────┐
│  reco   │  │     LLM Client (OpenAI)        │
└────┬────┘  │  structured output · retries   │
     │       └─────┬──────────────────────────┘
     │             │
┌────▼─────────────▼──────────────────────────┐
│          PostgreSQL (async)                  │
│  Campaign · CampaignMetrics · Audience      │
└─────────────────────────────────────────────┘
```

## Features

- **Natural Language Queries** — Ask questions about campaign performance in plain English via a LangGraph conversational agent.
- **Automated Report Generation** — Generate formatted PDF campaign reports combining LLM analysis with live metrics.
- **RAG-Powered Retrieval** — Semantic search over campaign data using ChromaDB embeddings for context-aware answers.
- **Structured LLM Output** — Type-safe responses using OpenAI structured output and Pydantic schemas.
- **Audience Recommendations** — AI-driven audience segment suggestions based on campaign history.
- **Streamlit Chat Interface** — Simple internal UI for non-technical team members.

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
| Reports      | FPDF2                                   |
| UI           | Streamlit                               |
| Infra        | Docker Compose                          |

## Quickstart

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenAI API key

### 1. Clone & configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 2. Start infrastructure

```bash
docker compose up -d postgres chromadb
```

### 3. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Run migrations & seed data

```bash
alembic upgrade head
python data/seed.py
```

### 5. Start the API

```bash
uvicorn app.main:app --reload
```

### 6. Launch the UI

```bash
streamlit run app/ui/streamlit_app.py
```

## Project Structure

```
app/
  main.py            → FastAPI application entry point
  config.py          → Environment configuration (Pydantic Settings)
  database.py        → Async SQLAlchemy engine & session factory
  models/            → SQLAlchemy ORM models & Pydantic schemas
  services/          → LLM client, RAG, report generation
  agents/            → LangGraph agent definition & tools
  api/               → FastAPI route handlers
  ui/                → Streamlit chat interface
data/                → Mock data & seed scripts
migrations/          → Alembic migration versions
tests/               → Pytest test suite
```

## Running Tests

```bash
pytest tests/ -v
```

---

## Built for InMarket AI Builder Role

This project demonstrates the architecture and engineering patterns needed for an AI-powered internal tool at an adtech company like InMarket. The business context:

- **Campaign Operations teams** spend significant time manually pulling metrics, formatting reports, and answering ad-hoc performance questions from stakeholders.
- **This tool replaces that workflow** with an AI agent that can query the campaign database, retrieve semantically relevant context, generate formatted reports, and recommend audience segments — all through a natural-language chat interface.
- **The architecture is production-oriented**: async database access, structured LLM outputs for reliability, vector-based retrieval for grounding, and a modular agent design that can be extended with new tools as business needs evolve.
- **Key adtech concepts modeled**: campaign lifecycle (draft → active → paused → completed), standard performance metrics (impressions, clicks, conversions, CTR, CPA, ROAS), and audience segmentation for targeting.

---

*Campaign Intelligence Assistant — internal tooling for smarter campaign operations.*
