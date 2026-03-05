# Architecture — Campaign Intelligence Assistant

This document explains the system design, technology choices, data flow, and production considerations for the Campaign Intelligence Assistant.

---

## System Overview

The Campaign Intelligence Assistant is an AI-powered internal tool that replaces manual campaign reporting workflows. It enables campaign operations teams to query performance data, generate client-ready reports, compare campaigns, and get audience recommendations — all through natural language.

```
┌───────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (:8501)                       │
│   Chat interface · Sidebar actions · Example queries              │
└──────────────────────────────┬────────────────────────────────────┘
                               │ HTTP (httpx)
                               │ reads API_BASE_URL from env
┌──────────────────────────────▼────────────────────────────────────┐
│                     FastAPI Backend (:8080)                        │
│                                                                    │
│  Middleware: CORS · Request logging · X-Processing-Time-Ms         │
│  Lifespan:  DB init · pgvector warmup · LLM client init           │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    API Routes (/api)                         │  │
│  │  POST /chat          → LangGraph agent invoke               │  │
│  │  GET  /campaigns     → Paginated list (filter by vertical)  │  │
│  │  GET  /campaigns/:id → Detail with metrics + segments       │  │
│  │  POST /reports/gen   → Markdown / PDF / Slack report        │  │
│  │  POST /reports/cmp   → Side-by-side campaign comparison     │  │
│  │  POST /audience/rec  → AI audience recommendations          │  │
│  │  GET  /health        → DB + pgvector + LLM status            │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌────────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │  LangGraph     │ │   Report     │ │    RAG Service           │ │
│  │  Agent         │ │   Generator  │ │    (pgvector)            │ │
│  │                │ │              │ │                          │ │
│  │  router ──┐    │ │  Markdown    │ │  embed_and_store()       │ │
│  │  tool_exec│    │ │  PDF (fpdf2) │ │  retrieve() with filters│ │
│  │  synthzr ─┘    │ │  Slack       │ │  hybrid_search()         │ │
│  │  error_hndlr   │ │  Comparison  │ │  refresh_index()         │ │
│  └───────┬────────┘ └──────┬───────┘ └────────────┬─────────────┘ │
│          │                 │                      │               │
│  ┌───────▼─────────────────▼──────────────────────▼─────────────┐ │
│  │                 LLM Client (Google Gemini)                    │ │
│  │  chat_completion · structured_output · embed_text/texts       │ │
│  │  tenacity retries · usage metadata · cost tracking            │ │
│  └──────────────────────────┬────────────────────────────────────┘ │
│                              │                                     │
│  ┌──────────────────────────▼────────────────────────────────────┐ │
│  │            PostgreSQL (async via asyncpg)                      │ │
│  │  Campaign · CampaignMetrics · AudienceSegment                 │ │
│  │  Indexes on vertical, status, client_name, campaign_id        │ │
│  └───────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### 1. Chat Query Flow

```
User types question
  → Streamlit POST /api/chat
    → invoke_agent(message)
      → LangGraph router_node (LLM decides which tools to call)
        → tool_executor_node (runs 1+ tools in sequence)
          → e.g. query_campaign_data:
              LLM generates SQL → validate SELECT-only → execute against PostgreSQL
          → e.g. search_similar_campaigns:
              embed query → pgvector semantic search → return ranked results
        → synthesizer_node (LLM combines tool results into natural response)
      → return {reply, sources, data}
    → HTTP 200 with ChatResponse
  → Next.js renders reply with tool badges
```

### 2. Report Generation Flow

```
User requests report (chat or sidebar)
  → POST /api/reports/generate {campaign_id, format}
    → Fetch campaign + metrics from PostgreSQL
    → LLM generates structured LCIReportSchema:
        - executive_summary
        - visit_lift_analysis (overall, market breakdown, daypart)
        - sales_lift_analysis (overall, basket size, frequency)
        - market_breakdown (per-market rankings)
        - recommendations (actionable list)
    → ReportGenerator formats output:
        - markdown: structured Markdown with metrics table
        - pdf: multi-page branded PDF via fpdf2
        - slack: <500 char Slack-formatted summary
    → HTTP 200 with report content
```

### 3. Seed / Indexing Flow

```
python -m data.seed
  → Read mock_campaigns.json (18 campaigns)
  → init_db() creates tables via SQLAlchemy metadata
  → Insert campaigns, metrics, audience segments (skip duplicates)
  → Build rich text representations per campaign
  → Embed via Gemini text-embedding-004
  → Store in pgvector with metadata filters
```

---

## Technology Choices

### FastAPI (API layer)

**Why:** Async-native, automatic OpenAPI docs, Pydantic validation built in. For an internal tool that needs to handle concurrent LLM calls without blocking, async support is essential. The auto-generated Swagger UI at `/docs` makes the API immediately explorable without separate documentation.

**Alternative considered:** Flask — rejected because async support is bolted on, and we'd lose automatic request validation and OpenAPI generation.

### LangGraph (agent framework)

**Why:** Provides explicit control over agent execution flow via a state graph. Unlike simple ReAct loops, the graph structure makes the routing logic (router → tool_executor → synthesizer → error_handler) inspectable and testable. Conditional edges let us handle error retries and multi-step tool chains deterministically.

**Alternative considered:** Raw LangChain AgentExecutor — rejected because it's a black box. We need to test routing decisions independently (e.g., "does the router send to error_handler after 3 failures?"), which requires explicit graph nodes.

### Gemini Structured Output (LLM responses)

**Why:** Using Gemini's `response_mime_type="application/json"` with schema in the system instruction ensures the LLM always returns data matching our Pydantic models (LCIReportSchema, CampaignComparisonSchema, AudienceRecommendationSchema). This eliminates parsing failures and makes report generation reliable.

**Alternative considered:** Prompt engineering with regex parsing — rejected because it's fragile and requires extensive error handling for malformed responses.

### pgvector (vector store)

**Why:** Runs as a PostgreSQL extension — no separate vector store service needed. Supports metadata filtering alongside cosine distance search. The hybrid search pattern (combine SQL results with vector results) provides both precision and recall. Simplifies deployment (single database) and enables transactional consistency between relational and vector data.

**Alternative considered:** ChromaDB — rejected because it requires a separate service in production. Pinecone — rejected because it requires external infrastructure and API keys.

### PostgreSQL + asyncpg (relational data)

**Why:** Industry standard for structured campaign data with complex relationships (campaign → metrics → segments). Async via asyncpg means database queries don't block the event loop during concurrent LLM calls. SQLAlchemy ORM provides type-safe queries and relationship loading.

**Alternative considered:** SQLite — rejected because it doesn't support concurrent async access well and lacks the feature set needed for production adtech data.

### FPDF2 (PDF reports)

**Why:** Pure Python, no system dependencies (no wkhtmltopdf, no LaTeX). Produces professional PDFs with styled tables, headers, and page numbers. Runs identically in Docker containers without binary dependency issues.

**Alternative considered:** WeasyPrint — rejected because it requires system-level dependencies (cairo, pango) that complicate Docker builds and CI pipelines.

### Streamlit (UI)

**Why:** Rapid development of internal tools. Chat interface, sidebar forms, and session state are first-class features. The team can iterate on the UI without frontend engineering. Communicates with FastAPI via HTTP, keeping concerns separated.

**Alternative considered:** React — rejected because it requires a separate build pipeline and frontend expertise. For an internal tool, Streamlit's development speed outweighs its limitations.

---

## Key Design Decisions

### Lazy Database Initialization

The database engine is created lazily on first access rather than at module import time. This prevents `asyncpg` from being imported when only testing report generation or schema validation — a critical pattern for keeping the test suite fast and dependency-free.

```python
# app/database.py
_engine: AsyncEngine | None = None

def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(settings.database_url, ...)
    return _engine
```

### SQL Injection Prevention

The `query_campaign_data` tool generates SQL via the LLM, which is inherently dangerous. All generated queries are validated before execution:

```python
async def _execute_readonly_sql(query: str) -> list[dict]:
    normalized = query.strip().upper()
    if not normalized.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed")
```

### Structured Output with Fallback

LLM responses use Gemini's `response_mime_type="application/json"` for reliable JSON output, with the Pydantic schema injected into the system instruction for guidance.

### Error Budget in Agent

The agent graph includes an error counter that limits retries to `MAX_RETRIES=2`. After exhausting retries, the error handler produces a graceful failure message rather than looping indefinitely. This prevents runaway API costs.

---

## How This Would Scale in Production

### Database

- **Connection pooling**: asyncpg pool with `pool_size=20, max_overflow=10` (already configured).
- **Read replicas**: Route agent SQL queries to read replicas to protect the primary.
- **Partitioning**: Partition `campaign_metrics` by date range for large-scale historical data.
- **Alembic migrations**: The migration infrastructure is in place (`alembic.ini`, `migrations/`). Production deployments would use `alembic upgrade head` instead of `create_all`.

### Vector Store

- **Incremental indexing**: The `refresh_index()` method already supports re-embedding only new or modified campaigns.
- **HNSW indexing**: For larger datasets, add an HNSW index on the embedding column for faster approximate nearest neighbor search.

### LLM Layer

- **Model routing**: Use Gemini Pro for complex analysis (report generation, comparisons) and Gemini Flash for simple tasks (SQL generation, classification) to optimize cost and latency.
- **Response caching**: Cache LLM responses for identical queries with a TTL. Campaign data changes infrequently, so most reports can be cached.
- **Token budgets**: The token counting infrastructure is already in place. Add per-user or per-team daily token limits.

### API Layer

- **Rate limiting**: Add per-user rate limits on `/api/chat` to prevent abuse.
- **Background tasks**: Move report generation to Celery/ARQ workers for long-running PDF generation.
- **CDN for PDFs**: Store generated PDFs in S3 with CloudFront for reuse.

### Observability

- **Structured logging**: Replace print-based logging with structured JSON logs (already using Python logging).
- **Tracing**: Add OpenTelemetry spans for LLM calls, DB queries, and vector searches.
- **Metrics**: Expose Prometheus metrics for response times, token usage, and error rates.

---

## What to Add Next

### Near-term

1. **Authentication & RBAC** — OAuth2 or SSO integration. Role-based access: analysts can query, managers can generate reports, admins can manage campaigns.

2. **Redis caching** — Cache frequent queries, LLM responses, and generated reports. A `/api/chat` response for "top campaigns by ROAS" doesn't change hourly.

3. **Webhook notifications** — Push Slack summaries automatically when campaigns complete or hit performance thresholds. The `generate_slack_summary` method is already built.

4. **Conversation memory** — Store chat history per user in PostgreSQL. The agent currently treats each message independently; memory would enable multi-turn conversations like "drill down on the New York market from my last query."

### Medium-term

5. **More agent tools** — Budget optimizer (suggest reallocation), anomaly detection (flag unusual metric changes), creative performance analyzer (compare ad creatives).

6. **Scheduled reports** — Cron-based report generation with email delivery. Weekly performance digests for campaign managers.

7. **Data connectors** — Ingest real campaign data from ad platforms (Google Ads, Meta, DV360) instead of mock data. The schema already models the right fields.

8. **Multi-tenant support** — Isolate data by client/team. The `client_name` field is already indexed for filtering.

### Long-term

9. **Fine-tuned models** — Train a domain-specific model on historical campaign reports for higher-quality analysis and lower latency.

10. **Real-time dashboards** — WebSocket-based live metrics updates alongside the chat interface.

---

## Test Architecture

```
tests/
  conftest.py       → Shared fixtures
                       MockLLMClient (deterministic, keyword-based)
                       MockRAGService (canned search results)
                       Sample data (metrics, campaigns, reports, comparisons)
                       Async HTTP client (deferred import to avoid DB deps)

  test_agent.py     → 20 tests
                       Tool unit tests (SQL validation, serialization)
                       Graph structure (compilation, node presence, tool registry)
                       Routing logic (tool calls, no tools, error limits)
                       invoke_agent (return shape, exception handling)

  test_report.py    → 30 tests
                       Markdown (11): headers, metrics, sections, edge cases
                       PDF (5): bytes, header, size, pages, no-metrics
                       Comparison (6): names, table, winners, differences
                       Slack (8): formatting, length, emoji, no-markdown-bold

  test_api.py       → 12 tests
                       Smoke tests for all endpoints
                       Input validation (empty message, invalid format)
                       Response shape (health endpoint fields)
                       Middleware (processing time header)
                       DB-dependent tests skip gracefully without PostgreSQL
```

Tests are designed to run without external services. The mock LLM and RAG fixtures ensure deterministic behavior. DB-dependent API tests use `pytest.skip()` when PostgreSQL is unavailable, keeping CI green without Docker services.

---

*This architecture reflects production engineering practices: async I/O, structured LLM outputs, defensive error handling, and clear separation of concerns between the API, agent, and service layers.*
