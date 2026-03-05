# Demo Script — Campaign Intelligence Assistant

A step-by-step guide for demonstrating the system's capabilities.

---

## Setup (before demo)

```bash
# Start infrastructure
docker compose up -d postgres

# Seed data (18 campaigns across 15 brands, 5 verticals)
make seed

# Start API server
make run

# In a separate terminal, start the UI
make ui
```

Verify: Open http://localhost:8501 — the sidebar should show a green "Online" status.

---

## Demo Flow

### 1. Basic Campaign Query (Agent + SQL Tool)

**Type in chat:**
> "What are the top 5 campaigns by ROAS?"

**What happens:**
- The LangGraph agent routes to the `query_campaign_data` tool
- LLM generates a SQL query against the campaign schema
- Results are returned as a formatted table
- Tool badge appears showing which tools were used

**Talking point:** The agent translates natural language into safe, read-only SQL. All generated queries are validated to be SELECT-only before execution.

---

### 2. Semantic Search (RAG Tool)

**Type in chat:**
> "Find campaigns similar to a QSR holiday promotion"

**What happens:**
- Agent uses `search_similar_campaigns` tool
- pgvector performs semantic search over campaign embeddings
- Results include similarity scores and key metrics

**Talking point:** RAG retrieval enables fuzzy matching — you don't need exact campaign names or IDs. The system uses Gemini embeddings stored in pgvector with hybrid search combining vector similarity and SQL results.

---

### 3. Campaign Comparison (Compare Tool)

**Option A — via chat:**
> "Compare the Dunkin' Q3 Summer Iced Coffee campaign with the Q4 Holiday Favorites"

**Option B — via sidebar:**
1. Click "Compare Campaigns" in the sidebar
2. Enter Campaign ID 1 and Campaign ID 2
3. Click "Compare"

**What happens:**
- Agent fetches both campaigns with full metrics
- LLM generates a structured `CampaignComparisonSchema`
- Output includes a metric-by-metric table with winners bolded
- Key differences and a recommendation are provided

**Talking point:** Structured output (Pydantic schemas) ensures the LLM response always has the right shape — no parsing failures, no missing fields.

---

### 4. Report Generation (Report Tool)

**Option A — via chat:**
> "Generate an LCI report for campaign 2"

**Option B — via sidebar:**
1. Click "Generate Report" in the sidebar
2. Enter Campaign ID: 2
3. Select format: Markdown, PDF, or Slack
4. Click "Generate"

**What happens:**
- Agent fetches campaign data and metrics
- LLM generates structured `LCIReportSchema` with executive summary, visit/sales lift analysis, market breakdown, and recommendations
- ReportGenerator formats into the selected output (Markdown/PDF/Slack)

**Talking point:** The PDF report includes a branded title page, styled metrics table, and professional formatting — ready to share with clients. The Slack summary is under 500 characters, ideal for channel notifications.

---

### 5. Audience Recommendation (Audience Tool)

**Type in chat:**
> "Recommend audience segments for a grocery chain launching a summer grilling promotion in Texas"

**What happens:**
- Agent searches similar campaigns via RAG
- Queries existing audience segments from the database
- LLM generates `AudienceRecommendationSchema` with segment suggestions

**Talking point:** The system combines historical campaign performance with semantic understanding to suggest relevant audience segments.

---

### 6. Health Check & API (optional)

**Show the API docs:**
- Open http://localhost:8080/docs (Swagger UI)
- Show the health endpoint: GET `/api/health` — displays DB, pgvector, and LLM connectivity

**Show a direct API call:**
```bash
curl -s http://localhost:8080/api/campaigns?limit=3 | python -m json.tool
```

---

## Architecture Highlights to Mention

1. **LangGraph Agent** — Multi-step reasoning with router → tool executor → synthesizer flow. Error handling with retry logic (max 2 retries).

2. **Structured Output** — All LLM responses use Pydantic schemas via Gemini's `response_mime_type="application/json"` parameter for reliable JSON output.

3. **Async Everything** — FastAPI with async SQLAlchemy (asyncpg), async LLM calls, async RAG retrieval. No blocking I/O.

4. **Hybrid Search** — Combines SQL database queries with pgvector semantic search, deduplicates, and boosts results found in both sources.

5. **62 Tests** — Comprehensive test suite covering agent routing, tool execution, report generation (all 4 formats), and API endpoints. Mock LLM and RAG services for deterministic testing.

6. **Production Patterns** — Request timing middleware, structured logging, health checks, pagination, input validation, CORS configuration.

---

## Sample Queries to Try

- "Which verticals have the highest average ROAS?"
- "Show me all active campaigns"
- "What's the average visit lift for QSR campaigns?"
- "Compare campaign 3 with campaign 5"
- "Generate a report for the Toyota campaign"
- "What audience should I target for a CPG snack brand in the Midwest?"
