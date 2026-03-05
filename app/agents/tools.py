"""Agent tools for campaign data operations.

Each tool is a LangGraph-compatible function decorated with @tool that the
campaign agent can invoke. Tools handle: SQL querying, semantic search,
campaign comparison, LCI report generation, and audience recommendations.
"""

from __future__ import annotations

import json
import logging
import uuid

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.database import async_session_factory
from app.models.schemas import (
    AudienceRecommendationSchema,
    CampaignComparisonSchema,
    LCIReportSchema,
)
from app.services.llm_client import get_llm_client
from app.services.rag import get_rag_service

logger = logging.getLogger(__name__)

# ── Table schema description injected into SQL-generation prompts ─────

TABLE_SCHEMA = """
Tables:
  campaigns (
    id INTEGER PRIMARY KEY,
    campaign_id UUID,
    campaign_name VARCHAR(255),
    client_name VARCHAR(255),
    vertical VARCHAR (one of: QSR, Automotive, CPG, Retail, Entertainment),
    start_date DATE,
    end_date DATE,
    budget FLOAT,
    status VARCHAR (one of: completed, active, planned, paused),
    targeting_type VARCHAR (one of: Moments, Predictive Moments, GeoLink, Audience-based),
    campaign_summary TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
  )

  campaign_metrics (
    id INTEGER PRIMARY KEY,
    campaign_id INTEGER REFERENCES campaigns(id),
    impressions INTEGER,
    visit_lift_percent FLOAT,
    sales_lift_percent FLOAT,
    incremental_roas FLOAT,
    incremental_visits INTEGER,
    incremental_sales_dollars FLOAT,
    avg_basket_size FLOAT,
    purchase_frequency FLOAT,
    top_markets TEXT[],
    top_performing_creative VARCHAR(500),
    control_group_size INTEGER,
    exposed_group_size INTEGER
  )

  audience_segments (
    id INTEGER PRIMARY KEY,
    campaign_id INTEGER REFERENCES campaigns(id),
    segment_name VARCHAR(255)
  )
"""

SQL_GENERATION_SYSTEM_PROMPT = f"""You are a SQL query generator for a campaign analytics database.
Given a natural language question, generate a read-only PostgreSQL SELECT query.

{TABLE_SCHEMA}

Rules:
- ONLY generate SELECT statements. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, or any DDL.
- Always JOIN campaign_metrics ON campaign_metrics.campaign_id = campaigns.id when metrics are needed.
- Use ILIKE for text matching.
- For date ranges: Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec.
- Return ONLY the SQL query, no explanation or markdown.
- Limit results to 20 rows unless the user asks for more.
"""


async def _execute_readonly_sql(query: str) -> list[dict]:
    """Execute a read-only SQL query and return rows as dicts.

    Validates the query is a SELECT before executing.

    Args:
        query: SQL query string (must be a SELECT).

    Returns:
        List of dicts, one per row.

    Raises:
        ValueError: If the query is not a SELECT statement.
    """
    cleaned = query.strip().rstrip(";").strip()
    if not cleaned.upper().startswith("SELECT"):
        raise ValueError(f"Only SELECT queries are allowed. Got: {cleaned[:50]}")

    async with async_session_factory() as session:
        result = await session.execute(text(cleaned))
        columns = list(result.keys())
        rows = result.fetchall()
        return [dict(zip(columns, row)) for row in rows]


def _serialize_row(row: dict) -> dict:
    """Serialize a database row dict for JSON output.

    Converts non-serializable types (UUID, date, etc.) to strings.

    Args:
        row: A dict from a database query.

    Returns:
        JSON-safe dict.
    """
    out = {}
    for k, v in row.items():
        if isinstance(v, (uuid.UUID,)):
            out[k] = str(v)
        elif hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


async def _fetch_campaign_with_metrics(campaign_id: int) -> dict | None:
    """Fetch a single campaign with all metrics and segments by database id.

    Args:
        campaign_id: The integer PK of the campaign.

    Returns:
        Dict with campaign data, metrics, and segments, or None if not found.
    """
    query = """
        SELECT
            c.id, c.campaign_id, c.campaign_name, c.client_name, c.vertical,
            c.start_date, c.end_date, c.budget, c.status, c.targeting_type,
            c.campaign_summary,
            m.impressions, m.visit_lift_percent, m.sales_lift_percent,
            m.incremental_roas, m.incremental_visits, m.incremental_sales_dollars,
            m.avg_basket_size, m.purchase_frequency, m.top_markets,
            m.top_performing_creative, m.control_group_size, m.exposed_group_size
        FROM campaigns c
        LEFT JOIN campaign_metrics m ON m.campaign_id = c.id
        WHERE c.id = :cid
    """
    async with async_session_factory() as session:
        result = await session.execute(text(query), {"cid": campaign_id})
        row = result.fetchone()
        if not row:
            return None
        data = _serialize_row(dict(zip(result.keys(), row)))

    # Fetch audience segments
    seg_query = "SELECT segment_name FROM audience_segments WHERE campaign_id = :cid"
    async with async_session_factory() as session:
        seg_result = await session.execute(text(seg_query), {"cid": campaign_id})
        data["audience_segments"] = [r[0] for r in seg_result.fetchall()]

    return data


# ── Tool 1: Query Campaign Data ──────────────────────────────────────


class QueryCampaignInput(BaseModel):
    """Input schema for query_campaign_data tool."""

    query: str = Field(
        ...,
        description="Natural language query about campaign data, e.g. 'all completed QSR campaigns in Q4 2025'",
    )


@tool("query_campaign_data", args_schema=QueryCampaignInput)
async def query_campaign_data(query: str) -> str:
    """Query the campaign database using natural language.

    Translates the query to SQL, executes it read-only against PostgreSQL,
    and returns formatted results. Use this for specific data lookups like
    filtering by client, vertical, date range, status, or metric thresholds.
    """
    llm = get_llm_client()

    try:
        # Generate SQL from natural language
        sql = await llm.chat_completion(
            messages=[
                {"role": "system", "content": SQL_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
        )

        # Clean markdown fencing if present
        sql = sql.strip()
        if sql.startswith("```"):
            lines = sql.split("\n")
            sql = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            ).strip()

        logger.info("Generated SQL: %s", sql)

        # Execute query
        rows = await _execute_readonly_sql(sql)
        serialized = [_serialize_row(r) for r in rows]

        if not serialized:
            return json.dumps(
                {"result": "No campaigns found matching your query.", "sql": sql}
            )

        return json.dumps(
            {"campaigns": serialized, "count": len(serialized), "sql": sql}, default=str
        )

    except ValueError as e:
        logger.warning("SQL validation failed: %s", e)
        return json.dumps({"error": f"Generated an unsafe query: {e}"})
    except Exception as e:
        logger.error("query_campaign_data failed: %s", e, exc_info=True)
        return json.dumps({"error": f"Query failed: {e}"})


# ── Tool 2: Search Similar Campaigns ─────────────────────────────────


class SearchSimilarInput(BaseModel):
    """Input schema for search_similar_campaigns tool."""

    query: str = Field(
        ...,
        description="Description of what kind of campaign to find, e.g. 'high-performing QSR campaigns with strong visit lift'",
    )


@tool("search_similar_campaigns", args_schema=SearchSimilarInput)
async def search_similar_campaigns(query: str) -> str:
    """Search for campaigns semantically similar to the query description.

    Uses vector similarity search over campaign embeddings in pgvector.
    Best for exploratory queries like 'find campaigns similar to X' or
    'which campaigns had the best ROAS'.
    """
    rag = get_rag_service()

    try:
        results = await rag.retrieve(query, n_results=5)

        if not results:
            return json.dumps({"result": "No similar campaigns found.", "count": 0})

        output = []
        for r in results:
            output.append(
                {
                    "campaign_name": r["metadata"].get("campaign_name", "Unknown"),
                    "client_name": r["metadata"].get("client_name", "Unknown"),
                    "vertical": r["metadata"].get("vertical", "Unknown"),
                    "status": r["metadata"].get("status", "Unknown"),
                    "incremental_roas": r["metadata"].get("incremental_roas", 0.0),
                    "visit_lift_percent": r["metadata"].get("visit_lift_percent", 0.0),
                    "sales_lift_percent": r["metadata"].get("sales_lift_percent", 0.0),
                    "similarity_score": round(1.0 - r["distance"], 3)
                    if r["distance"] < 1.0
                    else 0.0,
                    "summary_snippet": r["document"][:300] if r.get("document") else "",
                    "campaign_id": r["metadata"].get("campaign_id", ""),
                }
            )

        return json.dumps({"campaigns": output, "count": len(output)})

    except Exception as e:
        logger.error("search_similar_campaigns failed: %s", e, exc_info=True)
        return json.dumps({"error": f"Search failed: {e}"})


# ── Tool 3: Compare Campaigns ────────────────────────────────────────


class CompareCampaignsInput(BaseModel):
    """Input schema for compare_campaigns tool."""

    campaign_id_1: int = Field(
        ..., description="Database ID of the first campaign to compare."
    )
    campaign_id_2: int = Field(
        ..., description="Database ID of the second campaign to compare."
    )


@tool("compare_campaigns", args_schema=CompareCampaignsInput)
async def compare_campaigns(campaign_id_1: int, campaign_id_2: int) -> str:
    """Compare two campaigns side-by-side with detailed metric analysis.

    Fetches both campaigns from the database and uses the LLM to generate
    a structured comparison including per-metric winners and strategic insights.
    Use this when the user wants to compare Q3 vs Q4, or two different clients.
    """
    llm = get_llm_client()

    try:
        # Fetch both campaigns
        camp_a = await _fetch_campaign_with_metrics(campaign_id_1)
        camp_b = await _fetch_campaign_with_metrics(campaign_id_2)

        if not camp_a:
            return json.dumps({"error": f"Campaign {campaign_id_1} not found."})
        if not camp_b:
            return json.dumps({"error": f"Campaign {campaign_id_2} not found."})

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an adtech campaign analyst. Compare these two campaigns "
                    "and provide a detailed, data-driven comparison. Be specific with numbers."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Compare these two campaigns:\n\n"
                    f"Campaign A:\n{json.dumps(camp_a, indent=2, default=str)}\n\n"
                    f"Campaign B:\n{json.dumps(camp_b, indent=2, default=str)}"
                ),
            },
        ]

        comparison = await llm.structured_output(
            messages=messages,
            response_schema=CampaignComparisonSchema,
        )

        return comparison.model_dump_json(indent=2)

    except Exception as e:
        logger.error("compare_campaigns failed: %s", e, exc_info=True)
        return json.dumps({"error": f"Comparison failed: {e}"})


# ── Tool 4: Generate LCI Report ──────────────────────────────────────


class GenerateLCIReportInput(BaseModel):
    """Input schema for generate_lci_report tool."""

    campaign_id: int = Field(
        ..., description="Database ID of the campaign to generate a report for."
    )


@tool("generate_lci_report", args_schema=GenerateLCIReportInput)
async def generate_lci_report(campaign_id: int) -> str:
    """Generate a Location Conversion Index (LCI) attribution report for a campaign.

    Produces a structured report with executive summary, visit lift analysis,
    sales lift analysis, market breakdown, and strategic recommendations.
    This is the standard InMarket campaign performance report format.
    """
    llm = get_llm_client()

    try:
        campaign = await _fetch_campaign_with_metrics(campaign_id)

        if not campaign:
            return json.dumps({"error": f"Campaign {campaign_id} not found."})

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior campaign analyst at InMarket, an adtech company "
                    "specializing in location-based attribution. Generate a detailed "
                    "Location Conversion Index (LCI) report for the following campaign. "
                    "Use specific numbers from the data. Be professional and analytical. "
                    "The report should be suitable for sharing with the client's marketing team."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Generate an LCI report for this campaign:\n\n"
                    f"{json.dumps(campaign, indent=2, default=str)}"
                ),
            },
        ]

        report = await llm.structured_output(
            messages=messages,
            response_schema=LCIReportSchema,
        )

        return report.model_dump_json(indent=2)

    except Exception as e:
        logger.error("generate_lci_report failed: %s", e, exc_info=True)
        return json.dumps({"error": f"Report generation failed: {e}"})


# ── Tool 5: Recommend Audience ────────────────────────────────────────


class RecommendAudienceInput(BaseModel):
    """Input schema for recommend_audience tool."""

    description: str = Field(
        ...,
        description=(
            "Natural language description of the target audience or campaign context, "
            "e.g. 'lunch-time diners in Texas for a QSR brand' or "
            "'suburban families for a CPG back-to-school campaign'"
        ),
    )


@tool("recommend_audience", args_schema=RecommendAudienceInput)
async def recommend_audience(description: str) -> str:
    """Recommend audience segments based on a description and historical data.

    Searches existing audience segments across past campaigns, finds what
    worked well for similar campaigns, and generates AI-powered recommendations
    with confidence scores and supporting evidence.
    """
    llm = get_llm_client()
    rag = get_rag_service()

    try:
        # Find similar campaigns for context
        similar = await rag.retrieve(description, n_results=5)

        # Gather all audience segments from the database
        seg_query = """
            SELECT DISTINCT
                a.segment_name,
                c.vertical,
                c.targeting_type,
                m.incremental_roas,
                m.visit_lift_percent,
                c.campaign_name
            FROM audience_segments a
            JOIN campaigns c ON c.id = a.campaign_id
            LEFT JOIN campaign_metrics m ON m.campaign_id = c.id
            ORDER BY m.incremental_roas DESC NULLS LAST
        """
        async with async_session_factory() as session:
            result = await session.execute(text(seg_query))
            all_segments = [
                _serialize_row(dict(zip(result.keys(), row)))
                for row in result.fetchall()
            ]

        # Build context from similar campaigns
        similar_context = ""
        for s in similar:
            meta = s.get("metadata", {})
            similar_context += (
                f"- {meta.get('campaign_name', 'Unknown')}: "
                f"ROAS=${meta.get('incremental_roas', 0):.1f}, "
                f"Visit Lift={meta.get('visit_lift_percent', 0)}%\n"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an adtech audience strategist at InMarket. "
                    "Based on historical campaign performance data, recommend the "
                    "best audience segments for the described campaign. "
                    "Provide confidence scores based on how well similar segments "
                    "performed in past campaigns. Be specific and data-driven."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Target audience description: {description}\n\n"
                    f"Similar past campaigns:\n{similar_context}\n\n"
                    f"All available audience segments and their performance:\n"
                    f"{json.dumps(all_segments[:30], indent=2, default=str)}\n\n"
                    f"Recommend the best audience segments for this campaign."
                ),
            },
        ]

        recommendation = await llm.structured_output(
            messages=messages,
            response_schema=AudienceRecommendationSchema,
        )

        return recommendation.model_dump_json(indent=2)

    except Exception as e:
        logger.error("recommend_audience failed: %s", e, exc_info=True)
        return json.dumps({"error": f"Audience recommendation failed: {e}"})


# ── Tool registry ─────────────────────────────────────────────────────

ALL_TOOLS = [
    query_campaign_data,
    search_similar_campaigns,
    compare_campaigns,
    generate_lci_report,
    recommend_audience,
]
"""All agent tools in a list, for binding to the LLM."""
