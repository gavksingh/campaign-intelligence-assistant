"""FastAPI route definitions for the Campaign Intelligence Assistant.

Endpoints:
    POST /api/chat              — Natural-language query via the LangGraph agent.
    GET  /api/campaigns         — List campaigns with filters and pagination.
    GET  /api/campaigns/{id}    — Single campaign with full metrics.
    POST /api/reports/generate  — Generate LCI report (markdown, PDF, or Slack).
    POST /api/reports/compare   — Compare two campaigns side-by-side.
    POST /api/audience/recommend — AI audience segment recommendations.
    GET  /api/health            — Dependency health check.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from math import ceil

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.database import get_session_factory
from app.models.campaign import (
    Campaign,
    CampaignStatus,
    Vertical,
)
from app.models.schemas import (
    AudienceRecommendRequest,
    CampaignResponse,
    ChatRequest,
    ChatResponse,
    CompareRequest,
    HealthResponse,
    PaginatedCampaigns,
    ReportGenerateRequest,
    ReportTextResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["api"])


# ── Helpers ───────────────────────────────────────────────────────────


def _campaign_to_response(campaign: Campaign) -> CampaignResponse:
    """Convert a Campaign ORM object to a CampaignResponse schema."""
    return CampaignResponse(
        id=campaign.id,
        campaign_id=str(campaign.campaign_id),
        campaign_name=campaign.campaign_name,
        client_name=campaign.client_name,
        vertical=campaign.vertical,
        start_date=campaign.start_date,
        end_date=campaign.end_date,
        budget=campaign.budget,
        status=campaign.status,
        targeting_type=campaign.targeting_type,
        campaign_summary=campaign.campaign_summary,
        created_at=campaign.created_at,
        metrics=campaign.metrics,
        audience_segments=campaign.audience_segments or [],
    )


# ── POST /api/chat ────────────────────────────────────────────────────


@router.post(
    "/chat",
    summary="Chat with the campaign intelligence agent",
    description="Send a natural-language query and receive an AI-powered response "
    "grounded in campaign performance data. Set stream=true for SSE streaming.",
)
async def chat(
    request: ChatRequest,
    stream: bool = Query(False, description="Enable SSE streaming response"),
):
    """Process a natural-language query through the LangGraph campaign agent."""
    conversation_id = request.conversation_id or str(uuid.uuid4())

    logger.info(
        "POST /api/chat | conversation_id=%s | stream=%s | message='%s'",
        conversation_id,
        stream,
        request.message[:80],
    )

    if stream:
        return StreamingResponse(
            _stream_chat(request.message, conversation_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    from app.agents.campaign_agent import invoke_agent

    start = time.monotonic()

    try:
        result = await invoke_agent(request.message, session_id=conversation_id)
    except Exception as e:
        logger.error("Agent invocation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    elapsed_ms = int((time.monotonic() - start) * 1000)

    # Extract tool names from result data if present
    tools_used: list[str] = []
    if result.get("data"):
        data = result["data"]
        if "comparison_summary" in data:
            tools_used.append("compare_campaigns")
        elif "executive_summary" in data:
            tools_used.append("generate_lci_report")
        elif "recommended_segments" in data:
            tools_used.append("recommend_audience")

    logger.info(
        "POST /api/chat completed | conversation_id=%s | elapsed_ms=%d",
        conversation_id,
        elapsed_ms,
    )

    return ChatResponse(
        response=result.get("reply", "No response generated."),
        conversation_id=conversation_id,
        tools_used=tools_used,
        sources=result.get("sources", []),
        processing_time_ms=elapsed_ms,
        data=result.get("data"),
    )


async def _stream_chat(message: str, conversation_id: str):
    """SSE async generator for streaming chat responses.

    Yields SSE-formatted events with JSON data chunks.
    """
    from app.agents.campaign_agent import stream_agent

    try:
        async for chunk in stream_agent(message, session_id=conversation_id):
            yield f"data: {json.dumps(chunk)}\n\n"
    except Exception as e:
        logger.error("SSE stream error: %s", e, exc_info=True)
        yield f"data: {json.dumps({'content': 'Stream error occurred.', 'done': True})}\n\n"


# ── GET /api/campaigns ────────────────────────────────────────────────


@router.get(
    "/campaigns",
    response_model=PaginatedCampaigns,
    summary="List campaigns with optional filters",
    description="Retrieve campaigns filtered by vertical, status, or client name. "
    "Supports pagination.",
)
async def list_campaigns(
    vertical: str | None = Query(
        None,
        description="Filter by vertical (QSR, Automotive, CPG, Retail, Entertainment)",
    ),
    status: str | None = Query(
        None, description="Filter by status (completed, active, planned, paused)"
    ),
    client: str | None = Query(
        None, description="Filter by client name (case-insensitive partial match)"
    ),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Results per page"),
) -> PaginatedCampaigns:
    """Return a paginated list of campaigns with optional filters."""
    logger.info(
        "GET /api/campaigns | vertical=%s status=%s client=%s page=%d limit=%d",
        vertical,
        status,
        client,
        page,
        limit,
    )

    try:
        factory = get_session_factory()
        async with factory() as session:
            query = select(Campaign).options(
                selectinload(Campaign.metrics),
                selectinload(Campaign.audience_segments),
            )
            count_query = select(func.count(Campaign.id))

            # Apply filters
            if vertical:
                try:
                    v = Vertical(vertical)
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid vertical '{vertical}'. Must be one of: {[e.value for e in Vertical]}",
                    )
                query = query.where(Campaign.vertical == v)
                count_query = count_query.where(Campaign.vertical == v)

            if status:
                try:
                    s = CampaignStatus(status)
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid status '{status}'. Must be one of: {[e.value for e in CampaignStatus]}",
                    )
                query = query.where(Campaign.status == s)
                count_query = count_query.where(Campaign.status == s)

            if client:
                query = query.where(Campaign.client_name.ilike(f"%{client}%"))
                count_query = count_query.where(
                    Campaign.client_name.ilike(f"%{client}%")
                )

            # Get total count
            total_result = await session.execute(count_query)
            total = total_result.scalar_one()

            # Apply pagination
            offset = (page - 1) * limit
            query = query.order_by(Campaign.id).offset(offset).limit(limit)

            result = await session.execute(query)
            campaigns = result.scalars().all()

            campaign_responses = [_campaign_to_response(c) for c in campaigns]

        return PaginatedCampaigns(
            campaigns=campaign_responses,
            total=total,
            page=page,
            limit=limit,
            pages=ceil(total / limit) if total > 0 else 0,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("list_campaigns failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list campaigns: {e}")


# ── GET /api/campaigns/{campaign_id} ─────────────────────────────────


@router.get(
    "/campaigns/{campaign_id}",
    response_model=CampaignResponse,
    summary="Get a single campaign with full details",
    description="Retrieve a campaign by its database ID, including metrics and audience segments.",
)
async def get_campaign(campaign_id: int) -> CampaignResponse:
    """Return full campaign details by database ID."""
    logger.info("GET /api/campaigns/%d", campaign_id)

    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(Campaign)
            .options(
                selectinload(Campaign.metrics),
                selectinload(Campaign.audience_segments),
            )
            .where(Campaign.id == campaign_id)
        )
        campaign = result.scalar_one_or_none()
        if not campaign:
            raise HTTPException(
                status_code=404, detail=f"Campaign {campaign_id} not found."
            )
        return _campaign_to_response(campaign)


# ── POST /api/reports/generate ────────────────────────────────────────


@router.post(
    "/reports/generate",
    summary="Generate an LCI report for a campaign",
    description="Generate a campaign performance report in markdown, PDF, or Slack format.",
)
async def generate_report(request: ReportGenerateRequest):
    """Generate an LCI report using the agent and report service."""
    from app.agents.tools import generate_lci_report
    from app.services.report_gen import ReportGenerator

    logger.info(
        "POST /api/reports/generate | campaign_id=%d format=%s",
        request.campaign_id,
        request.format,
    )

    # Fetch campaign for the report generator
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(Campaign)
            .options(
                selectinload(Campaign.metrics),
                selectinload(Campaign.audience_segments),
            )
            .where(Campaign.id == request.campaign_id)
        )
        campaign = result.scalar_one_or_none()
        if not campaign:
            raise HTTPException(
                status_code=404, detail=f"Campaign {request.campaign_id} not found."
            )
        campaign_response = _campaign_to_response(campaign)
        campaign_name = campaign.campaign_name

    # Generate structured report data via the agent tool
    try:
        raw_report = await generate_lci_report.ainvoke(
            {"campaign_id": request.campaign_id}
        )
        report_data_dict = json.loads(raw_report)
    except Exception as e:
        logger.error("Report generation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    if "error" in report_data_dict:
        raise HTTPException(status_code=500, detail=report_data_dict["error"])

    # Parse into LCIReportSchema
    from app.models.schemas import LCIReportSchema

    try:
        report_schema = LCIReportSchema.model_validate(report_data_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse report data: {e}")

    generator = ReportGenerator()

    if request.format == "pdf":
        pdf_bytes = generator.generate_pdf_report(report_schema, campaign_response)
        filename = f"report_{campaign_name.replace(' ', '_')}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    if request.format == "slack":
        content = generator.generate_slack_summary(report_schema, campaign_response)
    else:
        content = generator.generate_markdown_report(report_schema, campaign_response)

    return ReportTextResponse(
        campaign_id=request.campaign_id,
        campaign_name=campaign_name,
        format=request.format,
        content=content,
        generated_at=datetime.now(timezone.utc),
    )


# ── POST /api/reports/compare ─────────────────────────────────────────


@router.post(
    "/reports/compare",
    summary="Compare two campaigns side-by-side",
    description="Generate a detailed comparison between two campaigns with metric-level analysis.",
)
async def compare_campaigns_endpoint(request: CompareRequest):
    """Compare two campaigns and return a formatted comparison report."""
    from app.agents.tools import compare_campaigns
    from app.services.report_gen import ReportGenerator

    logger.info(
        "POST /api/reports/compare | campaign_id_1=%d campaign_id_2=%d",
        request.campaign_id_1,
        request.campaign_id_2,
    )

    try:
        raw_result = await compare_campaigns.ainvoke(
            {
                "campaign_id_1": request.campaign_id_1,
                "campaign_id_2": request.campaign_id_2,
            }
        )
        comparison_dict = json.loads(raw_result)
    except Exception as e:
        logger.error("Comparison failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")

    if "error" in comparison_dict:
        raise HTTPException(status_code=500, detail=comparison_dict["error"])

    from app.models.schemas import CampaignComparisonSchema

    try:
        comparison = CampaignComparisonSchema.model_validate(comparison_dict)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to parse comparison data: {e}"
        )

    generator = ReportGenerator()
    markdown = generator.generate_comparison_report(comparison)

    return {
        "campaign_a": comparison.campaign_a_name,
        "campaign_b": comparison.campaign_b_name,
        "comparison_summary": comparison.comparison_summary,
        "markdown_report": markdown,
        "structured_data": comparison.model_dump(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ── POST /api/audience/recommend ──────────────────────────────────────


@router.post(
    "/audience/recommend",
    summary="Get AI-powered audience segment recommendations",
    description="Describe your target audience and get data-driven segment recommendations "
    "based on historical campaign performance.",
)
async def recommend_audience_endpoint(request: AudienceRecommendRequest):
    """Return audience recommendations based on description and history."""
    from app.agents.tools import recommend_audience

    logger.info(
        "POST /api/audience/recommend | description='%s' vertical=%s",
        request.description[:80],
        request.vertical,
    )

    # Build the description with vertical context if provided
    description = request.description
    if request.vertical:
        description = f"{description} (vertical: {request.vertical})"

    try:
        raw_result = await recommend_audience.ainvoke({"description": description})
        recommendation_dict = json.loads(raw_result)
    except Exception as e:
        logger.error("Audience recommendation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {e}")

    if "error" in recommendation_dict:
        raise HTTPException(status_code=500, detail=recommendation_dict["error"])

    return {
        "recommendation": recommendation_dict,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ── GET /api/debug/campaigns ──────────────────────────────────────────


@router.get("/debug/campaigns")
async def debug_campaigns():
    """Debug endpoint to test campaign query step by step."""
    steps = {}
    try:
        factory = get_session_factory()
        steps["factory"] = "ok"

        async with factory() as session:
            steps["session"] = "ok"

            # Step 1: simple count
            result = await session.execute(select(func.count(Campaign.id)))
            count = result.scalar_one()
            steps["count"] = count

            # Step 2: fetch one campaign without relationships
            result = await session.execute(
                select(Campaign).order_by(Campaign.id).limit(1)
            )
            campaign = result.scalar_one_or_none()
            if campaign:
                steps["campaign_found"] = True
                steps["campaign_id_type"] = type(campaign.campaign_id).__name__
                steps["campaign_name"] = campaign.campaign_name
                steps["created_at"] = str(campaign.created_at)
                steps["created_at_type"] = type(campaign.created_at).__name__

                # Step 3: try model_validate
                try:
                    resp = CampaignResponse(
                        id=campaign.id,
                        campaign_id=str(campaign.campaign_id),
                        campaign_name=campaign.campaign_name,
                        client_name=campaign.client_name,
                        vertical=campaign.vertical,
                        start_date=campaign.start_date,
                        end_date=campaign.end_date,
                        budget=campaign.budget,
                        status=campaign.status,
                        targeting_type=campaign.targeting_type,
                        campaign_summary=campaign.campaign_summary,
                        created_at=campaign.created_at,
                    )
                    steps["pydantic_basic"] = "ok"
                except Exception as e:
                    steps["pydantic_basic_error"] = str(e)

            # Step 4: fetch with relationships
            result = await session.execute(
                select(Campaign)
                .options(
                    selectinload(Campaign.metrics),
                    selectinload(Campaign.audience_segments),
                )
                .order_by(Campaign.id)
                .limit(1)
            )
            campaign = result.scalar_one_or_none()
            if campaign:
                steps["with_relationships"] = True
                steps["metrics_type"] = type(campaign.metrics).__name__
                steps["metrics_is_none"] = campaign.metrics is None
                steps["segments_count"] = (
                    len(campaign.audience_segments) if campaign.audience_segments else 0
                )

                try:
                    resp = _campaign_to_response(campaign)
                    steps["full_response"] = "ok"
                    steps["response_preview"] = resp.model_dump(
                        include={"id", "campaign_name"}
                    )
                except Exception as e:
                    steps["full_response_error"] = f"{type(e).__name__}: {e}"

    except Exception as e:
        steps["error"] = f"{type(e).__name__}: {e}"

    return steps


# ── GET /api/health ───────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Check the health of all dependencies: database, vector store, and LLM.",
)
async def health_check() -> HealthResponse:
    """Check connectivity to database, vector store, and LLM."""
    db_status = "unknown"
    vector_status = "unknown"
    llm_status = "unknown"

    # Check database
    try:
        factory = get_session_factory()
        async with factory() as session:
            await session.execute(select(func.count(Campaign.id)))
        db_status = "connected"
    except Exception as e:
        logger.warning("Health check — database failed: %s", e)
        db_status = f"error: {e}"

    # Check vector store (pgvector)
    try:
        from app.services.rag import get_rag_service

        rag = get_rag_service()
        stats = await rag.get_collection_stats()
        vector_status = f"ready ({stats['document_count']} documents)"
    except Exception as e:
        logger.warning("Health check — vector store failed: %s", e)
        vector_status = f"error: {e}"

    # Check LLM availability (just verify client can be constructed)
    try:
        from app.services.llm_client import get_llm_client

        client = get_llm_client()
        llm_status = f"available (model: {client._default_model})"
    except Exception as e:
        logger.warning("Health check — LLM failed: %s", e)
        llm_status = f"error: {e}"

    overall = "healthy" if "error" not in db_status else "degraded"

    return HealthResponse(
        status=overall,
        database=db_status,
        vector_store=vector_status,
        llm=llm_status,
    )
