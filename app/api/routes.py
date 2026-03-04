"""FastAPI route definitions.

Endpoints:
    POST /chat          — Send a natural-language query to the campaign agent.
    GET  /report/{id}   — Retrieve or generate a campaign performance report.
    GET  /health        — Health check returning service status.
"""

from fastapi import APIRouter

from app.models.schemas import ChatRequest, ChatResponse, HealthResponse, ReportResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return service health status."""
    return HealthResponse()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a natural-language query through the campaign agent.

    Accepts a user message, runs it through the LangGraph agent with
    access to campaign tools, and returns the agent's response.
    """
    raise NotImplementedError("Chat endpoint not yet implemented.")


@router.get("/report/{campaign_id}", response_model=ReportResponse)
async def get_report(campaign_id: int) -> ReportResponse:
    """Generate or retrieve a campaign performance report.

    Triggers the report generation pipeline for the specified campaign
    and returns metadata with a download link.
    """
    raise NotImplementedError("Report endpoint not yet implemented.")
