"""Pydantic models for API request/response validation and structured LLM output.

Groups:
    API schemas — ChatRequest, ChatResponse, CampaignCreate, CampaignResponse.
    LLM structured output — LCIReportSchema, CampaignComparisonSchema,
                             AudienceRecommendationSchema.
"""

from datetime import date, datetime

from pydantic import BaseModel, Field

from app.models.campaign import CampaignStatus, TargetingType, Vertical


# ---------------------------------------------------------------------------
# API request / response schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Incoming chat message from the user."""

    message: str = Field(..., min_length=1, description="User's natural-language query.")
    conversation_id: str | None = Field(None, description="Optional conversation session ID.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"message": "How did Dunkin's Q4 campaign perform?"},
                {
                    "message": "Compare Dunkin Q3 vs Q4",
                    "conversation_id": "abc-123",
                },
            ]
        }
    }


class ChatResponse(BaseModel):
    """Response returned from the /chat endpoint."""

    response: str = Field(..., description="Agent's natural-language response.")
    conversation_id: str = Field(..., description="Conversation session ID.")
    tools_used: list[str] = Field(
        default_factory=list, description="Names of agent tools invoked."
    )
    sources: list[str] = Field(
        default_factory=list, description="Source campaign names referenced."
    )
    processing_time_ms: int = Field(
        ..., description="Total processing time in milliseconds."
    )
    data: dict | None = Field(
        None, description="Optional structured data payload."
    )


class MetricsOut(BaseModel):
    """Campaign metrics returned by the API."""

    impressions: int
    visit_lift_percent: float
    sales_lift_percent: float
    incremental_roas: float
    incremental_visits: int
    incremental_sales_dollars: float
    avg_basket_size: float
    purchase_frequency: float
    top_markets: list[str]
    top_performing_creative: str | None = None
    control_group_size: int
    exposed_group_size: int

    model_config = {"from_attributes": True}


class AudienceSegmentOut(BaseModel):
    """Audience segment returned by the API."""

    id: int
    segment_name: str

    model_config = {"from_attributes": True}


class CampaignCreate(BaseModel):
    """Schema for creating a new campaign via the API."""

    campaign_name: str = Field(..., max_length=255)
    client_name: str = Field(..., max_length=255)
    vertical: Vertical
    start_date: date
    end_date: date | None = None
    budget: float = Field(..., gt=0)
    status: CampaignStatus = CampaignStatus.PLANNED
    targeting_type: TargetingType
    campaign_summary: str | None = None
    audience_segments: list[str] = Field(
        default_factory=list, description="Segment names to attach."
    )


class CampaignResponse(BaseModel):
    """Full campaign record returned by the API."""

    id: int
    campaign_id: str
    campaign_name: str
    client_name: str
    vertical: Vertical
    start_date: date
    end_date: date | None = None
    budget: float
    status: CampaignStatus
    targeting_type: TargetingType
    campaign_summary: str | None = None
    created_at: datetime
    metrics: MetricsOut | None = None
    audience_segments: list[AudienceSegmentOut] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class ReportResponse(BaseModel):
    """Metadata for a generated report."""

    report_id: str
    campaign_id: int
    generated_at: datetime
    download_url: str


class PaginatedCampaigns(BaseModel):
    """Paginated list of campaigns."""

    campaigns: list["CampaignResponse"] = Field(default_factory=list)
    total: int
    page: int
    limit: int
    pages: int


class ReportGenerateRequest(BaseModel):
    """Request to generate a campaign report."""

    campaign_id: int = Field(..., description="Database ID of the campaign.")
    format: str = Field(
        "markdown",
        description="Output format: 'markdown', 'pdf', or 'slack'.",
        pattern="^(markdown|pdf|slack)$",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"campaign_id": 2, "format": "markdown"},
                {"campaign_id": 5, "format": "pdf"},
            ]
        }
    }


class CompareRequest(BaseModel):
    """Request to compare two campaigns."""

    campaign_id_1: int = Field(..., description="Database ID of the first campaign.")
    campaign_id_2: int = Field(..., description="Database ID of the second campaign.")


class AudienceRecommendRequest(BaseModel):
    """Request for audience segment recommendation."""

    description: str = Field(
        ...,
        min_length=5,
        description="Natural language description of target audience or campaign context.",
    )
    vertical: str | None = Field(
        None, description="Optional vertical filter (QSR, Automotive, CPG, Retail, Entertainment)."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "lunch-time diners in Texas for a QSR brand",
                    "vertical": "QSR",
                },
            ]
        }
    }


class ReportTextResponse(BaseModel):
    """Response containing a text-format report."""

    campaign_id: int
    campaign_name: str
    format: str
    content: str
    generated_at: datetime


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "0.1.0"
    database: str = "unknown"
    vector_store: str = "unknown"
    llm: str = "unknown"


# ---------------------------------------------------------------------------
# Structured LLM output schemas
# ---------------------------------------------------------------------------


class VisitLiftAnalysis(BaseModel):
    """Detailed visit lift analysis section for LCI reports."""

    overall_lift: str = Field(
        ..., description="Summary of visit lift performance."
    )
    market_breakdown: list[str] = Field(
        ..., description="Per-market visit lift observations."
    )
    daypart_insights: str = Field(
        ..., description="Insights on which dayparts drove visits."
    )


class SalesLiftAnalysis(BaseModel):
    """Detailed sales lift analysis section for LCI reports."""

    overall_lift: str = Field(
        ..., description="Summary of sales lift performance."
    )
    basket_size_analysis: str = Field(
        ..., description="Analysis of average basket size trends."
    )
    purchase_frequency_insight: str = Field(
        ..., description="Insight on repeat purchase behavior."
    )


class MarketBreakdown(BaseModel):
    """Per-market performance breakdown."""

    market_name: str
    performance_summary: str
    relative_ranking: str = Field(
        ..., description="How this market ranks vs. others (top, average, below)."
    )


class LCIReportSchema(BaseModel):
    """Structured output schema for a full Location Conversion Index report.

    The LLM fills each section; the report generator renders it to PDF.
    """

    campaign_name: str
    client_name: str
    report_date: str
    executive_summary: str = Field(
        ...,
        description="2-3 paragraph executive summary of campaign performance.",
    )
    visit_lift_analysis: VisitLiftAnalysis
    sales_lift_analysis: SalesLiftAnalysis
    market_breakdown: list[MarketBreakdown] = Field(
        ..., description="Performance breakdown for each top market."
    )
    recommendations: list[str] = Field(
        ...,
        description="3-5 actionable recommendations for future campaigns.",
    )
    methodology_note: str = Field(
        default="Results measured using InMarket's deterministic location data with exposed vs. control group methodology.",
        description="Standard methodology disclaimer.",
    )


class CampaignComparisonSchema(BaseModel):
    """Structured output for comparing two campaigns side by side."""

    campaign_a_name: str
    campaign_b_name: str
    comparison_summary: str = Field(
        ...,
        description="Overall comparison narrative highlighting which campaign performed better and why.",
    )
    metric_comparisons: list[dict[str, str]] = Field(
        ...,
        description="List of {metric, campaign_a_value, campaign_b_value, winner, insight} dicts.",
    )
    key_differences: list[str] = Field(
        ..., description="Top 3-5 factors explaining the performance difference."
    )
    recommendation: str = Field(
        ..., description="Which approach to favor in future campaigns and why."
    )


class AudienceSegmentRecommendation(BaseModel):
    """A single audience segment recommendation."""

    segment_name: str
    rationale: str = Field(
        ..., description="Why this segment is recommended for the campaign."
    )
    estimated_reach: str = Field(
        ..., description="Estimated reachable audience size."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score 0-1."
    )
    supporting_evidence: str = Field(
        ..., description="Historical campaign data supporting this recommendation."
    )


class AudienceRecommendationSchema(BaseModel):
    """Structured output for audience segment recommendations."""

    campaign_name: str
    vertical: str
    recommended_segments: list[AudienceSegmentRecommendation] = Field(
        ..., description="Ranked list of recommended audience segments."
    )
    segments_to_avoid: list[str] = Field(
        default_factory=list,
        description="Segments that historically underperformed for this vertical.",
    )
    overall_strategy: str = Field(
        ..., description="High-level targeting strategy recommendation."
    )
