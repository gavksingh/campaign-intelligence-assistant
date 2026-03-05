"""Shared pytest fixtures for the Campaign Intelligence Assistant test suite.

Provides:
    - Mock LLM client with deterministic responses.
    - In-memory SQLite async database for isolated tests.
    - Mock RAG service returning canned results.
    - Sample campaign data matching the mock_campaigns.json schema.
    - Async HTTP test client for FastAPI endpoint tests.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
import pytest_asyncio
from pydantic import BaseModel

from app.models.campaign import CampaignStatus, TargetingType, Vertical
from app.models.schemas import (
    AudienceSegmentOut,
    CampaignComparisonSchema,
    CampaignResponse,
    LCIReportSchema,
    MarketBreakdown,
    MetricsOut,
    SalesLiftAnalysis,
    VisitLiftAnalysis,
)


# ── Pytest config ─────────────────────────────────────────────────────


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ── Sample data fixtures ──────────────────────────────────────────────


@pytest.fixture
def sample_metrics() -> MetricsOut:
    """Realistic campaign metrics for a completed QSR campaign."""
    return MetricsOut(
        impressions=24000000,
        visit_lift_percent=18.2,
        sales_lift_percent=14.5,
        incremental_roas=42.30,
        incremental_visits=218000,
        incremental_sales_dollars=951750.00,
        avg_basket_size=11.25,
        purchase_frequency=3.8,
        top_markets=["New York", "Boston", "Chicago", "Philadelphia", "Hartford"],
        top_performing_creative="Holiday Peppermint Mocha 15s",
        control_group_size=300000,
        exposed_group_size=1500000,
    )


@pytest.fixture
def sample_campaign(sample_metrics: MetricsOut) -> CampaignResponse:
    """Full campaign response object for testing."""
    return CampaignResponse(
        id=2,
        campaign_id="a1b2c3d4-1111-4000-8000-000000000002",
        campaign_name="Dunkin' Q4 Holiday Favorites",
        client_name="Dunkin'",
        vertical=Vertical.QSR,
        start_date=date(2025, 10, 15),
        end_date=date(2025, 12, 31),
        budget=225000.00,
        status=CampaignStatus.COMPLETED,
        targeting_type=TargetingType.MOMENTS,
        campaign_summary="Dunkin' Q4 Holiday Favorites significantly outperformed Q3.",
        created_at=datetime(2025, 10, 1, tzinfo=timezone.utc),
        metrics=sample_metrics,
        audience_segments=[
            AudienceSegmentOut(id=1, segment_name="Morning Commuters"),
            AudienceSegmentOut(id=2, segment_name="Holiday Shoppers"),
        ],
    )


@pytest.fixture
def sample_campaign_b() -> CampaignResponse:
    """Second campaign for comparison tests."""
    return CampaignResponse(
        id=1,
        campaign_id="a1b2c3d4-1111-4000-8000-000000000001",
        campaign_name="Dunkin' Q3 Summer Iced Coffee",
        client_name="Dunkin'",
        vertical=Vertical.QSR,
        start_date=date(2025, 7, 1),
        end_date=date(2025, 9, 30),
        budget=175000.00,
        status=CampaignStatus.COMPLETED,
        targeting_type=TargetingType.MOMENTS,
        campaign_summary="Moderate summer performance.",
        created_at=datetime(2025, 7, 1, tzinfo=timezone.utc),
        metrics=MetricsOut(
            impressions=18500000,
            visit_lift_percent=12.4,
            sales_lift_percent=8.7,
            incremental_roas=28.50,
            incremental_visits=142000,
            incremental_sales_dollars=498750.00,
            avg_basket_size=8.75,
            purchase_frequency=3.2,
            top_markets=["New York", "Boston", "Philadelphia"],
            top_performing_creative="Iced Coffee Morning Commute 30s",
            control_group_size=250000,
            exposed_group_size=1200000,
        ),
        audience_segments=[],
    )


@pytest.fixture
def sample_report_data() -> LCIReportSchema:
    """Structured LCI report content for testing."""
    return LCIReportSchema(
        campaign_name="Dunkin' Q4 Holiday Favorites",
        client_name="Dunkin'",
        report_date="2026-01-15",
        executive_summary=(
            "The Dunkin' Q4 Holiday Favorites campaign delivered exceptional performance "
            "across all key metrics. Visit lift of 18.2% significantly exceeded the QSR "
            "category benchmark of 10%."
        ),
        visit_lift_analysis=VisitLiftAnalysis(
            overall_lift="18.2% visit lift, nearly double the category average.",
            market_breakdown=[
                "New York: 20.1% visit lift (top performer)",
                "Boston: 19.5% visit lift",
                "Chicago: 17.8% visit lift",
            ],
            daypart_insights="Morning daypart (6-10am) drove 62% of incremental visits.",
        ),
        sales_lift_analysis=SalesLiftAnalysis(
            overall_lift="Sales lift of 14.5% driven by traffic and basket size.",
            basket_size_analysis="Average basket size of $11.25, 28.6% higher than Q3.",
            purchase_frequency_insight="3.8x purchase frequency shows strong repeat engagement.",
        ),
        market_breakdown=[
            MarketBreakdown(
                market_name="New York",
                performance_summary="20.1% visit lift",
                relative_ranking="Top",
            ),
            MarketBreakdown(
                market_name="Boston",
                performance_summary="19.5% visit lift",
                relative_ranking="Top",
            ),
        ],
        recommendations=[
            "Increase Q4 holiday budget by 20%.",
            "Extend peppermint mocha creative into January.",
            "Test expanded markets (DC, Baltimore).",
        ],
    )


@pytest.fixture
def sample_comparison() -> CampaignComparisonSchema:
    """Structured comparison data for testing."""
    return CampaignComparisonSchema(
        campaign_a_name="Dunkin' Q3 Summer Iced Coffee",
        campaign_b_name="Dunkin' Q4 Holiday Favorites",
        comparison_summary="Q4 significantly outperformed Q3 across all metrics.",
        metric_comparisons=[
            {
                "metric": "Visit Lift",
                "campaign_a_value": "12.4%",
                "campaign_b_value": "18.2%",
                "winner": "Dunkin' Q4 Holiday Favorites",
                "insight": "Q4 delivered 47% higher visit lift",
            },
            {
                "metric": "ROAS",
                "campaign_a_value": "$28.50",
                "campaign_b_value": "$42.30",
                "winner": "Dunkin' Q4 Holiday Favorites",
                "insight": "Holiday creative drove stronger conversion",
            },
        ],
        key_differences=[
            "Holiday creative outperformed summer creative.",
            "Q4 basket size was 28.6% higher.",
        ],
        recommendation="Allocate larger share to Q4 holiday campaigns.",
    )


@pytest.fixture
def sample_campaign_dict() -> dict:
    """Raw campaign dict matching mock_campaigns.json format."""
    return {
        "campaign_id": "a1b2c3d4-1111-4000-8000-000000000002",
        "campaign_name": "Dunkin' Q4 Holiday Favorites",
        "client_name": "Dunkin'",
        "vertical": "QSR",
        "start_date": "2025-10-15",
        "end_date": "2025-12-31",
        "budget": 225000.00,
        "status": "completed",
        "targeting_type": "Moments",
        "metrics": {
            "impressions": 24000000,
            "visit_lift_percent": 18.2,
            "sales_lift_percent": 14.5,
            "incremental_roas": 42.30,
            "incremental_visits": 218000,
            "incremental_sales_dollars": 951750.00,
            "avg_basket_size": 11.25,
            "purchase_frequency": 3.8,
            "top_markets": ["New York", "Boston", "Chicago"],
            "top_performing_creative": "Holiday Peppermint Mocha 15s",
            "control_group_size": 300000,
            "exposed_group_size": 1500000,
        },
        "campaign_summary": "Strong Q4 performance with 18.2% visit lift.",
        "audience_segments": ["Morning Commuters", "Holiday Shoppers"],
    }


# ── Mock LLM client ──────────────────────────────────────────────────


class MockLLMClient:
    """Deterministic LLM client for testing.

    Returns canned responses based on message content keywords.
    """

    def __init__(self) -> None:
        self._default_model = "llama-3.3-70b-mock"
        self._embedding_model = "gemini-embedding-mock"
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
    ) -> str:
        """Return a deterministic response based on message content."""
        content = messages[-1].get("content", "").lower()

        if "sql" in content or "select" in content or "generate" in content:
            return "SELECT c.*, m.* FROM campaigns c LEFT JOIN campaign_metrics m ON m.campaign_id = c.id LIMIT 10"

        return "Based on the campaign data, here is my analysis."

    async def structured_output(
        self,
        messages: list[dict[str, str]],
        response_schema: type[BaseModel],
        model: str | None = None,
        temperature: float = 0.3,
    ) -> BaseModel:
        """Return a canned instance of the requested schema."""
        schema_name = response_schema.__name__

        if schema_name == "LCIReportSchema":
            return LCIReportSchema(
                campaign_name="Test Campaign",
                client_name="Test Client",
                report_date="2026-01-01",
                executive_summary="Test executive summary.",
                visit_lift_analysis=VisitLiftAnalysis(
                    overall_lift="10% visit lift.",
                    market_breakdown=["Market A: 12%"],
                    daypart_insights="Morning drove 60%.",
                ),
                sales_lift_analysis=SalesLiftAnalysis(
                    overall_lift="8% sales lift.",
                    basket_size_analysis="$10 average.",
                    purchase_frequency_insight="2.5x frequency.",
                ),
                market_breakdown=[
                    MarketBreakdown(
                        market_name="Market A",
                        performance_summary="Strong",
                        relative_ranking="Top",
                    )
                ],
                recommendations=["Increase budget.", "Expand markets."],
            )

        if schema_name == "CampaignComparisonSchema":
            return CampaignComparisonSchema(
                campaign_a_name="Campaign A",
                campaign_b_name="Campaign B",
                comparison_summary="Campaign B outperformed.",
                metric_comparisons=[
                    {
                        "metric": "ROAS",
                        "campaign_a_value": "$20",
                        "campaign_b_value": "$40",
                        "winner": "Campaign B",
                        "insight": "Double the ROAS",
                    }
                ],
                key_differences=["Higher budget drove results."],
                recommendation="Favor Campaign B approach.",
            )

        # Generic fallback
        raise ValueError(f"MockLLMClient has no canned response for {schema_name}")

    async def embed_text(self, text: str) -> list[float]:
        """Return a deterministic embedding vector."""
        return [0.1] * 3072

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic embedding vectors."""
        return [[0.1] * 3072 for _ in texts]

    @property
    def cumulative_stats(self) -> dict:
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_cost_usd": 0.0,
        }


@pytest.fixture
def mock_llm() -> MockLLMClient:
    """A mock LLM client returning deterministic responses."""
    return MockLLMClient()


# ── Mock RAG service ──────────────────────────────────────────────────


class MockRAGService:
    """Deterministic RAG service for testing."""

    def __init__(self, sample_data: list[dict] | None = None) -> None:
        self._data = sample_data or []

    async def retrieve(
        self, query: str, n_results: int = 5, filters: dict | None = None
    ) -> list[dict]:
        """Return canned search results."""
        return [
            {
                "id": "test-campaign-id",
                "document": "Test campaign document text",
                "metadata": {
                    "campaign_id": "a1b2c3d4-1111-4000-8000-000000000002",
                    "campaign_name": "Dunkin' Q4 Holiday Favorites",
                    "client_name": "Dunkin'",
                    "vertical": "QSR",
                    "incremental_roas": 42.30,
                    "visit_lift_percent": 18.2,
                    "sales_lift_percent": 14.5,
                },
                "distance": 0.15,
            }
        ]

    async def embed_and_store(self, campaigns: list[dict], **kwargs) -> int:
        return len(campaigns)

    async def hybrid_search(
        self, query: str, sql_results: list[dict], n_results: int = 5
    ) -> list[dict]:
        return await self.retrieve(query, n_results)

    async def get_collection_stats(self) -> dict:
        return {"collection_name": "campaign_embeddings", "document_count": 18}


@pytest.fixture
def mock_rag() -> MockRAGService:
    """A mock RAG service returning deterministic results."""
    return MockRAGService()


# ── Async test client (deferred import) ───────────────────────────────


@pytest_asyncio.fixture
async def client():
    """Async HTTP test client for FastAPI endpoint tests.

    Defers app import to avoid requiring asyncpg at collection time.
    """
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
