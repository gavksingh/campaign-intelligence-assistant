"""Tests for the report generation service.

Covers all four output methods: Markdown, PDF, comparison, and Slack summary.
"""

from datetime import date, datetime, timezone

import pytest

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
from app.services.report_gen import ReportGenerator


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_metrics() -> MetricsOut:
    """Sample campaign metrics for testing."""
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
    """Sample campaign response for testing."""
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
def sample_report_data() -> LCIReportSchema:
    """Sample LCI report schema for testing."""
    return LCIReportSchema(
        campaign_name="Dunkin' Q4 Holiday Favorites",
        client_name="Dunkin'",
        report_date="2026-01-15",
        executive_summary=(
            "The Dunkin' Q4 Holiday Favorites campaign delivered exceptional performance "
            "across all key metrics. Visit lift of 18.2% significantly exceeded the QSR "
            "category benchmark of 10%, while the $42.30 incremental ROAS represents "
            "outstanding return on the $225K investment.\n\n"
            "Holiday-themed creative resonated strongly with consumers, driving higher "
            "basket sizes and purchase frequency compared to the Q3 summer campaign."
        ),
        visit_lift_analysis=VisitLiftAnalysis(
            overall_lift="The campaign drove an 18.2% visit lift, nearly doubling the category average.",
            market_breakdown=[
                "New York: 20.1% visit lift (top performer)",
                "Boston: 19.5% visit lift",
                "Chicago: 17.8% visit lift",
                "Philadelphia: 16.2% visit lift",
                "Hartford: 15.9% visit lift",
            ],
            daypart_insights="Morning daypart (6-10am) drove 62% of incremental visits.",
        ),
        sales_lift_analysis=SalesLiftAnalysis(
            overall_lift="Sales lift of 14.5% was driven by both increased traffic and higher basket sizes.",
            basket_size_analysis="Average basket size of $11.25 was 28.6% higher than Q3's $8.75.",
            purchase_frequency_insight="Purchase frequency of 3.8x indicates strong repeat engagement.",
        ),
        market_breakdown=[
            MarketBreakdown(
                market_name="New York",
                performance_summary="20.1% visit lift, highest incremental visits",
                relative_ranking="Top",
            ),
            MarketBreakdown(
                market_name="Boston",
                performance_summary="19.5% visit lift, strong brand loyalty market",
                relative_ranking="Top",
            ),
            MarketBreakdown(
                market_name="Chicago",
                performance_summary="17.8% visit lift, above average",
                relative_ranking="Above average",
            ),
        ],
        recommendations=[
            "Increase Q4 holiday budget by 20% based on strong ROAS performance.",
            "Extend peppermint mocha creative into January for post-holiday momentum.",
            "Test expanded markets (DC, Baltimore) given strong Northeast performance.",
            "Add Lapsed Dunkin' Visitors segment which showed 22% higher conversion.",
        ],
    )


@pytest.fixture
def sample_comparison() -> CampaignComparisonSchema:
    """Sample campaign comparison for testing."""
    return CampaignComparisonSchema(
        campaign_a_name="Dunkin' Q3 Summer Iced Coffee",
        campaign_b_name="Dunkin' Q4 Holiday Favorites",
        comparison_summary=(
            "The Q4 Holiday Favorites campaign significantly outperformed Q3 across "
            "all key metrics, with 47% higher visit lift and 48% higher ROAS."
        ),
        metric_comparisons=[
            {
                "metric": "Visit Lift",
                "campaign_a_value": "12.4%",
                "campaign_b_value": "18.2%",
                "winner": "Dunkin' Q4 Holiday Favorites",
                "insight": "Q4 delivered 47% higher visit lift",
            },
            {
                "metric": "Incremental ROAS",
                "campaign_a_value": "$28.50",
                "campaign_b_value": "$42.30",
                "winner": "Dunkin' Q4 Holiday Favorites",
                "insight": "Holiday creative drove stronger conversion",
            },
            {
                "metric": "Impressions",
                "campaign_a_value": "18.5M",
                "campaign_b_value": "24M",
                "winner": "Dunkin' Q4 Holiday Favorites",
                "insight": "Higher budget enabled broader reach",
            },
        ],
        key_differences=[
            "Holiday-themed creative outperformed summer creative on engagement.",
            "Q4 basket size ($11.25) was 28.6% higher than Q3 ($8.75).",
            "Q4 purchase frequency (3.8x) exceeded Q3 (3.2x).",
        ],
        recommendation=(
            "Allocate a larger share of annual budget to Q4 holiday campaigns "
            "based on the consistently stronger ROAS and visit lift metrics."
        ),
    )


@pytest.fixture
def generator() -> ReportGenerator:
    """A ReportGenerator instance."""
    return ReportGenerator()


# ── Markdown report tests ─────────────────────────────────────────────


class TestMarkdownReport:
    """Tests for generate_markdown_report."""

    def test_contains_campaign_name(
        self, generator: ReportGenerator, sample_report_data, sample_campaign
    ):
        md = generator.generate_markdown_report(sample_report_data, sample_campaign)
        assert "Dunkin' Q4 Holiday Favorites" in md

    def test_contains_header_metadata(
        self, generator: ReportGenerator, sample_report_data, sample_campaign
    ):
        md = generator.generate_markdown_report(sample_report_data, sample_campaign)
        assert "**Client:** Dunkin'" in md
        assert "**Vertical:** QSR" in md
        assert "**Targeting:** Moments" in md
        assert "$225,000.00" in md

    def test_contains_executive_summary(
        self, generator: ReportGenerator, sample_report_data, sample_campaign
    ):
        md = generator.generate_markdown_report(sample_report_data, sample_campaign)
        assert "## Executive Summary" in md
        assert "exceptional performance" in md

    def test_contains_metrics_table(
        self, generator: ReportGenerator, sample_report_data, sample_campaign
    ):
        md = generator.generate_markdown_report(sample_report_data, sample_campaign)
        assert "## Key Metrics Dashboard" in md
        assert "18.2%" in md
        assert "$42.30" in md
        assert "218,000" in md

    def test_contains_visit_lift_analysis(
        self, generator: ReportGenerator, sample_report_data, sample_campaign
    ):
        md = generator.generate_markdown_report(sample_report_data, sample_campaign)
        assert "## Visit Lift Analysis" in md
        assert "New York: 20.1%" in md
        assert "Daypart Insights" in md

    def test_contains_sales_lift_analysis(
        self, generator: ReportGenerator, sample_report_data, sample_campaign
    ):
        md = generator.generate_markdown_report(sample_report_data, sample_campaign)
        assert "## Sales Lift Analysis" in md
        assert "Basket Size Analysis" in md

    def test_contains_markets_table(
        self, generator: ReportGenerator, sample_report_data, sample_campaign
    ):
        md = generator.generate_markdown_report(sample_report_data, sample_campaign)
        assert "## Top Markets Performance" in md
        assert "| New York |" in md

    def test_contains_recommendations(
        self, generator: ReportGenerator, sample_report_data, sample_campaign
    ):
        md = generator.generate_markdown_report(sample_report_data, sample_campaign)
        assert "## Recommendations" in md
        assert "1." in md
        assert "Increase Q4 holiday budget" in md

    def test_contains_footer(
        self, generator: ReportGenerator, sample_report_data, sample_campaign
    ):
        md = generator.generate_markdown_report(sample_report_data, sample_campaign)
        assert "Generated:" in md
        assert "methodology" in md.lower()

    def test_handles_no_end_date(
        self, generator: ReportGenerator, sample_report_data, sample_campaign
    ):
        sample_campaign.end_date = None
        md = generator.generate_markdown_report(sample_report_data, sample_campaign)
        assert "ongoing" in md

    def test_handles_no_metrics(
        self, generator: ReportGenerator, sample_report_data, sample_campaign
    ):
        sample_campaign.metrics = None
        md = generator.generate_markdown_report(sample_report_data, sample_campaign)
        assert "## Executive Summary" in md
        assert "Key Metrics Dashboard" not in md


# ── PDF report tests ──────────────────────────────────────────────────


class TestPdfReport:
    """Tests for generate_pdf_report."""

    def test_returns_bytes(
        self, generator: ReportGenerator, sample_report_data, sample_campaign
    ):
        result = generator.generate_pdf_report(sample_report_data, sample_campaign)
        assert isinstance(result, bytes)

    def test_pdf_starts_with_header(
        self, generator: ReportGenerator, sample_report_data, sample_campaign
    ):
        result = generator.generate_pdf_report(sample_report_data, sample_campaign)
        assert result[:5] == b"%PDF-"

    def test_pdf_has_reasonable_size(
        self, generator: ReportGenerator, sample_report_data, sample_campaign
    ):
        result = generator.generate_pdf_report(sample_report_data, sample_campaign)
        # A multi-page report should be at least a few KB
        assert len(result) > 2000

    def test_pdf_is_multi_page(
        self, generator: ReportGenerator, sample_report_data, sample_campaign
    ):
        result = generator.generate_pdf_report(sample_report_data, sample_campaign)
        # Count page objects — a full report should have multiple pages
        page_count = result.count(b"/Type /Page\n")
        assert page_count >= 2

    def test_pdf_handles_no_metrics(
        self, generator: ReportGenerator, sample_report_data, sample_campaign
    ):
        sample_campaign.metrics = None
        result = generator.generate_pdf_report(sample_report_data, sample_campaign)
        assert result[:5] == b"%PDF-"


# ── Comparison report tests ───────────────────────────────────────────


class TestComparisonReport:
    """Tests for generate_comparison_report."""

    def test_contains_both_campaign_names(
        self, generator: ReportGenerator, sample_comparison
    ):
        md = generator.generate_comparison_report(sample_comparison)
        assert "Dunkin' Q3 Summer Iced Coffee" in md
        assert "Dunkin' Q4 Holiday Favorites" in md

    def test_contains_comparison_table(
        self, generator: ReportGenerator, sample_comparison
    ):
        md = generator.generate_comparison_report(sample_comparison)
        assert "| Metric |" in md
        assert "| Visit Lift |" in md

    def test_highlights_winners(self, generator: ReportGenerator, sample_comparison):
        md = generator.generate_comparison_report(sample_comparison)
        # Winner values should be bold
        assert "**18.2%**" in md
        assert "**$42.30**" in md

    def test_contains_key_differences(
        self, generator: ReportGenerator, sample_comparison
    ):
        md = generator.generate_comparison_report(sample_comparison)
        assert "## Key Differences" in md
        assert "Holiday-themed creative" in md

    def test_contains_recommendation(
        self, generator: ReportGenerator, sample_comparison
    ):
        md = generator.generate_comparison_report(sample_comparison)
        assert "## Recommendation" in md
        assert "larger share" in md

    def test_contains_footer_timestamp(
        self, generator: ReportGenerator, sample_comparison
    ):
        md = generator.generate_comparison_report(sample_comparison)
        assert "Generated:" in md


# ── Slack summary tests ───────────────────────────────────────────────


class TestSlackSummary:
    """Tests for generate_slack_summary."""

    def test_contains_campaign_name_bold(
        self, generator: ReportGenerator, sample_report_data, sample_campaign
    ):
        slack = generator.generate_slack_summary(sample_report_data, sample_campaign)
        # Slack bold uses single *
        assert "*Campaign Report: Dunkin' Q4 Holiday Favorites*" in slack

    def test_contains_key_metrics(
        self, generator: ReportGenerator, sample_report_data, sample_campaign
    ):
        slack = generator.generate_slack_summary(sample_report_data, sample_campaign)
        assert "18.2%" in slack
        assert "14.5%" in slack
        assert "$42.30" in slack

    def test_contains_verdict(
        self, generator: ReportGenerator, sample_report_data, sample_campaign
    ):
        slack = generator.generate_slack_summary(sample_report_data, sample_campaign)
        # Should have the first sentence of executive summary
        assert ":memo:" in slack

    def test_contains_recommendation(
        self, generator: ReportGenerator, sample_report_data, sample_campaign
    ):
        slack = generator.generate_slack_summary(sample_report_data, sample_campaign)
        assert ":bulb:" in slack
        assert "Increase Q4 holiday budget by 20%" in slack

    def test_line_count(
        self, generator: ReportGenerator, sample_report_data, sample_campaign
    ):
        slack = generator.generate_slack_summary(sample_report_data, sample_campaign)
        lines = [line for line in slack.strip().split("\n") if line.strip()]
        assert 4 <= len(lines) <= 7

    def test_uses_slack_bold_not_markdown(
        self, generator: ReportGenerator, sample_report_data, sample_campaign
    ):
        slack = generator.generate_slack_summary(sample_report_data, sample_campaign)
        # Should use Slack *bold*, not markdown **bold**
        assert "**" not in slack

    def test_contains_budget_and_dates(
        self, generator: ReportGenerator, sample_report_data, sample_campaign
    ):
        slack = generator.generate_slack_summary(sample_report_data, sample_campaign)
        assert "$225,000.00" in slack
        assert "2025-10-15" in slack

    def test_slack_under_500_characters(
        self, generator: ReportGenerator, sample_report_data, sample_campaign
    ):
        slack = generator.generate_slack_summary(sample_report_data, sample_campaign)
        assert len(slack) < 500, (
            f"Slack summary is {len(slack)} chars, should be under 500"
        )
