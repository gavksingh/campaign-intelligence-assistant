"""Campaign report generation service.

Produces formatted Markdown, PDF, comparison, and Slack summary outputs
from structured LCI report data and campaign metadata.

Usage::

    from app.services.report_gen import ReportGenerator

    gen = ReportGenerator()
    md = gen.generate_markdown_report(report_data, campaign)
    pdf_bytes = gen.generate_pdf_report(report_data, campaign)
    slack = gen.generate_slack_summary(report_data, campaign)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from io import BytesIO

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from app.models.schemas import (
    CampaignComparisonSchema,
    CampaignResponse,
    LCIReportSchema,
    MetricsOut,
)

logger = logging.getLogger(__name__)


def _fmt_number(value: int | float) -> str:
    """Format a number with commas for readability.

    Args:
        value: Numeric value.

    Returns:
        Formatted string (e.g. '1,234,567').
    """
    if isinstance(value, float) and value == int(value):
        return f"{int(value):,}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return f"{value:,}"


def _fmt_currency(value: float) -> str:
    """Format a float as USD currency.

    Args:
        value: Dollar amount.

    Returns:
        Formatted string (e.g. '$1,234.56').
    """
    return f"${value:,.2f}"


def _fmt_pct(value: float) -> str:
    """Format a float as a percentage.

    Args:
        value: Percentage value (e.g. 12.4 for 12.4%).

    Returns:
        Formatted string (e.g. '12.4%').
    """
    return f"{value:.1f}%"


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


class ReportGenerator:
    """Generates formatted campaign reports in multiple output formats."""

    # ── Markdown report ───────────────────────────────────────────────

    def generate_markdown_report(
        self,
        report_data: LCIReportSchema,
        campaign: CampaignResponse,
    ) -> str:
        """Create a beautifully formatted Markdown LCI report.

        Args:
            report_data: Structured report content from the LLM.
            campaign: Campaign metadata and metrics from the API/DB.

        Returns:
            Full Markdown report string.
        """
        m = campaign.metrics
        sections: list[str] = []

        # ── Header ────────────────────────────────────────────────────
        sections.append(f"# Campaign Intelligence Report: {report_data.campaign_name}")
        sections.append("")
        sections.append(f"**Client:** {report_data.client_name}  ")
        sections.append(f"**Vertical:** {campaign.vertical.value}  ")
        sections.append(f"**Targeting:** {campaign.targeting_type.value}  ")
        sections.append(
            f"**Flight Dates:** {campaign.start_date.isoformat()}"
            + (f" — {campaign.end_date.isoformat()}" if campaign.end_date else " — ongoing")
            + "  "
        )
        sections.append(f"**Budget:** {_fmt_currency(campaign.budget)}  ")
        sections.append(f"**Status:** {campaign.status.value.title()}  ")
        sections.append(f"**Report Date:** {report_data.report_date}")
        sections.append("")
        sections.append("---")
        sections.append("")

        # ── Executive Summary ─────────────────────────────────────────
        sections.append("## Executive Summary")
        sections.append("")
        sections.append(report_data.executive_summary)
        sections.append("")

        # ── Key Metrics Dashboard ─────────────────────────────────────
        if m:
            sections.append("## Key Metrics Dashboard")
            sections.append("")
            sections.append("| Metric | Value |")
            sections.append("|--------|-------|")
            sections.append(f"| Visit Lift | {_fmt_pct(m.visit_lift_percent)} |")
            sections.append(f"| Sales Lift | {_fmt_pct(m.sales_lift_percent)} |")
            sections.append(f"| Incremental ROAS | {_fmt_currency(m.incremental_roas)} |")
            sections.append(f"| Incremental Visits | {_fmt_number(m.incremental_visits)} |")
            sections.append(f"| Incremental Sales | {_fmt_currency(m.incremental_sales_dollars)} |")
            sections.append(f"| Impressions | {_fmt_number(m.impressions)} |")
            sections.append(f"| Avg Basket Size | {_fmt_currency(m.avg_basket_size)} |")
            sections.append(f"| Purchase Frequency | {m.purchase_frequency:.1f}x |")
            sections.append("")

        # ── Visit Lift Analysis ───────────────────────────────────────
        vla = report_data.visit_lift_analysis
        sections.append("## Visit Lift Analysis")
        sections.append("")
        sections.append(vla.overall_lift)
        sections.append("")
        if vla.market_breakdown:
            sections.append("### Market-Level Visit Lift")
            sections.append("")
            for observation in vla.market_breakdown:
                sections.append(f"- {observation}")
            sections.append("")
        sections.append(f"**Daypart Insights:** {vla.daypart_insights}")
        sections.append("")

        # ── Sales Lift Analysis ───────────────────────────────────────
        sla = report_data.sales_lift_analysis
        sections.append("## Sales Lift Analysis")
        sections.append("")
        sections.append(sla.overall_lift)
        sections.append("")
        sections.append(f"**Basket Size Analysis:** {sla.basket_size_analysis}")
        sections.append("")
        sections.append(f"**Purchase Frequency:** {sla.purchase_frequency_insight}")
        sections.append("")

        # ── Top Markets Performance ───────────────────────────────────
        if report_data.market_breakdown:
            sections.append("## Top Markets Performance")
            sections.append("")
            sections.append("| Market | Performance | Ranking |")
            sections.append("|--------|-------------|---------|")
            for mb in report_data.market_breakdown:
                sections.append(
                    f"| {mb.market_name} | {mb.performance_summary} | {mb.relative_ranking} |"
                )
            sections.append("")

        # ── Recommendations ───────────────────────────────────────────
        sections.append("## Recommendations")
        sections.append("")
        for i, rec in enumerate(report_data.recommendations, 1):
            sections.append(f"{i}. {rec}")
        sections.append("")

        # ── Footer ────────────────────────────────────────────────────
        sections.append("---")
        sections.append("")
        sections.append(f"*Generated: {_now_iso()}*  ")
        sections.append(f"*{report_data.methodology_note}*")

        return "\n".join(sections)

    # ── PDF report ────────────────────────────────────────────────────

    def generate_pdf_report(
        self,
        report_data: LCIReportSchema,
        campaign: CampaignResponse,
    ) -> bytes:
        """Produce a professional PDF campaign report.

        Args:
            report_data: Structured report content from the LLM.
            campaign: Campaign metadata and metrics from the API/DB.

        Returns:
            PDF file contents as bytes.
        """
        m = campaign.metrics
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)

        # ── Page 1: Title page ────────────────────────────────────────
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 28)
        pdf.cell(0, 40, "", new_x=XPos.LMARGIN, new_y=YPos.NEXT)  # spacer
        pdf.cell(0, 15, "Campaign Intelligence Report", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font("Helvetica", "", 16)
        pdf.cell(0, 12, report_data.campaign_name, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 10, f"Client: {report_data.client_name}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(0, 8, f"Vertical: {campaign.vertical.value}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        date_range = campaign.start_date.isoformat()
        if campaign.end_date:
            date_range += f"  to  {campaign.end_date.isoformat()}"
        pdf.cell(0, 8, f"Flight: {date_range}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(0, 8, f"Budget: {_fmt_currency(campaign.budget)}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.cell(0, 20, "", new_x=XPos.LMARGIN, new_y=YPos.NEXT)  # spacer

        # Branding placeholder
        pdf.set_draw_color(180, 180, 180)
        pdf.set_fill_color(245, 245, 245)
        pdf.rect(60, pdf.get_y(), 90, 25, style="DF")
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 25, "[Company Logo Placeholder]", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)

        pdf.cell(0, 15, "", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Report Date: {report_data.report_date}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # ── Page 2+: Content ──────────────────────────────────────────
        pdf.add_page()
        self._pdf_section_header(pdf, "Executive Summary")
        self._pdf_body_text(pdf, report_data.executive_summary)

        # Key Metrics table
        if m:
            self._pdf_section_header(pdf, "Key Metrics Dashboard")
            metrics_rows = [
                ("Visit Lift", _fmt_pct(m.visit_lift_percent)),
                ("Sales Lift", _fmt_pct(m.sales_lift_percent)),
                ("Incremental ROAS", _fmt_currency(m.incremental_roas)),
                ("Incremental Visits", _fmt_number(m.incremental_visits)),
                ("Incremental Sales", _fmt_currency(m.incremental_sales_dollars)),
                ("Impressions", _fmt_number(m.impressions)),
                ("Avg Basket Size", _fmt_currency(m.avg_basket_size)),
                ("Purchase Frequency", f"{m.purchase_frequency:.1f}x"),
            ]
            self._pdf_table(pdf, ["Metric", "Value"], metrics_rows)

        # Visit Lift Analysis
        self._pdf_section_header(pdf, "Visit Lift Analysis")
        self._pdf_body_text(pdf, report_data.visit_lift_analysis.overall_lift)
        for obs in report_data.visit_lift_analysis.market_breakdown:
            self._pdf_bullet(pdf, obs)
        pdf.ln(3)
        self._pdf_body_text(
            pdf, f"Daypart Insights: {report_data.visit_lift_analysis.daypart_insights}"
        )

        # Sales Lift Analysis
        self._pdf_section_header(pdf, "Sales Lift Analysis")
        self._pdf_body_text(pdf, report_data.sales_lift_analysis.overall_lift)
        self._pdf_body_text(
            pdf,
            f"Basket Size Analysis: {report_data.sales_lift_analysis.basket_size_analysis}",
        )
        self._pdf_body_text(
            pdf,
            f"Purchase Frequency: {report_data.sales_lift_analysis.purchase_frequency_insight}",
        )

        # Top Markets
        if report_data.market_breakdown:
            self._pdf_section_header(pdf, "Top Markets Performance")
            market_rows = [
                (mb.market_name, mb.performance_summary, mb.relative_ranking)
                for mb in report_data.market_breakdown
            ]
            self._pdf_table(pdf, ["Market", "Performance", "Ranking"], market_rows)

        # Recommendations
        self._pdf_section_header(pdf, "Recommendations")
        for i, rec in enumerate(report_data.recommendations, 1):
            self._pdf_bullet(pdf, f"{i}. {rec}")

        # Footer on every page
        total_pages = pdf.pages_count
        for page_num in range(1, total_pages + 1):
            pdf.page = page_num
            pdf.set_y(-15)
            pdf.set_font("Helvetica", "I", 8)
            pdf.cell(0, 10, f"Page {page_num} of {total_pages}", align="C")

        # Methodology disclaimer on last page
        pdf.page = total_pages
        pdf.set_y(-25)
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(120, 120, 120)
        pdf.multi_cell(0, 4, report_data.methodology_note, align="C")
        pdf.set_text_color(0, 0, 0)

        # Output to bytes
        buf = BytesIO()
        pdf.output(buf)
        pdf_bytes = buf.getvalue()

        logger.info(
            "Generated PDF report for '%s' (%d bytes, %d pages)",
            report_data.campaign_name,
            len(pdf_bytes),
            total_pages,
        )
        return pdf_bytes

    # ── Comparison report ─────────────────────────────────────────────

    def generate_comparison_report(
        self,
        comparison: CampaignComparisonSchema,
    ) -> str:
        """Create a Markdown comparison report between two campaigns.

        Args:
            comparison: Structured comparison data from the LLM.

        Returns:
            Formatted Markdown string with side-by-side analysis.
        """
        sections: list[str] = []

        sections.append(
            f"# Campaign Comparison: {comparison.campaign_a_name} vs {comparison.campaign_b_name}"
        )
        sections.append("")
        sections.append("---")
        sections.append("")

        # Overall summary
        sections.append("## Summary")
        sections.append("")
        sections.append(comparison.comparison_summary)
        sections.append("")

        # Metrics table
        sections.append("## Metric-by-Metric Comparison")
        sections.append("")
        sections.append(
            f"| Metric | {comparison.campaign_a_name} | {comparison.campaign_b_name} | Winner | Insight |"
        )
        sections.append("|--------|" + "--------|" * 4)

        for mc in comparison.metric_comparisons:
            metric = mc.get("metric", "—")
            val_a = mc.get("campaign_a_value", "—")
            val_b = mc.get("campaign_b_value", "—")
            winner = mc.get("winner", "—")
            insight = mc.get("insight", "")

            # Highlight the winner
            if winner == comparison.campaign_a_name:
                val_a = f"**{val_a}**"
            elif winner == comparison.campaign_b_name:
                val_b = f"**{val_b}**"

            sections.append(f"| {metric} | {val_a} | {val_b} | {winner} | {insight} |")

        sections.append("")

        # Key differences
        sections.append("## Key Differences")
        sections.append("")
        for diff in comparison.key_differences:
            sections.append(f"- {diff}")
        sections.append("")

        # Recommendation
        sections.append("## Recommendation")
        sections.append("")
        sections.append(comparison.recommendation)
        sections.append("")

        # Footer
        sections.append("---")
        sections.append(f"*Generated: {_now_iso()}*")

        return "\n".join(sections)

    # ── Slack summary ─────────────────────────────────────────────────

    def generate_slack_summary(
        self,
        report_data: LCIReportSchema,
        campaign: CampaignResponse,
    ) -> str:
        """Create a short Slack-formatted campaign summary.

        Uses Slack markdown conventions (* for bold, not **).

        Args:
            report_data: Structured report content from the LLM.
            campaign: Campaign metadata and metrics.

        Returns:
            5-7 line Slack message string.
        """
        m = campaign.metrics
        lines: list[str] = []

        lines.append(
            f":bar_chart: *Campaign Report: {report_data.campaign_name}* ({report_data.client_name})"
        )

        if m:
            lines.append(
                f":chart_with_upwards_trend: Visit Lift: *{_fmt_pct(m.visit_lift_percent)}* | "
                f"Sales Lift: *{_fmt_pct(m.sales_lift_percent)}* | "
                f"ROAS: *{_fmt_currency(m.incremental_roas)}*"
            )
            lines.append(
                f":busts_in_silhouette: Visits: *{_fmt_number(m.incremental_visits)}* | "
                f"Sales: *{_fmt_currency(m.incremental_sales_dollars)}*"
            )

        # One-line verdict: extract the first sentence of the executive summary
        summary = report_data.executive_summary
        first_sentence = summary.split(".")[0].strip() + "." if "." in summary else summary[:150]
        lines.append(f":memo: {first_sentence}")

        if report_data.recommendations:
            lines.append(f":bulb: {report_data.recommendations[0]}")

        lines.append(
            f":calendar: Flight: {campaign.start_date.isoformat()}"
            + (f" — {campaign.end_date.isoformat()}" if campaign.end_date else "")
            + f" | Budget: {_fmt_currency(campaign.budget)}"
        )

        return "\n".join(lines)

    # ── PDF helper methods ────────────────────────────────────────────

    def _pdf_section_header(self, pdf: FPDF, title: str) -> None:
        """Render a section header in the PDF.

        Args:
            pdf: FPDF instance.
            title: Section title text.
        """
        pdf.ln(6)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(30, 60, 110)
        pdf.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        # Underline
        pdf.set_draw_color(30, 60, 110)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(4)
        pdf.set_text_color(0, 0, 0)

    def _pdf_body_text(self, pdf: FPDF, text: str) -> None:
        """Render body text in the PDF with word wrapping.

        Args:
            pdf: FPDF instance.
            text: Body text content.
        """
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, text)
        pdf.ln(3)

    def _pdf_bullet(self, pdf: FPDF, text: str) -> None:
        """Render a bullet-point line in the PDF.

        Args:
            pdf: FPDF instance.
            text: Bullet text.
        """
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(6, 5, "-")
        pdf.multi_cell(0, 5, text)
        pdf.ln(1)

    def _pdf_table(
        self,
        pdf: FPDF,
        headers: list[str],
        rows: list[tuple[str, ...]],
    ) -> None:
        """Render a table in the PDF.

        Args:
            pdf: FPDF instance.
            headers: Column header strings.
            rows: List of row tuples, each matching the number of headers.
        """
        num_cols = len(headers)
        available_width = pdf.w - pdf.l_margin - pdf.r_margin
        col_width = available_width / num_cols

        # Header row
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(30, 60, 110)
        pdf.set_text_color(255, 255, 255)
        for header in headers:
            pdf.cell(col_width, 8, header, border=1, fill=True, align="C")
        pdf.ln()

        # Data rows
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(0, 0, 0)
        fill = False
        for row in rows:
            if fill:
                pdf.set_fill_color(240, 240, 245)
            else:
                pdf.set_fill_color(255, 255, 255)

            for i, cell_val in enumerate(row):
                align = "L" if i == 0 else "R"
                pdf.cell(col_width, 7, str(cell_val), border=1, fill=True, align=align)
            pdf.ln()
            fill = not fill

        pdf.ln(4)
