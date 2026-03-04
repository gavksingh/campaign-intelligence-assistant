"""Campaign report generation service.

Combines LLM-generated analysis with campaign metrics to produce
formatted PDF reports using FPDF2.
"""

from fpdf import FPDF

from app.models.schemas import ReportContent


async def generate_report_content(campaign_id: int) -> ReportContent:
    """Use the LLM to generate structured report content for a campaign.

    Fetches campaign data and metrics from the database, builds a prompt,
    and returns structured report content.

    Args:
        campaign_id: Database ID of the campaign.

    Returns:
        ReportContent with title, executive summary, sections, and conclusion.
    """
    raise NotImplementedError("Report content generation not yet implemented.")


def render_pdf(content: ReportContent, output_path: str) -> str:
    """Render a ReportContent object to a PDF file.

    Args:
        content: Structured report content from the LLM.
        output_path: File path to write the PDF to.

    Returns:
        The output file path.
    """
    raise NotImplementedError("PDF rendering not yet implemented.")


async def generate_and_save_report(campaign_id: int) -> dict:
    """End-to-end report generation: LLM content → PDF file.

    Args:
        campaign_id: Database ID of the campaign.

    Returns:
        Dict with report_id, campaign_id, generated_at, and download_url.
    """
    raise NotImplementedError("Full report pipeline not yet implemented.")
