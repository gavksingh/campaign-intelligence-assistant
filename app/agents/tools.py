"""Agent tools for campaign data operations.

Each tool is a callable that the LangGraph agent can invoke to query data,
search the vector store, generate reports, or recommend audience segments.
"""

from langchain_core.tools import tool


@tool
async def query_campaigns(query: str) -> str:
    """Query the campaign database using natural language.

    Translates a natural-language question into a database query against
    the campaigns and metrics tables, then returns formatted results.

    Args:
        query: Natural-language question about campaign data.

    Returns:
        Formatted string of matching campaign data.
    """
    raise NotImplementedError("query_campaigns tool not yet implemented.")


@tool
async def search_similar(query: str, n_results: int = 5) -> str:
    """Search for campaigns similar to the query using vector similarity.

    Uses the RAG service to find semantically similar campaign data
    in the ChromaDB vector store.

    Args:
        query: Natural-language search query.
        n_results: Number of results to return.

    Returns:
        Formatted string of similar campaign documents.
    """
    raise NotImplementedError("search_similar tool not yet implemented.")


@tool
async def generate_report(campaign_id: int) -> str:
    """Generate a performance report for the specified campaign.

    Triggers the report generation pipeline (LLM analysis + PDF creation)
    and returns a link to the generated report.

    Args:
        campaign_id: ID of the campaign to report on.

    Returns:
        Report metadata including download URL.
    """
    raise NotImplementedError("generate_report tool not yet implemented.")


@tool
async def recommend_audience(campaign_id: int) -> str:
    """Recommend audience segments for a campaign based on historical performance.

    Uses the LLM with campaign context to suggest new targeting segments.

    Args:
        campaign_id: ID of the campaign to generate recommendations for.

    Returns:
        Formatted audience segment recommendations.
    """
    raise NotImplementedError("recommend_audience tool not yet implemented.")
