"""RAG service using pgvector for campaign data retrieval.

Provides a RAGService class that manages:
    - Embedding and storing campaign data with rich text representations.
    - Semantic search with optional metadata filters via pgvector cosine distance.
    - Hybrid search combining vector results with SQL query results.
    - Index refresh/rebuild operations.

Uses Gemini embeddings via the LLMClient.

Usage::

    from app.services.rag import RAGService

    rag = RAGService()
    await rag.embed_and_store(campaign_dicts)
    results = await rag.retrieve("best performing QSR campaign")
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import delete, func, select, text

from app.config import Settings, settings
from app.database import get_session_factory
from app.models.campaign import CampaignEmbedding
from app.services.llm_client import LLMClient, get_llm_client

logger = logging.getLogger(__name__)


def _build_document_text(campaign: dict) -> str:
    """Build a rich text representation of a campaign for vector embedding.

    Combines campaign metadata, all metrics, and the narrative summary into
    a single string optimized for semantic retrieval.

    Args:
        campaign: A campaign dict (from mock_campaigns.json or DB serialization).

    Returns:
        Formatted text suitable for embedding.
    """
    m = campaign.get("metrics", {})
    segments = ", ".join(campaign.get("audience_segments", []))
    markets = ", ".join(m.get("top_markets", []))

    return (
        f"Campaign: {campaign['campaign_name']}\n"
        f"Client: {campaign['client_name']} | Vertical: {campaign['vertical']}\n"
        f"Status: {campaign['status']} | Targeting: {campaign['targeting_type']}\n"
        f"Budget: ${campaign['budget']:,.0f} | "
        f"Dates: {campaign['start_date']} to {campaign.get('end_date', 'ongoing')}\n"
        f"\nPerformance Metrics:\n"
        f"  Impressions: {m.get('impressions', 0):,}\n"
        f"  Visit Lift: {m.get('visit_lift_percent', 0)}%\n"
        f"  Sales Lift: {m.get('sales_lift_percent', 0)}%\n"
        f"  Incremental ROAS: ${m.get('incremental_roas', 0):.2f}\n"
        f"  Incremental Visits: {m.get('incremental_visits', 0):,}\n"
        f"  Incremental Sales: ${m.get('incremental_sales_dollars', 0):,.0f}\n"
        f"  Avg Basket Size: ${m.get('avg_basket_size', 0):.2f}\n"
        f"  Purchase Frequency: {m.get('purchase_frequency', 0):.1f}x\n"
        f"\nTop Markets: {markets}\n"
        f"Top Creative: {m.get('top_performing_creative', 'N/A')}\n"
        f"Audience Segments: {segments}\n"
        f"\nSummary: {campaign.get('campaign_summary', '')}"
    )


def _build_metadata(campaign: dict) -> dict[str, Any]:
    """Build a metadata dict for a campaign embedding document.

    Args:
        campaign: A campaign dict.

    Returns:
        Metadata dict with key campaign attributes.
    """
    m = campaign.get("metrics", {})
    return {
        "campaign_id": str(campaign.get("campaign_id", "")),
        "campaign_name": campaign["campaign_name"],
        "client_name": campaign["client_name"],
        "vertical": campaign["vertical"],
        "status": campaign["status"],
        "targeting_type": campaign["targeting_type"],
        "budget": float(campaign["budget"]),
        "start_date": campaign["start_date"],
        "end_date": campaign.get("end_date", ""),
        "impressions": int(m.get("impressions", 0)),
        "visit_lift_percent": float(m.get("visit_lift_percent", 0.0)),
        "sales_lift_percent": float(m.get("sales_lift_percent", 0.0)),
        "incremental_roas": float(m.get("incremental_roas", 0.0)),
        "incremental_visits": int(m.get("incremental_visits", 0)),
        "incremental_sales_dollars": float(m.get("incremental_sales_dollars", 0.0)),
    }


class RAGService:
    """Manages pgvector storage and retrieval for campaign data.

    Uses Gemini embeddings via LLMClient and stores vectors in the
    campaign_embeddings table with cosine distance search.

    Args:
        cfg: Application settings. Defaults to the global singleton.
        llm_client: LLMClient for Gemini embeddings. Defaults to the global singleton.
    """

    def __init__(
        self,
        cfg: Settings | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._cfg = cfg or settings
        self._llm = llm_client

    @property
    def llm(self) -> LLMClient:
        """Lazily resolve the LLM client."""
        if self._llm is None:
            self._llm = get_llm_client()
        return self._llm

    # ── Store ─────────────────────────────────────────────────────────

    async def embed_and_store(
        self,
        campaigns: list[dict],
        **kwargs,
    ) -> int:
        """Embed campaign data and store in pgvector.

        Builds a rich text representation for each campaign, generates
        embeddings via Gemini, and upserts into the campaign_embeddings table.
        Upsert pattern: delete existing rows for each campaign_id, then insert.

        Args:
            campaigns: List of campaign dicts (from JSON or DB).

        Returns:
            Number of documents stored.
        """
        if not campaigns:
            logger.warning("embed_and_store called with empty campaign list")
            return 0

        documents: list[str] = []
        metadatas: list[dict] = []
        campaign_db_ids: list[int | None] = []

        for c in campaigns:
            doc = _build_document_text(c)
            meta = _build_metadata(c)
            documents.append(doc)
            metadatas.append(meta)
            # The campaign dict may have 'id' (DB PK) or we look it up
            campaign_db_ids.append(c.get("id"))

        logger.info("Embedding %d campaign documents...", len(documents))
        embeddings = await self.llm.embed_texts(documents)

        factory = get_session_factory()
        async with factory() as session:
            # Delete existing embeddings for these campaigns (upsert)
            valid_ids = [cid for cid in campaign_db_ids if cid is not None]
            if valid_ids:
                await session.execute(
                    delete(CampaignEmbedding).where(
                        CampaignEmbedding.campaign_id.in_(valid_ids)
                    )
                )

            # Insert new embeddings
            for i, doc in enumerate(documents):
                db_id = campaign_db_ids[i]
                if db_id is None:
                    continue
                embedding_row = CampaignEmbedding(
                    campaign_id=db_id,
                    document_text=doc,
                    embedding=embeddings[i],
                    metadata_json=json.dumps(metadatas[i]),
                )
                session.add(embedding_row)

            await session.commit()

        logger.info("Stored %d documents in campaign_embeddings", len(documents))
        return len(documents)

    # ── Retrieve ──────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        n_results: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Semantic search over campaign documents via pgvector cosine distance.

        Args:
            query: Natural-language search query.
            n_results: Maximum number of results.
            filters: Optional metadata filters applied post-query. Supports:
                - {"vertical": "QSR"} — exact match
                - {"client_name": "Dunkin'"} — exact match

        Returns:
            List of dicts with 'document', 'metadata', 'distance', and 'id' keys,
            sorted by relevance (lowest distance first).
        """
        logger.info(
            "retrieve | query='%s' | n_results=%d | filters=%s",
            query[:80],
            n_results,
            filters,
        )

        # Embed the query
        query_embedding = await self.llm.embed_text(query)

        # Fetch more results if we need to filter post-query
        fetch_limit = n_results * 3 if filters else n_results

        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                text(
                    "SELECT id, campaign_id, document_text, metadata_json, "
                    "embedding <=> :query_vec AS distance "
                    "FROM campaign_embeddings "
                    "ORDER BY embedding <=> :query_vec "
                    "LIMIT :limit"
                ),
                {"query_vec": str(query_embedding), "limit": fetch_limit},
            )
            rows = result.fetchall()

        output: list[dict] = []
        for row in rows:
            row_id, campaign_id, doc_text, meta_json, distance = row
            metadata = json.loads(meta_json) if meta_json else {}

            # Apply post-query metadata filters
            if filters:
                match = True
                for key, value in filters.items():
                    if isinstance(value, dict):
                        # Numeric comparison (e.g., {"$gte": 30.0})
                        meta_val = metadata.get(key, 0)
                        for op, threshold in value.items():
                            if op == "$gte" and meta_val < threshold:
                                match = False
                            elif op == "$lte" and meta_val > threshold:
                                match = False
                            elif op == "$gt" and meta_val <= threshold:
                                match = False
                            elif op == "$lt" and meta_val >= threshold:
                                match = False
                    else:
                        if metadata.get(key) != value:
                            match = False
                if not match:
                    continue

            output.append(
                {
                    "id": str(metadata.get("campaign_id", row_id)),
                    "document": doc_text,
                    "metadata": metadata,
                    "distance": float(distance),
                }
            )

            if len(output) >= n_results:
                break

        logger.info("retrieve returned %d results", len(output))
        return output

    # ── Hybrid search ─────────────────────────────────────────────────

    async def hybrid_search(
        self,
        query: str,
        sql_results: list[dict],
        n_results: int = 5,
    ) -> list[dict]:
        """Combine vector search with SQL query results, deduplicate, and re-rank.

        Retrieves vector results for the query, merges with provided SQL results
        (matched by campaign_id or campaign_name), and returns a unified ranked list.

        Args:
            query: Natural-language search query.
            sql_results: Campaign dicts from a SQL query (must have 'campaign_id'
                         or 'campaign_name').
            n_results: Maximum results to return.

        Returns:
            Merged and re-ranked list of campaign result dicts.
        """
        logger.info(
            "hybrid_search | query='%s' | sql_results=%d | n_results=%d",
            query[:80],
            len(sql_results),
            n_results,
        )

        # Get vector results
        vector_results = await self.retrieve(query, n_results=n_results * 2)

        # Build a lookup by campaign identifier
        seen: dict[str, dict] = {}

        # SQL results get a base score (lower is better for consistency with distance)
        for i, sr in enumerate(sql_results):
            key = str(sr.get("campaign_id", sr.get("campaign_name", f"sql_{i}")))
            seen[key] = {
                "id": key,
                "document": sr.get("campaign_summary", ""),
                "metadata": sr,
                "distance": 0.0,  # Exact SQL match = best score
                "source": "sql",
            }

        # Merge vector results — add if new, keep best score if duplicate
        for vr in vector_results:
            key = vr["id"]
            if key in seen:
                seen[key]["source"] = "both"
                seen[key]["distance"] = min(seen[key]["distance"], vr["distance"] * 0.5)
            else:
                vr["source"] = "vector"
                seen[key] = vr

        # Sort by distance (lower = more relevant) and truncate
        merged = sorted(seen.values(), key=lambda x: x["distance"])[:n_results]

        logger.info(
            "hybrid_search merged %d results (sql=%d, vector=%d, both=%d)",
            len(merged),
            sum(1 for r in merged if r.get("source") == "sql"),
            sum(1 for r in merged if r.get("source") == "vector"),
            sum(1 for r in merged if r.get("source") == "both"),
        )
        return merged

    # ── Index management ──────────────────────────────────────────────

    async def refresh_index(self, campaigns: list[dict]) -> int:
        """Delete and rebuild the entire vector index.

        Args:
            campaigns: Full list of campaign dicts to re-embed.

        Returns:
            Number of documents in the rebuilt index.
        """
        logger.info("Refreshing vector index...")

        factory = get_session_factory()
        async with factory() as session:
            await session.execute(delete(CampaignEmbedding))
            await session.commit()
        logger.info("Deleted all existing embeddings")

        count = await self.embed_and_store(campaigns)
        logger.info("Index refreshed with %d documents", count)
        return count

    async def get_collection_stats(self) -> dict:
        """Return basic stats about the campaign embeddings.

        Returns:
            Dict with count and collection name.
        """
        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(select(func.count(CampaignEmbedding.id)))
            count = result.scalar_one()

        return {
            "collection_name": "campaign_embeddings",
            "document_count": count,
        }


# ── Module-level helpers ──────────────────────────────────────────────

_default_rag: RAGService | None = None


def get_rag_service() -> RAGService:
    """Return a module-level singleton RAGService.

    Returns:
        The shared RAGService instance.
    """
    global _default_rag
    if _default_rag is None:
        _default_rag = RAGService()
    return _default_rag
