"""RAG service using ChromaDB for campaign data retrieval.

Provides a RAGService class that manages:
    - Embedding and storing campaign data with rich text representations.
    - Semantic search with optional metadata filters.
    - Hybrid search combining vector results with SQL query results.
    - Index refresh/rebuild operations.

Uses OpenAI embeddings as primary, with sentence-transformers local fallback.

Usage::

    from app.services.rag import RAGService

    rag = RAGService()
    await rag.embed_and_store(campaign_dicts)
    results = await rag.retrieve("best performing QSR campaign")
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from app.config import Settings, settings
from app.services.llm_client import LLMClient, get_llm_client

logger = logging.getLogger(__name__)

COLLECTION_NAME = "campaign_data"


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
    """Build a flat metadata dict for a ChromaDB document.

    ChromaDB metadata values must be str, int, float, or bool.

    Args:
        campaign: A campaign dict.

    Returns:
        Flat metadata dict with key campaign attributes.
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
    """Manages ChromaDB vector storage and retrieval for campaign data.

    Supports OpenAI embeddings (primary) with sentence-transformers fallback.

    Args:
        cfg: Application settings. Defaults to the global singleton.
        llm_client: LLMClient for OpenAI embeddings. Defaults to the global singleton.
        chroma_client: Optional pre-configured ChromaDB client for testing.
    """

    def __init__(
        self,
        cfg: Settings | None = None,
        llm_client: LLMClient | None = None,
        chroma_client: chromadb.ClientAPI | None = None,
    ) -> None:
        self._cfg = cfg or settings
        self._llm = llm_client
        self._chroma = chroma_client
        self._collection: Collection | None = None
        self._local_embedder: Any = None  # Lazy-loaded sentence-transformer

    @property
    def llm(self) -> LLMClient:
        """Lazily resolve the LLM client."""
        if self._llm is None:
            self._llm = get_llm_client()
        return self._llm

    @property
    def chroma(self) -> chromadb.ClientAPI:
        """Return or create the ChromaDB client."""
        if self._chroma is None:
            self._chroma = chromadb.PersistentClient(path=self._cfg.chroma_persist_dir)
        return self._chroma

    @property
    def collection(self) -> Collection:
        """Return or create the campaign data collection."""
        if self._collection is None:
            self._collection = self.chroma.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Campaign performance data for RAG retrieval"},
            )
        return self._collection

    # ── Embedding ─────────────────────────────────────────────────────

    async def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the OpenAI embeddings API.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors.
        """
        return await self.llm.embed_texts(texts)

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using a local sentence-transformers model as fallback.

        Lazily loads the model on first call.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors.
        """
        if self._local_embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._local_embedder = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Loaded local embedding model: all-MiniLM-L6-v2")
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "Install it or provide an OpenAI API key."
                )
        embeddings = self._local_embedder.encode(texts, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using OpenAI (primary) or sentence-transformers (fallback).

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors.
        """
        if self._cfg.openai_api_key:
            try:
                return await self._embed_openai(texts)
            except Exception as exc:
                logger.warning(
                    "OpenAI embedding failed (%s), falling back to local model", exc
                )

        logger.info("Using local sentence-transformer embeddings")
        return self._embed_local(texts)

    # ── Store ─────────────────────────────────────────────────────────

    async def embed_and_store(
        self,
        campaigns: list[dict],
        collection_name: str = COLLECTION_NAME,
    ) -> int:
        """Embed campaign data and store in ChromaDB.

        Builds a rich text representation for each campaign, generates
        embeddings, and upserts into the specified collection.

        Args:
            campaigns: List of campaign dicts (from JSON or DB).
            collection_name: ChromaDB collection to store in.

        Returns:
            Number of documents stored.
        """
        if not campaigns:
            logger.warning("embed_and_store called with empty campaign list")
            return 0

        if collection_name != COLLECTION_NAME:
            self._collection = self.chroma.get_or_create_collection(
                name=collection_name
            )

        documents: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []

        for c in campaigns:
            doc = _build_document_text(c)
            meta = _build_metadata(c)
            doc_id = str(c.get("campaign_id", c["campaign_name"]))

            documents.append(doc)
            metadatas.append(meta)
            ids.append(doc_id)

        logger.info("Embedding %d campaign documents...", len(documents))
        embeddings = await self._embed(documents)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "Stored %d documents in collection '%s'", len(documents), collection_name
        )
        return len(documents)

    # ── Retrieve ──────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        n_results: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Semantic search over campaign documents.

        Args:
            query: Natural-language search query.
            n_results: Maximum number of results.
            filters: Optional ChromaDB where-clause filters. Supports:
                - {"vertical": "QSR"} — exact match
                - {"client_name": "Dunkin'"} — exact match
                - {"incremental_roas": {"$gte": 30.0}} — numeric comparison

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
        query_embedding = await self._embed([query])

        # Build ChromaDB query kwargs
        query_kwargs: dict[str, Any] = {
            "query_embeddings": query_embedding,
            "n_results": min(n_results, self.collection.count() or n_results),
        }

        if filters:
            # Wrap multiple filters in $and for ChromaDB
            if len(filters) > 1:
                where_clauses = []
                for key, value in filters.items():
                    if isinstance(value, dict):
                        where_clauses.append({key: value})
                    else:
                        where_clauses.append({key: value})
                query_kwargs["where"] = {"$and": where_clauses}
            else:
                query_kwargs["where"] = filters

        # Guard against empty collection
        if self.collection.count() == 0:
            logger.warning("Collection is empty, returning no results")
            return []

        results = self.collection.query(**query_kwargs)

        # Reshape results into a list of dicts
        output: list[dict] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            output.append(
                {
                    "id": doc_id,
                    "document": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "distance": distances[i] if i < len(distances) else 999.0,
                }
            )

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
                # Keep the entry but note it appeared in both sources
                seen[key]["source"] = "both"
                # Boost relevance: halve the vector distance for dual-source matches
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

        try:
            self.chroma.delete_collection(COLLECTION_NAME)
            logger.info("Deleted existing collection '%s'", COLLECTION_NAME)
        except ValueError:
            logger.info("No existing collection to delete")

        self._collection = None  # Reset cached collection reference
        count = await self.embed_and_store(campaigns)
        logger.info("Index refreshed with %d documents", count)
        return count

    def get_collection_stats(self) -> dict:
        """Return basic stats about the current collection.

        Returns:
            Dict with count and collection name.
        """
        return {
            "collection_name": COLLECTION_NAME,
            "document_count": self.collection.count(),
        }


# ── Module-level helpers (keep backward compat with scaffold) ─────────

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


def get_collection() -> Collection:
    """Return the ChromaDB collection (backward compat with main.py lifespan).

    Returns:
        The ChromaDB Collection object.
    """
    return get_rag_service().collection
