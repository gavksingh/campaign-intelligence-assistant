"""RAG service using ChromaDB for campaign data retrieval.

Handles embedding generation, document storage, and semantic search over
campaign descriptions, metrics summaries, and audience segment data.
"""

import chromadb

from app.config import settings

_chroma_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None

COLLECTION_NAME = "campaign_data"


def get_chroma_client() -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _chroma_client


def get_collection() -> chromadb.Collection:
    """Return the campaign data collection, creating it if needed."""
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


async def embed_campaign_data(
    documents: list[str],
    metadatas: list[dict],
    ids: list[str],
) -> None:
    """Embed and store campaign documents in ChromaDB.

    Args:
        documents: Text content to embed (campaign descriptions, metric summaries).
        metadatas: Metadata dicts associated with each document.
        ids: Unique IDs for each document.
    """
    raise NotImplementedError("Campaign data embedding not yet implemented.")


async def search_similar(query: str, n_results: int = 5) -> list[dict]:
    """Search for campaign documents semantically similar to the query.

    Args:
        query: Natural-language search query.
        n_results: Maximum number of results to return.

    Returns:
        List of dicts with 'document', 'metadata', and 'distance' keys.
    """
    raise NotImplementedError("Similarity search not yet implemented.")


async def delete_collection() -> None:
    """Delete the campaign data collection. Used for reseeding."""
    raise NotImplementedError("Collection deletion not yet implemented.")
