"""Seed script to populate the database and vector store with mock campaign data.

Usage::

    python data/seed.py

Reads mock_campaigns.json, inserts records into PostgreSQL, and embeds
campaign descriptions into ChromaDB for RAG retrieval.
"""

import asyncio
import json
from pathlib import Path

MOCK_DATA_PATH = Path(__file__).parent / "mock_campaigns.json"


async def seed_database() -> None:
    """Insert mock campaign data into PostgreSQL.

    Reads from mock_campaigns.json and creates Campaign, CampaignMetrics,
    and AudienceSegment records.
    """
    raise NotImplementedError("Database seeding not yet implemented.")


async def seed_vector_store() -> None:
    """Embed and store campaign data in ChromaDB.

    Generates text representations of campaigns and their metrics,
    then embeds them into the vector store for RAG retrieval.
    """
    raise NotImplementedError("Vector store seeding not yet implemented.")


async def main() -> None:
    """Run the full seed pipeline."""
    data = json.loads(MOCK_DATA_PATH.read_text())
    if not data:
        print("No mock data found in mock_campaigns.json. Skipping seed.")
        return

    print("Seeding database...")
    await seed_database()
    print("Seeding vector store...")
    await seed_vector_store()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
