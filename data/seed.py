"""Seed script to populate PostgreSQL with mock campaign data and pgvector embeddings.

Usage::

    python -m data.seed

Reads data/mock_campaigns.json, creates all tables, inserts campaign records
into PostgreSQL, and embeds campaign summaries with key metrics into pgvector
for RAG retrieval.
"""

import asyncio
import json
import sys
import uuid
from datetime import date
from pathlib import Path

from sqlalchemy import select

# Allow running as `python -m data.seed` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import get_session_factory, dispose_engine, init_db
from app.models.campaign import (
    AudienceSegment,
    Campaign,
    CampaignMetrics,
    CampaignStatus,
    TargetingType,
    Vertical,
)

MOCK_DATA_PATH = Path(__file__).parent / "mock_campaigns.json"


def _parse_date(date_str: str) -> date:
    """Parse an ISO date string to a date object.

    Args:
        date_str: Date string in YYYY-MM-DD format.

    Returns:
        A date object.
    """
    return date.fromisoformat(date_str)


def _parse_enum(value: str, enum_cls: type) -> object:
    """Match a string value to an enum member by value (case-insensitive).

    Args:
        value: The raw string value from JSON.
        enum_cls: The enum class to match against.

    Returns:
        The matching enum member.

    Raises:
        ValueError: If no matching enum member is found.
    """
    for member in enum_cls:
        if member.value.lower() == value.lower():
            return member
    raise ValueError(f"No {enum_cls.__name__} member matching '{value}'")


async def seed_database(data: list[dict]) -> dict[str, int]:
    """Insert mock campaign data into PostgreSQL.

    Creates Campaign, CampaignMetrics, and AudienceSegment records from
    the parsed JSON data.

    Args:
        data: List of campaign dicts from mock_campaigns.json.

    Returns:
        Dict with counts of inserted campaigns, metrics, and segments.
    """
    counts = {"campaigns": 0, "metrics": 0, "segments": 0}

    factory = get_session_factory()
    async with factory() as session:
        for item in data:
            # Check if campaign already exists
            existing = await session.execute(
                select(Campaign).where(
                    Campaign.campaign_id == uuid.UUID(item["campaign_id"])
                )
            )
            if existing.scalar_one_or_none():
                print(f"  Skipping existing campaign: {item['campaign_name']}")
                continue

            # Create campaign
            campaign = Campaign(
                campaign_id=uuid.UUID(item["campaign_id"]),
                campaign_name=item["campaign_name"],
                client_name=item["client_name"],
                vertical=_parse_enum(item["vertical"], Vertical),
                start_date=_parse_date(item["start_date"]),
                end_date=_parse_date(item["end_date"])
                if item.get("end_date")
                else None,
                budget=item["budget"],
                status=_parse_enum(item["status"], CampaignStatus),
                targeting_type=_parse_enum(item["targeting_type"], TargetingType),
                campaign_summary=item.get("campaign_summary"),
            )
            session.add(campaign)
            await session.flush()  # Get campaign.id for FK references
            counts["campaigns"] += 1

            # Create metrics
            m = item.get("metrics", {})
            metrics = CampaignMetrics(
                campaign_id=campaign.id,
                impressions=m.get("impressions", 0),
                visit_lift_percent=m.get("visit_lift_percent", 0.0),
                sales_lift_percent=m.get("sales_lift_percent", 0.0),
                incremental_roas=m.get("incremental_roas", 0.0),
                incremental_visits=m.get("incremental_visits", 0),
                incremental_sales_dollars=m.get("incremental_sales_dollars", 0.0),
                avg_basket_size=m.get("avg_basket_size", 0.0),
                purchase_frequency=m.get("purchase_frequency", 0.0),
                top_markets=m.get("top_markets", []),
                top_performing_creative=m.get("top_performing_creative"),
                control_group_size=m.get("control_group_size", 0),
                exposed_group_size=m.get("exposed_group_size", 0),
            )
            session.add(metrics)
            counts["metrics"] += 1

            # Create audience segments
            for segment_name in item.get("audience_segments", []):
                segment = AudienceSegment(
                    campaign_id=campaign.id,
                    segment_name=segment_name,
                )
                session.add(segment)
                counts["segments"] += 1

            print(f"  Inserted: {item['campaign_name']}")

        await session.commit()

    return counts


async def seed_vector_store(data: list[dict]) -> int:
    """Embed and store campaign data in pgvector via RAGService.

    Requires GOOGLE_API_KEY to be set for embedding generation.

    Args:
        data: List of campaign dicts from mock_campaigns.json.

    Returns:
        Number of documents embedded.
    """
    from app.services.rag import RAGService

    rag = RAGService()

    # Look up DB integer IDs for each campaign by UUID
    factory = get_session_factory()
    campaign_dicts_with_ids: list[dict] = []

    async with factory() as session:
        for item in data:
            result = await session.execute(
                select(Campaign.id).where(
                    Campaign.campaign_id == uuid.UUID(item["campaign_id"])
                )
            )
            db_id = result.scalar_one_or_none()
            if db_id is not None:
                enriched = {**item, "id": db_id}
                campaign_dicts_with_ids.append(enriched)
            else:
                print(
                    f"  Warning: no DB record for {item['campaign_name']}, skipping embed"
                )

    count = await rag.embed_and_store(campaign_dicts_with_ids)
    print(f"  Embedded {count} documents into pgvector.")
    return count


async def main() -> None:
    """Run the full seed pipeline: create tables -> insert DB -> embed vectors."""
    # Load mock data
    raw = MOCK_DATA_PATH.read_text(encoding="utf-8")
    data = json.loads(raw)

    if not data:
        print("No mock data found in mock_campaigns.json. Skipping seed.")
        return

    print(f"Loaded {len(data)} campaigns from mock_campaigns.json\n")

    # Step 1: Create tables
    print("Creating database tables...")
    await init_db()
    print("Tables created.\n")

    # Step 2: Seed PostgreSQL
    print("Seeding PostgreSQL...")
    counts = await seed_database(data)
    print(
        f"\nPostgreSQL seeded: "
        f"{counts['campaigns']} campaigns, "
        f"{counts['metrics']} metrics, "
        f"{counts['segments']} audience segments.\n"
    )

    # Step 3: Seed pgvector embeddings
    print("Seeding pgvector embeddings...")
    num_embedded = await seed_vector_store(data)
    print(f"\npgvector seeded: {num_embedded} documents.\n")

    # Cleanup
    await dispose_engine()
    print("Done! Database and vector store are ready.")


if __name__ == "__main__":
    asyncio.run(main())
