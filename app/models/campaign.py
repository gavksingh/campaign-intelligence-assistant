"""SQLAlchemy ORM models for campaign data.

Models:
    Campaign — Core campaign entity with name, status, budget, date range, and targeting.
    CampaignMetrics — Performance metrics (impressions, visits, sales, ROAS).
    AudienceSegment — Targeting segments associated with campaigns.
"""

import enum
import uuid
from datetime import date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class CampaignStatus(str, enum.Enum):
    """Lifecycle status of a campaign."""

    COMPLETED = "completed"
    ACTIVE = "active"
    PLANNED = "planned"
    PAUSED = "paused"


class Vertical(str, enum.Enum):
    """Industry vertical for the campaign's client."""

    QSR = "QSR"
    AUTOMOTIVE = "Automotive"
    CPG = "CPG"
    RETAIL = "Retail"
    ENTERTAINMENT = "Entertainment"


class TargetingType(str, enum.Enum):
    """InMarket targeting methodology."""

    MOMENTS = "Moments"
    PREDICTIVE_MOMENTS = "Predictive Moments"
    GEOLINK = "GeoLink"
    AUDIENCE_BASED = "Audience-based"


class Campaign(Base):
    """A single advertising campaign with client and targeting metadata."""

    __tablename__ = "campaigns"
    __table_args__ = (
        Index("ix_campaigns_client_name", "client_name"),
        Index("ix_campaigns_vertical", "vertical"),
        Index("ix_campaigns_status", "status"),
        Index("ix_campaigns_start_date", "start_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    campaign_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), unique=True, nullable=False, default=uuid.uuid4
    )
    campaign_name: Mapped[str] = mapped_column(String(255), nullable=False)
    client_name: Mapped[str] = mapped_column(String(255), nullable=False)
    vertical: Mapped[Vertical] = mapped_column(Enum(Vertical), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    budget: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[CampaignStatus] = mapped_column(
        Enum(CampaignStatus), default=CampaignStatus.PLANNED
    )
    targeting_type: Mapped[TargetingType] = mapped_column(
        Enum(TargetingType), nullable=False
    )
    campaign_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    metrics: Mapped["CampaignMetrics"] = relationship(
        back_populates="campaign", uselist=False, cascade="all, delete-orphan"
    )
    audience_segments: Mapped[list["AudienceSegment"]] = relationship(
        back_populates="campaign", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Campaign(id={self.id}, name='{self.campaign_name}', status={self.status.value})>"


class CampaignMetrics(Base):
    """Aggregated performance metrics for a campaign.

    Stores InMarket-standard KPIs: visit lift, sales lift, incremental ROAS,
    incremental visits/sales, and market-level breakdowns.
    """

    __tablename__ = "campaign_metrics"
    __table_args__ = (Index("ix_campaign_metrics_campaign_id", "campaign_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    campaign_id: Mapped[int] = mapped_column(
        ForeignKey("campaigns.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    impressions: Mapped[int] = mapped_column(Integer, default=0)
    visit_lift_percent: Mapped[float] = mapped_column(Float, default=0.0)
    sales_lift_percent: Mapped[float] = mapped_column(Float, default=0.0)
    incremental_roas: Mapped[float] = mapped_column(Float, default=0.0)
    incremental_visits: Mapped[int] = mapped_column(Integer, default=0)
    incremental_sales_dollars: Mapped[float] = mapped_column(Float, default=0.0)
    avg_basket_size: Mapped[float] = mapped_column(Float, default=0.0)
    purchase_frequency: Mapped[float] = mapped_column(Float, default=0.0)
    top_markets: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    top_performing_creative: Mapped[str | None] = mapped_column(
        String(500), nullable=True
    )
    control_group_size: Mapped[int] = mapped_column(Integer, default=0)
    exposed_group_size: Mapped[int] = mapped_column(Integer, default=0)

    # Relationship
    campaign: Mapped["Campaign"] = relationship(back_populates="metrics")

    @property
    def ctr(self) -> float:
        """Click-through rate (visits / impressions as percentage)."""
        return (
            (self.incremental_visits / self.impressions * 100)
            if self.impressions
            else 0.0
        )

    @property
    def cost_per_visit(self) -> float:
        """Cost per incremental visit."""
        if self.incremental_visits and self.campaign:
            return self.campaign.budget / self.incremental_visits
        return 0.0

    def __repr__(self) -> str:
        return (
            f"<CampaignMetrics(campaign_id={self.campaign_id}, "
            f"roas={self.incremental_roas}, visit_lift={self.visit_lift_percent}%)>"
        )


class AudienceSegment(Base):
    """An audience targeting segment linked to a campaign."""

    __tablename__ = "audience_segments"
    __table_args__ = (
        Index("ix_audience_segments_campaign_id", "campaign_id"),
        Index("ix_audience_segments_name", "segment_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    campaign_id: Mapped[int] = mapped_column(
        ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False
    )
    segment_name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Relationship
    campaign: Mapped["Campaign"] = relationship(back_populates="audience_segments")

    def __repr__(self) -> str:
        return f"<AudienceSegment(id={self.id}, name='{self.segment_name}')>"
