"""SQLAlchemy ORM models for campaign data.

Models:
    Campaign — Core campaign entity with name, status, budget, and date range.
    CampaignMetrics — Performance metrics (impressions, clicks, conversions, spend, revenue).
    AudienceSegment — Targeting segments associated with campaigns.
"""

import enum
from datetime import date, datetime

from sqlalchemy import Date, DateTime, Enum, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class CampaignStatus(str, enum.Enum):
    """Lifecycle status of a campaign."""

    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


class Campaign(Base):
    """A single advertising campaign."""

    __tablename__ = "campaigns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[CampaignStatus] = mapped_column(
        Enum(CampaignStatus), default=CampaignStatus.DRAFT
    )
    budget: Mapped[float] = mapped_column(Float, nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    metrics: Mapped[list["CampaignMetrics"]] = relationship(back_populates="campaign")
    audience_segments: Mapped[list["AudienceSegment"]] = relationship(
        back_populates="campaign"
    )


class CampaignMetrics(Base):
    """Daily performance metrics for a campaign."""

    __tablename__ = "campaign_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    impressions: Mapped[int] = mapped_column(Integer, default=0)
    clicks: Mapped[int] = mapped_column(Integer, default=0)
    conversions: Mapped[int] = mapped_column(Integer, default=0)
    spend: Mapped[float] = mapped_column(Float, default=0.0)
    revenue: Mapped[float] = mapped_column(Float, default=0.0)

    campaign: Mapped["Campaign"] = relationship(back_populates="metrics")

    @property
    def ctr(self) -> float:
        """Click-through rate."""
        return (self.clicks / self.impressions * 100) if self.impressions else 0.0

    @property
    def cpa(self) -> float:
        """Cost per acquisition."""
        return (self.spend / self.conversions) if self.conversions else 0.0

    @property
    def roas(self) -> float:
        """Return on ad spend."""
        return (self.revenue / self.spend) if self.spend else 0.0


class AudienceSegment(Base):
    """An audience targeting segment linked to a campaign."""

    __tablename__ = "audience_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    size: Mapped[int] = mapped_column(Integer, default=0)

    campaign: Mapped["Campaign"] = relationship(back_populates="audience_segments")
