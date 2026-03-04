"""Async SQLAlchemy database engine and session factory.

Provides an async engine configured from DATABASE_URL and an async session
maker for use in FastAPI dependency injection.
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

engine = create_async_engine(settings.database_url, echo=False)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""


async def get_session() -> AsyncSession:
    """FastAPI dependency that yields an async database session.

    Usage::

        @router.get("/example")
        async def example(session: AsyncSession = Depends(get_session)):
            ...
    """
    async with async_session() as session:
        yield session
