"""Async SQLAlchemy database engine and session factory.

Provides:
    - Lazy-initialized async engine configured from DATABASE_URL.
    - Async session maker for use in FastAPI dependency injection.
    - Base class for ORM model declarations.
    - get_db() async generator for FastAPI Depends().

Engine creation is deferred to first use so that importing this module
(and thus the ORM models) does not require asyncpg to be installed
at import time (e.g. during tests that only need the schemas/enums).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""


# ── Lazy engine + session factory ─────────────────────────────────────

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return the async engine, creating it on first call.

    Returns:
        The shared AsyncEngine instance.
    """
    global _engine
    if _engine is None:
        from app.config import settings

        _engine = create_async_engine(
            settings.database_url,
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the async session factory, creating it on first call.

    Returns:
        The shared async_sessionmaker instance.
    """
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


# Backward-compatible aliases used by agents/tools and seed script
def _get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    return get_session_factory()


# Property-style access for code that references `async_session_factory` directly
class _SessionFactoryProxy:
    """Proxy that lazily initializes the session factory on first call."""

    def __call__(self, *args, **kwargs):
        return get_session_factory()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(get_session_factory(), name)


async_session_factory = _SessionFactoryProxy()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session.

    The session is automatically closed when the request completes.

    Yields:
        An async SQLAlchemy session.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Create all tables defined by ORM models.

    Intended for development and seeding. In production, use Alembic migrations.
    Enables the pgvector extension before creating tables.
    """
    from sqlalchemy import text

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    """Dispose of the async engine and release all connections."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
