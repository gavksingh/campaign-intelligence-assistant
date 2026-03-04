"""Shared pytest fixtures for the test suite.

Provides async HTTP test client and common test configuration.
Imports are deferred so tests that don't need the full app (e.g. report
generation tests) can run without asyncpg/chromadb installed.
"""

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    """Async HTTP test client for the FastAPI app.

    Only import when actually used — avoids pulling in asyncpg/chromadb
    for unit tests that don't need the API.
    """
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
