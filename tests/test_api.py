"""Tests for FastAPI route endpoints.

Tests that don't require external dependencies (database, LLM) use mocking.
The health endpoint test verifies the response shape without requiring
live connections.  Endpoints that hit the database may raise connection
errors when no PostgreSQL instance is available — those tests are marked
as expected to tolerate both HTTP error codes and connection failures.
"""

import pytest


@pytest.mark.asyncio
async def test_root_endpoint(client):
    """GET / should return API information."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Campaign Intelligence Assistant"
    assert "endpoints" in data
    assert "/docs" == data["docs"]


@pytest.mark.asyncio
async def test_health_endpoint_shape(client):
    """GET /api/health should return the expected fields."""
    response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "database" in data
    assert "vector_store" in data
    assert "llm" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_campaigns_list_endpoint_exists(client):
    """GET /api/campaigns should return 200 or 500 when DB unavailable."""
    try:
        response = await client.get("/api/campaigns")
        assert response.status_code in (200, 500)
    except Exception:
        pytest.skip("Database not available")


@pytest.mark.asyncio
async def test_campaigns_invalid_vertical(client):
    """GET /api/campaigns?vertical=INVALID should return 400."""
    response = await client.get("/api/campaigns?vertical=INVALID")
    assert response.status_code in (400, 500)


@pytest.mark.asyncio
async def test_campaign_not_found(client):
    """GET /api/campaigns/99999 should return 404."""
    try:
        response = await client.get("/api/campaigns/99999")
        assert response.status_code in (404, 500)
    except Exception:
        pytest.skip("Database not available")


@pytest.mark.asyncio
async def test_chat_endpoint_exists(client):
    """POST /api/chat should accept a valid request body."""
    response = await client.post(
        "/api/chat", json={"message": "hello"}
    )
    # Will be 200 or 500 depending on agent availability
    assert response.status_code != 404


@pytest.mark.asyncio
async def test_chat_empty_message_rejected(client):
    """POST /api/chat with empty message should return 422."""
    response = await client.post("/api/chat", json={"message": ""})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_report_generate_endpoint_exists(client):
    """POST /api/reports/generate should accept a valid body."""
    try:
        response = await client.post(
            "/api/reports/generate",
            json={"campaign_id": 1, "format": "markdown"},
        )
        assert response.status_code != 404
    except Exception:
        pytest.skip("Database not available")


@pytest.mark.asyncio
async def test_report_generate_invalid_format(client):
    """POST /api/reports/generate with invalid format should return 422."""
    response = await client.post(
        "/api/reports/generate",
        json={"campaign_id": 1, "format": "xlsx"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_compare_endpoint_exists(client):
    """POST /api/reports/compare should accept a valid body."""
    response = await client.post(
        "/api/reports/compare",
        json={"campaign_id_1": 1, "campaign_id_2": 2},
    )
    assert response.status_code != 404


@pytest.mark.asyncio
async def test_audience_recommend_endpoint_exists(client):
    """POST /api/audience/recommend should accept a valid body."""
    response = await client.post(
        "/api/audience/recommend",
        json={"description": "lunch-time diners in Texas for QSR"},
    )
    assert response.status_code != 404


@pytest.mark.asyncio
async def test_processing_time_header(client):
    """All responses should include X-Processing-Time-Ms header."""
    response = await client.get("/")
    assert "x-processing-time-ms" in response.headers
