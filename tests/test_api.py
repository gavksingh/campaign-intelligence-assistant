"""Tests for FastAPI route endpoints."""

import pytest


@pytest.mark.asyncio
async def test_health_check(client):
    """GET /health should return 200 with status ok."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_chat_endpoint_exists(client):
    """POST /chat should exist (returns 500/501 since not implemented, not 404)."""
    response = await client.post("/chat", json={"message": "hello"})
    assert response.status_code != 404


@pytest.mark.asyncio
async def test_report_endpoint_exists(client):
    """GET /report/{id} should exist (returns 500/501 since not implemented, not 404)."""
    response = await client.get("/report/1")
    assert response.status_code != 404
