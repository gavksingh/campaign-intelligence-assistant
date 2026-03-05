"""Vercel serverless entry point.

Exposes the FastAPI application for Vercel's Python runtime.
"""

from app.main import app  # noqa: F401
