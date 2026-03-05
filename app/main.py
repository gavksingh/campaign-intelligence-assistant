"""FastAPI application entry point.

Configures the app with CORS middleware, lifespan events for startup/shutdown
(database connections, vector store initialization), and includes API routes.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown.

    Startup: create database tables, warm up vector store, init LLM client.
    Shutdown: dispose of database engine.
    """
    from app.database import dispose_engine, init_db
    from app.services.llm_client import get_llm_client
    from app.services.rag import get_rag_service

    logger.info("Starting up — initializing services...")

    await init_db()
    logger.info("Database tables ensured.")

    rag = get_rag_service()
    stats = await rag.get_collection_stats()
    logger.info("Vector store ready: %d documents", stats["document_count"])

    get_llm_client()
    logger.info("LLM client initialized (model: %s)", settings.llm_model)

    yield

    await dispose_engine()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Campaign Intelligence Assistant",
    description=(
        "AI-powered campaign analytics and reporting API. "
        "Query campaign data with natural language, generate LCI reports, "
        "compare campaigns, and get audience recommendations."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with method, path, status, and timing."""
    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "%s %s -> %d (%dms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Processing-Time-Ms"] = str(elapsed_ms)
    return response


# ── Root endpoint ─────────────────────────────────────────────────────


@app.get("/", tags=["root"])
async def root():
    """Return API information."""
    return {
        "name": "Campaign Intelligence Assistant",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/health",
        "endpoints": [
            "POST /api/chat",
            "GET  /api/campaigns",
            "GET  /api/campaigns/{id}",
            "POST /api/reports/generate",
            "POST /api/reports/compare",
            "POST /api/audience/recommend",
            "GET  /api/health",
        ],
    }


# ── Global exception handler ──────────────────────────────────────────

from fastapi.responses import JSONResponse  # noqa: E402


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions and return JSON instead of crashing."""
    logger.error("Unhandled exception: %s: %s", type(exc).__name__, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )


# ── Include API routes ────────────────────────────────────────────────

from app.api.routes import router  # noqa: E402

app.include_router(router)
