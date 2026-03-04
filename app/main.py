"""FastAPI application entry point.

Configures the app with CORS middleware, lifespan events for startup/shutdown
(database connections, vector store initialization), and includes API routes.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings
from app.database import dispose_engine, init_db
from app.services.rag import get_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown.

    Startup: initialize database connection pool, ensure tables exist,
             warm up ChromaDB collection.
    Shutdown: dispose of database engine.
    """
    await init_db()
    get_collection()  # Warm up ChromaDB client + collection
    yield
    await dispose_engine()


app = FastAPI(
    title="Campaign Intelligence Assistant",
    description="AI-powered campaign analytics and reporting API.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
