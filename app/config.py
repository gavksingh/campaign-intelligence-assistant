"""Application configuration loaded from environment variables.

Uses pydantic-settings to validate and type-check all configuration values
at startup. Reads from .env file automatically.
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings populated from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Groq (chat / structured output)
    groq_api_key: str = ""

    # Google Gemini (embeddings only)
    google_api_key: str = ""

    # Database
    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/campaign_intel"
    )

    # LLM
    llm_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "gemini-embedding-001"

    # CORS
    allowed_origins: list[str] = ["*"]

    @field_validator(
        "groq_api_key",
        "google_api_key",
        "database_url",
        "llm_model",
        "embedding_model",
        mode="before",
    )
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """Strip leading/trailing whitespace from string env vars."""
        return v.strip() if isinstance(v, str) else v


settings = Settings()
