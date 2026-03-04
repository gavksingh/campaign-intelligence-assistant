"""Application configuration loaded from environment variables.

Uses pydantic-settings to validate and type-check all configuration values
at startup. Reads from .env file automatically.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings populated from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str = ""

    # Database
    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/campaign_intel"
    )

    # ChromaDB
    chroma_persist_dir: str = "./chroma_data"

    # LLM
    llm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    # CORS
    allowed_origins: list[str] = ["*"]


settings = Settings()
