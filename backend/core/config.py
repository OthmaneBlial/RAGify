from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    # Database
    database_url: str = (
        "postgresql+asyncpg://ragify:RagifyStrongPass2023@localhost/ragify"
    )

    # Security
    secret_key: str = "your-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # API
    api_v1_str: str = "/api/v1"

    # Vector
    vector_dimension: int = 384  # Must match SentenceTransformer dimension

    # Caching / message bus
    redis_url: Optional[str] = None

    # Model Provider API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None  # Alias for google_api_key
    openrouter_api_key: Optional[str] = None

    # Model Provider Settings
    openai_rate_limit_requests: int = 60
    openai_rate_limit_window: int = 60
    anthropic_rate_limit_requests: int = 60
    anthropic_rate_limit_window: int = 60

    # Default Model Settings
    default_provider: str = "openrouter"
    default_model: str = "openai/gpt-5-nano"
    default_temperature: float = 0.7
    default_max_tokens: int = 4096

    model_config = SettingsConfigDict(
        extra="ignore", env_file=".env", case_sensitive=False
    )


settings = Settings()
