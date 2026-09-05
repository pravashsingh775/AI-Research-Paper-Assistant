from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://research:research@localhost:5432/research"
    redis_url: str = "redis://localhost:6379/0"
    jwt_secret: str = "change-me-in-production"
    jwt_expire_minutes: int = 60
    cors_origins: str = "http://localhost:3000"
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-5-20250929"
    anthropic_api_key: str = ""
    semantic_scholar_api_key: str = ""
    openalex_email: str = ""
    embedding_provider: str = "local"
    embedding_model: str = ""
    embedding_dimension: int = 384
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    app_env: str = "development"
    object_storage_endpoint: str = "http://localhost:9000"
    object_storage_bucket: str = "research-papers"
    object_storage_access_key: str = "minio"
    object_storage_secret_key: str = "miniosecret"
    object_storage_region: str = "us-east-1"
    storage_backend: str = "auto"
    storage_local_dir: str = "data/storage"

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", case_sensitive=False
    )

    def check_production_guard(self) -> None:
        if self.app_env.lower() in ("production", "prod"):
            if (
                not self.jwt_secret
                or self.jwt_secret == "change-me-in-production"
                or len(self.jwt_secret) < 32
            ):
                raise RuntimeError(
                    "FATAL: In production, JWT_SECRET must be set to a secure secret with at least 32 characters."
                )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.check_production_guard()
    return settings
