"""Конфигурация приложения через переменные окружения (pydantic-settings)."""
from __future__ import annotations

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field
    _HAS_PYDANTIC = True
except Exception:  # pragma: no cover — окружение без pydantic
    _HAS_PYDANTIC = False


if _HAS_PYDANTIC:
    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

        # PostgreSQL (производственная БД). Драйвер psycopg v3.
        database_url: str = Field(
            default="postgresql+psycopg://recsys:recsys@localhost:5432/recsys",
            description="SQLAlchemy URL подключения к PostgreSQL",
        )
        api_title: str = "Unified Cross-Platform Hotel RecSys"
        api_version: str = "1.0.0"

        top_k: int = 5
        recompute_interval_hours: int = 12
        cache_ttl_hours: int = 12

        # для удобной локальной отладки можно переопределить на sqlite:
        #   DATABASE_URL=sqlite:///./dev.db
        sql_echo: bool = False

    settings = Settings()
else:  # упрощённый fallback, чтобы модуль импортировался без pydantic
    class _Fallback:
        database_url = "postgresql+psycopg://recsys:recsys@localhost:5432/recsys"
        api_title = "Unified Cross-Platform Hotel RecSys"
        api_version = "1.0.0"
        top_k = 5
        recompute_interval_hours = 12
        cache_ttl_hours = 12
        sql_echo = False

    settings = _Fallback()
