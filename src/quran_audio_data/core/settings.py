from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QAD_", extra="ignore")

    request_timeout_s: float = 20.0
    request_retries: int = 5
    retry_backoff_s: float = 1.0
    retry_max_backoff_s: float = 30.0
    retry_jitter_s: float = 0.4

    cache_dir: str = ".cache/timings/v3"
    mfa_cache_dir: str = ".cache/mfa"

    engine_order: list[str] = Field(default_factory=lambda: ["nemo", "whisperx", "mfa"])
    required_engines: list[str] = Field(default_factory=list)

    use_webrtcvad: bool = True
    use_librosa: bool = True

    @field_validator("engine_order", "required_engines", mode="before")
    @classmethod
    def _parse_csv_list(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
