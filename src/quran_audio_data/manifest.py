from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator


class ManifestRowModel(BaseModel):
    audio_path: str
    reciter_id: str
    surah: int
    ayah: int | None = None
    source_url: str | None = None
    sha256: str | None = None
    language: str = "ar"
    riwaya: str | None = None
    text_variant: str | None = None
    reference_split: str | None = None

    @field_validator("audio_path", "reciter_id", mode="before")
    @classmethod
    def _strip_required_text(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("value is required")
        return text

    @field_validator("source_url", "sha256", "riwaya", "text_variant", "reference_split", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> str | None:
        text = str(value).strip() if value is not None else ""
        return text or None

    @field_validator("language", mode="before")
    @classmethod
    def _strip_language(cls, value: Any) -> str:
        text = str(value).strip() if value is not None else ""
        return text or "ar"

    @field_validator("ayah", mode="before")
    @classmethod
    def _parse_optional_ayah(cls, value: Any) -> int | None:
        if value is None:
            return None
        text = str(value).strip()
        if text == "":
            return None
        return int(text)

    @property
    def audio_path_obj(self) -> Path:
        return Path(self.audio_path)


class ManifestValidationError(RuntimeError):
    def __init__(self, row_index: int, error: ValidationError) -> None:
        self.row_index = row_index
        self.error = error
        super().__init__(f"manifest row {row_index}: {error.errors(include_url=False)}")
