from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DeviceOption = Literal["auto", "cpu", "cuda"]
EngineOption = Literal["nemo", "whisperx", "mfa"]
AccuracyMode = Literal["standard", "strict"]
EngineAvailabilityPolicy = Literal["best_effort", "require_requested", "require_all"]


class PipelineError(RuntimeError):
    pass


@dataclass(slots=True)
class ManifestRow:
    audio_path: Path
    reciter_id: str
    surah: int
    ayah: int | None
    source_url: str | None
    sha256: str | None
    language: str
    riwaya: str | None
    text_variant: str | None
    gold_split: str | None


@dataclass(slots=True)
class ProcessedFile:
    row: ManifestRow
    output_json: Path
    output_ayah_csv: Path
    output_words_csv: Path
    qc_report_json: Path
    source: str
    fallback_used: bool
    elapsed_s: float


@dataclass(slots=True)
class PipelineErrorDetail:
    key: str
    message: str
    attempted_engines: list[str]


@dataclass(slots=True)
class PipelineReportV2:
    total: int
    succeeded: int
    failed: int
    existing_resolved: int
    aligned: int
    fallback_used: int
    elapsed_s: float
    outputs: list[ProcessedFile]
    errors: list[str]
    error_details: list[PipelineErrorDetail]
    attempted_engines: list[str]
    schema_version: Literal["v2"] = "v2"


ProcessingSummary = PipelineReportV2


def default_cache_dir() -> Path:
    from quran_audio_data.core.settings import get_settings

    return Path(get_settings().cache_dir)
