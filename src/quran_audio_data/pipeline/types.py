from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DeviceOption = Literal["auto", "cpu", "cuda"]
EngineOption = Literal["nemo", "whisperx", "mfa"]
AccuracyMode = Literal["strict"]
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
    reference_split: str | None


@dataclass(slots=True)
class ProcessedFile:
    row: ManifestRow
    output_json: Path
    output_ayah_csv: Path
    output_words_csv: Path
    qc_report_json: Path
    text_audit_json: Path | None
    source: str
    fallback_used: bool
    elapsed_s: float


@dataclass(slots=True)
class PipelineErrorDetail:
    key: str
    message: str
    attempted_engines: list[str]


@dataclass(slots=True)
class PipelineReportV3:
    total: int
    succeeded: int
    failed: int
    aligned: int
    fallback_used: int
    elapsed_s: float
    outputs: list[ProcessedFile]
    errors: list[str]
    error_details: list[PipelineErrorDetail]
    attempted_engines: list[str]
    priors_used: int
    schema_version: Literal["v3"] = "v3"


ProcessingSummary = PipelineReportV3


def default_cache_dir() -> Path:
    from quran_audio_data.core.settings import get_settings

    return Path(get_settings().cache_dir)
