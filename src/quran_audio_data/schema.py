from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median
import csv
from typing import Any, Literal

import orjson
from pydantic import BaseModel, Field, model_validator


InputMode = Literal["full_surah", "ayah_file"]
TimingSource = Literal["existing", "aligned", "fallback"]
AlignmentOrigin = Literal["native", "interpolated", "distributed"]


class AudioMetadata(BaseModel):
    path: str
    duration_s: float
    sample_rate: int
    channels: int


class MetaInfo(BaseModel):
    reciter_id: str
    surah: int
    input_mode: InputMode


class EngineInfo(BaseModel):
    name: str
    model: str
    device: str
    fallback_used: bool = False


class AyahTiming(BaseModel):
    surah: int
    ayah: int
    start_s: float
    end_s: float
    source: TimingSource

    @model_validator(mode="after")
    def _validate_timing(self) -> "AyahTiming":
        if self.end_s < self.start_s:
            raise ValueError("ayah end_s must be >= start_s")
        return self


class WordTiming(BaseModel):
    surah: int
    ayah: int
    word_index_global: int
    word_index_in_ayah: int
    text_uthmani: str
    text_norm: str
    start_s: float
    end_s: float
    confidence: float | None = None
    alignment_origin: AlignmentOrigin = "native"
    match_score: float | None = None
    engine_candidate: str | None = None

    @model_validator(mode="after")
    def _validate_timing(self) -> "WordTiming":
        if self.end_s < self.start_s:
            raise ValueError("word end_s must be >= start_s")
        return self


class QCReport(BaseModel):
    coverage: float
    monotonic: bool
    duration_match: bool
    warnings: list[str] = Field(default_factory=list)
    zero_or_negative_ratio: float = 0.0
    median_confidence: float | None = None
    interpolated_ratio: float = 0.0
    lexical_match_ratio: float | None = None
    speech_end_delta_ratio: float = 1.0
    quantization_step_ms: float | None = None
    engine_candidate_scores: dict[str, float] = Field(default_factory=dict)


class TimingResult(BaseModel):
    schema_version: Literal["v1"] = "v1"
    audio: AudioMetadata
    meta: MetaInfo
    engine: EngineInfo
    ayahs: list[AyahTiming]
    words: list[WordTiming]
    qc: QCReport

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json_bytes(self) -> bytes:
        return orjson.dumps(self.to_dict(), option=orjson.OPT_INDENT_2)

    def write_json(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(self.to_json_bytes())
        return out_path

    def write_csvs(self, stem_path: str | Path) -> tuple[Path, Path]:
        stem = Path(stem_path)
        stem.parent.mkdir(parents=True, exist_ok=True)

        ayah_csv = stem.with_name(f"{stem.name}_ayah.csv")
        with ayah_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["surah", "ayah", "start_s", "end_s", "source"],
            )
            writer.writeheader()
            for ayah in self.ayahs:
                writer.writerow(
                    {
                        "surah": ayah.surah,
                        "ayah": ayah.ayah,
                        "start_s": round(ayah.start_s, 3),
                        "end_s": round(ayah.end_s, 3),
                        "source": ayah.source,
                    }
                )

        words_csv = stem.with_name(f"{stem.name}_words.csv")
        with words_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "surah",
                    "ayah",
                    "word_index_in_ayah",
                    "text",
                    "start_s",
                    "end_s",
                    "confidence",
                    "alignment_origin",
                    "match_score",
                    "engine_candidate",
                ],
            )
            writer.writeheader()
            for word in self.words:
                writer.writerow(
                    {
                        "surah": word.surah,
                        "ayah": word.ayah,
                        "word_index_in_ayah": word.word_index_in_ayah,
                        "text": word.text_uthmani,
                        "start_s": round(word.start_s, 3),
                        "end_s": round(word.end_s, 3),
                        "confidence": None
                        if word.confidence is None
                        else round(word.confidence, 4),
                        "alignment_origin": word.alignment_origin,
                        "match_score": None
                        if word.match_score is None
                        else round(word.match_score, 2),
                        "engine_candidate": word.engine_candidate,
                    }
                )

        return ayah_csv, words_csv

    @classmethod
    def read_json(cls, path: str | Path) -> "TimingResult":
        return cls.model_validate(orjson.loads(Path(path).read_bytes()))


@dataclass(slots=True)
class QCThresholds:
    min_coverage: float = 0.985
    max_zero_or_negative_ratio: float = 0.02
    min_median_confidence: float = 0.55
    max_duration_delta_ratio: float = 0.03
    max_interpolated_ratio: float = 0.25
    min_lexical_match_ratio: float = 0.0

    @classmethod
    def strict(cls) -> "QCThresholds":
        return cls(
            min_coverage=1.0,
            max_zero_or_negative_ratio=0.0,
            min_median_confidence=0.55,
            max_duration_delta_ratio=0.04,
            max_interpolated_ratio=0.03,
            min_lexical_match_ratio=0.97,
        )


def _estimate_quantization_step_ms(words: list[WordTiming]) -> float | None:
    boundaries: list[float] = []
    for word in words:
        boundaries.append(float(word.start_s))
        boundaries.append(float(word.end_s))
    if len(boundaries) < 3:
        return None

    boundaries = sorted(set(boundaries))
    deltas = [
        boundaries[idx + 1] - boundaries[idx]
        for idx in range(len(boundaries) - 1)
        if (boundaries[idx + 1] - boundaries[idx]) > 1e-6
    ]
    if not deltas:
        return None
    return median(deltas) * 1000.0


def compute_qc(
    *,
    words: list[WordTiming],
    expected_word_count: int,
    audio_duration_s: float,
    speech_end_s: float | None = None,
    thresholds: QCThresholds,
    candidate_scores: dict[str, float] | None = None,
) -> QCReport:
    warnings: list[str] = []

    aligned_words = [w for w in words if w.end_s >= w.start_s]
    coverage = 0.0
    if expected_word_count > 0:
        coverage = len(aligned_words) / expected_word_count

    monotonic = True
    previous = -1.0
    zero_or_negative = 0
    confidences: list[float] = []
    interpolated_count = 0
    lexical_matches = 0

    for word in words:
        if word.start_s < previous:
            monotonic = False
        previous = word.start_s
        if word.end_s <= word.start_s:
            zero_or_negative += 1
        if word.confidence is not None:
            confidences.append(word.confidence)
        if word.alignment_origin in {"interpolated", "distributed"}:
            interpolated_count += 1
        if word.match_score is not None and word.match_score >= 70.0:
            lexical_matches += 1

    zero_or_negative_ratio = 0.0
    interpolated_ratio = 0.0
    lexical_match_ratio: float | None = None

    if words:
        zero_or_negative_ratio = zero_or_negative / len(words)
        interpolated_ratio = interpolated_count / len(words)
        if any(word.match_score is not None for word in words):
            lexical_match_ratio = lexical_matches / len(words)

    duration_reference_s = audio_duration_s
    if speech_end_s is not None and speech_end_s > 0:
        duration_reference_s = speech_end_s

    if words:
        end_time = max(word.end_s for word in words)
        duration_delta_ratio = (
            abs(end_time - duration_reference_s) / duration_reference_s
            if duration_reference_s > 0
            else 0.0
        )
    else:
        duration_delta_ratio = 1.0

    duration_match = duration_delta_ratio <= thresholds.max_duration_delta_ratio
    median_confidence = median(confidences) if confidences else None
    quantization_step_ms = _estimate_quantization_step_ms(words)

    if coverage < thresholds.min_coverage:
        warnings.append(
            f"Coverage {coverage:.3f} below threshold {thresholds.min_coverage:.3f}"
        )
    if not monotonic:
        warnings.append("Non-monotonic word sequence detected")
    if zero_or_negative_ratio > thresholds.max_zero_or_negative_ratio:
        warnings.append(
            "Zero/negative duration ratio "
            f"{zero_or_negative_ratio:.3f} above {thresholds.max_zero_or_negative_ratio:.3f}"
        )
    if interpolated_ratio > thresholds.max_interpolated_ratio:
        warnings.append(
            "Interpolated/distributed ratio "
            f"{interpolated_ratio:.3f} above {thresholds.max_interpolated_ratio:.3f}"
        )
    if not duration_match:
        warnings.append(
            f"Speech/audio end mismatch ratio {duration_delta_ratio:.3f} exceeds "
            f"{thresholds.max_duration_delta_ratio:.3f}"
        )
    if (
        median_confidence is not None
        and median_confidence < thresholds.min_median_confidence
    ):
        warnings.append(
            f"Median confidence {median_confidence:.3f} below {thresholds.min_median_confidence:.3f}"
        )
    if (
        lexical_match_ratio is not None
        and lexical_match_ratio < thresholds.min_lexical_match_ratio
    ):
        warnings.append(
            f"Lexical match ratio {lexical_match_ratio:.3f} below {thresholds.min_lexical_match_ratio:.3f}"
        )

    return QCReport(
        coverage=coverage,
        monotonic=monotonic,
        duration_match=duration_match,
        warnings=warnings,
        zero_or_negative_ratio=zero_or_negative_ratio,
        median_confidence=median_confidence,
        interpolated_ratio=interpolated_ratio,
        lexical_match_ratio=lexical_match_ratio,
        speech_end_delta_ratio=duration_delta_ratio,
        quantization_step_ms=quantization_step_ms,
        engine_candidate_scores=candidate_scores or {},
    )


def qc_requires_fallback(qc: QCReport, thresholds: QCThresholds) -> bool:
    if qc.coverage < thresholds.min_coverage:
        return True
    if not qc.monotonic:
        return True
    if qc.zero_or_negative_ratio > thresholds.max_zero_or_negative_ratio:
        return True
    if not qc.duration_match:
        return True
    if qc.interpolated_ratio > thresholds.max_interpolated_ratio:
        return True
    if qc.median_confidence is not None and qc.median_confidence < thresholds.min_median_confidence:
        return True
    if (
        qc.lexical_match_ratio is not None
        and qc.lexical_match_ratio < thresholds.min_lexical_match_ratio
    ):
        return True
    return False
