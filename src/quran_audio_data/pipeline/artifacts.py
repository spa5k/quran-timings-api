from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import orjson

from quran_audio_data.schema import (
    AudioMetadata,
    AyahTiming,
    EngineInfo,
    MetaInfo,
    QCThresholds,
    TimingResult,
    WordTiming,
    compute_qc,
)
from .types import ManifestRow, ProcessedFile


def validate_outputs(input_path: str | Path) -> tuple[int, int, list[str]]:
    root = Path(input_path)
    files: list[Path]
    if root.is_dir():
        files = sorted(root.glob("*.json"))
    else:
        files = [root]

    valid = 0
    invalid = 0
    errors: list[str] = []

    for file_path in files:
        if file_path.name.endswith("_qc_report.json"):
            continue
        try:
            TimingResult.read_json(file_path)
            valid += 1
        except Exception as exc:
            invalid += 1
            errors.append(f"{file_path}: {exc}")

    return valid, invalid, errors


def derive_ayahs_from_words(words: list[WordTiming], *, source: str) -> list[AyahTiming]:
    grouped: dict[tuple[int, int], list[WordTiming]] = defaultdict(list)
    for word in words:
        grouped[(word.surah, word.ayah)].append(word)

    ayahs: list[AyahTiming] = []
    for (surah, ayah), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        ayahs.append(
            AyahTiming(
                surah=surah,
                ayah=ayah,
                start_s=min(item.start_s for item in group),
                end_s=max(item.end_s for item in group),
                source=source,
            )
        )
    return ayahs


def derive_ayahs_from_words_with_engine_sources(
    *,
    words: list[WordTiming],
    source_by_ayah: dict[int, str] | None = None,
    default_source: str = "aligned",
) -> list[AyahTiming]:
    grouped: dict[tuple[int, int], list[WordTiming]] = defaultdict(list)
    for word in words:
        grouped[(word.surah, word.ayah)].append(word)

    ayahs: list[AyahTiming] = []
    for (surah, ayah), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        source = source_by_ayah.get(ayah, default_source) if source_by_ayah is not None else default_source
        ayahs.append(
            AyahTiming(
                surah=surah,
                ayah=ayah,
                start_s=min(item.start_s for item in group),
                end_s=max(item.end_s for item in group),
                source=source,
            )
        )
    return ayahs


def build_result(
    *,
    row: ManifestRow,
    audio_info: AudioMetadata,
    engine_name: str,
    engine_model: str,
    device: str,
    fallback_used: bool,
    ayahs,
    words,
    expected_word_count: int,
    speech_end_s: float | None = None,
    thresholds: QCThresholds,
    candidate_scores: dict[str, float] | None = None,
) -> TimingResult:
    input_mode = "ayah_file" if row.ayah is not None else "full_surah"
    qc = compute_qc(
        words=words,
        expected_word_count=expected_word_count,
        audio_duration_s=audio_info.duration_s,
        speech_end_s=speech_end_s,
        thresholds=thresholds,
        candidate_scores=candidate_scores,
    )

    return TimingResult(
        audio=audio_info,
        meta=MetaInfo(reciter_id=row.reciter_id, surah=row.surah, input_mode=input_mode),
        engine=EngineInfo(
            name=engine_name,
            model=engine_model,
            device=device,
            fallback_used=fallback_used,
        ),
        ayahs=ayahs,
        words=words,
        qc=qc,
    )


def write_result_artifacts(
    *,
    result: TimingResult,
    row: ManifestRow,
    out_dir: Path,
    source: str,
    fallback_used: bool = False,
    elapsed_s: float = 0.0,
) -> ProcessedFile:
    stem = output_stem(row)
    json_path = out_dir / f"{stem}.json"
    result.write_json(json_path)
    ayah_csv, words_csv = result.write_csvs(out_dir / stem)

    qc_path = out_dir / f"{stem}_qc_report.json"
    qc_path.write_bytes(orjson.dumps(result.qc.model_dump(mode="json"), option=orjson.OPT_INDENT_2))

    return ProcessedFile(
        row=row,
        output_json=json_path,
        output_ayah_csv=ayah_csv,
        output_words_csv=words_csv,
        qc_report_json=qc_path,
        source=source,
        fallback_used=fallback_used,
        elapsed_s=elapsed_s,
    )


def write_cache_result(
    *,
    row: ManifestRow,
    result: TimingResult,
    cache_root: str | Path = ".cache/timings",
) -> None:
    cache_dir = Path(cache_root) / row.reciter_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    if row.ayah is None:
        surah_path = cache_dir / f"{row.surah:03d}.json"
        surah_path.write_bytes(result.to_json_bytes())
        return

    ayah_path = cache_dir / f"{row.surah:03d}_{row.ayah:03d}.json"
    ayah_path.write_bytes(result.to_json_bytes())


def output_stem(row: ManifestRow) -> str:
    if row.ayah is None:
        return f"{row.reciter_id}_s{row.surah:03d}_full"
    return f"{row.reciter_id}_s{row.surah:03d}_a{row.ayah:03d}"

__all__ = [
    "ManifestRow",
    "ProcessedFile",
    "build_result",
    "write_result_artifacts",
    "write_cache_result",
    "output_stem",
    "derive_ayahs_from_words",
    "derive_ayahs_from_words_with_engine_sources",
    "validate_outputs",
]
