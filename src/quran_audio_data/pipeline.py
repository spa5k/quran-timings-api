from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from typing import Any, Literal

import numpy as np
import orjson
import soundfile as sf

from quran_audio_data.alignment import (
    AlignmentError,
    AlignmentOutput,
    EngineUnavailable,
    MFAAligner,
    NemoAligner,
    WhisperXFallbackAligner,
)
from quran_audio_data.schema import (
    AudioMetadata,
    AyahTiming,
    EngineInfo,
    MetaInfo,
    QCThresholds,
    TimingResult,
    WordTiming,
    compute_qc,
    qc_requires_fallback,
)
from quran_audio_data.sources import ExistingTimingResolver
from quran_audio_data.text import CanonicalWord, QuranTextStore


DeviceOption = Literal["auto", "cpu", "cuda"]
EngineOption = Literal["nemo", "whisperx", "mfa"]
AccuracyMode = Literal["standard", "strict"]


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
class ProcessingSummary:
    total: int
    succeeded: int
    failed: int
    existing_resolved: int
    aligned: int
    fallback_used: int
    elapsed_s: float
    outputs: list[ProcessedFile]
    errors: list[str]


def read_manifest(manifest_path: str | Path) -> list[ManifestRow]:
    path = Path(manifest_path)
    if not path.exists():
        raise PipelineError(f"Manifest file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    required = {"audio_path", "reciter_id", "surah", "ayah"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise PipelineError(
            "Manifest missing required columns: " + ", ".join(sorted(missing))
        )

    parsed: list[ManifestRow] = []
    for idx, row in enumerate(rows, start=2):
        audio_path = Path(str(row.get("audio_path", "")).strip())
        reciter_id = str(row.get("reciter_id", "")).strip()
        if not reciter_id:
            raise PipelineError(f"manifest row {idx}: reciter_id is required")

        surah = _parse_int(row.get("surah"), f"manifest row {idx}: invalid surah")

        raw_ayah = str(row.get("ayah", "")).strip()
        ayah = None if raw_ayah == "" else _parse_int(raw_ayah, f"manifest row {idx}: invalid ayah")

        parsed.append(
            ManifestRow(
                audio_path=audio_path,
                reciter_id=reciter_id,
                surah=surah,
                ayah=ayah,
                source_url=_none_if_blank(row.get("source_url")),
                sha256=_none_if_blank(row.get("sha256")),
                language=_none_if_blank(row.get("language")) or "ar",
                riwaya=_none_if_blank(row.get("riwaya")),
                text_variant=_none_if_blank(row.get("text_variant")),
                gold_split=_none_if_blank(row.get("gold_split")),
            )
        )

    return parsed


def run_alignment_pipeline(
    *,
    manifest_path: str | Path,
    out_dir: str | Path,
    engine: EngineOption = "nemo",
    multi_engine: list[EngineOption] | None = None,
    accuracy_mode: AccuracyMode = "standard",
    device: DeviceOption = "auto",
    text_data: str | Path | None = None,
    cache_dir: str | Path = ".cache/timings",
    enable_remote: bool = True,
    sample_size: int | None = None,
    thresholds: QCThresholds | None = None,
) -> ProcessingSummary:
    started = time.time()
    if accuracy_mode not in {"standard", "strict"}:
        raise PipelineError(f"Unsupported accuracy mode: {accuracy_mode}")
    thresholds = thresholds or (
        QCThresholds.strict() if accuracy_mode == "strict" else QCThresholds()
    )
    rows = read_manifest(manifest_path)
    if sample_size is not None:
        rows = rows[:sample_size]

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    text_store = QuranTextStore(text_data)
    resolver = ExistingTimingResolver(cache_dir=cache_dir, enable_remote=enable_remote)
    nemo = NemoAligner()
    whisperx = WhisperXFallbackAligner()
    mfa = MFAAligner()
    selected_engines = multi_engine or ["nemo", "whisperx", "mfa"]
    if not mfa.is_available():
        raise PipelineError("MFA unavailable: " + mfa.availability_error())

    outputs: list[ProcessedFile] = []
    errors: list[str] = []
    existing_resolved = 0
    aligned = 0
    fallback_used_count = 0

    for row in rows:
        try:
            processed = _process_row(
                row=row,
                out_dir=out_root,
                text_store=text_store,
                resolver=resolver,
                nemo=nemo,
                whisperx=whisperx,
                mfa=mfa,
                requested_engine=engine,
                multi_engine=selected_engines,
                accuracy_mode=accuracy_mode,
                device=device,
                thresholds=thresholds,
                cache_dir=cache_dir,
            )
            outputs.append(processed)
            if processed.source == "existing":
                existing_resolved += 1
            else:
                aligned += 1
            if processed.fallback_used:
                fallback_used_count += 1
        except Exception as exc:
            key = f"{row.reciter_id}:{row.surah}:{row.ayah or 'full'}"
            errors.append(f"{key}: {exc}")

    elapsed = time.time() - started
    return ProcessingSummary(
        total=len(rows),
        succeeded=len(outputs),
        failed=len(errors),
        existing_resolved=existing_resolved,
        aligned=aligned,
        fallback_used=fallback_used_count,
        elapsed_s=elapsed,
        outputs=outputs,
        errors=errors,
    )


def run_resolve_existing_only(
    *,
    manifest_path: str | Path,
    out_dir: str | Path,
    text_data: str | Path | None = None,
    cache_dir: str | Path = ".cache/timings",
    enable_remote: bool = True,
    sample_size: int | None = None,
) -> ProcessingSummary:
    started = time.time()
    rows = read_manifest(manifest_path)
    if sample_size is not None:
        rows = rows[:sample_size]

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    text_store = QuranTextStore(text_data)
    resolver = ExistingTimingResolver(cache_dir=cache_dir, enable_remote=enable_remote)

    outputs: list[ProcessedFile] = []
    errors: list[str] = []

    for row in rows:
        try:
            audio_info = probe_audio(row.audio_path)
            canonical_words = _load_canonical_words(text_store=text_store, row=row)
            resolved = resolver.resolve(
                reciter_id=row.reciter_id,
                surah=row.surah,
                ayah=row.ayah,
                canonical_words=canonical_words,
                audio_duration_s=audio_info.duration_s,
                source_url=row.source_url,
            )
            if resolved is None:
                raise PipelineError("existing timing not found or failed validation")

            result = _build_result(
                row=row,
                audio_info=audio_info,
                engine_name="existing",
                engine_model=resolved.source_name,
                device="n/a",
                fallback_used=False,
                ayahs=resolved.ayahs,
                words=resolved.words,
                expected_word_count=len(canonical_words),
                thresholds=QCThresholds(),
            )
            outputs.append(_write_result_artifacts(result=result, row=row, out_dir=out_root, source="existing"))
        except Exception as exc:
            key = f"{row.reciter_id}:{row.surah}:{row.ayah or 'full'}"
            errors.append(f"{key}: {exc}")

    elapsed = time.time() - started
    return ProcessingSummary(
        total=len(rows),
        succeeded=len(outputs),
        failed=len(errors),
        existing_resolved=len(outputs),
        aligned=0,
        fallback_used=0,
        elapsed_s=elapsed,
        outputs=outputs,
        errors=errors,
    )


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
        # Skip qc companion report files.
        if file_path.name.endswith("_qc_report.json"):
            continue
        try:
            TimingResult.read_json(file_path)
            valid += 1
        except Exception as exc:
            invalid += 1
            errors.append(f"{file_path}: {exc}")

    return valid, invalid, errors


def benchmark_pipeline(
    *,
    manifest_path: str | Path,
    out_dir: str | Path,
    sample_size: int,
    engine: EngineOption = "nemo",
    multi_engine: list[EngineOption] | None = None,
    accuracy_mode: AccuracyMode = "standard",
    device: DeviceOption = "auto",
    text_data: str | Path | None = None,
    cache_dir: str | Path = ".cache/timings",
    enable_remote: bool = True,
) -> dict[str, Any]:
    start = time.time()
    summary = run_alignment_pipeline(
        manifest_path=manifest_path,
        out_dir=out_dir,
        engine=engine,
        multi_engine=multi_engine,
        accuracy_mode=accuracy_mode,
        device=device,
        text_data=text_data,
        cache_dir=cache_dir,
        enable_remote=enable_remote,
        sample_size=sample_size,
    )
    elapsed = time.time() - start

    avg_runtime = (
        sum(item.elapsed_s for item in summary.outputs) / len(summary.outputs)
        if summary.outputs
        else 0.0
    )

    return {
        "sample_size": sample_size,
        "total": summary.total,
        "succeeded": summary.succeeded,
        "failed": summary.failed,
        "existing_resolved": summary.existing_resolved,
        "aligned": summary.aligned,
        "fallback_used": summary.fallback_used,
        "elapsed_s": elapsed,
        "avg_file_runtime_s": avg_runtime,
        "errors": summary.errors,
    }


def _process_row(
    *,
    row: ManifestRow,
    out_dir: Path,
    text_store: QuranTextStore,
    resolver: ExistingTimingResolver,
    nemo: NemoAligner,
    whisperx: WhisperXFallbackAligner,
    mfa: MFAAligner,
    requested_engine: EngineOption,
    multi_engine: list[EngineOption],
    accuracy_mode: AccuracyMode,
    device: DeviceOption,
    thresholds: QCThresholds,
    cache_dir: str | Path,
) -> ProcessedFile:
    started = time.time()

    audio_info = probe_audio(row.audio_path)
    canonical_words = _load_canonical_words(text_store=text_store, row=row)

    resolved = resolver.resolve(
        reciter_id=row.reciter_id,
        surah=row.surah,
        ayah=row.ayah,
        canonical_words=canonical_words,
        audio_duration_s=audio_info.duration_s,
        source_url=row.source_url,
    )

    if resolved is not None:
        result = _build_result(
            row=row,
            audio_info=audio_info,
            engine_name="existing",
            engine_model=resolved.source_name,
            device="n/a",
            fallback_used=False,
            ayahs=resolved.ayahs,
            words=resolved.words,
            expected_word_count=len(canonical_words),
            thresholds=thresholds,
        )
        if accuracy_mode != "strict" or not qc_requires_fallback(result.qc, thresholds):
            return _write_result_artifacts(
                result=result,
                row=row,
                out_dir=out_dir,
                source="existing",
                fallback_used=False,
                elapsed_s=time.time() - started,
            )

    wav_path = ensure_wav_16k_mono(row.audio_path)
    speech_end_s = _estimate_speech_end_s(wav_path, fallback_duration_s=audio_info.duration_s)
    engines_to_try = _normalize_engines(
        requested_engine=requested_engine,
        multi_engine=multi_engine,
        accuracy_mode=accuracy_mode,
    )

    candidate_results: list[TimingResult] = []
    candidate_failures: list[str] = []
    fallback_used = False

    for engine_name in engines_to_try:
        try:
            output, refinement_warnings = _align_with_engine(
                engine_name=engine_name,
                row=row,
                wav_path=wav_path,
                canonical_words=canonical_words,
                nemo=nemo,
                whisperx=whisperx,
                mfa=mfa,
                audio_duration_s=audio_info.duration_s,
                device=device,
            )
            result = _build_result(
                row=row,
                audio_info=audio_info,
                engine_name=output.engine_name,
                engine_model=output.engine_model,
                device=output.device,
                fallback_used=False,
                ayahs=output.ayahs,
                words=output.words,
                expected_word_count=len(canonical_words),
                speech_end_s=speech_end_s,
                thresholds=thresholds,
            )
            if refinement_warnings:
                result.qc.warnings.extend(refinement_warnings)
            candidate_results.append(result)
        except (AlignmentError, EngineUnavailable, PipelineError) as exc:
            candidate_failures.append(f"{engine_name}: {exc}")

    if not candidate_results:
        if requested_engine == "nemo":
            try:
                fallback_output, _ = _align_with_engine(
                    engine_name="whisperx",
                    row=row,
                    wav_path=wav_path,
                    canonical_words=canonical_words,
                    nemo=nemo,
                    whisperx=whisperx,
                    mfa=mfa,
                    audio_duration_s=audio_info.duration_s,
                    device=device,
                )
                fallback_result = _build_result(
                    row=row,
                    audio_info=audio_info,
                    engine_name=fallback_output.engine_name,
                    engine_model=fallback_output.engine_model,
                    device=fallback_output.device,
                    fallback_used=True,
                    ayahs=fallback_output.ayahs,
                    words=fallback_output.words,
                    expected_word_count=len(canonical_words),
                    speech_end_s=speech_end_s,
                    thresholds=thresholds,
                )
                candidate_results.append(fallback_result)
            except (AlignmentError, EngineUnavailable, PipelineError) as exc:
                candidate_failures.append(f"whisperx: {exc}")

    if not candidate_results:
        raise PipelineError("no_alignment_candidates_succeeded: " + "; ".join(candidate_failures))

    candidate_scores = {
        candidate.engine.name: _score_timing_result(candidate)
        for candidate in candidate_results
    }

    if accuracy_mode == "strict" or len(candidate_results) > 1:
        result = _select_best_result_per_ayah(
            row=row,
            audio_info=audio_info,
            canonical_words=canonical_words,
            candidates=candidate_results,
            thresholds=thresholds,
            candidate_scores=candidate_scores,
        )
    else:
        result = max(candidate_results, key=_score_timing_result)
        result.qc.engine_candidate_scores = candidate_scores

    if result.engine.name != requested_engine:
        fallback_used = True

    selected_before_refinement = result
    if accuracy_mode == "strict":
        refined_words = _refine_word_boundaries(
            words=result.words,
            wav_path=wav_path,
            max_shift_s=0.12,
            min_duration_s=0.02,
        )
        if refined_words:
            source_by_ayah: dict[int, str] = {}
            for ayah, ayah_words in _words_by_ayah(refined_words).items():
                if any(word.engine_candidate == "whisperx" for word in ayah_words):
                    source_by_ayah[ayah] = "fallback"
                else:
                    source_by_ayah[ayah] = "aligned"
            refined_ayahs = _derive_ayahs_from_words_with_engine_sources(
                words=refined_words,
                source_by_ayah=source_by_ayah,
                default_source="aligned",
            )
            refined_result = _build_result(
                row=row,
                audio_info=audio_info,
                engine_name=result.engine.name,
                engine_model=result.engine.model,
                device=result.engine.device,
                fallback_used=fallback_used,
                ayahs=refined_ayahs,
                words=refined_words,
                expected_word_count=len(canonical_words),
                speech_end_s=speech_end_s,
                thresholds=thresholds,
                candidate_scores=candidate_scores,
            )
            if _should_accept_refinement(
                original=selected_before_refinement,
                refined=refined_result,
                thresholds=thresholds,
            ):
                result = refined_result
                result.qc.warnings.append("boundary_refinement_applied")
            else:
                result = selected_before_refinement
                result.qc.warnings.append("boundary_refinement_rejected")

    if qc_requires_fallback(result.qc, thresholds):
        if accuracy_mode == "strict":
            rescue_candidates = [
                result,
                selected_before_refinement,
                *candidate_results,
            ]
            rescue = _select_strict_rescue_candidate(
                candidates=rescue_candidates,
                thresholds=thresholds,
            )
            if rescue is not None:
                result = rescue
                result.qc.engine_candidate_scores = candidate_scores
                result.qc.warnings.append("strict_rescue_selected")
                fallback_used = result.engine.name != requested_engine
            else:
                raise PipelineError("strict_qc_failed: " + "; ".join(result.qc.warnings))
        if requested_engine != "whisperx" and "whisperx" not in engines_to_try:
            try:
                fallback_output, _ = _align_with_engine(
                    engine_name="whisperx",
                    row=row,
                    wav_path=wav_path,
                    canonical_words=canonical_words,
                    nemo=nemo,
                    whisperx=whisperx,
                    mfa=mfa,
                    audio_duration_s=audio_info.duration_s,
                    device=device,
                )
                fallback_used = True
                result = _build_result(
                    row=row,
                    audio_info=audio_info,
                    engine_name=fallback_output.engine_name,
                    engine_model=fallback_output.engine_model,
                    device=fallback_output.device,
                    fallback_used=True,
                    ayahs=fallback_output.ayahs,
                    words=fallback_output.words,
                    expected_word_count=len(canonical_words),
                    speech_end_s=speech_end_s,
                    thresholds=thresholds,
                    candidate_scores=candidate_scores,
                )
            except (AlignmentError, EngineUnavailable) as exc:
                result.qc.warnings.append(f"fallback_unavailable: {exc}")

    _write_cache_result(row=row, result=result, cache_root=cache_dir)

    return _write_result_artifacts(
        result=result,
        row=row,
        out_dir=out_dir,
        source=result.ayahs[0].source if result.ayahs else "aligned",
        fallback_used=fallback_used,
        elapsed_s=time.time() - started,
    )


def _normalize_engines(
    *,
    requested_engine: EngineOption,
    multi_engine: list[EngineOption],
    accuracy_mode: AccuracyMode,
) -> list[EngineOption]:
    ordered: list[EngineOption] = []
    for item in multi_engine:
        if item not in {"nemo", "whisperx", "mfa"}:
            continue
        if item not in ordered:
            ordered.append(item)
    for required in ("nemo", "whisperx", "mfa"):
        if required not in ordered:
            ordered.append(required)
    if requested_engine not in ordered:
        ordered.insert(0, requested_engine)

    # Standard and strict both evaluate the full engine set now.
    return ordered


def _align_with_engine(
    *,
    engine_name: EngineOption,
    row: ManifestRow,
    wav_path: Path,
    canonical_words: list[CanonicalWord],
    nemo: NemoAligner,
    whisperx: WhisperXFallbackAligner,
    mfa: MFAAligner,
    audio_duration_s: float,
    device: DeviceOption,
) -> tuple[AlignmentOutput, list[str]]:
    if engine_name == "nemo":
        if row.ayah is None:
            return _align_full_surah_quality(
                wav_path=wav_path,
                canonical_words=canonical_words,
                nemo=nemo,
                audio_duration_s=audio_duration_s,
                device=device,
            )
        return (
            nemo.align(
                audio_wav_path=str(wav_path),
                canonical_words=canonical_words,
                audio_duration_s=audio_duration_s,
                device=device,
            ),
            [],
        )

    if engine_name == "whisperx":
        return (
            whisperx.align(
                audio_wav_path=str(wav_path),
                canonical_words=canonical_words,
                audio_duration_s=audio_duration_s,
                device=device,
            ),
            [],
        )

    return (
        mfa.align(
            audio_wav_path=str(wav_path),
            canonical_words=canonical_words,
            audio_duration_s=audio_duration_s,
            device=device,
        ),
        [],
    )


def _score_words_slice(words: list[WordTiming], expected_count: int) -> float:
    if not words:
        return -10_000.0

    starts = [word.start_s for word in words]
    monotonic = starts == sorted(starts)
    non_positive = sum(1 for word in words if word.end_s <= word.start_s)
    interpolated = sum(
        1 for word in words if word.alignment_origin in {"interpolated", "distributed"}
    )
    interpolated_ratio = interpolated / max(1, len(words))
    count_ratio = min(len(words), expected_count) / max(1, expected_count)

    lexical_scores = [word.match_score for word in words if word.match_score is not None]
    lexical_component = (
        (sum(lexical_scores) / len(lexical_scores)) / 100.0 if lexical_scores else 0.0
    )
    confidence_scores = [word.confidence for word in words if word.confidence is not None]
    confidence_component = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

    score = 100.0
    score += count_ratio * 25.0
    score += lexical_component * 35.0
    score += confidence_component * 10.0
    score -= interpolated_ratio * 80.0
    score -= non_positive * 25.0
    if not monotonic:
        score -= 100.0
    return score


def _score_timing_result(result: TimingResult) -> float:
    qc = result.qc
    score = 100.0
    score += qc.coverage * 45.0
    score -= qc.interpolated_ratio * 80.0
    score -= qc.speech_end_delta_ratio * 50.0
    if not qc.monotonic:
        score -= 150.0
    if not qc.duration_match:
        score -= 100.0
    if qc.zero_or_negative_ratio > 0:
        score -= qc.zero_or_negative_ratio * 120.0
    if qc.median_confidence is not None:
        score += qc.median_confidence * 10.0
    if qc.lexical_match_ratio is not None:
        score += qc.lexical_match_ratio * 35.0
    if qc.quantization_step_ms is not None:
        score -= min(qc.quantization_step_ms, 300.0) * 0.05
    score -= len(qc.warnings) * 2.5
    return score


def _qc_violation_count(qc_report, thresholds: QCThresholds) -> int:
    violations = 0
    if qc_report.coverage < thresholds.min_coverage:
        violations += 1
    if not qc_report.monotonic:
        violations += 1
    if qc_report.zero_or_negative_ratio > thresholds.max_zero_or_negative_ratio:
        violations += 1
    if not qc_report.duration_match:
        violations += 1
    if qc_report.interpolated_ratio > thresholds.max_interpolated_ratio:
        violations += 1
    if (
        qc_report.lexical_match_ratio is not None
        and qc_report.lexical_match_ratio < thresholds.min_lexical_match_ratio
    ):
        violations += 1
    if (
        qc_report.median_confidence is not None
        and qc_report.median_confidence < thresholds.min_median_confidence
    ):
        violations += 1
    return violations


def _should_accept_refinement(
    *,
    original: TimingResult,
    refined: TimingResult,
    thresholds: QCThresholds,
) -> bool:
    original_violations = _qc_violation_count(original.qc, thresholds)
    refined_violations = _qc_violation_count(refined.qc, thresholds)
    if refined_violations < original_violations:
        return True
    if refined_violations > original_violations:
        return False
    return _score_timing_result(refined) >= _score_timing_result(original)


def _select_strict_rescue_candidate(
    *,
    candidates: list[TimingResult],
    thresholds: QCThresholds,
) -> TimingResult | None:
    unique: dict[str, TimingResult] = {}
    for candidate in candidates:
        key = f"{candidate.engine.name}:{candidate.engine.model}:{len(candidate.words)}"
        best = unique.get(key)
        if best is None or _score_timing_result(candidate) > _score_timing_result(best):
            unique[key] = candidate

    ordered = sorted(unique.values(), key=_score_timing_result, reverse=True)
    for candidate in ordered:
        if not qc_requires_fallback(candidate.qc, thresholds):
            return candidate
    return None


def _select_best_result_per_ayah(
    *,
    row: ManifestRow,
    audio_info: AudioMetadata,
    canonical_words: list[CanonicalWord],
    candidates: list[TimingResult],
    thresholds: QCThresholds,
    candidate_scores: dict[str, float],
) -> TimingResult:
    expected_by_ayah: dict[int, int] = defaultdict(int)
    for word in canonical_words:
        expected_by_ayah[word.ayah] += 1

    candidate_words: dict[str, dict[int, list[WordTiming]]] = {}
    for candidate in candidates:
        candidate_words[candidate.engine.name] = _words_by_ayah(candidate.words)

    selected_words: list[WordTiming] = []
    selected_sources: dict[int, str] = {}

    for ayah in sorted(expected_by_ayah):
        expected_count = expected_by_ayah[ayah]
        best_engine: str | None = None
        best_slice: list[WordTiming] = []
        best_score = -10_000.0

        for engine_name, words_by_ayah in candidate_words.items():
            group = words_by_ayah.get(ayah, [])
            score = _score_words_slice(group, expected_count)
            if score > best_score:
                best_score = score
                best_engine = engine_name
                best_slice = group

        if best_engine is None or not best_slice:
            raise PipelineError(f"unable to choose candidate for ayah {ayah}")

        selected_sources[ayah] = "fallback" if best_engine == "whisperx" else "aligned"
        selected_words.extend(
            [
                word.model_copy(update={"engine_candidate": best_engine})
                for word in best_slice
            ]
        )

    selected_words.sort(key=lambda word: word.word_index_global)
    ayahs = _derive_ayahs_from_words_with_engine_sources(
        words=selected_words,
        source_by_ayah=selected_sources,
        default_source="aligned",
    )

    return _build_result(
        row=row,
        audio_info=audio_info,
        engine_name="ensemble",
        engine_model="ayah-wise",
        device=candidates[0].engine.device,
        fallback_used=False,
        ayahs=ayahs,
        words=selected_words,
        expected_word_count=len(canonical_words),
        thresholds=thresholds,
        candidate_scores=candidate_scores,
    )


def _refine_word_boundaries(
    *,
    words: list[WordTiming],
    wav_path: Path,
    max_shift_s: float,
    min_duration_s: float,
) -> list[WordTiming]:
    if not words:
        return words

    try:
        audio, sample_rate = sf.read(str(wav_path))
    except Exception:
        return words

    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)
    if not isinstance(audio, np.ndarray):
        return words
    if audio.size <= 1 or sample_rate <= 0:
        return words

    envelope = np.abs(audio.astype(np.float32, copy=False))
    if float(np.max(envelope)) < 1e-6:
        return words
    window = max(1, int(sample_rate * 0.005))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.convolve(envelope, kernel, mode="same")

    max_shift = max(1, int(sample_rate * max_shift_s))

    def snap(ts: float) -> float:
        center = int(max(0, min(len(smoothed) - 1, round(ts * sample_rate))))
        left = max(0, center - max_shift)
        right = min(len(smoothed), center + max_shift + 1)
        region = smoothed[left:right]
        if region.size == 0:
            return ts
        offset = int(np.argmin(region))
        return (left + offset) / sample_rate

    refined: list[WordTiming] = []
    min_duration = max(0.001, min_duration_s)
    previous_start = 0.0
    previous_end = 0.0
    for word in words:
        start_s = snap(word.start_s)
        end_s = snap(word.end_s)
        start_s = max(previous_start, start_s)
        if end_s < start_s + min_duration:
            end_s = start_s + min_duration
        if start_s < previous_end:
            start_s = previous_end
            if end_s < start_s + min_duration:
                end_s = start_s + min_duration
        previous_start = start_s
        previous_end = end_s
        refined.append(
            word.model_copy(update={"start_s": start_s, "end_s": end_s})
        )

    return refined


def _estimate_speech_end_s(
    wav_path: Path,
    *,
    fallback_duration_s: float,
) -> float:
    try:
        audio, sample_rate = sf.read(str(wav_path))
    except Exception:
        return fallback_duration_s

    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)
    if not isinstance(audio, np.ndarray):
        return fallback_duration_s
    if audio.size <= 1 or sample_rate <= 0:
        return fallback_duration_s

    envelope = np.abs(audio.astype(np.float32, copy=False))
    peak = float(np.max(envelope))
    if peak <= 1e-6:
        return fallback_duration_s

    window = max(1, int(sample_rate * 0.02))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.convolve(envelope, kernel, mode="same")

    percentile = float(np.percentile(smoothed, 70))
    threshold = max(peak * 0.035, percentile * 0.5, 1e-4)
    active = np.where(smoothed >= threshold)[0]
    if active.size == 0:
        return fallback_duration_s

    end_index = int(active[-1])
    speech_end_s = end_index / float(sample_rate)
    if speech_end_s <= 0:
        return fallback_duration_s
    return min(max(speech_end_s, 0.01), fallback_duration_s)


def _align_full_surah_quality(
    *,
    wav_path: Path,
    canonical_words: list[CanonicalWord],
    nemo: NemoAligner,
    audio_duration_s: float,
    device: DeviceOption,
    max_passes: int = 3,
    base_overlap_s: float = 0.9,
) -> tuple[AlignmentOutput, list[str]]:
    """High-accuracy mode for full-surah files.

    Strategy:
    1. Full-surah forced alignment pass.
    2. Detect weak ayahs (collapsed/invalid timings).
    3. Re-align weak ayahs on overlapped chunks, iteratively.
    """

    warnings: list[str] = []

    primary = nemo.align(
        audio_wav_path=str(wav_path),
        canonical_words=canonical_words,
        audio_duration_s=audio_duration_s,
        device=device,
    )
    refined_words = sorted(primary.words, key=lambda word: word.word_index_global)

    expected_by_ayah: dict[int, list[CanonicalWord]] = defaultdict(list)
    for canon in canonical_words:
        expected_by_ayah[canon.ayah].append(canon)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        warnings.append("quality_refine_skipped: ffmpeg_unavailable")
        ayahs = _derive_ayahs_from_words(refined_words, source="aligned")
        return (
            AlignmentOutput(
                ayahs=ayahs,
                words=refined_words,
                engine_name="nemo",
                engine_model=nemo.model_name,
                device=primary.device,
                source="aligned",
            ),
            warnings,
        )

    with tempfile.TemporaryDirectory(prefix="qad_refine_") as temp_dir:
        temp_root = Path(temp_dir)
        for pass_index in range(max_passes):
            weak_ayahs = _find_weak_ayahs(
                words=refined_words,
                expected_by_ayah=expected_by_ayah,
            )
            if not weak_ayahs:
                break

            overlap_s = base_overlap_s * (pass_index + 1)
            warnings.append(
                f"quality_refine_pass_{pass_index + 1}: weak_ayahs={','.join(str(v) for v in weak_ayahs)} overlap_s={overlap_s:.2f}"
            )

            any_updated = False
            current_by_ayah = _words_by_ayah(refined_words)

            for ayah in weak_ayahs:
                expected_words = expected_by_ayah.get(ayah, [])
                if not expected_words:
                    continue

                current_words = current_by_ayah.get(ayah, [])
                if current_words:
                    base_start = min(word.start_s for word in current_words)
                    base_end = max(word.end_s for word in current_words)
                else:
                    ratio_start, ratio_end = _estimate_ayah_ratio_window(
                        ayah=ayah,
                        expected_by_ayah=expected_by_ayah,
                    )
                    base_start = ratio_start * audio_duration_s
                    base_end = ratio_end * audio_duration_s

                # Ensure we always provide a usable chunk.
                if base_end <= base_start:
                    base_end = min(audio_duration_s, base_start + 2.0)

                chunk_start = max(0.0, base_start - overlap_s)
                chunk_end = min(audio_duration_s, base_end + overlap_s)
                if chunk_end <= chunk_start:
                    continue

                chunk_path = temp_root / f"ayah_{ayah:03d}_pass_{pass_index + 1}.wav"
                _cut_audio_chunk(
                    ffmpeg=ffmpeg,
                    src_wav=wav_path,
                    dst_wav=chunk_path,
                    start_s=chunk_start,
                    end_s=chunk_end,
                )

                chunk_output = nemo.align(
                    audio_wav_path=str(chunk_path),
                    canonical_words=expected_words,
                    audio_duration_s=(chunk_end - chunk_start),
                    device=device,
                )

                shifted = [
                    word.model_copy(
                        update={
                            "start_s": word.start_s + chunk_start,
                            "end_s": word.end_s + chunk_start,
                        }
                    )
                    for word in chunk_output.words
                ]

                refined_words = [word for word in refined_words if word.ayah != ayah]
                refined_words.extend(shifted)
                refined_words.sort(key=lambda word: word.word_index_global)
                any_updated = True

            if not any_updated:
                break

    ayahs = _derive_ayahs_from_words(refined_words, source="aligned")
    return (
        AlignmentOutput(
            ayahs=ayahs,
            words=refined_words,
            engine_name="nemo",
            engine_model=nemo.model_name,
            device=primary.device,
            source="aligned",
        ),
        warnings,
    )


def _find_weak_ayahs(
    *,
    words: list[WordTiming],
    expected_by_ayah: dict[int, list[CanonicalWord]],
) -> list[int]:
    by_ayah = _words_by_ayah(words)
    weak: list[tuple[float, int]] = []

    for ayah in sorted(expected_by_ayah):
        expected_count = len(expected_by_ayah[ayah])
        actual_words = by_ayah.get(ayah, [])
        severity = 0.0
        is_weak = False

        if len(actual_words) != expected_count:
            severity += 100.0 + abs(expected_count - len(actual_words)) * 5.0
            is_weak = True

        if not actual_words:
            severity += 150.0
            is_weak = True
            weak.append((severity, ayah))
            continue

        starts = [word.start_s for word in actual_words]
        if starts != sorted(starts):
            severity += 80.0
            is_weak = True

        non_positive = sum(1 for word in actual_words if word.end_s <= word.start_s)
        if non_positive > 0:
            severity += (non_positive / len(actual_words)) * 120.0
            is_weak = True

        tiny = sum(1 for word in actual_words if (word.end_s - word.start_s) <= 0.03)
        if tiny / len(actual_words) > 0.12:
            severity += (tiny / len(actual_words)) * 60.0
            is_weak = True

        if is_weak:
            weak.append((severity, ayah))

    weak.sort(key=lambda item: (-item[0], item[1]))
    return [ayah for _, ayah in weak]


def _words_by_ayah(words: list[WordTiming]) -> dict[int, list[WordTiming]]:
    out: dict[int, list[WordTiming]] = defaultdict(list)
    for word in words:
        out[word.ayah].append(word)
    for ayah in out:
        out[ayah].sort(key=lambda word: word.word_index_in_ayah)
    return out


def _estimate_ayah_ratio_window(
    *,
    ayah: int,
    expected_by_ayah: dict[int, list[CanonicalWord]],
) -> tuple[float, float]:
    total = sum(len(group) for group in expected_by_ayah.values())
    if total <= 0:
        return 0.0, 1.0

    before = sum(len(expected_by_ayah[key]) for key in sorted(expected_by_ayah) if key < ayah)
    current = len(expected_by_ayah.get(ayah, []))
    start_ratio = before / total
    end_ratio = (before + current) / total
    return start_ratio, end_ratio


def _cut_audio_chunk(
    *,
    ffmpeg: str,
    src_wav: Path,
    dst_wav: Path,
    start_s: float,
    end_s: float,
) -> None:
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-to",
        f"{end_s:.3f}",
        "-i",
        str(src_wav),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(dst_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PipelineError(f"ffmpeg chunk extraction failed: {proc.stderr.strip()}")


def _derive_ayahs_from_words(words: list[WordTiming], *, source: str) -> list[AyahTiming]:
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


def _derive_ayahs_from_words_with_engine_sources(
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
        if source_by_ayah is not None:
            source = source_by_ayah.get(ayah, default_source)
        else:
            source = default_source
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


def _load_canonical_words(*, text_store: QuranTextStore, row: ManifestRow) -> list[CanonicalWord]:
    words = text_store.build_words(
        surah=row.surah,
        ayah=row.ayah,
        text_variant=row.text_variant,
        riwaya=row.riwaya,
    )
    if not words:
        raise PipelineError(f"No canonical words found for surah={row.surah} ayah={row.ayah}")
    return words


def _build_result(
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


def _write_result_artifacts(
    *,
    result: TimingResult,
    row: ManifestRow,
    out_dir: Path,
    source: str,
    fallback_used: bool = False,
    elapsed_s: float = 0.0,
) -> ProcessedFile:
    stem = _output_stem(row)
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


def _write_cache_result(
    *,
    row: ManifestRow,
    result: TimingResult,
    cache_root: str | Path = ".cache/timings",
) -> None:
    cache_dir = Path(cache_root) / row.reciter_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    path = cache_dir / f"{row.surah:03d}.json"
    path.write_bytes(result.to_json_bytes())

    if row.ayah is not None:
        ayah_path = cache_dir / f"{row.surah:03d}_{row.ayah:03d}.json"
        ayah_path.write_bytes(result.to_json_bytes())


def _output_stem(row: ManifestRow) -> str:
    if row.ayah is None:
        return f"{row.reciter_id}_s{row.surah:03d}_full"
    return f"{row.reciter_id}_s{row.surah:03d}_a{row.ayah:03d}"


def _none_if_blank(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _parse_int(value: Any, message: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise PipelineError(message) from exc


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_audio(path: str | Path) -> AudioMetadata:
    file_path = Path(path)
    if not file_path.exists():
        raise PipelineError(f"Audio file not found: {file_path}")

    try:
        info = sf.info(str(file_path))
        return AudioMetadata(
            path=str(file_path),
            duration_s=float(info.duration),
            sample_rate=int(info.samplerate),
            channels=int(info.channels),
        )
    except RuntimeError:
        pass

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise PipelineError(
            f"Unable to probe {file_path}. Install ffprobe or use wav/flac readable by soundfile."
        )

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(file_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PipelineError(f"ffprobe failed for {file_path}: {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    streams = payload.get("streams", [])
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    format_obj = payload.get("format", {})

    duration_s = float(format_obj.get("duration") or audio_stream.get("duration") or 0.0)
    sample_rate = int(audio_stream.get("sample_rate") or 16000)
    channels = int(audio_stream.get("channels") or 1)

    return AudioMetadata(
        path=str(file_path),
        duration_s=duration_s,
        sample_rate=sample_rate,
        channels=channels,
    )


def ensure_wav_16k_mono(path: str | Path) -> Path:
    src = Path(path)

    try:
        info = sf.info(str(src))
        if src.suffix.lower() == ".wav" and info.samplerate == 16000 and info.channels == 1:
            return src
    except RuntimeError:
        pass

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise PipelineError(
            "ffmpeg is required for converting input audio to mono WAV 16k."
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="qad_audio_"))
    dst = tmp_dir / f"{src.stem}_16k_mono.wav"

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PipelineError(f"ffmpeg conversion failed: {proc.stderr.strip()}")

    return dst
