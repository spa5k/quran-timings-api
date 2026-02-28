from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, cast

from quran_audio_data.alignment import (
    AlignmentError,
    AlignmentOutput,
    EngineUnavailable,
    MFAAligner,
    NemoAligner,
    WhisperXFallbackAligner,
)
from quran_audio_data.core.settings import get_settings
from quran_audio_data.schema import QCThresholds, TimingResult, WordTiming, qc_requires_fallback
from quran_audio_data.sources import ExistingTimingResolver
from quran_audio_data.text import CanonicalWord, QuranTextStore

from .artifacts import (
    build_result,
    derive_ayahs_from_words,
    derive_ayahs_from_words_with_engine_sources,
    write_cache_result,
    write_result_artifacts,
)
from .audio import (
    cut_audio_chunk,
    ensure_wav_16k_mono,
    estimate_speech_end_s,
    probe_audio,
    refine_word_boundaries,
)
from .engine_registry import EngineRegistry
from .manifest import read_manifest
from .scoring import (
    score_timing_result,
    select_best_result_per_ayah,
    select_strict_rescue_candidate,
    should_accept_refinement,
    words_by_ayah,
)
from .types import (
    AccuracyMode,
    DeviceOption,
    EngineAvailabilityPolicy,
    EngineOption,
    ManifestRow,
    PipelineError,
    PipelineErrorDetail,
    PipelineReportV2,
    default_cache_dir,
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
        required_engine = cast(EngineOption, required)
        if required_engine not in ordered:
            ordered.append(required_engine)
    if requested_engine not in ordered:
        ordered.insert(0, requested_engine)
    return ordered


def run_alignment_pipeline(
    *,
    manifest_path: str | Path,
    out_dir: str | Path,
    engine: EngineOption = "nemo",
    multi_engine: list[EngineOption] | None = None,
    accuracy_mode: AccuracyMode = "standard",
    device: DeviceOption = "auto",
    text_data: str | Path | None = None,
    cache_dir: str | Path | None = None,
    enable_remote: bool = True,
    sample_size: int | None = None,
    thresholds: QCThresholds | None = None,
    availability_policy: EngineAvailabilityPolicy = "best_effort",
    registry: EngineRegistry | None = None,
    refine_word_boundaries_fn=refine_word_boundaries,
) -> PipelineReportV2:
    started = time.time()
    if accuracy_mode not in {"standard", "strict"}:
        raise PipelineError(f"Unsupported accuracy mode: {accuracy_mode}")

    settings = get_settings()
    cache_root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    thresholds = thresholds or (
        QCThresholds.strict() if accuracy_mode == "strict" else QCThresholds()
    )

    rows = read_manifest(manifest_path)
    if sample_size is not None:
        rows = rows[:sample_size]

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    text_store = QuranTextStore(text_data)
    resolver = ExistingTimingResolver(cache_dir=cache_root, enable_remote=enable_remote)
    active_registry = registry or EngineRegistry(
        nemo=NemoAligner(),
        whisperx=WhisperXFallbackAligner(),
        mfa=MFAAligner(),
    )

    configured_engines = [
        cast(EngineOption, name)
        for name in settings.engine_order
        if name in {"nemo", "whisperx", "mfa"}
    ]
    requested_engines = multi_engine or configured_engines
    if not requested_engines:
        requested_engines = [cast(EngineOption, "nemo"), cast(EngineOption, "whisperx"), cast(EngineOption, "mfa")]

    selection = active_registry.select(
        requested_engine=engine,
        multi_engine=requested_engines,
        policy=availability_policy,
    )

    outputs = []
    errors: list[str] = []
    error_details: list[PipelineErrorDetail] = []
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
                nemo=active_registry.get("nemo"),
                whisperx=active_registry.get("whisperx"),
                mfa=active_registry.get("mfa"),
                requested_engine=selection.requested_engine,
                multi_engine=selection.engines_to_try,
                accuracy_mode=accuracy_mode,
                device=device,
                thresholds=thresholds,
                cache_dir=cache_root,
                refine_word_boundaries_fn=refine_word_boundaries_fn,
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
            message = f"{key}: {exc}"
            errors.append(message)
            error_details.append(
                PipelineErrorDetail(
                    key=key,
                    message=str(exc),
                    attempted_engines=list(selection.engines_to_try),
                )
            )

    elapsed = time.time() - started
    return PipelineReportV2(
        total=len(rows),
        succeeded=len(outputs),
        failed=len(errors),
        existing_resolved=existing_resolved,
        aligned=aligned,
        fallback_used=fallback_used_count,
        elapsed_s=elapsed,
        outputs=outputs,
        errors=errors,
        error_details=error_details,
        attempted_engines=list(selection.engines_to_try),
    )


def run_resolve_existing_only(
    *,
    manifest_path: str | Path,
    out_dir: str | Path,
    text_data: str | Path | None = None,
    cache_dir: str | Path | None = None,
    enable_remote: bool = True,
    sample_size: int | None = None,
) -> PipelineReportV2:
    started = time.time()
    rows = read_manifest(manifest_path)
    if sample_size is not None:
        rows = rows[:sample_size]

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    text_store = QuranTextStore(text_data)
    resolver = ExistingTimingResolver(
        cache_dir=cache_dir if cache_dir is not None else default_cache_dir(),
        enable_remote=enable_remote,
    )

    outputs = []
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

            result = build_result(
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
            outputs.append(write_result_artifacts(result=result, row=row, out_dir=out_root, source="existing"))
        except Exception as exc:
            key = f"{row.reciter_id}:{row.surah}:{row.ayah or 'full'}"
            errors.append(f"{key}: {exc}")

    elapsed = time.time() - started
    return PipelineReportV2(
        total=len(rows),
        succeeded=len(outputs),
        failed=len(errors),
        existing_resolved=len(outputs),
        aligned=0,
        fallback_used=0,
        elapsed_s=elapsed,
        outputs=outputs,
        errors=errors,
        error_details=[
            PipelineErrorDetail(
                key=message.split(": ", 1)[0],
                message=message.split(": ", 1)[1] if ": " in message else message,
                attempted_engines=[],
            )
            for message in errors
        ],
        attempted_engines=[],
    )


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
    cache_dir: str | Path | None = None,
    enable_remote: bool = True,
    availability_policy: EngineAvailabilityPolicy = "best_effort",
) -> dict[str, Any]:
    started = time.time()
    summary = run_alignment_pipeline(
        manifest_path=manifest_path,
        out_dir=out_dir,
        sample_size=sample_size,
        engine=engine,
        multi_engine=multi_engine,
        accuracy_mode=accuracy_mode,
        device=device,
        text_data=text_data,
        cache_dir=cache_dir,
        enable_remote=enable_remote,
        availability_policy=availability_policy,
    )
    elapsed = time.time() - started
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
        "attempted_engines": summary.attempted_engines,
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
    refine_word_boundaries_fn=refine_word_boundaries,
):
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
        result = build_result(
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
            return write_result_artifacts(
                result=result,
                row=row,
                out_dir=out_dir,
                source="existing",
                fallback_used=False,
                elapsed_s=time.time() - started,
            )

    wav_path = ensure_wav_16k_mono(row.audio_path)
    speech_end_s, speech_end_method = estimate_speech_end_s(
        wav_path,
        fallback_duration_s=audio_info.duration_s,
    )
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
            result = build_result(
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
            result.qc.speech_end_method = speech_end_method
            candidate_results.append(result)
        except (AlignmentError, EngineUnavailable, PipelineError) as exc:
            candidate_failures.append(f"{engine_name}: {exc}")

    if not candidate_results and requested_engine == "nemo":
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
            fallback_result = build_result(
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
            fallback_result.qc.speech_end_method = speech_end_method
            candidate_results.append(fallback_result)
        except (AlignmentError, EngineUnavailable, PipelineError) as exc:
            candidate_failures.append(f"whisperx: {exc}")

    if not candidate_results:
        raise PipelineError("no_alignment_candidates_succeeded: " + "; ".join(candidate_failures))

    candidate_scores = {
        candidate.engine.name: score_timing_result(candidate)
        for candidate in candidate_results
    }

    if accuracy_mode == "strict" or len(candidate_results) > 1:
        result = select_best_result_per_ayah(
            row=row,
            audio_info=audio_info,
            canonical_words=canonical_words,
            candidates=candidate_results,
            thresholds=thresholds,
            candidate_scores=candidate_scores,
        )
    else:
        result = max(candidate_results, key=score_timing_result)
        result.qc.engine_candidate_scores = candidate_scores

    if result.engine.name != requested_engine:
        fallback_used = True

    selected_before_refinement = result
    if accuracy_mode == "strict":
        refined_raw = refine_word_boundaries_fn(
            words=result.words,
            wav_path=wav_path,
            max_shift_s=0.12,
            min_duration_s=0.02,
        )
        if isinstance(refined_raw, tuple):
            refined_words, refine_method = refined_raw
        else:
            refined_words = refined_raw
            refine_method = "numpy"
        if refined_words:
            source_by_ayah: dict[int, str] = {}
            for ayah, ayah_words in words_by_ayah(refined_words).items():
                if any(word.engine_candidate == "whisperx" for word in ayah_words):
                    source_by_ayah[ayah] = "fallback"
                else:
                    source_by_ayah[ayah] = "aligned"
            refined_ayahs = derive_ayahs_from_words_with_engine_sources(
                words=refined_words,
                source_by_ayah=source_by_ayah,
                default_source="aligned",
            )
            refined_result = build_result(
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
            refined_result.qc.speech_end_method = speech_end_method
            refined_result.qc.boundary_refine_method = refine_method
            if should_accept_refinement(
                original=selected_before_refinement,
                refined=refined_result,
                thresholds=thresholds,
            ):
                result = refined_result
                result.qc.warnings.append("boundary_refinement_applied")
            else:
                result = selected_before_refinement
                result.qc.boundary_refine_method = "none"
                result.qc.warnings.append("boundary_refinement_rejected")
    else:
        result.qc.boundary_refine_method = "none"

    if qc_requires_fallback(result.qc, thresholds):
        rescue_candidates = [
            result,
            selected_before_refinement,
            *candidate_results,
        ]
        rescue = select_strict_rescue_candidate(
            candidates=rescue_candidates,
            thresholds=thresholds,
        )
        if rescue is None:
            raise PipelineError(
                "qc_failed_after_all_candidates: " + "; ".join(result.qc.warnings)
            )
        result = rescue
        result.qc.engine_candidate_scores = candidate_scores
        result.qc.warnings.append("qc_rescue_selected")
        fallback_used = result.engine.name != requested_engine

    result.qc.speech_end_method = speech_end_method
    if result.qc.boundary_refine_method is None:
        result.qc.boundary_refine_method = "none"

    write_cache_result(row=row, result=result, cache_root=cache_dir)

    return write_result_artifacts(
        result=result,
        row=row,
        out_dir=out_dir,
        source=result.ayahs[0].source if result.ayahs else "aligned",
        fallback_used=fallback_used,
        elapsed_s=time.time() - started,
    )


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
        ayahs = derive_ayahs_from_words(refined_words, source="aligned")
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
            current_by_ayah = words_by_ayah(refined_words)

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

                if base_end <= base_start:
                    base_end = min(audio_duration_s, base_start + 2.0)

                chunk_start = max(0.0, base_start - overlap_s)
                chunk_end = min(audio_duration_s, base_end + overlap_s)
                if chunk_end <= chunk_start:
                    continue

                chunk_path = temp_root / f"ayah_{ayah:03d}_pass_{pass_index + 1}.wav"
                cut_audio_chunk(
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

    ayahs = derive_ayahs_from_words(refined_words, source="aligned")
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
    by_ayah = words_by_ayah(words)
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

