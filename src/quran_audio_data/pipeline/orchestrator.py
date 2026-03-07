from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
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
    apply_supervision_overlay,
)
from quran_audio_data.core.settings import get_settings
from quran_audio_data.schema import (
    QCThresholds,
    SegmentSourceType,
    TimingResult,
    WordTiming,
    qc_requires_fallback,
)
from quran_audio_data.sources import ExistingTimingResolver
from quran_audio_data.supervision import (
    build_audio_url,
    fetch_best_verse_segments,
    fetch_chapter_recitation_by_chapter,
    normalize_segments,
    resolve_reciter_mapping,
)
from quran_audio_data.text import CanonicalWord, QuranTextStore, TextSanitizationAudit

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
    PipelineReportV3,
    default_cache_dir,
)


@dataclass(slots=True)
class SupervisionContext:
    sources: list[str]
    segment_source_type: SegmentSourceType
    word_bounds_by_ayah: dict[int, dict[int, tuple[float, float]]]


def _normalize_engines(
    *,
    requested_engine: EngineOption,
    multi_engine: list[EngineOption],
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
    accuracy_mode: AccuracyMode = "strict",
    device: DeviceOption = "auto",
    text_data: str | Path | None = None,
    cache_dir: str | Path | None = None,
    enable_remote: bool = True,
    sample_size: int | None = None,
    thresholds: QCThresholds | None = None,
    availability_policy: EngineAvailabilityPolicy = "best_effort",
    registry: EngineRegistry | None = None,
    refine_word_boundaries_fn=refine_word_boundaries,
) -> PipelineReportV3:
    started = time.time()
    if accuracy_mode != "strict":
        raise PipelineError("Only strict mode is supported in the v3 pipeline")

    settings = get_settings()
    cache_root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    thresholds = thresholds or QCThresholds.strict()

    rows = read_manifest(manifest_path)
    if sample_size is not None:
        rows = rows[:sample_size]

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    text_store = QuranTextStore(text_data)
    resolver = ExistingTimingResolver(cache_dir=cache_root, enable_remote=enable_remote)
    active_registry = registry or EngineRegistry(
        nemo=NemoAligner(model_name="nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"),
        whisperx=WhisperXFallbackAligner(),
        mfa=NemoAligner(model_name="nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0"),
    )

    configured_engines = [
        cast(EngineOption, name)
        for name in settings.engine_order
        if name in {"nemo", "whisperx", "mfa"}
    ]
    requested_engines = multi_engine or configured_engines
    if not requested_engines:
        requested_engines = [
            cast(EngineOption, "nemo"),
            cast(EngineOption, "mfa"),
            cast(EngineOption, "whisperx"),
        ]

    selection = active_registry.select(
        requested_engine=engine,
        multi_engine=requested_engines,
        policy=availability_policy,
    )

    outputs = []
    errors: list[str] = []
    error_details: list[PipelineErrorDetail] = []
    aligned = 0
    fallback_used_count = 0
    priors_used_count = 0

    for row in rows:
        try:
            processed, priors_used = _process_row(
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
                enable_remote=enable_remote,
                refine_word_boundaries_fn=refine_word_boundaries_fn,
            )
            outputs.append(processed)
            aligned += 1
            if priors_used:
                priors_used_count += 1
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
    return PipelineReportV3(
        total=len(rows),
        succeeded=len(outputs),
        failed=len(errors),
        aligned=aligned,
        fallback_used=fallback_used_count,
        elapsed_s=elapsed,
        outputs=outputs,
        errors=errors,
        error_details=error_details,
        attempted_engines=list(selection.engines_to_try),
        priors_used=priors_used_count,
    )


def benchmark_pipeline(
    *,
    manifest_path: str | Path,
    out_dir: str | Path,
    sample_size: int,
    engine: EngineOption = "nemo",
    multi_engine: list[EngineOption] | None = None,
    accuracy_mode: AccuracyMode = "strict",
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
        "aligned": summary.aligned,
        "fallback_used": summary.fallback_used,
        "priors_used": summary.priors_used,
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
    enable_remote: bool,
    refine_word_boundaries_fn=refine_word_boundaries,
) -> tuple[Any, bool]:
    started = time.time()

    pass_trace: list[str] = [
        "A_audio_normalization",
        "B_ayah_anchors",
        "C_word_candidates",
    ]

    audio_info = probe_audio(row.audio_path)
    canonical_words, text_audits = _load_canonical_words(text_store=text_store, row=row)

    resolved = resolver.resolve(
        reciter_id=row.reciter_id,
        surah=row.surah,
        ayah=row.ayah,
        canonical_words=canonical_words,
        audio_duration_s=audio_info.duration_s,
        source_url=row.source_url,
    )

    supervision = _load_supervision_context(
        row=row,
        enable_remote=enable_remote,
    )
    if resolved is not None:
        supervision.sources.append(f"existing_prior:{resolved.source_name}")

    wav_path = ensure_wav_16k_mono(row.audio_path)
    speech_end_s, speech_end_method = estimate_speech_end_s(
        wav_path,
        fallback_duration_s=audio_info.duration_s,
    )
    engines_to_try = _normalize_engines(
        requested_engine=requested_engine,
        multi_engine=multi_engine,
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
            supervised_words = _apply_supervision_to_words(
                words=output.words,
                supervision=supervision,
            )
            result = build_result(
                row=row,
                audio_info=audio_info,
                engine_name=output.engine_name,
                engine_model=output.engine_model,
                device=output.device,
                fallback_used=False,
                ayahs=derive_ayahs_from_words(supervised_words, source="aligned"),
                words=supervised_words,
                expected_word_count=len(canonical_words),
                speech_end_s=speech_end_s,
                thresholds=thresholds,
                attempted_engines=engines_to_try,
                supervision_sources=supervision.sources,
                segment_source_type=supervision.segment_source_type,
                pass_trace=pass_trace,
            )
            if refinement_warnings:
                result.qc.warnings.extend(refinement_warnings)
            result.qc.speech_end_method = speech_end_method
            candidate_results.append(result)
        except (AlignmentError, EngineUnavailable, PipelineError) as exc:
            candidate_failures.append(f"{engine_name}: {exc}")

    if not candidate_results:
        raise PipelineError("no_alignment_candidates_succeeded: " + "; ".join(candidate_failures))

    candidate_scores = _score_candidates_with_supervision(
        candidates=candidate_results,
        supervision=supervision,
    )

    pass_trace.append("D_candidate_fusion")
    if len(candidate_results) > 1:
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

    result.selected_candidate_engine = result.engine.name
    result.candidate_scores = candidate_scores
    result.supervision_sources = supervision.sources
    result.segment_source_type = supervision.segment_source_type
    result.pass_trace = list(pass_trace)

    if result.engine.name != requested_engine:
        fallback_used = True

    pass_trace.append("E_boundary_refinement")
    selected_before_refinement = result
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
            attempted_engines=engines_to_try,
            supervision_sources=supervision.sources,
            selected_candidate_engine=result.engine.name,
            pass_trace=pass_trace,
            segment_source_type=supervision.segment_source_type,
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

    pass_trace.append("F_qc_gates_and_rescue")
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
            # Never silently pass: keep best and mark explicit failure code.
            result = max(rescue_candidates, key=score_timing_result)
            result.qc.warnings.append("qc_failed_after_rescue_keep_best")
            result.qc.reason_codes.append("qc_failed_keep_best")
        else:
            result = rescue
            result.qc.engine_candidate_scores = candidate_scores
            result.qc.warnings.append("qc_rescue_selected")
            fallback_used = result.engine.name != requested_engine

    result.qc.speech_end_method = speech_end_method
    if result.qc.boundary_refine_method is None:
        result.qc.boundary_refine_method = "none"

    result.supervision_sources = supervision.sources
    result.segment_source_type = supervision.segment_source_type
    result.pass_trace = list(pass_trace)
    result.selected_candidate_engine = result.engine.name
    result.candidate_scores = candidate_scores

    write_cache_result(row=row, result=result, cache_root=cache_dir)

    text_audit_payload = {
        "surah": row.surah,
        "ayah": row.ayah,
        "audits": [audit.to_dict() for audit in text_audits],
    }

    processed = write_result_artifacts(
        result=result,
        row=row,
        out_dir=out_dir,
        source=result.ayahs[0].source if result.ayahs else "aligned",
        fallback_used=fallback_used,
        elapsed_s=time.time() - started,
        text_audit=text_audit_payload,
    )
    return processed, resolved is not None


def _load_supervision_context(*, row: ManifestRow, enable_remote: bool) -> SupervisionContext:
    mapping = resolve_reciter_mapping(row.reciter_id)
    sources: list[str] = []
    word_bounds_by_ayah: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    segment_source_type: SegmentSourceType = "none"

    if mapping.everyayah_subfolder is not None:
        if row.ayah is not None:
            sources.append(
                "everyayah:"
                + build_audio_url(
                    subfolder=mapping.everyayah_subfolder,
                    surah=row.surah,
                    ayah=row.ayah,
                )
            )
        else:
            sources.append(
                f"everyayah:subfolder={mapping.everyayah_subfolder}:surah={row.surah}:scope=full_surah"
            )

    if (
        not enable_remote
        or not mapping.qcom_word_supervision_supported
        or mapping.qcom_recitation_id is None
    ):
        return SupervisionContext(
            sources=sources,
            segment_source_type=segment_source_type,
            word_bounds_by_ayah=dict(word_bounds_by_ayah),
        )

    try:
        if row.ayah is not None:
            verse_key = f"{row.surah}:{row.ayah}"
            payload = fetch_best_verse_segments(
                recitation_id=mapping.qcom_recitation_id,
                chapter=row.surah,
                verse_key=verse_key,
            )
            if payload is not None:
                for seg in payload.segments:
                    word_bounds_by_ayah[row.ayah][seg.word_index] = (
                        seg.start_ms / 1000.0,
                        seg.end_ms / 1000.0,
                    )
                segment_source_type = cast(SegmentSourceType, payload.source_type)
                sources.append(
                    f"qcom:{payload.source_type}:{mapping.qcom_recitation_id}:shape={payload.segment_shape}"
                )
        else:
            chapter_payload = fetch_chapter_recitation_by_chapter(
                mapping.qcom_recitation_id,
                row.surah,
                include_segments=True,
            )
            audio_file = (
                chapter_payload.get("audio_file") if isinstance(chapter_payload, dict) else None
            )
            timestamps = audio_file.get("timestamps") if isinstance(audio_file, dict) else None
            observed_shape = "unknown"
            if isinstance(timestamps, list):
                for item in timestamps:
                    if not isinstance(item, dict):
                        continue
                    verse_key = str(item.get("verse_key") or "")
                    if ":" not in verse_key:
                        continue
                    _, ayah_str = verse_key.split(":", 1)
                    try:
                        ayah = int(ayah_str)
                    except ValueError:
                        continue
                    raw_segments = (
                        item.get("segments") if isinstance(item.get("segments"), list) else None
                    )
                    if observed_shape == "unknown" and isinstance(raw_segments, list):
                        for raw in raw_segments:
                            if not isinstance(raw, list):
                                continue
                            if len(raw) >= 4:
                                observed_shape = "4_field"
                                break
                            if len(raw) >= 3:
                                observed_shape = "3_field"
                                break
                    segments = normalize_segments(raw_segments)
                    for seg in segments:
                        word_bounds_by_ayah[ayah][seg.word_index] = (
                            seg.start_ms / 1000.0,
                            seg.end_ms / 1000.0,
                        )
                if word_bounds_by_ayah:
                    segment_source_type = "qcom_chapter"
                    sources.append(
                        f"qcom:qcom_chapter:{mapping.qcom_recitation_id}:shape={observed_shape}"
                    )
    except Exception:
        # Supervision is optional; model-only path continues.
        pass

    return SupervisionContext(
        sources=sources,
        segment_source_type=segment_source_type,
        word_bounds_by_ayah=dict(word_bounds_by_ayah),
    )


def _apply_supervision_to_words(
    *, words: list[WordTiming], supervision: SupervisionContext
) -> list[WordTiming]:
    source_provider = "quran_com" if supervision.segment_source_type.startswith("qcom_") else "none"
    return apply_supervision_overlay(
        words=words,
        supervision_word_bounds=supervision.word_bounds_by_ayah,
        model_weight=0.30,
        source_provider=source_provider,
    )


def _score_candidates_with_supervision(
    *,
    candidates: list[TimingResult],
    supervision: SupervisionContext,
) -> dict[str, float]:
    out: dict[str, float] = {}

    agreement_by_engine = _inter_engine_agreement_scores(candidates)

    for candidate in candidates:
        base = score_timing_result(candidate)
        supervision_bonus = _supervision_agreement_bonus(candidate, supervision)
        inter_engine_bonus = agreement_by_engine.get(candidate.engine.name, 0.0)
        out[candidate.engine.name] = base + supervision_bonus + inter_engine_bonus

    return out


def _supervision_agreement_bonus(result: TimingResult, supervision: SupervisionContext) -> float:
    if not supervision.word_bounds_by_ayah:
        return 0.0

    errors_ms: list[float] = []
    for word in result.words:
        ayah_map = supervision.word_bounds_by_ayah.get(word.ayah)
        if not ayah_map:
            continue
        target = ayah_map.get(word.word_index_in_ayah)
        if target is None:
            continue
        start_s, end_s = target
        errors_ms.append(abs(word.start_s - start_s) * 1000.0)
        errors_ms.append(abs(word.end_s - end_s) * 1000.0)

    if not errors_ms:
        return 0.0

    mean_err = sum(errors_ms) / len(errors_ms)
    return max(-25.0, 30.0 - (mean_err / 8.0))


def _inter_engine_agreement_scores(candidates: list[TimingResult]) -> dict[str, float]:
    if len(candidates) < 2:
        return {candidates[0].engine.name: 0.0} if candidates else {}

    by_engine_index: dict[str, dict[tuple[int, int, int], WordTiming]] = {}
    for candidate in candidates:
        by_engine_index[candidate.engine.name] = {
            (word.ayah, word.word_index_in_ayah, word.word_index_global): word
            for word in candidate.words
        }

    scores: dict[str, float] = {}
    for engine_name, index in by_engine_index.items():
        deltas: list[float] = []
        for other_engine, other_index in by_engine_index.items():
            if other_engine == engine_name:
                continue
            keys = set(index).intersection(other_index)
            for key in keys:
                word_a = index[key]
                word_b = other_index[key]
                deltas.append(abs(word_a.start_s - word_b.start_s) * 1000.0)
                deltas.append(abs(word_a.end_s - word_b.end_s) * 1000.0)
        if not deltas:
            scores[engine_name] = 0.0
            continue
        mean_delta = sum(deltas) / len(deltas)
        scores[engine_name] = max(-20.0, 18.0 - (mean_delta / 10.0))

    return scores


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

    output = mfa.align(
        audio_wav_path=str(wav_path),
        canonical_words=canonical_words,
        audio_duration_s=audio_duration_s,
        device=device,
    )
    return (
        AlignmentOutput(
            ayahs=output.ayahs,
            words=[word.model_copy(update={"engine_candidate": "mfa"}) for word in output.words],
            engine_name="mfa",
            engine_model=output.engine_model,
            device=output.device,
            source=output.source,
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


def _load_canonical_words(
    *,
    text_store: QuranTextStore,
    row: ManifestRow,
) -> tuple[list[CanonicalWord], list[TextSanitizationAudit]]:
    words, audits = text_store.build_words_with_audit(
        surah=row.surah,
        ayah=row.ayah,
        text_variant=row.text_variant,
        riwaya=row.riwaya,
    )
    if not words:
        raise PipelineError(f"No canonical words found for surah={row.surah} ayah={row.ayah}")
    return words, audits
