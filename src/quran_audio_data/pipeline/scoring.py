from __future__ import annotations

from collections import defaultdict

from quran_audio_data.schema import QCThresholds, TimingResult, WordTiming


def score_words_slice(words: list[WordTiming], expected_count: int) -> float:
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


def score_timing_result(result: TimingResult) -> float:
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


def qc_violation_count(qc_report, thresholds: QCThresholds) -> int:
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


def should_accept_refinement(
    *,
    original: TimingResult,
    refined: TimingResult,
    thresholds: QCThresholds,
) -> bool:
    original_violations = qc_violation_count(original.qc, thresholds)
    refined_violations = qc_violation_count(refined.qc, thresholds)
    if refined_violations < original_violations:
        return True
    if refined_violations > original_violations:
        return False
    return score_timing_result(refined) >= score_timing_result(original)


def select_strict_rescue_candidate(
    *,
    candidates: list[TimingResult],
    thresholds: QCThresholds,
) -> TimingResult | None:
    from quran_audio_data.schema import qc_requires_fallback

    unique: dict[str, TimingResult] = {}
    for candidate in candidates:
        key = f"{candidate.engine.name}:{candidate.engine.model}:{len(candidate.words)}"
        best = unique.get(key)
        if best is None or score_timing_result(candidate) > score_timing_result(best):
            unique[key] = candidate

    ordered = sorted(unique.values(), key=score_timing_result, reverse=True)
    for candidate in ordered:
        if not qc_requires_fallback(candidate.qc, thresholds):
            return candidate
    return None


def words_by_ayah(words: list[WordTiming]) -> dict[int, list[WordTiming]]:
    out: dict[int, list[WordTiming]] = defaultdict(list)
    for word in words:
        out[word.ayah].append(word)
    for ayah in out:
        out[ayah].sort(key=lambda word: word.word_index_in_ayah)
    return out


def select_best_result_per_ayah(
    *,
    row,
    audio_info,
    canonical_words,
    candidates: list[TimingResult],
    thresholds: QCThresholds,
    candidate_scores: dict[str, float],
) -> TimingResult:
    from .artifacts import build_result, derive_ayahs_from_words_with_engine_sources
    from .types import PipelineError

    expected_by_ayah: dict[int, int] = defaultdict(int)
    for word in canonical_words:
        expected_by_ayah[word.ayah] += 1

    candidate_words: dict[str, dict[int, list[WordTiming]]] = {}
    for candidate in candidates:
        candidate_words[candidate.engine.name] = words_by_ayah(candidate.words)

    selected_words: list[WordTiming] = []
    selected_sources: dict[int, str] = {}

    for ayah in sorted(expected_by_ayah):
        expected_count = expected_by_ayah[ayah]
        best_engine: str | None = None
        best_slice: list[WordTiming] = []
        best_score = -10_000.0

        for engine_name, grouped in candidate_words.items():
            group = grouped.get(ayah, [])
            score = score_words_slice(group, expected_count)
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
    ayahs = derive_ayahs_from_words_with_engine_sources(
        words=selected_words,
        source_by_ayah=selected_sources,
        default_source="aligned",
    )

    return build_result(
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

__all__ = [
    "score_words_slice",
    "score_timing_result",
    "qc_violation_count",
    "should_accept_refinement",
    "select_strict_rescue_candidate",
    "select_best_result_per_ayah",
    "words_by_ayah",
]
