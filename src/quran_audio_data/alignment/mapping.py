from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from rapidfuzz import fuzz

from quran_audio_data.schema import AyahTiming, WordTiming
from quran_audio_data.text.quran_text import CanonicalWord, normalize_arabic


Prediction = TypeVar("Prediction")


@dataclass(slots=True)
class PredictionSpan(Generic[Prediction]):
    prediction: Prediction
    text_norm: str
    start_s: float
    end_s: float
    confidence: float | None


@dataclass(slots=True)
class MappingConfig:
    engine_candidate: str
    search_window: int = 24
    exact_break_score: float = 99.0
    min_match_score: float = 55.0
    matched_origin: str = "native"
    unmatched_origin: str = "interpolated"


def to_prediction_spans(
    *,
    predicted_words: list[Prediction],
    text_getter: Callable[[Prediction], str | None],
    start_getter: Callable[[Prediction], float | None],
    end_getter: Callable[[Prediction], float | None],
    confidence_getter: Callable[[Prediction], float | None] | None = None,
) -> list[PredictionSpan[Prediction]]:
    spans: list[PredictionSpan[Prediction]] = []
    for predicted in predicted_words:
        text_raw = text_getter(predicted)
        start_s = start_getter(predicted)
        end_s = end_getter(predicted)
        if not text_raw or start_s is None or end_s is None:
            continue
        spans.append(
            PredictionSpan(
                prediction=predicted,
                text_norm=normalize_arabic(text_raw),
                start_s=float(start_s),
                end_s=float(end_s),
                confidence=confidence_getter(predicted) if confidence_getter else None,
            )
        )
    spans.sort(key=lambda item: item.start_s)
    return spans


def interpolate_slot(
    *,
    index: int,
    total: int,
    matched_idx: dict[int, int],
    predicted_words: list[PredictionSpan],
    audio_duration_s: float,
) -> tuple[float, float]:
    prev_i = max((i for i in matched_idx if i < index), default=None)
    next_i = min((i for i in matched_idx if i > index), default=None)

    if prev_i is not None:
        left = predicted_words[matched_idx[prev_i]].end_s
    else:
        left = 0.0

    if next_i is not None:
        right = predicted_words[matched_idx[next_i]].start_s
    else:
        right = max(audio_duration_s, predicted_words[-1].end_s) if predicted_words else audio_duration_s

    if prev_i is not None and next_i is not None:
        gap_count = max(1, next_i - prev_i - 1)
        rank = index - prev_i - 1
    elif prev_i is None and next_i is not None:
        gap_count = max(1, next_i)
        rank = index
    else:
        gap_count = max(1, total - (prev_i + 1 if prev_i is not None else 0))
        rank = index - (prev_i + 1 if prev_i is not None else 0)

    width = max(0.0, right - left)
    slot = width / gap_count
    start_s = left + (rank * slot)
    end_s = left + ((rank + 1) * slot)
    if end_s < start_s:
        end_s = start_s
    return start_s, end_s


def map_canonical_words(
    *,
    canonical_words: list[CanonicalWord],
    predicted_words: list[PredictionSpan],
    audio_duration_s: float,
    config: MappingConfig,
) -> list[WordTiming]:
    matched_idx: dict[int, int] = {}
    matched_scores: dict[int, float] = {}
    cursor = 0

    for index, canonical in enumerate(canonical_words):
        best_index: int | None = None
        best_score = -1.0

        search_end = min(len(predicted_words), cursor + config.search_window)
        for predicted_index in range(cursor, search_end):
            candidate = predicted_words[predicted_index]
            score = fuzz.ratio(canonical.text_norm, candidate.text_norm)
            if score > best_score:
                best_score = score
                best_index = predicted_index
            if score >= config.exact_break_score:
                break

        if best_index is not None and best_score >= config.min_match_score:
            matched_idx[index] = best_index
            matched_scores[index] = float(best_score)
            cursor = best_index + 1

    mapped: list[WordTiming] = []
    total = len(canonical_words)

    for index, canonical in enumerate(canonical_words):
        predicted_index = matched_idx.get(index)
        if predicted_index is not None:
            matched = predicted_words[predicted_index]
            start_s, end_s = matched.start_s, matched.end_s
            confidence = matched.confidence
            alignment_origin = config.matched_origin
            match_score = matched_scores.get(index)
        else:
            start_s, end_s = interpolate_slot(
                index=index,
                total=total,
                matched_idx=matched_idx,
                predicted_words=predicted_words,
                audio_duration_s=audio_duration_s,
            )
            confidence = None
            alignment_origin = config.unmatched_origin
            match_score = None

        mapped.append(
            WordTiming(
                surah=canonical.surah,
                ayah=canonical.ayah,
                word_index_global=canonical.word_index_global,
                word_index_in_ayah=canonical.word_index_in_ayah,
                text_uthmani=canonical.text_uthmani,
                text_norm=canonical.text_norm,
                start_s=start_s,
                end_s=end_s,
                confidence=confidence,
                alignment_origin=alignment_origin,  # type: ignore[arg-type]
                match_score=match_score,
                engine_candidate=config.engine_candidate,
            )
        )

    return mapped


def apply_supervision_overlay(
    *,
    words: list[WordTiming],
    supervision_word_bounds: dict[int, dict[int, tuple[float, float]]],
    model_weight: float = 0.30,
    source_provider: str = "quran_com",
) -> list[WordTiming]:
    """Blend model timings with external supervision bounds."""

    if not supervision_word_bounds:
        return words

    supervision_weight = max(0.0, min(1.0, 1.0 - model_weight))
    model_weight = 1.0 - supervision_weight
    overlaid: list[WordTiming] = []
    for word in words:
        ayah_bounds = supervision_word_bounds.get(word.ayah)
        if ayah_bounds is None:
            overlaid.append(word)
            continue
        supervised = ayah_bounds.get(word.word_index_in_ayah)
        if supervised is None:
            overlaid.append(word)
            continue
        sup_start, sup_end = supervised
        start = (model_weight * word.start_s) + (supervision_weight * sup_start)
        end = (model_weight * word.end_s) + (supervision_weight * sup_end)
        if end < start:
            end = start
        overlaid.append(
            word.model_copy(
                update={
                    "start_s": start,
                    "end_s": end,
                    "alignment_origin": "native",
                    "source_start_s": sup_start,
                    "source_end_s": sup_end,
                    "source_provider": source_provider,
                }
            )
        )
    return overlaid


def derive_ayahs_from_words(
    *,
    words: list[WordTiming],
    source: str,
    source_by_ayah: dict[int, str] | None = None,
) -> list[AyahTiming]:
    grouped: dict[tuple[int, int], list[WordTiming]] = defaultdict(list)
    for word in words:
        grouped[(word.surah, word.ayah)].append(word)

    ayahs: list[AyahTiming] = []
    for (surah, ayah), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        ayahs.append(
            AyahTiming(
                surah=surah,
                ayah=ayah,
                start_s=min(word.start_s for word in group),
                end_s=max(word.end_s for word in group),
                source=source_by_ayah.get(ayah, source) if source_by_ayah else source,  # type: ignore[arg-type]
            )
        )
    return ayahs
