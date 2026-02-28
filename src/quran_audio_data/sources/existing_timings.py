from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from quran_audio_data.core.http import get_json_or_none
from quran_audio_data.schema import AyahTiming, WordTiming
from quran_audio_data.text.quran_text import CanonicalWord


@dataclass(slots=True)
class ExternalValidationResult:
    ok: bool
    warnings: list[str]


@dataclass(slots=True)
class ResolvedTiming:
    ayahs: list[AyahTiming]
    words: list[WordTiming]
    source_name: str
    warnings: list[str]


@dataclass(slots=True)
class _NormalizedTiming:
    ayahs: list[AyahTiming]
    words: list[WordTiming]


class ExistingTimingResolver:
    """Resolve prior timings from local cache or explicit source URL.

    Legacy multi-source adapter stacks were removed intentionally to keep the v3
    code path minimal and deterministic.
    """

    def __init__(
        self,
        *,
        cache_dir: str | Path = ".cache/timings",
        timeout_s: float = 10.0,
        enable_remote: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.timeout_s = timeout_s
        self.enable_remote = enable_remote

    def resolve(
        self,
        *,
        reciter_id: str,
        surah: int,
        ayah: int | None,
        canonical_words: list[CanonicalWord],
        audio_duration_s: float,
        source_url: str | None = None,
    ) -> ResolvedTiming | None:
        expected_word_count = len(canonical_words)
        if expected_word_count == 0:
            return None

        for source_name, payload in self._build_candidates(
            reciter_id=reciter_id,
            surah=surah,
            ayah=ayah,
            source_url=source_url,
        ):
            if payload is None:
                continue
            normalized = _normalize_prior_payload(
                payload=payload,
                canonical_words=canonical_words,
                source_default="existing",
            )
            if normalized is None:
                continue

            validation = validate_external_timing(
                ayahs=normalized.ayahs,
                words=normalized.words,
                expected_word_count=expected_word_count,
                audio_duration_s=audio_duration_s,
                require_lexical_scores=False,
            )
            if validation.ok:
                return ResolvedTiming(
                    ayahs=normalized.ayahs,
                    words=normalized.words,
                    source_name=source_name,
                    warnings=validation.warnings,
                )

        return None

    def _build_candidates(
        self,
        *,
        reciter_id: str,
        surah: int,
        ayah: int | None,
        source_url: str | None,
    ) -> list[tuple[str, Any | None]]:
        candidates: list[tuple[str, Any | None]] = []

        for file_path in self._local_cache_candidates(reciter_id=reciter_id, surah=surah, ayah=ayah):
            payload = _read_json_file(file_path)
            if payload is not None:
                candidates.append((f"cache:{file_path}", payload))

        if source_url and self.enable_remote:
            candidates.append(
                (
                    f"source_url:{source_url}",
                    get_json_or_none(url=source_url, timeout_s=self.timeout_s),
                )
            )

        return candidates

    def _local_cache_candidates(self, *, reciter_id: str, surah: int, ayah: int | None) -> list[Path]:
        paths: list[Path] = []
        reciter_dir = self.cache_dir / reciter_id

        if ayah is not None:
            paths.append(reciter_dir / f"{surah:03d}_{ayah:03d}.json")
            paths.append(reciter_dir / f"{surah}_{ayah}.json")

        paths.append(reciter_dir / f"{surah:03d}.json")
        paths.append(reciter_dir / f"{surah}.json")
        paths.append(self.cache_dir / f"{reciter_id}_{surah:03d}.json")
        return paths


def _normalize_prior_payload(
    *,
    payload: Any,
    canonical_words: list[CanonicalWord],
    source_default: str,
) -> _NormalizedTiming | None:
    if not isinstance(payload, dict):
        return None

    words_payload = payload.get("words")
    if not isinstance(words_payload, list):
        return None

    canonical_by_key = {
        (word.ayah, word.word_index_in_ayah): word
        for word in canonical_words
    }

    words: list[WordTiming] = []
    for raw in words_payload:
        if not isinstance(raw, dict):
            continue

        ayah = _to_int(raw.get("ayah"))
        position = _to_int(raw.get("word_index_in_ayah") or raw.get("position"))
        start_s = _to_float(raw.get("start_s") if "start_s" in raw else raw.get("start"))
        end_s = _to_float(raw.get("end_s") if "end_s" in raw else raw.get("end"))
        if ayah is None or position is None or start_s is None or end_s is None:
            continue

        canon = canonical_by_key.get((ayah, position))
        if canon is None:
            continue

        origin_raw = str(raw.get("alignment_origin") or "native")
        alignment_origin = origin_raw if origin_raw in {"native", "interpolated", "distributed"} else "native"

        words.append(
            WordTiming(
                surah=canon.surah,
                ayah=canon.ayah,
                word_index_global=canon.word_index_global,
                word_index_in_ayah=canon.word_index_in_ayah,
                text_uthmani=canon.text_uthmani,
                text_norm=canon.text_norm,
                start_s=start_s,
                end_s=end_s,
                confidence=_to_float(raw.get("confidence")),
                alignment_origin=alignment_origin,
                match_score=_to_float(raw.get("match_score")),
                engine_candidate="existing",
            )
        )

    if not words:
        return None

    ayahs_payload = payload.get("ayahs")
    ayahs: list[AyahTiming] = []
    if isinstance(ayahs_payload, list):
        for raw in ayahs_payload:
            if not isinstance(raw, dict):
                continue
            ayah = _to_int(raw.get("ayah"))
            start_s = _to_float(raw.get("start_s") if "start_s" in raw else raw.get("start"))
            end_s = _to_float(raw.get("end_s") if "end_s" in raw else raw.get("end"))
            if ayah is None or start_s is None or end_s is None:
                continue
            ayahs.append(
                AyahTiming(
                    surah=canonical_words[0].surah,
                    ayah=ayah,
                    start_s=start_s,
                    end_s=end_s,
                    source=source_default,
                )
            )

    if not ayahs:
        grouped: dict[int, list[WordTiming]] = {}
        for word in words:
            grouped.setdefault(word.ayah, []).append(word)
        for ayah_num, group in sorted(grouped.items()):
            ayahs.append(
                AyahTiming(
                    surah=canonical_words[0].surah,
                    ayah=ayah_num,
                    start_s=min(word.start_s for word in group),
                    end_s=max(word.end_s for word in group),
                    source=source_default,
                )
            )

    return _NormalizedTiming(ayahs=ayahs, words=words)


def validate_external_timing(
    *,
    ayahs: list[AyahTiming],
    words: list[WordTiming],
    expected_word_count: int,
    audio_duration_s: float,
    max_duration_delta_ratio: float = 0.03,
    max_interpolated_ratio: float = 0.25,
    min_lexical_match_ratio: float = 0.85,
    require_lexical_scores: bool = True,
) -> ExternalValidationResult:
    warnings: list[str] = []

    if len(words) != expected_word_count:
        warnings.append(f"word_count_mismatch:{len(words)}!=expected:{expected_word_count}")

    monotonic = True
    last_start = -1.0
    for word in words:
        if word.end_s < word.start_s:
            warnings.append("negative_word_duration")
            monotonic = False
            break
        if word.start_s < last_start:
            warnings.append("non_monotonic_word_starts")
            monotonic = False
            break
        last_start = word.start_s

    if ayahs:
        for ayah in ayahs:
            if ayah.end_s < ayah.start_s:
                warnings.append("negative_ayah_duration")
                monotonic = False
                break

    if words and audio_duration_s > 0:
        max_end = max(word.end_s for word in words)
        delta = abs(max_end - audio_duration_s) / audio_duration_s
        if delta > max_duration_delta_ratio:
            warnings.append(f"duration_mismatch_ratio:{delta:.4f}")

    if words:
        interpolated_count = sum(
            1 for word in words if word.alignment_origin in {"interpolated", "distributed"}
        )
        interpolated_ratio = interpolated_count / len(words)
        if interpolated_ratio > max_interpolated_ratio:
            warnings.append(
                f"interpolated_ratio:{interpolated_ratio:.4f}>{max_interpolated_ratio:.4f}"
            )

        lexical_scores = [word.match_score for word in words if word.match_score is not None]
        if lexical_scores:
            lexical_match_ratio = sum(1 for score in lexical_scores if score >= 70.0) / len(words)
            if lexical_match_ratio < min_lexical_match_ratio:
                warnings.append(
                    f"lexical_match_ratio:{lexical_match_ratio:.4f}<{min_lexical_match_ratio:.4f}"
                )
        elif require_lexical_scores:
            warnings.append("missing_lexical_scores")

    ok = (
        len(warnings) == 0
        and len(words) == expected_word_count
        and monotonic
    )
    return ExternalValidationResult(ok=ok, warnings=warnings)


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_json_file(path: Path) -> Any | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return orjson.loads(path.read_bytes())
    except Exception:
        return None
