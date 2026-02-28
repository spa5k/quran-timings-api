from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from quran_audio_data.core.http import get_json_or_none
from quran_audio_data.sources.adapters import normalize_payload_with_adapters
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


class ExistingTimingResolver:
    QURAN_AUDIO_URL_TEMPLATES = (
        "https://quranaudio.pages.dev/timing/{reciter_id}/{surah}.json",
        "https://quranaudio.pages.dev/timing/{reciter_id}/{surah:03d}.json",
        "https://quranaudio.pages.dev/timing/{reciter_id}/{surah}/{ayah}.json",
    )

    QURAN_FOUNDATION_URL_TEMPLATES = (
        "https://api.quran.foundation/api/v4/chapter_recitations/{reciter_id}/{surah}",
        "https://api.quran.foundation/api/v4/chapter_recitations/{reciter_id}/{surah}:{ayah}",
    )

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

        candidates = self._build_candidates(
            reciter_id=reciter_id,
            surah=surah,
            ayah=ayah,
            source_url=source_url,
        )

        for source_name, payload in candidates:
            if payload is None:
                continue
            normalized = normalize_payload_with_adapters(
                payload=payload,
                canonical_words=canonical_words,
                source_name=source_name,
                source_default="existing",
            )
            if normalized is None:
                continue

            ayahs, words = normalized.ayahs, normalized.words
            validation = validate_external_timing(
                ayahs=ayahs,
                words=words,
                expected_word_count=expected_word_count,
                audio_duration_s=audio_duration_s,
                require_lexical_scores=not source_name.startswith("qf:"),
            )
            if validation.ok:
                return ResolvedTiming(
                    ayahs=ayahs,
                    words=words,
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

        if self.enable_remote:
            for template in self.QURAN_AUDIO_URL_TEMPLATES:
                url = template.format(reciter_id=reciter_id, surah=surah, ayah=ayah or "")
                candidates.append(
                    (f"quranaudio:{url}", get_json_or_none(url=url, timeout_s=self.timeout_s))
                )

            for template in self.QURAN_FOUNDATION_URL_TEMPLATES:
                url = template.format(reciter_id=reciter_id, surah=surah, ayah=ayah or "")
                candidates.append(
                    (f"qf:{url}", get_json_or_none(url=url, timeout_s=self.timeout_s))
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

def _read_json_file(path: Path) -> Any | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return orjson.loads(path.read_bytes())
    except Exception:
        return None
