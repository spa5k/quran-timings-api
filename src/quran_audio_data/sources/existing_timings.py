from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson
import requests

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
            normalized = _normalize_payload_to_schema(
                payload=payload,
                canonical_words=canonical_words,
                source_default="existing",
            )
            if normalized is None:
                continue

            ayahs, words = normalized
            validation = validate_external_timing(
                ayahs=ayahs,
                words=words,
                expected_word_count=expected_word_count,
                audio_duration_s=audio_duration_s,
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
            candidates.append((f"source_url:{source_url}", _http_get_json(source_url, timeout_s=self.timeout_s)))

        if self.enable_remote:
            for template in self.QURAN_AUDIO_URL_TEMPLATES:
                url = template.format(reciter_id=reciter_id, surah=surah, ayah=ayah or "")
                candidates.append((f"quranaudio:{url}", _http_get_json(url, timeout_s=self.timeout_s)))

            for template in self.QURAN_FOUNDATION_URL_TEMPLATES:
                url = template.format(reciter_id=reciter_id, surah=surah, ayah=ayah or "")
                candidates.append((f"qf:{url}", _http_get_json(url, timeout_s=self.timeout_s)))

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

    ok = (
        len(warnings) == 0
        and len(words) == expected_word_count
        and monotonic
    )
    return ExternalValidationResult(ok=ok, warnings=warnings)


def _normalize_payload_to_schema(
    *,
    payload: Any,
    canonical_words: list[CanonicalWord],
    source_default: str,
) -> tuple[list[AyahTiming], list[WordTiming]] | None:
    if not isinstance(payload, (dict, list)):
        return None

    # Pass-through for already-normalized schema payload.
    if isinstance(payload, dict) and isinstance(payload.get("ayahs"), list) and isinstance(payload.get("words"), list):
        try:
            ayahs = [AyahTiming.model_validate(item) for item in payload["ayahs"]]
            words = [WordTiming.model_validate(item) for item in payload["words"]]
            return ayahs, words
        except Exception:
            return None

    # Some APIs wrap output under data/verses keys.
    if isinstance(payload, dict) and isinstance(payload.get("verses"), list):
        ayah_entries = payload["verses"]
    elif isinstance(payload, dict) and isinstance(payload.get("data"), dict) and isinstance(payload["data"].get("verses"), list):
        ayah_entries = payload["data"]["verses"]
    elif isinstance(payload, list):
        ayah_entries = payload
    else:
        ayah_entries = None

    canonical_by_ayah: dict[int, list[CanonicalWord]] = {}
    for word in canonical_words:
        canonical_by_ayah.setdefault(word.ayah, []).append(word)

    if ayah_entries is not None:
        parsed_ayahs: list[AyahTiming] = []
        parsed_words: list[WordTiming] = []

        for entry in ayah_entries:
            ayah_number = _to_int(
                entry.get("ayah")
                or entry.get("verse_number")
                or entry.get("verse_key", "0:0").split(":")[-1]
            )
            if ayah_number is None or ayah_number not in canonical_by_ayah:
                continue

            ayah_words = canonical_by_ayah[ayah_number]
            start = _to_float(entry.get("start") or entry.get("start_s") or entry.get("timestamp_from"))
            end = _to_float(entry.get("end") or entry.get("end_s") or entry.get("timestamp_to"))

            words_raw = entry.get("words") if isinstance(entry.get("words"), list) else None
            if words_raw:
                for idx, canon in enumerate(ayah_words):
                    sample = words_raw[idx] if idx < len(words_raw) else None
                    w_start = _to_float(_safe_get(sample, "start", "start_s", "timestamp_from"))
                    w_end = _to_float(_safe_get(sample, "end", "end_s", "timestamp_to"))
                    if w_start is None or w_end is None:
                        if start is None or end is None:
                            continue
                        w_start, w_end = _distribute_slot(start, end, len(ayah_words), idx)

                    parsed_words.append(
                        WordTiming(
                            surah=canon.surah,
                            ayah=canon.ayah,
                            word_index_global=canon.word_index_global,
                            word_index_in_ayah=canon.word_index_in_ayah,
                            text_uthmani=canon.text_uthmani,
                            text_norm=canon.text_norm,
                            start_s=w_start,
                            end_s=w_end,
                            confidence=_to_float(_safe_get(sample, "confidence", "score")),
                        )
                    )

                if start is None:
                    start = parsed_words[-len(ayah_words)].start_s
                if end is None:
                    end = parsed_words[-1].end_s
            else:
                if start is None or end is None:
                    continue
                for idx, canon in enumerate(ayah_words):
                    w_start, w_end = _distribute_slot(start, end, len(ayah_words), idx)
                    parsed_words.append(
                        WordTiming(
                            surah=canon.surah,
                            ayah=canon.ayah,
                            word_index_global=canon.word_index_global,
                            word_index_in_ayah=canon.word_index_in_ayah,
                            text_uthmani=canon.text_uthmani,
                            text_norm=canon.text_norm,
                            start_s=w_start,
                            end_s=w_end,
                            confidence=None,
                        )
                    )

            if start is not None and end is not None:
                parsed_ayahs.append(
                    AyahTiming(
                        surah=ayah_words[0].surah,
                        ayah=ayah_number,
                        start_s=start,
                        end_s=end,
                        source=source_default,
                    )
                )

        if parsed_words and parsed_ayahs:
            parsed_words.sort(key=lambda x: (x.ayah, x.word_index_in_ayah))
            return parsed_ayahs, parsed_words

    # Fallback for compact map format: {"1": {"start":..., "end":...}}
    numeric_keys = [k for k in payload.keys() if isinstance(k, str) and k.isdigit()] if isinstance(payload, dict) else []
    if numeric_keys and isinstance(payload, dict):
        parsed_ayahs = []
        parsed_words = []
        for ayah_str in sorted(numeric_keys, key=int):
            ayah_number = int(ayah_str)
            if ayah_number not in canonical_by_ayah:
                continue
            row = payload[ayah_str]
            if not isinstance(row, dict):
                continue
            start = _to_float(row.get("start") or row.get("start_s"))
            end = _to_float(row.get("end") or row.get("end_s"))
            if start is None or end is None:
                continue

            ayah_words = canonical_by_ayah[ayah_number]
            for idx, canon in enumerate(ayah_words):
                w_start, w_end = _distribute_slot(start, end, len(ayah_words), idx)
                parsed_words.append(
                    WordTiming(
                        surah=canon.surah,
                        ayah=canon.ayah,
                        word_index_global=canon.word_index_global,
                        word_index_in_ayah=canon.word_index_in_ayah,
                        text_uthmani=canon.text_uthmani,
                        text_norm=canon.text_norm,
                        start_s=w_start,
                        end_s=w_end,
                        confidence=None,
                    )
                )
            parsed_ayahs.append(
                AyahTiming(
                    surah=ayah_words[0].surah,
                    ayah=ayah_number,
                    start_s=start,
                    end_s=end,
                    source=source_default,
                )
            )

        if parsed_words and parsed_ayahs:
            return parsed_ayahs, parsed_words

    return None


def _safe_get(obj: Any, *keys: str) -> Any:
    if not isinstance(obj, dict):
        return None
    for key in keys:
        if key in obj:
            return obj[key]
    return None


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


def _distribute_slot(start: float, end: float, count: int, idx: int) -> tuple[float, float]:
    if count <= 0:
        return start, end
    duration = max(0.0, end - start)
    slot = duration / count
    s = start + (idx * slot)
    e = start + ((idx + 1) * slot)
    return s, e


def _read_json_file(path: Path) -> Any | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return orjson.loads(path.read_bytes())
    except Exception:
        return None


def _http_get_json(url: str, *, timeout_s: float) -> Any | None:
    try:
        response = requests.get(url, timeout=timeout_s)
        if response.status_code >= 400:
            return None
        return response.json()
    except Exception:
        return None
