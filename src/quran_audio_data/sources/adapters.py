from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import AliasChoices, BaseModel, Field, field_validator
from rapidfuzz import fuzz

from quran_audio_data.core.parsing import safe_get, to_float, to_int
from quran_audio_data.schema import AyahTiming, WordTiming
from quran_audio_data.text.quran_text import CanonicalWord, normalize_arabic


@dataclass(slots=True)
class NormalizedTimingBundle:
    ayahs: list[AyahTiming]
    words: list[WordTiming]


class _AdapterWord(BaseModel):
    position: int | None = Field(default=None, validation_alias=AliasChoices("position", "word_index_in_ayah"))
    start_s: float | None = Field(default=None, validation_alias=AliasChoices("start", "start_s", "timestamp_from"))
    end_s: float | None = Field(default=None, validation_alias=AliasChoices("end", "end_s", "timestamp_to"))
    text: str | None = Field(default=None, validation_alias=AliasChoices("text", "word", "token", "text_uthmani"))
    confidence: float | None = Field(default=None, validation_alias=AliasChoices("confidence", "score"))
    match_score: float | None = None

    @field_validator("text", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: Any) -> str | None:
        text = str(value).strip() if value is not None else ""
        return text or None


class _AdapterAyahEntry(BaseModel):
    ayah: int | None = Field(default=None, validation_alias=AliasChoices("ayah", "verse_number"))
    verse_key: str | None = None
    start_s: float | None = Field(default=None, validation_alias=AliasChoices("start", "start_s", "timestamp_from"))
    end_s: float | None = Field(default=None, validation_alias=AliasChoices("end", "end_s", "timestamp_to"))
    words: list[_AdapterWord] = Field(default_factory=list)


class _TimingSchemaPayload(BaseModel):
    ayahs: list[AyahTiming]
    words: list[WordTiming]


class _CompactAyahEntry(BaseModel):
    start_s: float = Field(validation_alias=AliasChoices("start", "start_s"))
    end_s: float = Field(validation_alias=AliasChoices("end", "end_s"))


def normalize_payload_with_adapters(
    *,
    payload: Any,
    canonical_words: list[CanonicalWord],
    source_name: str,
    source_default: str,
) -> NormalizedTimingBundle | None:
    adapter = _pick_adapter(source_name)
    return adapter(payload=payload, canonical_words=canonical_words, source_default=source_default)


def _pick_adapter(source_name: str):
    if source_name.startswith("cache:"):
        return _normalize_cache_payload
    if source_name.startswith("quranaudio:"):
        return _normalize_quranaudio_payload
    if source_name.startswith("qf:"):
        return _normalize_quran_foundation_payload
    if source_name.startswith("source_url:"):
        return _normalize_source_url_payload
    return _normalize_source_url_payload


def _normalize_cache_payload(
    *,
    payload: Any,
    canonical_words: list[CanonicalWord],
    source_default: str,
) -> NormalizedTimingBundle | None:
    schema_bundle = _normalize_schema_payload(payload=payload, source_default=source_default)
    if schema_bundle is not None:
        return schema_bundle
    return _normalize_generic_payload(
        payload=payload,
        canonical_words=canonical_words,
        source_default=source_default,
    )


def _normalize_quranaudio_payload(
    *,
    payload: Any,
    canonical_words: list[CanonicalWord],
    source_default: str,
) -> NormalizedTimingBundle | None:
    schema_bundle = _normalize_schema_payload(payload=payload, source_default=source_default)
    if schema_bundle is not None:
        return schema_bundle
    return _normalize_generic_payload(
        payload=payload,
        canonical_words=canonical_words,
        source_default=source_default,
    )


def _normalize_quran_foundation_payload(
    *,
    payload: Any,
    canonical_words: list[CanonicalWord],
    source_default: str,
) -> NormalizedTimingBundle | None:
    schema_bundle = _normalize_schema_payload(payload=payload, source_default=source_default)
    if schema_bundle is not None:
        return schema_bundle

    chapters = _extract_quran_foundation_timestamps(payload)
    if chapters is not None:
        bundle = _entries_to_bundle(
            entries=chapters,
            canonical_words=canonical_words,
            source_default=source_default,
        )
        if bundle is not None:
            return bundle

    return _normalize_generic_payload(
        payload=payload,
        canonical_words=canonical_words,
        source_default=source_default,
    )


def _normalize_source_url_payload(
    *,
    payload: Any,
    canonical_words: list[CanonicalWord],
    source_default: str,
) -> NormalizedTimingBundle | None:
    schema_bundle = _normalize_schema_payload(payload=payload, source_default=source_default)
    if schema_bundle is not None:
        return schema_bundle
    return _normalize_generic_payload(
        payload=payload,
        canonical_words=canonical_words,
        source_default=source_default,
    )


def _normalize_schema_payload(*, payload: Any, source_default: str) -> NormalizedTimingBundle | None:
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("ayahs"), list) or not isinstance(payload.get("words"), list):
        return None
    try:
        schema_payload = _TimingSchemaPayload.model_validate(payload)
    except Exception:
        return None

    ayahs = [ayah.model_copy(update={"source": source_default}) for ayah in schema_payload.ayahs]
    words = [word.model_copy(update={"engine_candidate": "existing"}) for word in schema_payload.words]
    return NormalizedTimingBundle(ayahs=ayahs, words=words)


def _normalize_generic_payload(
    *,
    payload: Any,
    canonical_words: list[CanonicalWord],
    source_default: str,
) -> NormalizedTimingBundle | None:
    ayah_entries = _extract_ayah_entries(payload)
    if ayah_entries is not None:
        bundle = _entries_to_bundle(
            entries=ayah_entries,
            canonical_words=canonical_words,
            source_default=source_default,
        )
        if bundle is not None:
            return bundle

    compact = _extract_compact_ayah_map(payload)
    if compact is not None:
        bundle = _compact_to_bundle(
            compact=compact,
            canonical_words=canonical_words,
            source_default=source_default,
        )
        if bundle is not None:
            return bundle

    return None


def _extract_ayah_entries(payload: Any) -> list[_AdapterAyahEntry] | None:
    raw_entries: Any
    if isinstance(payload, dict) and isinstance(payload.get("verses"), list):
        raw_entries = payload.get("verses")
    elif (
        isinstance(payload, dict)
        and isinstance(payload.get("data"), dict)
        and isinstance(payload["data"].get("verses"), list)
    ):
        raw_entries = payload["data"]["verses"]
    elif isinstance(payload, list):
        raw_entries = payload
    else:
        return None

    parsed: list[_AdapterAyahEntry] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        try:
            parsed.append(_AdapterAyahEntry.model_validate(entry))
        except Exception:
            continue
    return parsed or None


def _extract_quran_foundation_timestamps(payload: Any) -> list[_AdapterAyahEntry] | None:
    if not isinstance(payload, dict):
        return None
    audio_file = payload.get("audio_file")
    if not isinstance(audio_file, dict):
        return None
    timestamps = audio_file.get("timestamps")
    if not isinstance(timestamps, list):
        return None

    chapter_anchor_ms: float | None = None
    for stamp in timestamps:
        if not isinstance(stamp, dict):
            continue
        anchor_candidate = to_float(stamp.get("timestamp_from"))
        if anchor_candidate is not None:
            chapter_anchor_ms = anchor_candidate
            break
    if chapter_anchor_ms is None:
        chapter_anchor_ms = 0.0

    out: list[_AdapterAyahEntry] = []
    for stamp in timestamps:
        if not isinstance(stamp, dict):
            continue
        verse_key = stamp.get("verse_key")
        if not isinstance(verse_key, str) or ":" not in verse_key:
            continue
        ayah = to_int(verse_key.split(":")[-1])
        if ayah is None:
            continue

        anchor_ms = to_float(stamp.get("timestamp_from"))
        if anchor_ms is None:
            continue
        end_ms = to_float(stamp.get("timestamp_to"))
        ayah_start_s = max(0.0, (anchor_ms - chapter_anchor_ms) / 1000.0)
        ayah_end_s = (
            max(ayah_start_s, (end_ms - chapter_anchor_ms) / 1000.0)
            if end_ms is not None
            else None
        )
        segments = stamp.get("segments")
        words: list[_AdapterWord] = []
        if isinstance(segments, list):
            for segment in segments:
                if not isinstance(segment, list) or len(segment) < 3:
                    continue
                position = to_int(segment[0])
                start_ms = to_float(segment[1])
                end_ms_segment = to_float(segment[2])
                if position is None or start_ms is None or end_ms_segment is None:
                    continue
                start_s, end_s = _normalize_qf_segment_window(
                    segment_start_ms=start_ms,
                    segment_end_ms=end_ms_segment,
                    chapter_anchor_ms=chapter_anchor_ms,
                    ayah_anchor_ms=anchor_ms,
                    ayah_start_s=ayah_start_s,
                )
                words.append(
                    _AdapterWord(
                        position=position,
                        start_s=start_s,
                        end_s=max(end_s, start_s + 0.001),
                    )
                )

        out.append(
            _AdapterAyahEntry(
                ayah=ayah,
                verse_key=verse_key,
                start_s=ayah_start_s,
                end_s=ayah_end_s,
                words=words,
            )
        )
    return out or None


def _normalize_qf_segment_window(
    *,
    segment_start_ms: float,
    segment_end_ms: float,
    chapter_anchor_ms: float,
    ayah_anchor_ms: float,
    ayah_start_s: float,
) -> tuple[float, float]:
    # Most chapter_recitation payloads are chapter-absolute milliseconds.
    chapter_start_s = (segment_start_ms - chapter_anchor_ms) / 1000.0
    chapter_end_s = (segment_end_ms - chapter_anchor_ms) / 1000.0
    if chapter_end_s >= chapter_start_s and chapter_end_s >= (ayah_start_s - 0.5):
        start_s = max(0.0, chapter_start_s)
        end_s = max(start_s, chapter_end_s)
        return start_s, end_s

    # Fallback for ayah-relative segment offsets.
    ayah_relative_start_s = ayah_start_s + max(0.0, (segment_start_ms - ayah_anchor_ms) / 1000.0)
    ayah_relative_end_s = ayah_start_s + max(0.0, (segment_end_ms - ayah_anchor_ms) / 1000.0)
    end_s = max(ayah_relative_start_s, ayah_relative_end_s)
    return ayah_relative_start_s, end_s


def _extract_compact_ayah_map(payload: Any) -> dict[int, _CompactAyahEntry] | None:
    if not isinstance(payload, dict):
        return None
    numeric_keys = [key for key in payload if isinstance(key, str) and key.isdigit()]
    if not numeric_keys:
        return None

    out: dict[int, _CompactAyahEntry] = {}
    for ayah_key in numeric_keys:
        value = payload.get(ayah_key)
        if not isinstance(value, dict):
            continue
        try:
            out[int(ayah_key)] = _CompactAyahEntry.model_validate(value)
        except Exception:
            continue
    return out or None


def _entries_to_bundle(
    *,
    entries: list[_AdapterAyahEntry],
    canonical_words: list[CanonicalWord],
    source_default: str,
) -> NormalizedTimingBundle | None:
    canonical_by_ayah: dict[int, list[CanonicalWord]] = {}
    for word in canonical_words:
        canonical_by_ayah.setdefault(word.ayah, []).append(word)

    parsed_ayahs: list[AyahTiming] = []
    parsed_words: list[WordTiming] = []

    for entry in entries:
        ayah_number = entry.ayah
        if ayah_number is None and entry.verse_key:
            ayah_number = to_int(entry.verse_key.split(":")[-1])
        if ayah_number is None or ayah_number not in canonical_by_ayah:
            continue

        ayah_words = canonical_by_ayah[ayah_number]
        ayah_start = entry.start_s
        ayah_end = entry.end_s

        if entry.words:
            indexed_words: dict[int, _AdapterWord] = {}
            ordered_words: list[_AdapterWord] = []
            for idx, candidate in enumerate(entry.words, start=1):
                position = candidate.position or idx
                indexed_words[position] = candidate
                ordered_words.append(candidate)

            for idx, canon in enumerate(ayah_words, start=1):
                sample = indexed_words.get(canon.word_index_in_ayah) or indexed_words.get(idx)
                if sample is None:
                    if ayah_start is None or ayah_end is None:
                        continue
                    word_start, word_end = _distribute_slot(ayah_start, ayah_end, len(ayah_words), idx - 1)
                    origin = "distributed"
                    confidence = None
                    match_score = None
                else:
                    word_start = sample.start_s
                    word_end = sample.end_s
                    confidence = sample.confidence
                    if word_start is None or word_end is None:
                        if ayah_start is None or ayah_end is None:
                            continue
                        word_start, word_end = _distribute_slot(ayah_start, ayah_end, len(ayah_words), idx - 1)
                        origin = "distributed"
                    else:
                        origin = "native"

                    sample_text_norm = normalize_arabic(sample.text) if sample.text else None
                    raw_match_score = sample.match_score
                    match_score = (
                        raw_match_score
                        if raw_match_score is not None
                        else (
                            float(fuzz.ratio(canon.text_norm, sample_text_norm))
                            if sample_text_norm
                            else None
                        )
                    )

                parsed_words.append(
                    WordTiming(
                        surah=canon.surah,
                        ayah=canon.ayah,
                        word_index_global=canon.word_index_global,
                        word_index_in_ayah=canon.word_index_in_ayah,
                        text_uthmani=canon.text_uthmani,
                        text_norm=canon.text_norm,
                        start_s=word_start,
                        end_s=word_end,
                        confidence=confidence,
                        alignment_origin=origin,
                        match_score=match_score,
                        engine_candidate="existing",
                    )
                )

            if ayah_start is None and len(parsed_words) >= len(ayah_words):
                ayah_start = parsed_words[-len(ayah_words)].start_s
            if ayah_end is None and parsed_words:
                ayah_end = parsed_words[-1].end_s
        else:
            if ayah_start is None or ayah_end is None:
                continue
            for idx, canon in enumerate(ayah_words):
                word_start, word_end = _distribute_slot(ayah_start, ayah_end, len(ayah_words), idx)
                parsed_words.append(
                    WordTiming(
                        surah=canon.surah,
                        ayah=canon.ayah,
                        word_index_global=canon.word_index_global,
                        word_index_in_ayah=canon.word_index_in_ayah,
                        text_uthmani=canon.text_uthmani,
                        text_norm=canon.text_norm,
                        start_s=word_start,
                        end_s=word_end,
                        confidence=None,
                        alignment_origin="distributed",
                        match_score=None,
                        engine_candidate="existing",
                    )
                )

        if ayah_start is not None and ayah_end is not None:
            parsed_ayahs.append(
                AyahTiming(
                    surah=ayah_words[0].surah,
                    ayah=ayah_number,
                    start_s=ayah_start,
                    end_s=ayah_end,
                    source=source_default,
                )
            )

    if not parsed_words or not parsed_ayahs:
        return None

    parsed_words.sort(key=lambda word: (word.ayah, word.word_index_in_ayah))
    return NormalizedTimingBundle(ayahs=parsed_ayahs, words=parsed_words)


def _compact_to_bundle(
    *,
    compact: dict[int, _CompactAyahEntry],
    canonical_words: list[CanonicalWord],
    source_default: str,
) -> NormalizedTimingBundle | None:
    canonical_by_ayah: dict[int, list[CanonicalWord]] = {}
    for word in canonical_words:
        canonical_by_ayah.setdefault(word.ayah, []).append(word)

    parsed_ayahs: list[AyahTiming] = []
    parsed_words: list[WordTiming] = []
    for ayah_number in sorted(compact):
        ayah_words = canonical_by_ayah.get(ayah_number)
        if not ayah_words:
            continue

        ayah_timing = compact[ayah_number]
        start = ayah_timing.start_s
        end = ayah_timing.end_s

        for idx, canon in enumerate(ayah_words):
            word_start, word_end = _distribute_slot(start, end, len(ayah_words), idx)
            parsed_words.append(
                WordTiming(
                    surah=canon.surah,
                    ayah=canon.ayah,
                    word_index_global=canon.word_index_global,
                    word_index_in_ayah=canon.word_index_in_ayah,
                    text_uthmani=canon.text_uthmani,
                    text_norm=canon.text_norm,
                    start_s=word_start,
                    end_s=word_end,
                    confidence=None,
                    alignment_origin="distributed",
                    match_score=None,
                    engine_candidate="existing",
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

    if not parsed_ayahs or not parsed_words:
        return None
    return NormalizedTimingBundle(ayahs=parsed_ayahs, words=parsed_words)


def _distribute_slot(start: float, end: float, count: int, idx: int) -> tuple[float, float]:
    if count <= 0:
        return start, end
    duration = max(0.0, end - start)
    slot = duration / count
    word_start = start + (idx * slot)
    word_end = start + ((idx + 1) * slot)
    return word_start, word_end
