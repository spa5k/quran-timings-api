from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quran_audio_data.core.http import get_json_with_retry

from .segment_normalizer import WordSegment, normalize_segments


QCOM_API_BASE = "https://api.quran.com/api/v4"
QCOM_VERSE_FILE_BASE = "https://verses.quran.com/"


@dataclass(slots=True)
class VerseSegmentPayload:
    verse_key: str
    segments: list[WordSegment]
    source_type: str
    segment_shape: str


def _join_verse_url(path_or_url: str | None) -> str | None:
    if not path_or_url:
        return None
    value = str(path_or_url)
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return QCOM_VERSE_FILE_BASE.rstrip("/") + "/" + value.lstrip("/")


def fetch_recitation_catalog(*, language: str = "en") -> dict[str, Any]:
    return get_json_with_retry(
        url=f"{QCOM_API_BASE}/resources/recitations",
        params={"language": language},
    )


def fetch_chapter_recitations(reciter_id: int | str) -> dict[str, Any]:
    return get_json_with_retry(
        url=f"{QCOM_API_BASE}/chapter_recitations/{reciter_id}",
    )


def fetch_chapter_recitation_by_chapter(
    reciter_id: int | str,
    chapter: int,
    *,
    include_segments: bool = True,
) -> dict[str, Any]:
    params = {"segments": "true"} if include_segments else None
    return get_json_with_retry(
        url=f"{QCOM_API_BASE}/chapter_recitations/{reciter_id}/{chapter}",
        params=params,
    )


def fetch_verse_recitations_by_chapter(
    recitation_id: int | str,
    chapter: int,
    *,
    fields: str = "segments,format,id,chapter_id",
) -> dict[str, Any]:
    return get_json_with_retry(
        url=f"{QCOM_API_BASE}/recitations/{recitation_id}/by_chapter/{chapter}",
        params={"fields": fields},
    )


def fetch_verse_recitation_by_ayah(
    recitation_id: int | str,
    verse_key: str,
    *,
    fields: str = "segments,format,id,chapter_id",
) -> dict[str, Any]:
    return get_json_with_retry(
        url=f"{QCOM_API_BASE}/recitations/{recitation_id}/by_ayah/{verse_key}",
        params={"fields": fields},
    )


def resolve_verse_audio_url(url_value: str | None) -> str | None:
    return _join_verse_url(url_value)


def extract_chapter_timestamp_segments(
    payload: dict[str, Any],
    *,
    verse_key: str,
) -> VerseSegmentPayload | None:
    audio_file = payload.get("audio_file") if isinstance(payload, dict) else None
    if not isinstance(audio_file, dict):
        return None
    timestamps = audio_file.get("timestamps")
    if not isinstance(timestamps, list):
        return None

    for item in timestamps:
        if not isinstance(item, dict):
            continue
        if str(item.get("verse_key") or "") != verse_key:
            continue
        raw_segments = item.get("segments") if isinstance(item.get("segments"), list) else None
        segments = normalize_segments(raw_segments)
        if not segments:
            return None
        segment_shape = _detect_segment_shape(raw_segments)
        return VerseSegmentPayload(
            verse_key=verse_key,
            segments=segments,
            source_type="qcom_chapter",
            segment_shape=segment_shape,
        )
    return None


def extract_verse_segments(
    payload: dict[str, Any],
    *,
    verse_key: str,
) -> VerseSegmentPayload | None:
    audio_files = payload.get("audio_files") if isinstance(payload, dict) else None
    if not isinstance(audio_files, list):
        return None

    for item in audio_files:
        if not isinstance(item, dict):
            continue
        if str(item.get("verse_key") or "") != verse_key:
            continue
        raw_segments = item.get("segments") if isinstance(item.get("segments"), list) else None
        segments = normalize_segments(raw_segments)
        if not segments:
            return None
        segment_shape = _detect_segment_shape(raw_segments)
        return VerseSegmentPayload(
            verse_key=verse_key,
            segments=segments,
            source_type="qcom_verse",
            segment_shape=segment_shape,
        )
    return None


def _detect_segment_shape(raw_segments: list[Any] | None) -> str:
    if not isinstance(raw_segments, list):
        return "unknown"
    for entry in raw_segments:
        if not isinstance(entry, list):
            continue
        if len(entry) >= 4:
            return "4_field"
        if len(entry) >= 3:
            return "3_field"
    return "unknown"


def fetch_best_verse_segments(
    *,
    recitation_id: int,
    chapter: int,
    verse_key: str,
) -> VerseSegmentPayload | None:
    chapter_payload = fetch_chapter_recitation_by_chapter(recitation_id, chapter, include_segments=True)
    from_chapter = extract_chapter_timestamp_segments(chapter_payload, verse_key=verse_key)
    if from_chapter is not None:
        return from_chapter

    by_chapter = fetch_verse_recitations_by_chapter(recitation_id, chapter)
    from_chapter_verse = extract_verse_segments(by_chapter, verse_key=verse_key)
    if from_chapter_verse is not None:
        return from_chapter_verse

    by_ayah = fetch_verse_recitation_by_ayah(recitation_id, verse_key)
    return extract_verse_segments(by_ayah, verse_key=verse_key)


__all__ = [
    "QCOM_API_BASE",
    "QCOM_VERSE_FILE_BASE",
    "VerseSegmentPayload",
    "fetch_recitation_catalog",
    "fetch_chapter_recitations",
    "fetch_chapter_recitation_by_chapter",
    "fetch_verse_recitations_by_chapter",
    "fetch_verse_recitation_by_ayah",
    "resolve_verse_audio_url",
    "extract_chapter_timestamp_segments",
    "extract_verse_segments",
    "fetch_best_verse_segments",
]
