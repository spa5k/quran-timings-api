from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class WordSegment:
    word_index: int
    start_ms: float
    end_ms: float


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


def normalize_segments(raw_segments: list[Any] | None) -> list[WordSegment]:
    """Normalize Quran.com segment shapes to canonical word segments.

    Supported shapes:
    - [word_position, start_ms, end_ms]
    - [segment_index, word_position, start_ms, end_ms]
    """

    if not isinstance(raw_segments, list):
        return []

    normalized: list[WordSegment] = []
    for entry in raw_segments:
        if not isinstance(entry, list):
            continue

        word_index: int | None = None
        start_ms: float | None = None
        end_ms: float | None = None

        if len(entry) >= 4:
            word_index = _to_int(entry[1])
            start_ms = _to_float(entry[2])
            end_ms = _to_float(entry[3])
        elif len(entry) >= 3:
            word_index = _to_int(entry[0])
            start_ms = _to_float(entry[1])
            end_ms = _to_float(entry[2])

        if word_index is None or start_ms is None or end_ms is None:
            continue
        if end_ms < start_ms:
            end_ms = start_ms

        normalized.append(
            WordSegment(
                word_index=word_index,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )

    normalized.sort(key=lambda item: (item.word_index, item.start_ms, item.end_ms))

    dedup: dict[int, WordSegment] = {}
    for item in normalized:
        current = dedup.get(item.word_index)
        if current is None:
            dedup[item.word_index] = item
            continue
        if item.end_ms - item.start_ms > current.end_ms - current.start_ms:
            dedup[item.word_index] = item

    return [dedup[idx] for idx in sorted(dedup)]


__all__ = ["WordSegment", "normalize_segments"]
