from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import orjson
import requests
from requests.exceptions import RequestException


QURAN_COM_CHAPTER_RECITATION_URL = "https://api.quran.com/api/v4/chapter_recitations/{reciter_id}/{surah}"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass(slots=True)
class AutoLabelSummary:
    files_total: int
    files_updated: int
    files_skipped_already_labeled: int
    files_missing_segments: int
    words_labeled: int
    words_missing_segments: int
    errors: list[str]

    @property
    def passes(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_total": self.files_total,
            "files_updated": self.files_updated,
            "files_skipped_already_labeled": self.files_skipped_already_labeled,
            "files_missing_segments": self.files_missing_segments,
            "words_labeled": self.words_labeled,
            "words_missing_segments": self.words_missing_segments,
            "errors": self.errors,
            "passes": self.passes,
        }


@dataclass(slots=True)
class VerseSegments:
    word_map: dict[int, tuple[float, float]]
    verse_start_s: float | None
    verse_end_s: float | None


def _sleep_for_retry(*, attempt: int, base_seconds: float) -> None:
    sleep_s = min(base_seconds * (2**attempt), 30.0)
    if sleep_s > 0:
        time.sleep(sleep_s)


def _http_get_with_retry(
    *,
    url: str,
    timeout_s: float,
    params: dict[str, str] | None = None,
    retries: int = 5,
    retry_backoff_s: float = 1.0,
) -> requests.Response:
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout_s)
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < retries:
                _sleep_for_retry(attempt=attempt, base_seconds=retry_backoff_s)
                continue
            response.raise_for_status()
            return response
        except RequestException:
            if attempt >= retries:
                raise
            _sleep_for_retry(attempt=attempt, base_seconds=retry_backoff_s)
    raise RuntimeError("HTTP retries exhausted")


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_labeled(words: list[dict[str, Any]]) -> bool:
    if not words:
        return False
    for word in words:
        if not isinstance(word, dict):
            return False
        if word.get("start_s") is None or word.get("end_s") is None:
            return False
    return True


def _fill_missing_word_spans(
    *,
    words: list[dict[str, Any]],
    verse_segments: VerseSegments,
) -> tuple[int, int]:
    # Returns (filled_count, missing_count_after_fill).
    labeled_intervals: list[tuple[float | None, float | None]] = []
    for idx, word in enumerate(words, start=1):
        if not isinstance(word, dict):
            labeled_intervals.append((None, None))
            continue
        existing_start = _as_float(word.get("start_s"))
        existing_end = _as_float(word.get("end_s"))
        if existing_start is not None and existing_end is not None and existing_end > existing_start:
            labeled_intervals.append((existing_start, existing_end))
            continue
        position = _as_int(word.get("word_index_in_ayah")) or idx
        segment = verse_segments.word_map.get(position)
        if segment is not None:
            labeled_intervals.append(segment)
        else:
            labeled_intervals.append((None, None))

    known_starts = [span[0] for span in labeled_intervals if span[0] is not None]
    known_ends = [span[1] for span in labeled_intervals if span[1] is not None]
    fallback_start = verse_segments.verse_start_s
    fallback_end = verse_segments.verse_end_s
    if fallback_start is None and known_starts:
        fallback_start = float(min(known_starts))
    if fallback_end is None and known_ends:
        fallback_end = float(max(known_ends))

    run_start = -1
    for idx, interval in enumerate(labeled_intervals + [(0.0, 0.0)]):
        missing = interval[0] is None or interval[1] is None
        if missing and run_start < 0:
            run_start = idx
            continue
        if missing or run_start < 0:
            continue

        run_end = idx - 1
        run_len = run_end - run_start + 1

        left_end = labeled_intervals[run_start - 1][1] if run_start > 0 else None
        right_start = labeled_intervals[idx][0] if idx < len(labeled_intervals) else None

        anchor_start = left_end if left_end is not None else fallback_start
        anchor_end = right_start if right_start is not None else fallback_end

        if anchor_start is None and anchor_end is None:
            anchor_start = 0.0
            anchor_end = max(0.05 * run_len, 0.05)
        elif anchor_start is None and anchor_end is not None:
            anchor_start = max(0.0, anchor_end - max(0.05 * run_len, 0.05))
        elif anchor_start is not None and anchor_end is None:
            anchor_end = anchor_start + max(0.05 * run_len, 0.05)

        assert anchor_start is not None
        assert anchor_end is not None
        if anchor_end <= anchor_start:
            anchor_end = anchor_start + max(0.01 * run_len, 0.01)

        slot = (anchor_end - anchor_start) / run_len
        for offset in range(run_len):
            start_s = anchor_start + slot * offset
            end_s = anchor_start + slot * (offset + 1)
            if end_s <= start_s:
                end_s = start_s + 0.001
            labeled_intervals[run_start + offset] = (start_s, end_s)

        run_start = -1

    filled_count = 0
    missing_count = 0
    prev_end = 0.0
    for idx, word in enumerate(words):
        if not isinstance(word, dict):
            continue
        start_s, end_s = labeled_intervals[idx]
        if start_s is None or end_s is None:
            missing_count += 1
            continue
        if start_s < prev_end:
            start_s = prev_end
        if end_s <= start_s:
            end_s = start_s + 0.001
        word["start_s"] = round(start_s, 3)
        word["end_s"] = round(end_s, 3)
        prev_end = float(word["end_s"])
        filled_count += 1

    return filled_count, missing_count


def _fetch_surah_segments(
    *,
    chapter_reciter_id: int,
    surah: int,
    timeout_s: float,
    request_retries: int,
    retry_backoff_s: float,
) -> dict[int, VerseSegments]:
    url = QURAN_COM_CHAPTER_RECITATION_URL.format(reciter_id=chapter_reciter_id, surah=surah)
    response = _http_get_with_retry(
        url=url,
        params={"segments": "true"},
        timeout_s=timeout_s,
        retries=request_retries,
        retry_backoff_s=retry_backoff_s,
    )
    payload = response.json()
    if not isinstance(payload, dict):
        return {}
    audio_file = payload.get("audio_file")
    if not isinstance(audio_file, dict):
        return {}

    timestamps = audio_file.get("timestamps")
    if not isinstance(timestamps, list):
        return {}

    out: dict[int, VerseSegments] = {}
    for timestamp in timestamps:
        if not isinstance(timestamp, dict):
            continue
        verse_key = timestamp.get("verse_key")
        if not isinstance(verse_key, str) or ":" not in verse_key:
            continue
        surah_str, ayah_str = verse_key.split(":", 1)
        ayah = _as_int(ayah_str)
        parsed_surah = _as_int(surah_str)
        if ayah is None or parsed_surah is None or parsed_surah != surah:
            continue

        timestamp_from = timestamp.get("timestamp_from")
        timestamp_to = timestamp.get("timestamp_to")
        verse_anchor_from_ms = 0.0
        if timestamp_from is not None:
            try:
                verse_anchor_from_ms = float(timestamp_from)
            except (TypeError, ValueError):
                verse_anchor_from_ms = 0.0

        segments = timestamp.get("segments")
        if not isinstance(segments, list):
            continue

        word_map: dict[int, tuple[float, float]] = {}
        for segment in segments:
            if not isinstance(segment, list) or len(segment) < 3:
                continue
            position = _as_int(segment[0])
            try:
                start_ms = float(segment[1])
                end_ms = float(segment[2])
            except (TypeError, ValueError):
                continue
            if position is None:
                continue
            # Quran.com chapter recitation segments are chapter-absolute ms.
            # Convert to ayah-relative seconds for ayah-level audio files.
            start_s = max(0.0, (start_ms - verse_anchor_from_ms) / 1000.0)
            end_s = max(0.0, (end_ms - verse_anchor_from_ms) / 1000.0)
            if end_s <= start_s:
                end_s = start_s + 0.001
            word_map[position] = (start_s, end_s)

        verse_start_s: float | None = 0.0
        verse_end_s: float | None = None
        if timestamp_to is not None:
            try:
                raw_to = float(timestamp_to)
                verse_end_s = max(0.0, (raw_to - verse_anchor_from_ms) / 1000.0)
            except (TypeError, ValueError):
                verse_end_s = None

        out[ayah] = VerseSegments(
            word_map=word_map,
            verse_start_s=verse_start_s,
            verse_end_s=verse_end_s,
        )

    return out


def auto_label_gold_from_quran_com(
    *,
    gold_dir: str | Path,
    chapter_reciter_id: int,
    overwrite_existing: bool = False,
    timeout_s: float = 20.0,
    request_retries: int = 5,
    retry_backoff_s: float = 1.0,
    max_errors: int = 100,
) -> AutoLabelSummary:
    root = Path(gold_dir)
    files = sorted(root.rglob("*.json")) if root.is_dir() else [root]

    files_updated = 0
    files_skipped_already_labeled = 0
    files_missing_segments = 0
    words_labeled = 0
    words_missing_segments = 0
    errors: list[str] = []

    surah_cache: dict[int, dict[int, VerseSegments]] = {}

    for file_path in files:
        try:
            payload = orjson.loads(file_path.read_bytes())
        except Exception as exc:
            if len(errors) < max_errors:
                errors.append(f"{file_path}: parse error: {exc}")
            continue

        if not isinstance(payload, dict):
            if len(errors) < max_errors:
                errors.append(f"{file_path}: payload is not an object")
            continue

        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        surah = _as_int(meta.get("surah") or payload.get("surah"))
        ayah = _as_int(meta.get("ayah") or payload.get("ayah"))
        words = payload.get("words")

        if surah is None or ayah is None:
            if len(errors) < max_errors:
                errors.append(f"{file_path}: missing surah/ayah in meta")
            continue
        if not isinstance(words, list) or not words:
            if len(errors) < max_errors:
                errors.append(f"{file_path}: missing or empty words list")
            continue

        if not overwrite_existing and _is_labeled(words):
            files_skipped_already_labeled += 1
            continue

        if surah not in surah_cache:
            try:
                surah_cache[surah] = _fetch_surah_segments(
                    chapter_reciter_id=chapter_reciter_id,
                    surah=surah,
                    timeout_s=timeout_s,
                    request_retries=request_retries,
                    retry_backoff_s=retry_backoff_s,
                )
            except Exception as exc:
                if len(errors) < max_errors:
                    errors.append(f"{file_path}: failed to fetch Quran.com segments for surah {surah}: {exc}")
                continue

        ayah_segments = surah_cache[surah].get(ayah)
        if not ayah_segments:
            files_missing_segments += 1
            if len(errors) < max_errors:
                errors.append(
                    f"{file_path}: no segments found for chapter_reciter_id={chapter_reciter_id} verse={surah}:{ayah}"
                )
            continue

        before_labeled = _is_labeled(words)
        missing_before = 0
        for word in words:
            if not isinstance(word, dict):
                continue
            if _as_float(word.get("start_s")) is None or _as_float(word.get("end_s")) is None:
                missing_before += 1
        if overwrite_existing:
            for word in words:
                if isinstance(word, dict):
                    word["start_s"] = None
                    word["end_s"] = None
            missing_before = sum(1 for word in words if isinstance(word, dict))

        _, remaining_missing = _fill_missing_word_spans(
            words=words,
            verse_segments=ayah_segments,
        )
        words_labeled += max(0, missing_before - remaining_missing)
        words_missing_segments += remaining_missing
        file_changed = overwrite_existing or not before_labeled

        if file_changed:
            file_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
            files_updated += 1

    return AutoLabelSummary(
        files_total=len(files),
        files_updated=files_updated,
        files_skipped_already_labeled=files_skipped_already_labeled,
        files_missing_segments=files_missing_segments,
        words_labeled=words_labeled,
        words_missing_segments=words_missing_segments,
        errors=errors,
    )
