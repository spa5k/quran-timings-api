from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from quran_audio_data.core.http import get_json_with_retry


DEFAULT_SOURCE_URL = (
    "https://cdn.jsdelivr.net/gh/fawazahmed0/quran-api@1/editions/ara-quranuthmanienc.json"
)


def download_source_json(
    *,
    source_url: str = DEFAULT_SOURCE_URL,
    raw_out: str | Path,
    timeout_s: float = 20.0,
    retries: int = 5,
    retry_backoff_s: float = 1.0,
) -> dict[str, Any]:
    payload = get_json_with_retry(
        url=source_url,
        timeout_s=timeout_s,
        retries=retries,
        retry_backoff_s=retry_backoff_s,
    )
    if not isinstance(payload, dict):
        raise ValueError("Source payload is not a JSON object")

    raw_path = Path(raw_out)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def transform_to_canonical(
    *,
    source_data: dict[str, Any],
    source_url: str,
) -> dict[str, Any]:
    verses = source_data.get("quran")
    if not isinstance(verses, list):
        raise ValueError("Source JSON missing 'quran' list")

    surahs: dict[str, dict[str, str]] = {}
    total_ayahs = 0

    for index, row in enumerate(verses, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid verse row at index {index}")
        chapter = row.get("chapter")
        verse = row.get("verse")
        text = row.get("text")

        if not isinstance(chapter, int) or not isinstance(verse, int) or not isinstance(text, str):
            raise ValueError(f"Malformed verse row at index {index}: {row}")

        chapter_key = str(chapter)
        verse_key = str(verse)
        surahs.setdefault(chapter_key, {})[verse_key] = text.strip()
        total_ayahs += 1

    if len(surahs) != 114:
        raise ValueError(f"Unexpected surah count: {len(surahs)} (expected 114)")
    if total_ayahs != 6236:
        raise ValueError(f"Unexpected ayah count: {total_ayahs} (expected 6236)")

    return {
        "metadata": {
            "version": "v1",
            "script": "Uthmani",
            "description": "Canonical Quran text snapshot for alignment",
            "attribution": "Source: fawazahmed0/quran-api (ara-quranuthmanienc)",
            "source_url": source_url,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "surah_count": len(surahs),
            "ayah_count": total_ayahs,
        },
        "surahs": surahs,
    }


def build_canonical_corpus(
    *,
    source_url: str = DEFAULT_SOURCE_URL,
    output: str | Path = "data/quran_text_uthmani_v1.json",
    raw_out: str | Path = "data/ara-quranuthmanienc.json",
    keep_raw: bool = False,
    timeout_s: float = 20.0,
    retries: int = 5,
    retry_backoff_s: float = 1.0,
) -> dict[str, Any]:
    output_path = Path(output)
    raw_path = Path(raw_out)

    source_data = download_source_json(
        source_url=source_url,
        raw_out=raw_path,
        timeout_s=timeout_s,
        retries=retries,
        retry_backoff_s=retry_backoff_s,
    )
    canonical = transform_to_canonical(source_data=source_data, source_url=source_url)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(canonical, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if not keep_raw and raw_path.exists():
        raw_path.unlink()

    return canonical
