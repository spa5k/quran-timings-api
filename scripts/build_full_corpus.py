#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, UTC
import json
from pathlib import Path
from urllib.request import urlopen

DEFAULT_SOURCE_URL = (
    "https://cdn.jsdelivr.net/gh/fawazahmed0/quran-api@1/editions/ara-quranuthmanienc.json"
)


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Build canonical Quran text snapshot in project schema from "
            "ara-quranuthmanienc source"
        )
    )
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--output", default="data/quran_text_uthmani_v1.json")
    parser.add_argument("--raw-out", default="data/ara-quranuthmanienc.json")
    parser.add_argument("--keep-raw", action="store_true")
    return parser


def download_json(source_url: str, raw_out: Path) -> dict:
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(source_url) as response:
        payload = response.read().decode("utf-8")
    raw_out.write_text(payload, encoding="utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("Source payload is not a JSON object")
    return data


def transform_to_canonical(source_data: dict, source_url: str) -> dict:
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


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    output_path = Path(args.output)
    raw_path = Path(args.raw_out)

    source_data = download_json(args.source_url, raw_path)
    canonical = transform_to_canonical(source_data, args.source_url)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(canonical, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if not args.keep_raw and raw_path.exists():
        raw_path.unlink()

    print(f"Wrote canonical corpus to {output_path}")
    print(f"Surahs: {canonical['metadata']['surah_count']}")
    print(f"Ayahs: {canonical['metadata']['ayah_count']}")


if __name__ == "__main__":
    main()
