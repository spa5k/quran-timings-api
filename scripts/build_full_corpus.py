#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

from quran_audio_data.corpus_builder import DEFAULT_SOURCE_URL, build_canonical_corpus


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Build canonical Quran text snapshot in project schema from ara-quranuthmanienc source"
        )
    )
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--output", default="data/quran_text_uthmani_v1.json")
    parser.add_argument("--raw-out", default="data/ara-quranuthmanienc.json")
    parser.add_argument("--keep-raw", action=BooleanOptionalAction, default=False)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--request-retries", type=int, default=5)
    parser.add_argument("--retry-backoff-s", type=float, default=1.0)
    return parser


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    canonical = build_canonical_corpus(
        source_url=args.source_url,
        output=Path(args.output),
        raw_out=Path(args.raw_out),
        keep_raw=args.keep_raw,
        timeout_s=args.timeout_s,
        retries=args.request_retries,
        retry_backoff_s=args.retry_backoff_s,
    )

    print(f"Wrote canonical corpus to {args.output}")
    print(f"Surahs: {canonical['metadata']['surah_count']}")
    print(f"Ayahs: {canonical['metadata']['ayah_count']}")


if __name__ == "__main__":
    main()
