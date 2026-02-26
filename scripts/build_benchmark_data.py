#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import orjson

from quran_audio_data.benchmark_data import prepare_benchmark_data


def _parse_csv_ints(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_strings(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = ArgumentParser(description="Build benchmark manifest + gold templates from Quran.com and EveryAyah")
    parser.add_argument("--out-dir", default="benchmarks/generated")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--surahs", default=None)
    parser.add_argument("--ayah-keys", default=None)
    parser.add_argument("--reciter-key", type=int, default=None)
    parser.add_argument("--reciter-subfolder", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-audio", action="store_true", default=False)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--request-retries", type=int, default=5)
    parser.add_argument("--retry-backoff-s", type=float, default=1.0)
    parser.add_argument("--resume", action=BooleanOptionalAction, default=True)
    parser.add_argument("--gold-split", default="benchmark")
    args = parser.parse_args()

    metadata = prepare_benchmark_data(
        out_dir=Path(args.out_dir),
        count=args.count,
        surahs=_parse_csv_ints(args.surahs),
        ayah_keys=_parse_csv_strings(args.ayah_keys),
        reciter_key=args.reciter_key,
        reciter_subfolder=args.reciter_subfolder,
        seed=args.seed,
        download_audio=args.download_audio,
        timeout_s=args.timeout_s,
        request_retries=args.request_retries,
        retry_backoff_s=args.retry_backoff_s,
        resume=args.resume,
        gold_split=args.gold_split,
    )
    print(orjson.dumps(metadata, option=orjson.OPT_INDENT_2).decode("utf-8"))


if __name__ == "__main__":
    main()
