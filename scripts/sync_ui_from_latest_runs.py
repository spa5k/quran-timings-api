#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from pathlib import Path
import json

from quran_audio_data.ui_sync import sync_ui_from_latest_runs


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Sync UI timing data from runs by selecting the newest run artifact "
            "per reciter+surah key."
        )
    )
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--ui-data-dir", default="ui/public/data")
    parser.add_argument("--catalog", default="ui/public/data/catalog.json")
    parser.add_argument("--dist-data-dir", default="ui/dist/data")
    parser.add_argument("--sync-dist", action=BooleanOptionalAction, default=True)
    parser.add_argument("--prune-ui", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = sync_ui_from_latest_runs(
        runs_root=Path(args.runs_root).resolve(),
        ui_data_dir=Path(args.ui_data_dir).resolve(),
        catalog_path=Path(args.catalog).resolve(),
        dist_data_dir=Path(args.dist_data_dir).resolve(),
        sync_dist=args.sync_dist,
        prune_ui=args.prune_ui,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
