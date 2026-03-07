from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
from typing import Any

from quran_audio_data.surah_runner import run_surah_for_reciter
from quran_audio_data.ui_sync import sync_ui_from_latest_runs


@dataclass(slots=True)
class BatchResult:
    reciter_id: str
    surah: int
    success: bool
    summary_path: str | None
    error: str | None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a surah-range batch for reciters that already exist in the UI catalog, "
            "then sync new outputs into ui/public/data and ui/dist/data."
        )
    )
    parser.add_argument("--surah-start", type=int, default=55)
    parser.add_argument("--surah-end", type=int, default=84)
    parser.add_argument("--ui-catalog-in", type=Path, default=Path("ui/public/data/catalog.json"))
    parser.add_argument(
        "--reciters",
        type=str,
        default="",
        help="Optional comma-separated list to restrict to specific reciter ids (must exist in UI catalog).",
    )
    parser.add_argument("--catalog", type=Path, default=Path("data/reciter_catalog.json"))
    parser.add_argument("--out-root", type=Path, default=Path("runs/s055_084_ui"))
    parser.add_argument("--runs-root-for-sync", type=Path, default=Path("runs"))
    parser.add_argument("--ui-data-dir", type=Path, default=Path("ui/public/data"))
    parser.add_argument("--ui-catalog", type=Path, default=Path("ui/public/data/catalog.json"))
    parser.add_argument("--dist-data-dir", type=Path, default=Path("ui/dist/data"))
    parser.add_argument(
        "--sync-every", type=int, default=25, help="Sync UI every N completed tasks."
    )
    parser.add_argument("--text-data", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/timings/v3"))
    parser.add_argument("--report", type=Path, default=Path("runs/s055_084_ui/report.json"))
    args = parser.parse_args()

    if args.surah_start < 1 or args.surah_end > 114 or args.surah_start > args.surah_end:
        raise SystemExit(f"invalid surah range: {args.surah_start}-{args.surah_end}")

    selected_reciters = _load_ui_reciters(args.ui_catalog_in)
    if not selected_reciters:
        raise SystemExit(f"no reciters found in ui catalog: {args.ui_catalog_in}")

    if args.reciters.strip():
        requested = {item.strip().lower() for item in args.reciters.split(",") if item.strip()}
        selected_reciters = [rid for rid in selected_reciters if rid in requested]
        if not selected_reciters:
            raise SystemExit(f"no requested reciters found in ui catalog: {sorted(requested)}")

    catalog = _load_json(args.catalog)
    enabled, missing = _enable_reciters_in_catalog(catalog, selected_reciters)
    if missing:
        print(f"[batch] warning: reciters missing from reciter_catalog.json: {len(missing)}")
        for rid in missing[:20]:
            print(f"  - {rid}")
        if len(missing) > 20:
            print("  ...")
    if enabled:
        _save_json(args.catalog, catalog)

    surahs = list(range(args.surah_start, args.surah_end + 1))
    tasks = [(reciter_id, surah) for reciter_id in selected_reciters for surah in surahs]
    total = len(tasks)
    print(f"[batch] reciters={len(selected_reciters)} surahs={len(surahs)} tasks={total}")
    print(f"[batch] out_root={args.out_root}")
    print(f"[batch] sync_runs_root={args.runs_root_for_sync}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    existing_full_json = {p.name for p in args.runs_root_for_sync.rglob("*_full.json")}

    results: list[BatchResult] = []
    completed = 0
    failures = 0

    for reciter_id, surah in tasks:
        completed += 1
        full_name = f"{reciter_id}_s{surah:03d}_full.json"
        if full_name in existing_full_json:
            print(f"[{completed}/{total}] skip {reciter_id} surah={surah} (already exists in runs)")
            results.append(
                BatchResult(
                    reciter_id=reciter_id,
                    surah=surah,
                    success=True,
                    summary_path=None,
                    error=None,
                )
            )
            _write_report(args.report, selected_reciters=selected_reciters, results=results)
            if args.sync_every > 0 and completed % args.sync_every == 0:
                _sync_ui(
                    runs_root=args.runs_root_for_sync,
                    ui_data_dir=args.ui_data_dir,
                    ui_catalog=args.ui_catalog,
                    dist_data_dir=args.dist_data_dir,
                    catalog=args.catalog,
                )
            continue

        print(f"[{completed}/{total}] run {reciter_id} surah={surah}")
        try:
            summary = run_surah_for_reciter(
                reciter_id=reciter_id,
                surah=surah,
                out_root=args.out_root,
                text_data=args.text_data,
                cache_dir=args.cache_dir,
                catalog_path=args.catalog,
            )
            paths = summary.get("paths") if isinstance(summary.get("paths"), dict) else {}
            summary_path = str(paths.get("summary_path")) if paths.get("summary_path") else None
            priors_used = (summary.get("pipeline") or {}).get("priors_used")
            print(f"  -> ok priors_used={priors_used} summary={summary_path}")
            results.append(
                BatchResult(
                    reciter_id=reciter_id,
                    surah=surah,
                    success=True,
                    summary_path=summary_path,
                    error=None,
                )
            )
            existing_full_json.add(full_name)
        except Exception as exc:
            failures += 1
            err = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            print(f"  -> failed: {err}")
            results.append(
                BatchResult(
                    reciter_id=reciter_id,
                    surah=surah,
                    success=False,
                    summary_path=None,
                    error=err,
                )
            )

        _write_report(args.report, selected_reciters=selected_reciters, results=results)
        if args.sync_every > 0 and completed % args.sync_every == 0:
            _sync_ui(
                runs_root=args.runs_root_for_sync,
                ui_data_dir=args.ui_data_dir,
                ui_catalog=args.ui_catalog,
                dist_data_dir=args.dist_data_dir,
                catalog=args.catalog,
            )

    _sync_ui(
        runs_root=args.runs_root_for_sync,
        ui_data_dir=args.ui_data_dir,
        ui_catalog=args.ui_catalog,
        dist_data_dir=args.dist_data_dir,
        catalog=args.catalog,
    )

    _write_report(args.report, selected_reciters=selected_reciters, results=results, done=True)
    print(f"[batch] finished total={total} failures={failures}")
    return 0 if failures == 0 else 1


def _sync_ui(
    *,
    runs_root: Path,
    ui_data_dir: Path,
    ui_catalog: Path,
    dist_data_dir: Path,
    catalog: Path,
) -> None:
    summary = sync_ui_from_latest_runs(
        runs_root=runs_root,
        ui_data_dir=ui_data_dir,
        catalog_path=ui_catalog,
        dist_data_dir=dist_data_dir,
        sync_dist=True,
        prune_ui=True,
        dry_run=False,
        reciter_catalog_path=catalog,
        prune_catalog_surahs=True,
    )
    print(
        "[sync]",
        f"keys={summary['keys_selected']}",
        f"ui_copied={summary['ui']['copied']}",
        f"ui_audio_copied={summary['ui']['audio_copied']}",
    )


def _write_report(
    path: Path,
    *,
    selected_reciters: list[str],
    results: list[BatchResult],
    done: bool = False,
) -> None:
    successes = sum(1 for item in results if item.success)
    failures = sum(1 for item in results if not item.success)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "done": done,
        "ui_catalog_reciters": selected_reciters,
        "totals": {
            "tasks": len(results),
            "successes": successes,
            "failures": failures,
        },
        "results": [
            {
                "reciter_id": item.reciter_id,
                "surah": item.surah,
                "success": item.success,
                "summary_path": item.summary_path,
                "error": item.error,
            }
            for item in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_ui_reciters(ui_catalog_path: Path) -> list[str]:
    payload = _load_json(ui_catalog_path)
    recitations = payload.get("recitations")
    if not isinstance(recitations, list):
        return []
    out: list[str] = []
    for item in recitations:
        if not isinstance(item, dict):
            continue
        rid = item.get("id")
        if isinstance(rid, str) and rid.strip():
            out.append(rid.strip().lower())
    return sorted(set(out))


def _enable_reciters_in_catalog(
    catalog: dict[str, Any], reciter_ids: list[str]
) -> tuple[bool, list[str]]:
    configured = catalog.get("configured_reciters")
    if not isinstance(configured, list):
        raise ValueError("reciter catalog missing configured_reciters")

    configured_by_id: dict[str, dict[str, Any]] = {}
    for item in configured:
        if not isinstance(item, dict):
            continue
        rid = str(item.get("manifest_reciter_id") or "").strip().lower()
        if rid:
            configured_by_id[rid] = item

    changed = False
    missing: list[str] = []
    for rid in reciter_ids:
        entry = configured_by_id.get(rid)
        if entry is None:
            missing.append(rid)
            continue
        if not bool(entry.get("enabled")):
            entry["enabled"] = True
            changed = True

    enabled_ids = sorted(
        {rid for rid, entry in configured_by_id.items() if bool(entry.get("enabled"))}
    )
    if catalog.get("enabled_manifest_reciters") != enabled_ids:
        catalog["enabled_manifest_reciters"] = enabled_ids
        changed = True

    counts = catalog.get("counts")
    if not isinstance(counts, dict):
        counts = {}
        catalog["counts"] = counts
        changed = True
    counts["configured_reciters"] = len(configured)
    counts["configured_enabled"] = sum(
        1 for item in configured if isinstance(item, dict) and bool(item.get("enabled"))
    )

    return changed, missing


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
