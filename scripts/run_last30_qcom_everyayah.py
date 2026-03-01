from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import traceback

from quran_audio_data.surah_runner import run_surah_for_reciter
from quran_audio_data.ui_sync import sync_ui_from_latest_runs


QCOM_RECITERS: list[tuple[str, int]] = [
    ("qcom_abdulbaset_abdulsamad_murattal", 2),
    ("qcom_abdurrahman_as_sudais", 3),
    ("qcom_abu_bakr_al_shatri", 4),
    ("qcom_hani_ar_rifai", 5),
    ("qcom_husary", 6),
    ("qcom_mishari_alafasy", 7),
    ("qcom_minshawi_murattal", 8),
    ("qcom_saud_ash_shuraym", 10),
    ("qcom_mohamed_al_tablawi", 11),
    ("qcom_husary_muallim", 12),
]


EVERYAYAH_RECITERS: list[tuple[str, str]] = [
    ("eya_maher_al_muaiqly", "MaherAlMuaiqly128kbps"),
    ("eya_minshawy_murattal", "Minshawy_Murattal_128kbps"),
    ("eya_muhammad_ayyoub", "Muhammad_Ayyoub_128kbps"),
    ("eya_muhammad_jibreel", "Muhammad_Jibreel_128kbps"),
    ("eya_salaah_bukhatir", "Salaah_AbdulRahman_Bukhatir_128kbps"),
    ("eya_salah_al_budair", "Salah_Al_Budair_128kbps"),
    ("eya_abdullah_matroud", "Abdullah_Matroud_128kbps"),
    ("eya_ahmed_neana", "Ahmed_Neana_128kbps"),
    ("eya_nasser_alqatami", "Nasser_Alqatami_128kbps"),
    ("eya_yaser_salamah", "Yaser_Salamah_128kbps"),
]


SURAHS_LAST_30 = list(range(85, 115))


@dataclass(slots=True)
class BatchResult:
    reciter_id: str
    surah: int
    success: bool
    summary_path: str | None
    error: str | None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run last-30-surah batch for 10 Quran.com + 10 EveryAyah reciters.")
    parser.add_argument("--catalog", type=Path, default=Path("data/reciter_catalog.json"))
    parser.add_argument("--out-root", type=Path, default=Path("runs/last30_qcom_everyayah"))
    parser.add_argument("--ui-data-dir", type=Path, default=Path("ui/public/data"))
    parser.add_argument("--ui-catalog", type=Path, default=Path("ui/public/data/catalog.json"))
    parser.add_argument("--dist-data-dir", type=Path, default=Path("ui/dist/data"))
    parser.add_argument("--sync-every", type=int, default=25, help="Sync UI every N completed items.")
    parser.add_argument("--text-data", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument("--report", type=Path, default=Path("runs/last30_qcom_everyayah/report.json"))
    args = parser.parse_args()

    catalog = _load_catalog(args.catalog)
    _upsert_selected_reciters(catalog)
    _save_catalog(args.catalog, catalog)

    selected_reciters = [item[0] for item in QCOM_RECITERS] + [item[0] for item in EVERYAYAH_RECITERS]
    tasks = [(reciter_id, surah) for reciter_id in selected_reciters for surah in SURAHS_LAST_30]
    total = len(tasks)
    print(f"[batch] reciters={len(selected_reciters)} surahs={len(SURAHS_LAST_30)} tasks={total}")
    print(f"[batch] out_root={args.out_root}")

    results: list[BatchResult] = []
    completed = 0
    failures = 0
    args.out_root.mkdir(parents=True, exist_ok=True)

    for reciter_id, surah in tasks:
        completed += 1
        existing_summary = args.out_root / f"{reciter_id}_s{surah:03d}" / "run_summary.json"
        if existing_summary.exists():
            print(f"[{completed}/{total}] skip {reciter_id} surah={surah} (already exists)")
            results.append(
                BatchResult(
                    reciter_id=reciter_id,
                    surah=surah,
                    success=True,
                    summary_path=str(existing_summary),
                    error=None,
                )
            )
            _write_report(args.report, selected_reciters=selected_reciters, results=results)
            if args.sync_every > 0 and completed % args.sync_every == 0:
                _sync_ui(
                    runs_root=args.out_root,
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
                runs_root=args.out_root,
                ui_data_dir=args.ui_data_dir,
                ui_catalog=args.ui_catalog,
                dist_data_dir=args.dist_data_dir,
                catalog=args.catalog,
            )

    _sync_ui(
        runs_root=args.out_root,
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
        "selected_reciters": selected_reciters,
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


def _load_catalog(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"catalog missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_catalog(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _upsert_selected_reciters(catalog: dict[str, Any]) -> None:
    everyayah_rows = catalog.get("everyayah_reciters")
    qcom_rows = catalog.get("quran_com_reciters")
    if not isinstance(everyayah_rows, list) or not isinstance(qcom_rows, list):
        raise ValueError("catalog missing everyayah_reciters or quran_com_reciters")

    by_subfolder = {
        str(item.get("subfolder")): item
        for item in everyayah_rows
        if isinstance(item, dict) and isinstance(item.get("subfolder"), str)
    }
    by_qcom_id = {
        int(item.get("id")): item
        for item in qcom_rows
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    }

    configured = catalog.get("configured_reciters")
    if not isinstance(configured, list):
        configured = []
        catalog["configured_reciters"] = configured

    configured_by_id: dict[str, dict[str, Any]] = {}
    for item in configured:
        if isinstance(item, dict):
            reciter_id = str(item.get("manifest_reciter_id") or "").strip().lower()
            if reciter_id:
                configured_by_id[reciter_id] = item

    selected_ids: set[str] = set()
    for manifest_id, qcom_id in QCOM_RECITERS:
        selected_ids.add(manifest_id)
        qcom_meta = by_qcom_id.get(qcom_id, {})
        display_name = (
            str(qcom_meta.get("translated_name") or "").strip()
            or str(qcom_meta.get("reciter_name") or "").strip()
            or f"Quran.com reciter {qcom_id}"
        )
        entry = configured_by_id.get(manifest_id, {})
        entry.update(
            {
                "manifest_reciter_id": manifest_id,
                "enabled": True,
                "check_type": "word_by_word",
                "checks": {"ayah_by_ayah": False, "word_by_word": True},
                "sources": {"everyayah": False, "quran_com": True},
                "everyayah": {"subfolder": None, "reciter_key": None, "name": None},
                "quran_com": {"recitation_id": qcom_id, "name": display_name},
                "qcom_word_supervision_supported": True,
            }
        )
        configured_by_id[manifest_id] = entry

    for manifest_id, subfolder in EVERYAYAH_RECITERS:
        selected_ids.add(manifest_id)
        eya_meta = by_subfolder.get(subfolder, {})
        display_name = str(eya_meta.get("name") or subfolder).strip()
        reciter_key = eya_meta.get("reciter_key") if isinstance(eya_meta.get("reciter_key"), int) else None
        entry = configured_by_id.get(manifest_id, {})
        entry.update(
            {
                "manifest_reciter_id": manifest_id,
                "enabled": True,
                "check_type": "ayah_by_ayah",
                "checks": {"ayah_by_ayah": True, "word_by_word": False},
                "sources": {"everyayah": True, "quran_com": False},
                "everyayah": {"subfolder": subfolder, "reciter_key": reciter_key, "name": display_name},
                "quran_com": {"recitation_id": None, "name": None},
                "qcom_word_supervision_supported": False,
            }
        )
        configured_by_id[manifest_id] = entry

    # Keep previous configured entries, but ensure selected set is enabled.
    configured_out = sorted(configured_by_id.values(), key=lambda item: str(item.get("manifest_reciter_id") or ""))
    catalog["configured_reciters"] = configured_out

    enabled = {
        str(item.get("manifest_reciter_id") or "").strip()
        for item in configured_out
        if bool(item.get("enabled"))
    }
    catalog["enabled_manifest_reciters"] = sorted(value for value in enabled if value)

    counts = catalog.get("counts")
    if not isinstance(counts, dict):
        counts = {}
        catalog["counts"] = counts
    counts["configured_reciters"] = len(configured_out)
    counts["configured_enabled"] = sum(1 for item in configured_out if bool(item.get("enabled")))


if __name__ == "__main__":
    raise SystemExit(main())
