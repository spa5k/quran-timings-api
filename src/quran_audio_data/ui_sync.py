from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import filecmp
import json
import re
import shutil


FULL_JSON_RE = re.compile(r"^(?P<reciter_id>.+)_s(?P<surah>\d{3})_full\.json$")


@dataclass(frozen=True, slots=True)
class RunCandidate:
    reciter_id: str
    surah: int
    path: Path
    mtime_ns: int


def _safe_surah_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def discover_latest_candidates(runs_root: Path) -> dict[tuple[str, int], RunCandidate]:
    latest: dict[tuple[str, int], RunCandidate] = {}
    for path in runs_root.rglob("*_full.json"):
        match = FULL_JSON_RE.fullmatch(path.name)
        if match is None:
            continue
        reciter_id = match.group("reciter_id")
        surah = int(match.group("surah"))
        mtime_ns = path.stat().st_mtime_ns
        candidate = RunCandidate(reciter_id=reciter_id, surah=surah, path=path, mtime_ns=mtime_ns)
        key = (reciter_id, surah)
        current = latest.get(key)
        if current is None or candidate.mtime_ns > current.mtime_ns:
            latest[key] = candidate
            continue
        if candidate.mtime_ns == current.mtime_ns and str(candidate.path) > str(current.path):
            latest[key] = candidate
    return latest


def needs_copy(source: Path, target: Path) -> bool:
    if not target.exists():
        return True
    return not filecmp.cmp(source, target, shallow=False)


def sync_data_files(
    latest_candidates: dict[tuple[str, int], RunCandidate],
    target_dir: Path,
    dry_run: bool,
    prune: bool,
) -> tuple[int, int, int]:
    copied = 0
    unchanged = 0
    pruned = 0
    target_dir.mkdir(parents=True, exist_ok=True)

    selected_names = {candidate.path.name for candidate in latest_candidates.values()}
    for candidate in sorted(latest_candidates.values(), key=lambda item: (item.reciter_id, item.surah)):
        target = target_dir / candidate.path.name
        if needs_copy(candidate.path, target):
            copied += 1
            if not dry_run:
                shutil.copy2(candidate.path, target)
        else:
            unchanged += 1

    if not prune:
        return copied, unchanged, pruned

    for existing in target_dir.glob("*_full.json"):
        if existing.name in selected_names:
            continue
        pruned += 1
        if not dry_run:
            existing.unlink(missing_ok=True)

    return copied, unchanged, pruned


def update_catalog(
    catalog_path: Path,
    latest_candidates: dict[tuple[str, int], RunCandidate],
    dry_run: bool,
) -> tuple[bool, int, int, int]:
    raw = catalog_path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    recitations = payload.get("recitations")
    if not isinstance(recitations, list):
        raise ValueError("catalog.json missing recitations list")

    recitation_by_id: dict[str, dict[str, Any]] = {}
    for item in recitations:
        if isinstance(item, dict):
            reciter_id = str(item.get("id", "")).strip()
            if reciter_id:
                recitation_by_id[reciter_id] = item

    added = 0
    updated = 0
    skipped = 0

    for candidate in sorted(latest_candidates.values(), key=lambda item: (item.reciter_id, item.surah)):
        recitation = recitation_by_id.get(candidate.reciter_id)
        if recitation is None:
            skipped += 1
            continue

        surahs = recitation.get("surahs")
        if not isinstance(surahs, list):
            surahs = []
            recitation["surahs"] = surahs

        timing_json = f"/data/{candidate.path.name}"
        audio_src = f"https://download.quranicaudio.com/quran/{candidate.reciter_id}/{candidate.surah:03d}.mp3"

        existing_entry: dict[str, Any] | None = None
        for entry in surahs:
            if not isinstance(entry, dict):
                continue
            entry_surah = _safe_surah_int(entry.get("surah"))
            if entry_surah == candidate.surah:
                existing_entry = entry
                break

        if existing_entry is None:
            surahs.append(
                {
                    "surah": candidate.surah,
                    "title": f"Surah {candidate.surah}",
                    "audioSrc": audio_src,
                    "timingJson": timing_json,
                }
            )
            added += 1
            continue

        changed = False
        if existing_entry.get("timingJson") != timing_json:
            existing_entry["timingJson"] = timing_json
            changed = True
        if existing_entry.get("audioSrc") != audio_src:
            existing_entry["audioSrc"] = audio_src
            changed = True
        if not str(existing_entry.get("title", "")).strip():
            existing_entry["title"] = f"Surah {candidate.surah}"
            changed = True
        if changed:
            updated += 1

    for recitation in recitation_by_id.values():
        surahs = recitation.get("surahs")
        if not isinstance(surahs, list):
            continue
        surahs.sort(
            key=lambda entry: (
                _safe_surah_int(entry.get("surah"))
                if isinstance(entry, dict)
                else None
            )
            or 0
        )

    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    changed = rendered != raw
    if changed and not dry_run:
        catalog_path.write_text(rendered, encoding="utf-8")

    return changed, added, updated, skipped


def sync_ui_from_latest_runs(
    runs_root: Path,
    ui_data_dir: Path,
    catalog_path: Path,
    dist_data_dir: Path,
    *,
    sync_dist: bool = True,
    prune_ui: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    latest_candidates = discover_latest_candidates(runs_root=runs_root)
    if not latest_candidates:
        raise RuntimeError(f"No *_full.json files found under {runs_root}")

    ui_copied, ui_unchanged, ui_pruned = sync_data_files(
        latest_candidates=latest_candidates,
        target_dir=ui_data_dir,
        dry_run=dry_run,
        prune=prune_ui,
    )
    catalog_changed, catalog_added, catalog_updated, catalog_skipped = update_catalog(
        catalog_path=catalog_path,
        latest_candidates=latest_candidates,
        dry_run=dry_run,
    )

    dist_copied = 0
    dist_unchanged = 0
    dist_pruned = 0
    if sync_dist:
        dist_copied, dist_unchanged, dist_pruned = sync_data_files(
            latest_candidates=latest_candidates,
            target_dir=dist_data_dir,
            dry_run=dry_run,
            prune=prune_ui,
        )
        if catalog_changed and not dry_run:
            shutil.copy2(catalog_path, dist_data_dir / "catalog.json")

    return {
        "dry_run": dry_run,
        "keys_selected": len(latest_candidates),
        "ui": {
            "copied": ui_copied,
            "unchanged": ui_unchanged,
            "pruned": ui_pruned,
            "target_dir": str(ui_data_dir),
        },
        "catalog": {
            "changed": catalog_changed,
            "added_surahs": catalog_added,
            "updated_surahs": catalog_updated,
            "skipped_missing_reciter": catalog_skipped,
            "path": str(catalog_path),
        },
        "dist": {
            "enabled": sync_dist,
            "copied": dist_copied,
            "unchanged": dist_unchanged,
            "pruned": dist_pruned,
            "target_dir": str(dist_data_dir),
        },
    }
