from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import filecmp
import json
import re
import shutil

import orjson


FULL_JSON_RE = re.compile(r"^(?P<reciter_id>.+)_s(?P<surah>\d{3})_full\.json$")


@dataclass(frozen=True, slots=True)
class RunCandidate:
    reciter_id: str
    surah: int
    path: Path
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class AudioAsset:
    reciter_id: str
    surah: int
    source_path: Path
    target_name: str
    web_src: str


def _safe_surah_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def ensure_catalog(
    *,
    catalog_path: Path,
    reciter_catalog_path: Path | None = None,
    enabled_only: bool = True,
    dry_run: bool = False,
) -> tuple[bool, int]:
    recitations: list[dict[str, Any]] = []
    existing_by_id: dict[str, dict[str, Any]] = {}

    if catalog_path.exists():
        try:
            existing_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
            existing_recitations = existing_payload.get("recitations")
            if isinstance(existing_recitations, list):
                for item in existing_recitations:
                    if isinstance(item, dict):
                        reciter_id = str(item.get("id") or "").strip()
                        if reciter_id:
                            existing_by_id[reciter_id] = item
        except Exception:
            existing_by_id = {}

    source_entries: list[dict[str, Any]] = []
    if reciter_catalog_path is not None and reciter_catalog_path.exists():
        try:
            reciter_payload = orjson.loads(reciter_catalog_path.read_bytes())
        except orjson.JSONDecodeError:
            reciter_payload = {}
        if isinstance(reciter_payload, dict):
            configured = reciter_payload.get("configured_reciters")
            if isinstance(configured, list):
                for item in configured:
                    if isinstance(item, dict):
                        source_entries.append(item)

    added = 0
    if source_entries:
        for item in source_entries:
            if enabled_only and not bool(item.get("enabled")):
                continue
            reciter_id = str(item.get("manifest_reciter_id") or "").strip()
            if not reciter_id:
                continue

            existing = existing_by_id.pop(reciter_id, None)
            everyayah = item.get("everyayah") if isinstance(item.get("everyayah"), dict) else {}
            qcom = item.get("quran_com") if isinstance(item.get("quran_com"), dict) else {}
            display_name = (
                str(qcom.get("name") or "").strip()
                or str(everyayah.get("name") or "").strip()
                or reciter_id
            )
            surahs = existing.get("surahs") if isinstance(existing, dict) and isinstance(existing.get("surahs"), list) else []
            if existing is None:
                added += 1
            recitations.append(
                {
                    "id": reciter_id,
                    "name": display_name,
                    "surahs": surahs,
                }
            )

    for reciter_id, existing in existing_by_id.items():
        recitations.append(
            {
                "id": reciter_id,
                "name": str(existing.get("name") or reciter_id),
                "surahs": existing.get("surahs") if isinstance(existing.get("surahs"), list) else [],
            }
        )

    recitations.sort(key=lambda item: str(item.get("id") or ""))
    payload = {"recitations": recitations}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    previous = catalog_path.read_text(encoding="utf-8") if catalog_path.exists() else ""
    changed = rendered != previous
    if changed and not dry_run:
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_text(rendered, encoding="utf-8")
    return changed, added


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


def extract_audio_assets(
    latest_candidates: dict[tuple[str, int], RunCandidate],
) -> dict[tuple[str, int], AudioAsset]:
    assets: dict[tuple[str, int], AudioAsset] = {}
    for key, candidate in latest_candidates.items():
        try:
            payload = orjson.loads(candidate.path.read_bytes())
        except orjson.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        audio = payload.get("audio")
        if not isinstance(audio, dict):
            continue
        audio_path_raw = audio.get("path")
        if not isinstance(audio_path_raw, str) or not audio_path_raw.strip():
            continue
        source_path = Path(audio_path_raw)
        if not source_path.exists() or not source_path.is_file():
            continue
        suffix = source_path.suffix.lower() or ".mp3"
        target_name = f"{candidate.reciter_id}_s{candidate.surah:03d}{suffix}"
        assets[key] = AudioAsset(
            reciter_id=candidate.reciter_id,
            surah=candidate.surah,
            source_path=source_path,
            target_name=target_name,
            web_src=f"/data/audio/{target_name}",
        )
    return assets


def sync_audio_assets(
    assets: dict[tuple[str, int], AudioAsset],
    target_dir: Path,
    dry_run: bool,
    prune: bool,
) -> tuple[int, int, int]:
    copied = 0
    unchanged = 0
    pruned = 0
    target_dir.mkdir(parents=True, exist_ok=True)

    selected_names = {asset.target_name for asset in assets.values()}
    for asset in sorted(assets.values(), key=lambda item: (item.reciter_id, item.surah)):
        target = target_dir / asset.target_name
        if needs_copy(asset.source_path, target):
            copied += 1
            if not dry_run:
                shutil.copy2(asset.source_path, target)
        else:
            unchanged += 1

    if not prune:
        return copied, unchanged, pruned

    for existing in target_dir.glob("*"):
        if not existing.is_file():
            continue
        if existing.name in selected_names:
            continue
        pruned += 1
        if not dry_run:
            existing.unlink(missing_ok=True)

    return copied, unchanged, pruned


def update_catalog(
    catalog_path: Path,
    latest_candidates: dict[tuple[str, int], RunCandidate],
    audio_src_by_key: dict[tuple[str, int], str] | None = None,
    dry_run: bool = False,
    prune_surahs: bool = False,
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
    selected_by_reciter: dict[str, set[int]] = {}
    source_map = audio_src_by_key or {}

    for candidate in sorted(latest_candidates.values(), key=lambda item: (item.reciter_id, item.surah)):
        key = (candidate.reciter_id, candidate.surah)
        selected_by_reciter.setdefault(candidate.reciter_id, set()).add(candidate.surah)
        recitation = recitation_by_id.get(candidate.reciter_id)
        if recitation is None:
            skipped += 1
            continue

        surahs = recitation.get("surahs")
        if not isinstance(surahs, list):
            surahs = []
            recitation["surahs"] = surahs

        timing_json = f"/data/{candidate.path.name}"
        audio_src = source_map.get(key)
        if audio_src is None:
            audio_src = ""

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

    if prune_surahs:
        for reciter_id, recitation in recitation_by_id.items():
            surahs = recitation.get("surahs")
            if not isinstance(surahs, list):
                continue
            selected = selected_by_reciter.get(reciter_id, set())
            filtered: list[dict[str, Any]] = []
            for entry in surahs:
                if not isinstance(entry, dict):
                    continue
                surah_value = _safe_surah_int(entry.get("surah"))
                if surah_value is None:
                    continue
                if surah_value in selected:
                    filtered.append(entry)
            recitation["surahs"] = filtered

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

    # Keep UI catalog focused on runnable artifacts only: hide reciters with no synced surahs.
    filtered_recitations: list[dict[str, Any]] = []
    for item in recitations:
        if not isinstance(item, dict):
            continue
        surahs = item.get("surahs")
        if isinstance(surahs, list) and surahs:
            filtered_recitations.append(item)
    payload["recitations"] = filtered_recitations

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
    reciter_catalog_path: Path | None = None,
    prune_catalog_surahs: bool = False,
) -> dict[str, Any]:
    latest_candidates = discover_latest_candidates(runs_root=runs_root)
    if not latest_candidates:
        raise RuntimeError(f"No *_full.json files found under {runs_root}")

    catalog_bootstrap_changed, catalog_bootstrap_added = ensure_catalog(
        catalog_path=catalog_path,
        reciter_catalog_path=reciter_catalog_path,
        enabled_only=True,
        dry_run=dry_run,
    )

    ui_copied, ui_unchanged, ui_pruned = sync_data_files(
        latest_candidates=latest_candidates,
        target_dir=ui_data_dir,
        dry_run=dry_run,
        prune=prune_ui,
    )
    audio_assets = extract_audio_assets(latest_candidates)
    audio_src_by_key = {
        key: asset.web_src for key, asset in audio_assets.items()
    }
    ui_audio_copied, ui_audio_unchanged, ui_audio_pruned = sync_audio_assets(
        assets=audio_assets,
        target_dir=ui_data_dir / "audio",
        dry_run=dry_run,
        prune=prune_ui,
    )
    catalog_changed, catalog_added, catalog_updated, catalog_skipped = update_catalog(
        catalog_path=catalog_path,
        latest_candidates=latest_candidates,
        audio_src_by_key=audio_src_by_key,
        dry_run=dry_run,
        prune_surahs=prune_catalog_surahs,
    )

    dist_copied = 0
    dist_unchanged = 0
    dist_pruned = 0
    dist_audio_copied = 0
    dist_audio_unchanged = 0
    dist_audio_pruned = 0
    if sync_dist:
        dist_copied, dist_unchanged, dist_pruned = sync_data_files(
            latest_candidates=latest_candidates,
            target_dir=dist_data_dir,
            dry_run=dry_run,
            prune=prune_ui,
        )
        dist_audio_copied, dist_audio_unchanged, dist_audio_pruned = sync_audio_assets(
            assets=audio_assets,
            target_dir=dist_data_dir / "audio",
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
            "audio_copied": ui_audio_copied,
            "audio_unchanged": ui_audio_unchanged,
            "audio_pruned": ui_audio_pruned,
            "target_dir": str(ui_data_dir),
        },
        "catalog": {
            "changed": catalog_changed or catalog_bootstrap_changed,
            "bootstrap_added_reciters": catalog_bootstrap_added,
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
            "audio_copied": dist_audio_copied,
            "audio_unchanged": dist_audio_unchanged,
            "audio_pruned": dist_audio_pruned,
            "target_dir": str(dist_data_dir),
        },
    }
