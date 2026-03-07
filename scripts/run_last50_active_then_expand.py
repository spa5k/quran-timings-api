from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import traceback
from typing import Any

from quran_audio_data.surah_runner import run_surah_for_reciter


ACTIVE_MAPPING_ALIASES: dict[str, dict[str, Any]] = {
    "eya_nasser_alqatami": {"everyayah_subfolder": "Nasser_Alqatami_128kbps"},
    "eya_salah_al_budair": {"everyayah_subfolder": "Salah_Al_Budair_128kbps"},
    "qcom_husary": {"qcom_recitation_id": 6},
    "qcom_mishari_alafasy": {"qcom_recitation_id": 7},
}


@dataclass(slots=True)
class RunResult:
    reciter_id: str
    surah: int
    status: str
    summary_path: str | None
    error: str | None


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid JSON object in: {path}")
    return payload


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _ensure_reciter_fields(reciter: dict[str, Any]) -> None:
    source = reciter.get("source")
    if not isinstance(source, dict):
        source = {}
        reciter["source"] = source
    everyayah = source.get("everyayah")
    if not isinstance(everyayah, dict):
        everyayah = {"subfolder": None, "reciter_key": None, "name": None}
        source["everyayah"] = everyayah
    quran_com = source.get("quran_com")
    if not isinstance(quran_com, dict):
        quran_com = {"recitation_id": None, "name": None}
        source["quran_com"] = quran_com

    capabilities = reciter.get("capabilities")
    if not isinstance(capabilities, dict):
        capabilities = {"ayah_by_ayah": False, "word_by_word": False}
        reciter["capabilities"] = capabilities

    endpoints = reciter.get("endpoints")
    if not isinstance(endpoints, dict):
        endpoints = {}
        reciter["endpoints"] = endpoints
    if not isinstance(endpoints.get("metadata"), str) or not endpoints.get("metadata"):
        slug = str(reciter.get("slug") or "").strip().lower()
        endpoints["metadata"] = f"/data/reciters/{slug}/metadata.json"


def _set_check_type(reciter: dict[str, Any]) -> None:
    capabilities = reciter.get("capabilities") if isinstance(reciter.get("capabilities"), dict) else {}
    ayah_by_ayah = bool(capabilities.get("ayah_by_ayah"))
    word_by_word = bool(capabilities.get("word_by_word"))
    if ayah_by_ayah and word_by_word:
        reciter["check_type"] = "both"
    elif ayah_by_ayah:
        reciter["check_type"] = "ayah_by_ayah"
    elif word_by_word:
        reciter["check_type"] = "word_by_word"
    else:
        reciter["check_type"] = "model_only"


def _hydrate_active_reciter_mappings(catalog: dict[str, Any]) -> list[str]:
    reciters = catalog.get("reciters")
    if not isinstance(reciters, list):
        raise ValueError("catalog is missing reciters list")

    everyayah_rows = catalog.get("everyayah_reciters")
    qcom_rows = catalog.get("quran_com_reciters")
    if not isinstance(everyayah_rows, list) or not isinstance(qcom_rows, list):
        raise ValueError("catalog is missing everyayah_reciters/quran_com_reciters source pools")

    everyayah_by_subfolder: dict[str, dict[str, Any]] = {}
    everyayah_by_slug: dict[str, dict[str, Any]] = {}
    for row in everyayah_rows:
        if not isinstance(row, dict):
            continue
        subfolder = row.get("subfolder")
        if isinstance(subfolder, str) and subfolder.strip():
            everyayah_by_subfolder[subfolder.strip()] = row
        name = row.get("name")
        if isinstance(name, str) and name.strip():
            slug = f"eya_{_slugify(name)}"
            everyayah_by_slug.setdefault(slug, row)

    qcom_by_id: dict[int, dict[str, Any]] = {}
    qcom_by_slug: dict[str, dict[str, Any]] = {}
    for row in qcom_rows:
        if not isinstance(row, dict):
            continue
        recitation_id = row.get("id")
        if not isinstance(recitation_id, int):
            continue
        qcom_by_id[recitation_id] = row
        display_name = str(row.get("translated_name") or row.get("reciter_name") or "").strip()
        if display_name:
            slug = f"qcom_{_slugify(display_name)}"
            qcom_by_slug.setdefault(slug, row)

    updated: list[str] = []
    for reciter in reciters:
        if not isinstance(reciter, dict):
            continue
        if not bool(reciter.get("enabled")):
            continue

        _ensure_reciter_fields(reciter)
        slug = str(reciter.get("slug") or "").strip().lower()
        if not slug:
            continue

        source = reciter["source"]
        everyayah = source["everyayah"]
        quran_com = source["quran_com"]

        has_everyayah = isinstance(everyayah.get("subfolder"), str) and everyayah.get("subfolder")
        has_qcom = isinstance(quran_com.get("recitation_id"), int)
        if has_everyayah or has_qcom:
            continue

        changed = False
        alias = ACTIVE_MAPPING_ALIASES.get(slug, {})

        alias_subfolder = alias.get("everyayah_subfolder")
        if isinstance(alias_subfolder, str) and alias_subfolder in everyayah_by_subfolder:
            row = everyayah_by_subfolder[alias_subfolder]
            everyayah["subfolder"] = alias_subfolder
            everyayah["reciter_key"] = row.get("reciter_key") if isinstance(row.get("reciter_key"), int) else None
            everyayah["name"] = row.get("name") if isinstance(row.get("name"), str) else alias_subfolder
            reciter["capabilities"]["ayah_by_ayah"] = True
            changed = True
        elif slug.startswith("eya_") and slug in everyayah_by_slug:
            row = everyayah_by_slug[slug]
            subfolder = row.get("subfolder")
            if isinstance(subfolder, str) and subfolder.strip():
                everyayah["subfolder"] = subfolder.strip()
                everyayah["reciter_key"] = row.get("reciter_key") if isinstance(row.get("reciter_key"), int) else None
                everyayah["name"] = row.get("name") if isinstance(row.get("name"), str) else subfolder.strip()
                reciter["capabilities"]["ayah_by_ayah"] = True
                changed = True

        alias_qcom_id = alias.get("qcom_recitation_id")
        if isinstance(alias_qcom_id, int) and alias_qcom_id in qcom_by_id:
            row = qcom_by_id[alias_qcom_id]
            quran_com["recitation_id"] = alias_qcom_id
            quran_com["name"] = row.get("translated_name") or row.get("reciter_name") or str(alias_qcom_id)
            reciter["capabilities"]["word_by_word"] = True
            changed = True
        elif slug.startswith("qcom_") and slug in qcom_by_slug:
            row = qcom_by_slug[slug]
            recitation_id = row.get("id")
            if isinstance(recitation_id, int):
                quran_com["recitation_id"] = recitation_id
                quran_com["name"] = row.get("translated_name") or row.get("reciter_name") or str(recitation_id)
                reciter["capabilities"]["word_by_word"] = True
                changed = True

        if changed:
            _set_check_type(reciter)
            updated.append(slug)

    return sorted(set(updated))


def _update_catalog_counts(catalog: dict[str, Any]) -> None:
    reciters = catalog.get("reciters")
    if not isinstance(reciters, list):
        return
    counts = catalog.get("counts")
    if not isinstance(counts, dict):
        counts = {}
        catalog["counts"] = counts
    counts["configured_reciters"] = len(reciters)
    counts["enabled_reciters"] = sum(
        1 for item in reciters if isinstance(item, dict) and bool(item.get("enabled"))
    )
    catalog["generated_at"] = _now_iso()


def _create_qcom_entry(row: dict[str, Any]) -> dict[str, Any] | None:
    recitation_id = row.get("id")
    if not isinstance(recitation_id, int):
        return None
    name = str(row.get("translated_name") or row.get("reciter_name") or "").strip()
    if not name:
        return None
    slug = f"qcom_{_slugify(name)}"
    display_name = name
    return {
        "slug": slug,
        "name": display_name,
        "enabled": True,
        "check_type": "word_by_word",
        "capabilities": {
            "ayah_by_ayah": False,
            "word_by_word": True,
        },
        "source": {
            "everyayah": {
                "subfolder": None,
                "reciter_key": None,
                "name": None,
            },
            "quran_com": {
                "recitation_id": recitation_id,
                "name": display_name,
            },
        },
        "surahs_available": [],
        "surah_count": 0,
        "endpoints": {
            "metadata": f"/data/reciters/{slug}/metadata.json",
        },
    }


def _create_everyayah_entry(row: dict[str, Any]) -> dict[str, Any] | None:
    subfolder = row.get("subfolder")
    name = row.get("name")
    if not isinstance(subfolder, str) or not subfolder.strip():
        return None
    if not isinstance(name, str) or not name.strip():
        return None
    slug = f"eya_{_slugify(name)}"
    display_name = name.strip()
    return {
        "slug": slug,
        "name": display_name,
        "enabled": True,
        "check_type": "ayah_by_ayah",
        "capabilities": {
            "ayah_by_ayah": True,
            "word_by_word": False,
        },
        "source": {
            "everyayah": {
                "subfolder": subfolder.strip(),
                "reciter_key": row.get("reciter_key") if isinstance(row.get("reciter_key"), int) else None,
                "name": display_name,
            },
            "quran_com": {
                "recitation_id": None,
                "name": None,
            },
        },
        "surahs_available": [],
        "surah_count": 0,
        "endpoints": {
            "metadata": f"/data/reciters/{slug}/metadata.json",
        },
    }


def _expand_catalog_reciters(
    catalog: dict[str, Any],
    *,
    add_qcom: int,
    add_everyayah: int,
) -> list[str]:
    reciters = catalog.get("reciters")
    everyayah_rows = catalog.get("everyayah_reciters")
    qcom_rows = catalog.get("quran_com_reciters")
    if not isinstance(reciters, list):
        raise ValueError("catalog is missing reciters list")
    if not isinstance(everyayah_rows, list) or not isinstance(qcom_rows, list):
        raise ValueError("catalog is missing everyayah_reciters/quran_com_reciters source pools")

    existing_slugs = {
        str(item.get("slug") or "").strip().lower()
        for item in reciters
        if isinstance(item, dict)
    }
    added: list[str] = []

    if add_qcom > 0:
        qcom_candidates: list[dict[str, Any]] = []
        seen_candidate_slugs: set[str] = set()
        for row in qcom_rows:
            if not isinstance(row, dict):
                continue
            candidate = _create_qcom_entry(row)
            if candidate is None:
                continue
            slug = str(candidate["slug"]).lower()
            if slug in existing_slugs or slug in seen_candidate_slugs:
                continue
            seen_candidate_slugs.add(slug)
            qcom_candidates.append(candidate)
        for candidate in qcom_candidates[:add_qcom]:
            slug = str(candidate["slug"]).lower()
            reciters.append(candidate)
            existing_slugs.add(slug)
            added.append(slug)

    if add_everyayah > 0:
        eya_candidates: list[dict[str, Any]] = []
        seen_candidate_slugs = set()
        for row in everyayah_rows:
            if not isinstance(row, dict):
                continue
            candidate = _create_everyayah_entry(row)
            if candidate is None:
                continue
            slug = str(candidate["slug"]).lower()
            if slug in existing_slugs or slug in seen_candidate_slugs:
                continue
            seen_candidate_slugs.add(slug)
            eya_candidates.append(candidate)
        for candidate in eya_candidates[:add_everyayah]:
            slug = str(candidate["slug"]).lower()
            reciters.append(candidate)
            existing_slugs.add(slug)
            added.append(slug)

    reciters.sort(key=lambda item: str(item.get("slug") or "") if isinstance(item, dict) else "")
    _update_catalog_counts(catalog)
    return added


def _write_report(
    path: Path,
    *,
    catalog: Path,
    surah_start: int,
    surah_end: int,
    active_reciters: list[str],
    mapped_active_reciters: list[str],
    results: list[RunResult],
    added_reciters: list[str],
    done: bool,
) -> None:
    status_counts: dict[str, int] = {}
    for row in results:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1

    payload = {
        "updated_at": _now_iso(),
        "done": done,
        "catalog_path": str(catalog),
        "surah_range": [surah_start, surah_end],
        "active_reciters": active_reciters,
        "mapped_active_reciters": mapped_active_reciters,
        "added_reciters": added_reciters,
        "totals": {
            "tasks": len(results),
            "status_counts": status_counts,
        },
        "results": [
            {
                "reciter_id": row.reciter_id,
                "surah": row.surah,
                "status": row.status,
                "summary_path": row.summary_path,
                "error": row.error,
            }
            for row in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run surahs 65-114 one-by-one for currently enabled reciters, "
            "then expand reciters from existing catalog source pools."
        )
    )
    parser.add_argument("--catalog", type=Path, default=Path("data/reciters.json"))
    parser.add_argument("--surah-start", type=int, default=65)
    parser.add_argument("--surah-end", type=int, default=114)
    parser.add_argument("--out-root", type=Path, default=Path("runs/last50_active"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument("--text-data", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=Path("runs/last50_active/report.json"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--add-qcom", type=int, default=5)
    parser.add_argument("--add-everyayah", type=int, default=5)
    args = parser.parse_args()

    if args.surah_start < 1 or args.surah_end > 114 or args.surah_start > args.surah_end:
        raise SystemExit(f"invalid surah range: {args.surah_start}-{args.surah_end}")

    catalog = _load_json(args.catalog)
    reciters = catalog.get("reciters")
    if not isinstance(reciters, list):
        raise SystemExit("catalog missing reciters list")

    active_reciters = sorted(
        {
            str(item.get("slug") or "").strip().lower()
            for item in reciters
            if isinstance(item, dict) and bool(item.get("enabled")) and str(item.get("slug") or "").strip()
        }
    )
    if not active_reciters:
        raise SystemExit("no enabled reciters found in catalog")

    mapped_active_reciters = _hydrate_active_reciter_mappings(catalog)
    _update_catalog_counts(catalog)
    _save_json(args.catalog, catalog)

    surahs = list(range(args.surah_start, args.surah_end + 1))
    tasks = [(reciter_id, surah) for reciter_id in active_reciters for surah in surahs]
    total = len(tasks)
    print(f"[batch] active_reciters={len(active_reciters)} surahs={len(surahs)} tasks={total}")
    print(f"[batch] out_root={args.out_root}")
    if mapped_active_reciters:
        print(f"[batch] hydrated mappings for: {', '.join(mapped_active_reciters)}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    added_reciters: list[str] = []
    failures = 0

    for index, (reciter_id, surah) in enumerate(tasks, start=1):
        summary_path = args.out_root / f"{reciter_id}_s{surah:03d}" / "run_summary.json"
        if summary_path.exists() and not args.force:
            print(f"[{index}/{total}] skip {reciter_id} surah={surah} (summary exists)")
            results.append(
                RunResult(
                    reciter_id=reciter_id,
                    surah=surah,
                    status="skipped_existing",
                    summary_path=str(summary_path),
                    error=None,
                )
            )
            _write_report(
                args.report,
                catalog=args.catalog,
                surah_start=args.surah_start,
                surah_end=args.surah_end,
                active_reciters=active_reciters,
                mapped_active_reciters=mapped_active_reciters,
                results=results,
                added_reciters=added_reciters,
                done=False,
            )
            continue

        print(f"[{index}/{total}] run {reciter_id} surah={surah}")
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
            summary_path_value = (
                str(paths.get("summary_path")) if paths.get("summary_path") else str(summary_path)
            )
            results.append(
                RunResult(
                    reciter_id=reciter_id,
                    surah=surah,
                    status="ok",
                    summary_path=summary_path_value,
                    error=None,
                )
            )
        except Exception as exc:
            failures += 1
            error_message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            print(f"  -> failed: {error_message}")
            results.append(
                RunResult(
                    reciter_id=reciter_id,
                    surah=surah,
                    status="failed",
                    summary_path=None,
                    error=error_message,
                )
            )

        _write_report(
            args.report,
            catalog=args.catalog,
            surah_start=args.surah_start,
            surah_end=args.surah_end,
            active_reciters=active_reciters,
            mapped_active_reciters=mapped_active_reciters,
            results=results,
            added_reciters=added_reciters,
            done=False,
        )

    added_reciters = _expand_catalog_reciters(
        catalog,
        add_qcom=max(0, args.add_qcom),
        add_everyayah=max(0, args.add_everyayah),
    )
    _save_json(args.catalog, catalog)
    if added_reciters:
        print(f"[batch] added reciters ({len(added_reciters)}): {', '.join(added_reciters)}")
    else:
        print("[batch] no additional reciters were added")

    _write_report(
        args.report,
        catalog=args.catalog,
        surah_start=args.surah_start,
        surah_end=args.surah_end,
        active_reciters=active_reciters,
        mapped_active_reciters=mapped_active_reciters,
        results=results,
        added_reciters=added_reciters,
        done=True,
    )

    print(f"[batch] finished tasks={len(results)} failures={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
