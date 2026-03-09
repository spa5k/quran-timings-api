from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any
import unicodedata

import orjson

from quran_audio_data.reciters import fetch_quranicaudio_reciters

from .everyayah import fetch_catalog as fetch_everyayah_catalog
from .qcom_audio import fetch_recitation_catalog
from .reciter_defaults import (
    DEFAULT_ENABLED_RECITERS,
    EVERYAYAH_SUBFOLDER_BY_RECITER_DEFAULT,
    QCOM_RECITATION_ID_BY_RECITER_DEFAULT,
    UNSUPPORTED_QCOM_WORD_SUPERVISION_DEFAULT,
)


DEFAULT_RECITER_CATALOG_PATH = Path("data/reciters.json")


def _slugify(value: str) -> str:
    raw = unicodedata.normalize("NFKD", str(value or "").strip().lower())
    raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
    raw = re.sub(r"[^a-z0-9]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw or "reciter"


def _parse_everyayah_catalog(payload: dict[str, Any]) -> list[dict[str, Any]]:
    reciters: list[dict[str, Any]] = []
    for key, value in payload.items():
        if key == "ayahCount" or not str(key).isdigit() or not isinstance(value, dict):
            continue
        subfolder = value.get("subfolder")
        name = value.get("name")
        if not isinstance(subfolder, str) or not isinstance(name, str):
            continue
        reciters.append(
            {
                "reciter_key": int(key),
                "name": name,
                "subfolder": subfolder,
                "bitrate": value.get("bitrate") if isinstance(value.get("bitrate"), str) else None,
            }
        )
    reciters.sort(key=lambda item: int(item["reciter_key"]))
    return reciters


def _parse_qcom_catalog(payload: dict[str, Any]) -> list[dict[str, Any]]:
    recitations = payload.get("recitations")
    if not isinstance(recitations, list):
        return []

    out: list[dict[str, Any]] = []
    for item in recitations:
        if not isinstance(item, dict):
            continue
        reciter_id = item.get("id")
        if not isinstance(reciter_id, int):
            continue

        translated = item.get("translated_name")
        translated_name = translated.get("name") if isinstance(translated, dict) else None
        style = item.get("style")
        style_name = style.get("name") if isinstance(style, dict) else None
        out.append(
            {
                "id": reciter_id,
                "reciter_name": item.get("reciter_name"),
                "translated_name": translated_name,
                "style": style_name,
            }
        )
    out.sort(key=lambda item: int(item["id"]))
    return out


def _empty_reciter_row(
    *,
    slug: str,
    name: str,
    enabled_reciters: set[str],
) -> dict[str, Any]:
    return {
        "slug": slug,
        "name": name,
        "enabled": slug in enabled_reciters,
        "check_type": "model_only",
        "capabilities": {
            "ayah_by_ayah": False,
            "word_by_word": False,
        },
        "source": {
            "everyayah": {"subfolder": None, "reciter_key": None, "name": None},
            "quran_com": {"recitation_id": None, "name": None},
            "quranicaudio": {"path": None, "name": None},
        },
        "endpoints": {
            "metadata": f"/data/reciters/{slug}/metadata.json",
        },
        "surahs_available": [],
        "surah_count": 0,
    }


def _finalize_check_type(row: dict[str, Any]) -> None:
    capabilities = row.get("capabilities") if isinstance(row.get("capabilities"), dict) else {}
    ayah_by_ayah = bool(capabilities.get("ayah_by_ayah"))
    word_by_word = bool(capabilities.get("word_by_word"))
    if ayah_by_ayah and word_by_word:
        row["check_type"] = "both"
    elif ayah_by_ayah:
        row["check_type"] = "ayah_by_ayah"
    elif word_by_word:
        row["check_type"] = "word_by_word"
    else:
        row["check_type"] = "model_only"


def _build_source_reciters(
    *,
    enabled_reciters: set[str],
    everyayah_reciters: list[dict[str, Any]],
    qcom_reciters: list[dict[str, Any]],
    quranicaudio_reciters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    default_slug_by_subfolder = {
        subfolder: slug
        for slug, subfolder in EVERYAYAH_SUBFOLDER_BY_RECITER_DEFAULT.items()
        if subfolder
    }
    default_slug_by_qcom_id = {
        int(recitation_id): slug
        for slug, recitation_id in QCOM_RECITATION_ID_BY_RECITER_DEFAULT.items()
        if recitation_id is not None
    }

    rows_by_slug: dict[str, dict[str, Any]] = {}

    for item in everyayah_reciters:
        subfolder = str(item.get("subfolder") or "").strip()
        if not subfolder:
            continue
        slug = default_slug_by_subfolder.get(subfolder) or f"eya_{_slugify(subfolder)}"
        name = str(item.get("name") or subfolder).strip() or subfolder
        row = rows_by_slug.get(slug)
        if row is None:
            row = _empty_reciter_row(slug=slug, name=name, enabled_reciters=enabled_reciters)
            rows_by_slug[slug] = row
        row["name"] = name or row["name"]
        row["enabled"] = bool(row.get("enabled")) or slug in enabled_reciters
        row["capabilities"]["ayah_by_ayah"] = True
        row["source"]["everyayah"] = {
            "subfolder": subfolder,
            "reciter_key": item.get("reciter_key")
            if isinstance(item.get("reciter_key"), int)
            else None,
            "name": name,
        }

    seen_qcom_slugs: dict[str, int] = {}
    for item in qcom_reciters:
        recitation_id = item.get("id")
        if not isinstance(recitation_id, int):
            continue
        label = str(
            item.get("translated_name") or item.get("reciter_name") or recitation_id
        ).strip()
        base_slug = default_slug_by_qcom_id.get(recitation_id) or f"qcom_{_slugify(label)}"
        slug = base_slug
        if slug in rows_by_slug:
            existing_qcom = (
                rows_by_slug[slug].get("source", {}).get("quran_com", {}).get("recitation_id")
                if isinstance(rows_by_slug[slug].get("source"), dict)
                else None
            )
            if existing_qcom not in {None, recitation_id}:
                slug = f"{base_slug}_{recitation_id}"
        if slug in seen_qcom_slugs and seen_qcom_slugs[slug] != recitation_id:
            slug = f"{base_slug}_{recitation_id}"
        seen_qcom_slugs[slug] = recitation_id

        row = rows_by_slug.get(slug)
        if row is None:
            row = _empty_reciter_row(slug=slug, name=label, enabled_reciters=enabled_reciters)
            rows_by_slug[slug] = row
        row["name"] = label or row["name"]
        row["enabled"] = bool(row.get("enabled")) or slug in enabled_reciters
        row["capabilities"]["word_by_word"] = slug not in UNSUPPORTED_QCOM_WORD_SUPERVISION_DEFAULT
        row["source"]["quran_com"] = {
            "recitation_id": recitation_id,
            "name": label,
        }

    for item in quranicaudio_reciters:
        slug = str(item.get("id") or "").strip().lower()
        if not slug:
            continue
        name = str(item.get("name") or slug).strip() or slug
        notes = str(item.get("notes") or "").strip()
        relative_path = None
        if "relative_path=" in notes:
            relative_path = notes.split("relative_path=", 1)[1].strip().rstrip("/")
        row = rows_by_slug.get(slug)
        if row is None:
            row = _empty_reciter_row(slug=slug, name=name, enabled_reciters=enabled_reciters)
            rows_by_slug[slug] = row
        row["enabled"] = bool(row.get("enabled")) or slug in enabled_reciters
        row["source"]["quranicaudio"] = {
            "path": relative_path,
            "name": name,
        }

    for row in rows_by_slug.values():
        _finalize_check_type(row)

    out = sorted(rows_by_slug.values(), key=lambda item: str(item.get("slug") or ""))
    return out


def build_reciter_catalog_payload(
    *,
    enabled_reciters: set[str] | None = None,
) -> dict[str, Any]:
    enabled = {
        item.strip().lower()
        for item in (enabled_reciters or DEFAULT_ENABLED_RECITERS)
        if item.strip()
    }
    everyayah_payload = fetch_everyayah_catalog()
    qcom_payload = fetch_recitation_catalog(language="en")

    everyayah_reciters = _parse_everyayah_catalog(everyayah_payload)
    qcom_reciters = _parse_qcom_catalog(qcom_payload)
    quranicaudio_reciters = fetch_quranicaudio_reciters()
    configured = _build_source_reciters(
        enabled_reciters=enabled,
        everyayah_reciters=everyayah_reciters,
        qcom_reciters=qcom_reciters,
        quranicaudio_reciters=quranicaudio_reciters,
    )

    return {
        "schema_version": "v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "everyayah_source_reciters": len(everyayah_reciters),
            "quran_com_source_reciters": len(qcom_reciters),
            "quranicaudio_source_reciters": len(quranicaudio_reciters),
            "configured_reciters": len(configured),
            "enabled_reciters": sum(1 for item in configured if item.get("enabled")),
        },
        "sources": {
            "everyayah_catalog_url": "https://everyayah.com/data/recitations.js",
            "quran_com_recitations_url": "https://api.quran.com/api/v4/resources/recitations?language=en",
            "quranicaudio_catalog_url": "https://quranicaudio.com",
        },
        "reciters": configured,
        # Keep source catalogs for tooling that inspects external mapping pools.
        "everyayah_reciters": everyayah_reciters,
        "quran_com_reciters": qcom_reciters,
        "quranicaudio_reciters": quranicaudio_reciters,
    }


def write_reciter_catalog(
    *,
    path: str | Path = DEFAULT_RECITER_CATALOG_PATH,
    enabled_reciters: set[str] | None = None,
) -> dict[str, Any]:
    payload = build_reciter_catalog_payload(enabled_reciters=enabled_reciters)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    return payload


def read_reciter_catalog(path: str | Path = DEFAULT_RECITER_CATALOG_PATH) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    try:
        payload = orjson.loads(target.read_bytes())
    except orjson.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    reciters = payload.get("reciters")
    if not isinstance(reciters, list):
        return None
    return payload


def get_configured_reciter_entry(
    reciter_id: str,
    *,
    catalog_path: str | Path = DEFAULT_RECITER_CATALOG_PATH,
) -> dict[str, Any] | None:
    payload = read_reciter_catalog(catalog_path)
    if payload is None:
        return None
    configured = payload.get("reciters")
    if not isinstance(configured, list):
        return None

    key = reciter_id.strip().lower()
    for item in configured:
        if not isinstance(item, dict):
            continue
        value = str(item.get("slug") or "").strip().lower()
        if value == key:
            return item
    return None


__all__ = [
    "DEFAULT_RECITER_CATALOG_PATH",
    "build_reciter_catalog_payload",
    "write_reciter_catalog",
    "read_reciter_catalog",
    "get_configured_reciter_entry",
]
