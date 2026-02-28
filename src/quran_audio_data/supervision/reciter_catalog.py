from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson

from .everyayah import fetch_catalog as fetch_everyayah_catalog
from .qcom_audio import fetch_recitation_catalog
from .reciter_defaults import (
    DEFAULT_ENABLED_RECITERS,
    EVERYAYAH_SUBFOLDER_BY_RECITER_DEFAULT,
    QCOM_RECITATION_ID_BY_RECITER_DEFAULT,
    UNSUPPORTED_QCOM_WORD_SUPERVISION_DEFAULT,
)


DEFAULT_RECITER_CATALOG_PATH = Path("data/reciter_catalog.json")


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


def _build_configured_reciters(
    *,
    enabled_reciters: set[str],
    everyayah_reciters: list[dict[str, Any]],
    qcom_reciters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_subfolder = {
        str(item.get("subfolder")): item
        for item in everyayah_reciters
        if isinstance(item.get("subfolder"), str)
    }
    by_qcom_id = {
        int(item["id"]): item
        for item in qcom_reciters
        if isinstance(item.get("id"), int)
    }

    configured_ids = sorted(
        set(EVERYAYAH_SUBFOLDER_BY_RECITER_DEFAULT)
        | set(QCOM_RECITATION_ID_BY_RECITER_DEFAULT)
        | set(UNSUPPORTED_QCOM_WORD_SUPERVISION_DEFAULT)
    )

    out: list[dict[str, Any]] = []
    for reciter_id in configured_ids:
        everyayah_subfolder = EVERYAYAH_SUBFOLDER_BY_RECITER_DEFAULT.get(reciter_id)
        qcom_recitation_id = QCOM_RECITATION_ID_BY_RECITER_DEFAULT.get(reciter_id)
        qcom_word_supervision_supported = (
            reciter_id not in UNSUPPORTED_QCOM_WORD_SUPERVISION_DEFAULT
            and qcom_recitation_id is not None
        )

        ayah_by_ayah = everyayah_subfolder is not None
        word_by_word = qcom_word_supervision_supported
        if ayah_by_ayah and word_by_word:
            check_type = "both"
        elif ayah_by_ayah:
            check_type = "ayah_by_ayah"
        elif word_by_word:
            check_type = "word_by_word"
        else:
            check_type = "model_only"

        everyayah_meta = by_subfolder.get(everyayah_subfolder) if everyayah_subfolder else None
        qcom_meta = by_qcom_id.get(qcom_recitation_id) if qcom_recitation_id else None
        out.append(
            {
                "manifest_reciter_id": reciter_id,
                "enabled": reciter_id in enabled_reciters,
                "check_type": check_type,
                "checks": {
                    "ayah_by_ayah": ayah_by_ayah,
                    "word_by_word": word_by_word,
                },
                "sources": {
                    "everyayah": everyayah_subfolder is not None,
                    "quran_com": qcom_recitation_id is not None,
                },
                "everyayah": {
                    "subfolder": everyayah_subfolder,
                    "reciter_key": everyayah_meta.get("reciter_key") if everyayah_meta else None,
                    "name": everyayah_meta.get("name") if everyayah_meta else None,
                },
                "quran_com": {
                    "recitation_id": qcom_recitation_id,
                    "name": (
                        qcom_meta.get("translated_name")
                        or qcom_meta.get("reciter_name")
                        if qcom_meta
                        else None
                    ),
                },
                "qcom_word_supervision_supported": qcom_word_supervision_supported,
            }
        )
    return out


def build_reciter_catalog_payload(
    *,
    enabled_reciters: set[str] | None = None,
) -> dict[str, Any]:
    enabled = {item.strip().lower() for item in (enabled_reciters or DEFAULT_ENABLED_RECITERS) if item.strip()}
    everyayah_payload = fetch_everyayah_catalog()
    qcom_payload = fetch_recitation_catalog(language="en")

    everyayah_reciters = _parse_everyayah_catalog(everyayah_payload)
    qcom_reciters = _parse_qcom_catalog(qcom_payload)
    configured = _build_configured_reciters(
        enabled_reciters=enabled,
        everyayah_reciters=everyayah_reciters,
        qcom_reciters=qcom_reciters,
    )

    return {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "enabled_manifest_reciters": sorted(enabled),
        "counts": {
            "everyayah_reciters": len(everyayah_reciters),
            "quran_com_reciters": len(qcom_reciters),
            "configured_reciters": len(configured),
            "configured_enabled": sum(1 for item in configured if item.get("enabled")),
        },
        "sources": {
            "everyayah_catalog_url": "https://everyayah.com/data/recitations.js",
            "quran_com_recitations_url": "https://api.quran.com/api/v4/resources/recitations?language=en",
        },
        "configured_reciters": configured,
        "everyayah_reciters": everyayah_reciters,
        "quran_com_reciters": qcom_reciters,
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
    return payload


def get_configured_reciter_entry(
    reciter_id: str,
    *,
    catalog_path: str | Path = DEFAULT_RECITER_CATALOG_PATH,
) -> dict[str, Any] | None:
    payload = read_reciter_catalog(catalog_path)
    if payload is None:
        return None
    configured = payload.get("configured_reciters")
    if not isinstance(configured, list):
        return None

    key = reciter_id.strip().lower()
    for item in configured:
        if not isinstance(item, dict):
            continue
        value = str(item.get("manifest_reciter_id") or "").strip().lower()
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

