from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from .reciter_defaults import (
    DEFAULT_ENABLED_RECITERS,
    EVERYAYAH_SUBFOLDER_BY_RECITER_DEFAULT,
    QCOM_RECITATION_ID_BY_RECITER_DEFAULT,
    UNSUPPORTED_QCOM_WORD_SUPERVISION_DEFAULT,
)


@dataclass(frozen=True, slots=True)
class ReciterMapping:
    manifest_reciter_id: str
    everyayah_subfolder: str | None
    qcom_recitation_id: int | None
    qcom_word_supervision_supported: bool


DEFAULT_RECITER_CATALOG_PATH = Path("data/reciters.json")

# Resolved maps (defaults + optional catalog overrides).
EVERYAYAH_SUBFOLDER_BY_RECITER: dict[str, str] = dict(EVERYAYAH_SUBFOLDER_BY_RECITER_DEFAULT)
QCOM_RECITATION_ID_BY_RECITER: dict[str, int] = dict(QCOM_RECITATION_ID_BY_RECITER_DEFAULT)
UNSUPPORTED_QCOM_WORD_SUPERVISION: set[str] = set(UNSUPPORTED_QCOM_WORD_SUPERVISION_DEFAULT)


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_catalog_overrides(
    catalog_path: str | Path | None = None,
) -> tuple[dict[str, str], dict[str, int], set[str], set[str], bool]:
    everyayah_map = dict(EVERYAYAH_SUBFOLDER_BY_RECITER_DEFAULT)
    qcom_map = dict(QCOM_RECITATION_ID_BY_RECITER_DEFAULT)
    unsupported = set(UNSUPPORTED_QCOM_WORD_SUPERVISION_DEFAULT)
    enabled = set(DEFAULT_ENABLED_RECITERS)

    target = Path(catalog_path) if catalog_path is not None else DEFAULT_RECITER_CATALOG_PATH
    if not target.exists():
        return everyayah_map, qcom_map, unsupported, enabled, False

    try:
        payload = orjson.loads(target.read_bytes())
    except orjson.JSONDecodeError:
        return everyayah_map, qcom_map, unsupported, enabled, False
    if not isinstance(payload, dict):
        return everyayah_map, qcom_map, unsupported, enabled, False

    configured = payload.get("reciters")
    if not isinstance(configured, list):
        return everyayah_map, qcom_map, unsupported, enabled, False

    has_entries = False
    for item in configured:
        if not isinstance(item, dict):
            continue
        reciter_id = str(item.get("slug") or "").strip().lower()
        if not reciter_id:
            continue
        has_entries = True

        source = item.get("source") if isinstance(item.get("source"), dict) else {}
        everyayah_section = (
            source.get("everyayah") if isinstance(source.get("everyayah"), dict) else {}
        )
        subfolder = everyayah_section.get("subfolder")
        if isinstance(subfolder, str) and subfolder.strip():
            everyayah_map[reciter_id] = subfolder.strip()
        elif reciter_id in everyayah_map:
            del everyayah_map[reciter_id]

        quran_section = source.get("quran_com") if isinstance(source.get("quran_com"), dict) else {}
        qcom_id = _to_int(quran_section.get("recitation_id"))
        capabilities = (
            item.get("capabilities") if isinstance(item.get("capabilities"), dict) else {}
        )
        qcom_supported = bool(capabilities.get("word_by_word"))
        if qcom_supported and qcom_id is not None:
            qcom_map[reciter_id] = qcom_id
            unsupported.discard(reciter_id)
        else:
            if reciter_id in qcom_map:
                del qcom_map[reciter_id]
            unsupported.add(reciter_id)

    if has_entries:
        enabled.clear()
        for item in configured:
            if not isinstance(item, dict):
                continue
            reciter_id = str(item.get("slug") or "").strip().lower()
            if reciter_id and bool(item.get("enabled")):
                enabled.add(reciter_id)

    return everyayah_map, qcom_map, unsupported, enabled, has_entries


def resolve_reciter_mapping(
    reciter_id: str,
    *,
    catalog_path: str | Path | None = None,
) -> ReciterMapping:
    reciter_key = reciter_id.strip().lower()
    everyayah_map, qcom_map, unsupported, _, _ = _load_catalog_overrides(catalog_path)
    if reciter_key in unsupported:
        return ReciterMapping(
            manifest_reciter_id=reciter_id,
            everyayah_subfolder=everyayah_map.get(reciter_key),
            qcom_recitation_id=None,
            qcom_word_supervision_supported=False,
        )

    return ReciterMapping(
        manifest_reciter_id=reciter_id,
        everyayah_subfolder=everyayah_map.get(reciter_key),
        qcom_recitation_id=qcom_map.get(reciter_key),
        qcom_word_supervision_supported=reciter_key in qcom_map,
    )


def is_qcom_word_supervision_supported(
    reciter_id: str,
    *,
    catalog_path: str | Path | None = None,
) -> bool:
    return resolve_reciter_mapping(
        reciter_id, catalog_path=catalog_path
    ).qcom_word_supervision_supported


def is_reciter_enabled(
    reciter_id: str,
    *,
    catalog_path: str | Path | None = None,
) -> bool:
    key = reciter_id.strip().lower()
    _, _, _, enabled, _ = _load_catalog_overrides(catalog_path)
    return key in enabled


__all__ = [
    "ReciterMapping",
    "DEFAULT_RECITER_CATALOG_PATH",
    "EVERYAYAH_SUBFOLDER_BY_RECITER",
    "QCOM_RECITATION_ID_BY_RECITER",
    "UNSUPPORTED_QCOM_WORD_SUPERVISION",
    "resolve_reciter_mapping",
    "is_qcom_word_supervision_supported",
    "is_reciter_enabled",
]
