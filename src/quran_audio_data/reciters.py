from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import unicodedata
from typing import Any

import orjson

from quran_audio_data.core.http import get_bytes_with_retry
from quran_audio_data.supervision.everyayah import fetch_catalog as fetch_everyayah_catalog
from quran_audio_data.supervision.qcom_audio import fetch_recitation_catalog


DEFAULT_RECITERS_PATH = Path("data/reciters.json")
QURANICAUDIO_HOME_URL = "https://quranicaudio.com"

_QURANICAUDIO_QARI_PATTERN = re.compile(
    r'\{id:(?P<id>\d+),name:"(?P<name>(?:[^"\\]|\\.)*)",arabic_name:"(?P<arabic>(?:[^"\\]|\\.)*)",relative_path:"(?P<path>[^"]+)"'
)


def normalize_reciter_id(value: str) -> str:
    raw = unicodedata.normalize("NFKD", str(value or "").strip().lower())
    raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
    raw = raw.replace("/", " ")
    raw = re.sub(r"[^a-z0-9]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    if not raw:
        raise ValueError("reciter_id is required")
    return raw


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_scoped_reciter_id(source: str, value: str | int) -> str:
    source_key = normalize_reciter_id(source)
    value_key = normalize_reciter_id(str(value))
    return f"{source_key}_{value_key}"


def _default_registry() -> dict[str, Any]:
    return {
        "version": 1,
        "updated_at": _utc_now(),
        "reciters": [],
    }


def load_registry(path: str | Path = DEFAULT_RECITERS_PATH) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return _default_registry()
    try:
        payload = orjson.loads(target.read_bytes())
    except orjson.JSONDecodeError:
        return _default_registry()

    if not isinstance(payload, dict):
        return _default_registry()

    reciters_raw = payload.get("reciters")
    if not isinstance(reciters_raw, list):
        return _default_registry()

    reciters: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in reciters_raw:
        if not isinstance(item, dict):
            continue
        try:
            rid = normalize_reciter_id(str(item.get("id") or ""))
        except ValueError:
            continue
        if rid in seen:
            continue
        seen.add(rid)
        reciters.append(
            {
                "id": rid,
                "name": str(item.get("name") or rid.replace("_", " ").title()).strip(),
                "source": str(item.get("source") or "custom").strip() or "custom",
                "notes": str(item.get("notes") or "").strip(),
                "created_at": str(item.get("created_at") or _utc_now()),
                "updated_at": str(item.get("updated_at") or _utc_now()),
            }
        )

    return {
        "version": 1,
        "updated_at": str(payload.get("updated_at") or _utc_now()),
        "reciters": sorted(reciters, key=lambda row: str(row["id"])),
    }


def save_registry(payload: dict[str, Any], path: str | Path = DEFAULT_RECITERS_PATH) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def list_reciters(path: str | Path = DEFAULT_RECITERS_PATH) -> list[dict[str, Any]]:
    payload = load_registry(path)
    reciters = payload.get("reciters")
    if not isinstance(reciters, list):
        return []
    return [row for row in reciters if isinstance(row, dict)]


def get_reciter(reciter_id: str, path: str | Path = DEFAULT_RECITERS_PATH) -> dict[str, Any] | None:
    key = normalize_reciter_id(reciter_id)
    for item in list_reciters(path):
        if str(item.get("id") or "") == key:
            return item
    return None


def upsert_reciter(
    *,
    reciter_id: str,
    name: str,
    source: str,
    notes: str | None = None,
    path: str | Path = DEFAULT_RECITERS_PATH,
) -> dict[str, Any]:
    rid = normalize_reciter_id(reciter_id)
    now = _utc_now()
    payload = load_registry(path)
    reciters = list_reciters(path)

    created_at = now
    updated: dict[str, Any] | None = None
    for item in reciters:
        if str(item.get("id") or "") != rid:
            continue
        created_at = str(item.get("created_at") or now)
        updated = item
        break

    row = {
        "id": rid,
        "name": str(name).strip() or rid.replace("_", " ").title(),
        "source": str(source).strip() or "custom",
        "notes": str(notes or "").strip(),
        "created_at": created_at,
        "updated_at": now,
    }

    if updated is None:
        reciters.append(row)
    else:
        idx = reciters.index(updated)
        reciters[idx] = row

    payload["version"] = 1
    payload["updated_at"] = now
    payload["reciters"] = sorted(reciters, key=lambda item: str(item.get("id") or ""))
    save_registry(payload, path)
    return row


def reciter_exists(
    reciter_id: str,
    *,
    path: str | Path = DEFAULT_RECITERS_PATH,
) -> bool:
    rid = normalize_reciter_id(reciter_id)
    return get_reciter(rid, path) is not None


def fetch_everyayah_reciters() -> list[dict[str, str]]:
    payload = fetch_everyayah_catalog()
    rows: list[dict[str, str]] = []
    for key, value in payload.items():
        if key == "ayahCount" or not str(key).isdigit() or not isinstance(value, dict):
            continue
        subfolder = str(value.get("subfolder") or "").strip()
        if not subfolder:
            continue
        reciter_id = source_scoped_reciter_id("everyayah", subfolder)
        name = str(value.get("name") or reciter_id.replace("_", " ").title()).strip()
        rows.append(
            {
                "id": reciter_id,
                "name": name,
                "source": "everyayah",
                "notes": f"everyayah_key={key}; subfolder={subfolder}",
            }
        )
    return rows


def fetch_quran_com_reciters() -> list[dict[str, str]]:
    payload = fetch_recitation_catalog(language="en")
    recitations = payload.get("recitations") if isinstance(payload, dict) else None
    if not isinstance(recitations, list):
        return []

    rows: list[dict[str, str]] = []
    for item in recitations:
        if not isinstance(item, dict):
            continue
        recitation_id = item.get("id")
        if not isinstance(recitation_id, int):
            continue
        translated = item.get("translated_name")
        translated_name = translated.get("name") if isinstance(translated, dict) else None
        reciter_name = str(item.get("reciter_name") or "").strip()
        label = str(translated_name or reciter_name or f"Quran.com recitation {recitation_id}")
        style = item.get("style")
        style_name = style.get("name") if isinstance(style, dict) else None
        rows.append(
            {
                "id": source_scoped_reciter_id("quran.com", recitation_id),
                "name": label,
                "source": "quran.com",
                "notes": (
                    f"recitation_id={recitation_id}; reciter_name={reciter_name}; style={style_name or '-'}"
                ),
            }
        )
    return rows


def _extract_quranicaudio_qaris_array(html: str) -> str:
    marker = "qaris:["
    start = html.find(marker)
    if start == -1:
        raise ValueError("unable to find qaris payload on quranicaudio homepage")

    begin = html.find("[", start)
    if begin == -1:
        raise ValueError("unable to parse qaris payload on quranicaudio homepage")

    depth = 0
    in_string = False
    escaping = False
    end = -1
    for idx, ch in enumerate(html[begin:], start=begin):
        if in_string:
            if escaping:
                escaping = False
                continue
            if ch == "\\":
                escaping = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "[":
            depth += 1
            continue
        if ch == "]":
            depth -= 1
            if depth == 0:
                end = idx
                break

    if end == -1:
        raise ValueError("unable to parse qaris payload boundaries")
    return html[begin : end + 1]


def fetch_quranicaudio_reciters() -> list[dict[str, str]]:
    html = get_bytes_with_retry(url=QURANICAUDIO_HOME_URL).decode("utf-8", "ignore")
    array_payload = _extract_quranicaudio_qaris_array(html)
    rows: list[dict[str, str]] = []
    for match in _QURANICAUDIO_QARI_PATTERN.finditer(array_payload):
        reciter_num = match.group("id")
        raw_name = match.group("name").replace('\\"', '"').strip()
        raw_path = match.group("path").strip().strip("/")
        if not raw_path:
            continue
        reciter_id = source_scoped_reciter_id("quranicaudio", raw_path)
        rows.append(
            {
                "id": reciter_id,
                "name": raw_name or reciter_id.replace("_", " ").title(),
                "source": "quranicaudio",
                "notes": f"quranicaudio_id={reciter_num}; relative_path={raw_path}/",
            }
        )
    return rows


def prefill_registry_from_sources(
    path: str | Path = DEFAULT_RECITERS_PATH,
    *,
    preserve_existing: bool = True,
) -> dict[str, Any]:
    now = _utc_now()
    merged: dict[str, dict[str, str]] = {}

    if preserve_existing:
        for item in list_reciters(path):
            rid = str(item.get("id") or "").strip()
            if not rid:
                continue
            merged[rid] = {
                "id": rid,
                "name": str(item.get("name") or rid.replace("_", " ").title()).strip(),
                "source": str(item.get("source") or "custom").strip() or "custom",
                "notes": str(item.get("notes") or "").strip(),
            }

    source_rows = (
        fetch_quranicaudio_reciters()
        + fetch_everyayah_reciters()
        + fetch_quran_com_reciters()
    )
    for item in source_rows:
        rid = normalize_reciter_id(item["id"])
        merged[rid] = {
            "id": rid,
            "name": str(item.get("name") or rid.replace("_", " ").title()).strip(),
            "source": str(item.get("source") or "custom").strip() or "custom",
            "notes": str(item.get("notes") or "").strip(),
        }

    reciters_payload = [
        {
            "id": item["id"],
            "name": item["name"],
            "source": item["source"],
            "notes": item["notes"],
            "created_at": now,
            "updated_at": now,
        }
        for item in sorted(merged.values(), key=lambda row: row["id"])
    ]
    payload: dict[str, Any] = {
        "version": 1,
        "updated_at": now,
        "reciters": reciters_payload,
    }
    save_registry(payload, path)
    return payload


__all__ = [
    "DEFAULT_RECITERS_PATH",
    "QURANICAUDIO_HOME_URL",
    "fetch_everyayah_reciters",
    "fetch_quran_com_reciters",
    "fetch_quranicaudio_reciters",
    "get_reciter",
    "list_reciters",
    "load_registry",
    "normalize_reciter_id",
    "source_scoped_reciter_id",
    "prefill_registry_from_sources",
    "reciter_exists",
    "save_registry",
    "upsert_reciter",
]
