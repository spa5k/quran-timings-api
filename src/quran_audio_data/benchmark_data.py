from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import random
import re
from typing import Any

import orjson

from quran_audio_data.core.http import get_bytes_with_retry, get_json_with_retry


EVERYAYAH_RECITATIONS_URL = "https://everyayah.com/data/recitations.js"
QURAN_COM_VERSE_BY_KEY_URL = "https://api.quran.com/api/v4/verses/by_key/{surah}:{ayah}"
EVERYAYAH_AUDIO_URL_TEMPLATE = "https://everyayah.com/data/{subfolder}/{surah:03d}{ayah:03d}.mp3"


@dataclass(slots=True)
class EveryAyahReciter:
    reciter_key: int
    name: str
    subfolder: str
    bitrate: str | None


def _parse_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for {field}: {value!r}") from exc


def _sha256_bytes(content: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(content)
    return digest.hexdigest()


def _sha256_file(file_path: Path) -> str:
    return _sha256_bytes(file_path.read_bytes())


def fetch_everyayah_catalog(
    *,
    timeout_s: float = 20.0,
    retries: int = 5,
    retry_backoff_s: float = 1.0,
) -> dict[str, Any]:
    payload = get_json_with_retry(
        url=EVERYAYAH_RECITATIONS_URL,
        timeout_s=timeout_s,
        retries=retries,
        retry_backoff_s=retry_backoff_s,
    )
    if not isinstance(payload, dict):
        raise ValueError("EveryAyah catalog payload is not a JSON object")
    return payload


def parse_everyayah_reciters(catalog: dict[str, Any]) -> list[EveryAyahReciter]:
    reciters: list[EveryAyahReciter] = []
    for key, value in catalog.items():
        if key == "ayahCount" or not re.fullmatch(r"\d+", str(key)):
            continue
        if not isinstance(value, dict):
            continue
        subfolder = value.get("subfolder")
        name = value.get("name")
        if not isinstance(subfolder, str) or not isinstance(name, str):
            continue
        reciters.append(
            EveryAyahReciter(
                reciter_key=int(key),
                name=name,
                subfolder=subfolder,
                bitrate=value.get("bitrate") if isinstance(value.get("bitrate"), str) else None,
            )
        )
    reciters.sort(key=lambda item: item.reciter_key)
    return reciters


def resolve_everyayah_reciter(
    *,
    catalog: dict[str, Any],
    reciter_key: int | None,
    reciter_subfolder: str | None,
) -> EveryAyahReciter:
    reciters = parse_everyayah_reciters(catalog)
    if not reciters:
        raise ValueError("No reciters found in EveryAyah catalog")

    if reciter_subfolder:
        for reciter in reciters:
            if reciter.subfolder == reciter_subfolder:
                return reciter
        raise ValueError(f"EveryAyah subfolder not found: {reciter_subfolder}")

    key = reciter_key if reciter_key is not None else 1
    for reciter in reciters:
        if reciter.reciter_key == key:
            return reciter
    raise ValueError(f"EveryAyah reciter key not found: {key}")


def get_ayah_count_map(catalog: dict[str, Any]) -> dict[int, int]:
    ayah_counts = catalog.get("ayahCount")
    if not isinstance(ayah_counts, list) or len(ayah_counts) != 114:
        raise ValueError("EveryAyah catalog missing valid ayahCount list")

    out: dict[int, int] = {}
    for index, value in enumerate(ayah_counts, start=1):
        out[index] = _parse_int(value, f"ayahCount[{index}]")
    return out


def fetch_quran_com_verse(
    *,
    surah: int,
    ayah: int,
    timeout_s: float = 20.0,
    retries: int = 5,
    retry_backoff_s: float = 1.0,
) -> dict[str, Any]:
    url = QURAN_COM_VERSE_BY_KEY_URL.format(surah=surah, ayah=ayah)
    payload = get_json_with_retry(
        url=url,
        timeout_s=timeout_s,
        params={
            "words": "true",
            "word_fields": "text_uthmani,verse_key,position",
            "language": "en",
        },
        retries=retries,
        retry_backoff_s=retry_backoff_s,
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("verse"), dict):
        raise ValueError(f"Quran.com verse payload malformed for {surah}:{ayah}")
    return payload["verse"]


def _build_ayah_pool(*, ayah_count_map: dict[int, int], surahs: list[int]) -> list[tuple[int, int]]:
    pool: list[tuple[int, int]] = []
    for surah in surahs:
        max_ayah = ayah_count_map.get(surah)
        if max_ayah is None:
            raise ValueError(f"Surah not present in ayah count map: {surah}")
        for ayah in range(1, max_ayah + 1):
            pool.append((surah, ayah))
    return pool


def _select_ayah_keys(
    *,
    ayah_count_map: dict[int, int],
    count: int,
    surahs: list[int],
    ayah_keys: list[str] | None,
    seed: int,
) -> list[tuple[int, int]]:
    if ayah_keys:
        out: list[tuple[int, int]] = []
        for key in ayah_keys:
            cleaned = key.strip()
            if not cleaned:
                continue
            if ":" not in cleaned:
                raise ValueError(f"Invalid ayah key (expected S:A): {cleaned}")
            surah_str, ayah_str = cleaned.split(":", 1)
            surah = _parse_int(surah_str, "surah")
            ayah = _parse_int(ayah_str, "ayah")
            max_ayah = ayah_count_map.get(surah)
            if max_ayah is None or ayah < 1 or ayah > max_ayah:
                raise ValueError(f"Ayah key out of range: {cleaned}")
            out.append((surah, ayah))
        if not out:
            raise ValueError("No valid ayah keys provided")
        return out

    pool = _build_ayah_pool(ayah_count_map=ayah_count_map, surahs=surahs)
    if not pool:
        raise ValueError("Ayah pool is empty")

    target = min(max(1, count), len(pool))
    rng = random.Random(seed)
    rng.shuffle(pool)
    return sorted(pool[:target])


def _download_everyayah_audio(
    *,
    subfolder: str,
    surah: int,
    ayah: int,
    out_dir: Path,
    timeout_s: float,
    retries: int,
    retry_backoff_s: float,
    reuse_existing: bool,
) -> tuple[Path, str, str]:
    url = EVERYAYAH_AUDIO_URL_TEMPLATE.format(subfolder=subfolder, surah=surah, ayah=ayah)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"{subfolder}_{surah:03d}{ayah:03d}.mp3"
    if reuse_existing and file_path.exists():
        return file_path, url, _sha256_file(file_path)

    content = get_bytes_with_retry(
        url=url,
        timeout_s=timeout_s,
        retries=retries,
        retry_backoff_s=retry_backoff_s,
    )

    file_path.write_bytes(content)
    return file_path, url, _sha256_bytes(content)


def _sanitize_reciter_id(subfolder: str) -> str:
    value = subfolder.lower().replace("/", "_").replace(" ", "_")
    value = re.sub(r"[^a-z0-9_\-]", "", value)
    return value


def prepare_benchmark_data(
    *,
    out_dir: str | Path,
    count: int = 200,
    surahs: list[int] | None = None,
    ayah_keys: list[str] | None = None,
    reciter_key: int | None = None,
    reciter_subfolder: str | None = None,
    seed: int = 42,
    download_audio: bool = True,
    timeout_s: float = 20.0,
    manifest_reciter_id: str | None = None,
    reference_split: str = "benchmark",
    request_retries: int = 5,
    retry_backoff_s: float = 1.0,
    resume: bool = True,
) -> dict[str, Any]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    catalog = fetch_everyayah_catalog(
        timeout_s=timeout_s,
        retries=request_retries,
        retry_backoff_s=retry_backoff_s,
    )
    ayah_count_map = get_ayah_count_map(catalog)
    reciter = resolve_everyayah_reciter(
        catalog=catalog,
        reciter_key=reciter_key,
        reciter_subfolder=reciter_subfolder,
    )

    selected_surahs = sorted(set(surahs or list(range(1, 115))))
    for surah in selected_surahs:
        if surah < 1 or surah > 114:
            raise ValueError(f"Surah out of range: {surah}")

    keys = _select_ayah_keys(
        ayah_count_map=ayah_count_map,
        count=count,
        surahs=selected_surahs,
        ayah_keys=ayah_keys,
        seed=seed,
    )

    audio_dir = out_root / "audio"
    reference_dir = out_root / "reference_templates"
    reference_dir.mkdir(parents=True, exist_ok=True)

    reciter_id = (
        str(manifest_reciter_id).strip().lower()
        if manifest_reciter_id is not None and str(manifest_reciter_id).strip()
        else _sanitize_reciter_id(reciter.subfolder)
    )
    manifest_path = out_root / "benchmark_manifest.csv"

    rows: list[dict[str, str]] = []
    for surah, ayah in keys:
        source_url = EVERYAYAH_AUDIO_URL_TEMPLATE.format(
            subfolder=reciter.subfolder,
            surah=surah,
            ayah=ayah,
        )
        verse_key = f"{surah}:{ayah}"
        reference_path = reference_dir / f"{reciter_id}_s{surah:03d}_a{ayah:03d}.json"

        if not (resume and reference_path.exists()):
            verse = fetch_quran_com_verse(
                surah=surah,
                ayah=ayah,
                timeout_s=timeout_s,
                retries=request_retries,
                retry_backoff_s=retry_backoff_s,
            )

            words = verse.get("words") if isinstance(verse.get("words"), list) else []
            reference_words: list[dict[str, Any]] = []
            for word in words:
                if not isinstance(word, dict):
                    continue
                position = _parse_int(word.get("position"), "word.position")
                text_uthmani = str(word.get("text_uthmani") or word.get("text") or "").strip()
                if not text_uthmani:
                    continue
                reference_words.append(
                    {
                        "ayah": ayah,
                        "word_index_in_ayah": position,
                        "text_uthmani": text_uthmani,
                        "start_s": None,
                        "end_s": None,
                    }
                )

            verse_key = str(verse.get("verse_key") or verse_key)
            reference_payload = {
                "meta": {
                    "reciter_id": reciter_id,
                    "surah": surah,
                    "ayah": ayah,
                    "verse_key": verse_key,
                    "everyayah_subfolder": reciter.subfolder,
                    "everyayah_name": reciter.name,
                    "source": {
                        "quran_com": QURAN_COM_VERSE_BY_KEY_URL.format(surah=surah, ayah=ayah),
                        "everyayah": source_url,
                    },
                },
                "words": reference_words,
            }
            reference_path.write_bytes(orjson.dumps(reference_payload, option=orjson.OPT_INDENT_2))
        elif reference_path.exists():
            try:
                existing_reference = orjson.loads(reference_path.read_bytes())
                if isinstance(existing_reference, dict):
                    meta = existing_reference.get("meta")
                    if isinstance(meta, dict):
                        verse_key = str(meta.get("verse_key") or verse_key)
            except orjson.JSONDecodeError:
                pass

        if download_audio:
            audio_path, source_url, sha256 = _download_everyayah_audio(
                subfolder=reciter.subfolder,
                surah=surah,
                ayah=ayah,
                out_dir=audio_dir,
                timeout_s=timeout_s,
                retries=request_retries,
                retry_backoff_s=retry_backoff_s,
                reuse_existing=resume,
            )
            audio_path_str = str(audio_path)
        else:
            sha256 = ""
            audio_path_str = ""

        rows.append(
            {
                "audio_path": audio_path_str,
                "reciter_id": reciter_id,
                "surah": str(surah),
                "ayah": str(ayah),
                "source_url": source_url,
                "sha256": sha256,
                "language": "ar",
                "riwaya": "",
                "text_variant": "",
                "reference_split": reference_split,
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "audio_path",
                "reciter_id",
                "surah",
                "ayah",
                "source_url",
                "sha256",
                "language",
                "riwaya",
                "text_variant",
                "reference_split",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "count": len(keys),
        "reciter": {
            "key": reciter.reciter_key,
            "name": reciter.name,
            "subfolder": reciter.subfolder,
            "bitrate": reciter.bitrate,
        },
        "manifest_path": str(manifest_path),
        "reference_dir": str(reference_dir),
        "audio_dir": str(audio_dir) if download_audio else None,
        "download_audio": download_audio,
        "seed": seed,
        "request_retries": request_retries,
        "retry_backoff_s": retry_backoff_s,
        "resume": resume,
        "sources": {
            "quran_com": "https://api.quran.com/api/v4/verses/by_key/{surah}:{ayah}",
            "everyayah_catalog": EVERYAYAH_RECITATIONS_URL,
            "everyayah_audio": EVERYAYAH_AUDIO_URL_TEMPLATE,
        },
    }
    metadata_path = out_root / "benchmark_metadata.json"
    metadata_path.write_bytes(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))

    return metadata
