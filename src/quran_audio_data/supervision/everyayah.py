from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib

from quran_audio_data.core.http import get_bytes_with_retry, get_json_with_retry


EVERYAYAH_RECITATIONS_URL = "https://everyayah.com/data/recitations.js"
EVERYAYAH_AUDIO_URL_TEMPLATE = "https://everyayah.com/data/{subfolder}/{surah:03d}{ayah:03d}.mp3"


@dataclass(slots=True)
class EveryAyahAsset:
    surah: int
    ayah: int
    url: str
    sha256: str
    local_path: Path


def fetch_catalog() -> dict[str, Any]:
    payload = get_json_with_retry(url=EVERYAYAH_RECITATIONS_URL)
    if not isinstance(payload, dict):
        raise ValueError("EveryAyah catalog payload is not a JSON object")
    return payload


def build_audio_url(*, subfolder: str, surah: int, ayah: int) -> str:
    return EVERYAYAH_AUDIO_URL_TEMPLATE.format(subfolder=subfolder, surah=surah, ayah=ayah)


def download_ayah_asset(
    *,
    subfolder: str,
    surah: int,
    ayah: int,
    out_dir: str | Path,
    reuse_existing: bool = True,
) -> EveryAyahAsset:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    url = build_audio_url(subfolder=subfolder, surah=surah, ayah=ayah)
    file_name = f"{subfolder}_{surah:03d}{ayah:03d}.mp3"
    target = out_path / file_name

    if reuse_existing and target.exists():
        sha = _sha256(target.read_bytes())
        return EveryAyahAsset(surah=surah, ayah=ayah, url=url, sha256=sha, local_path=target)

    content = get_bytes_with_retry(url=url)
    target.write_bytes(content)
    return EveryAyahAsset(
        surah=surah,
        ayah=ayah,
        url=url,
        sha256=_sha256(content),
        local_path=target,
    )


def _sha256(content: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(content)
    return digest.hexdigest()


__all__ = [
    "EVERYAYAH_RECITATIONS_URL",
    "EVERYAYAH_AUDIO_URL_TEMPLATE",
    "EveryAyahAsset",
    "fetch_catalog",
    "build_audio_url",
    "download_ayah_asset",
]
