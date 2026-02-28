from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata
from typing import Any

import orjson


_AR_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_AR_PUNCT = re.compile(r"[\u0600-\u0605\u061B\u061F\u066A-\u066D\u06D4]")
_MULTISPACE = re.compile(r"\s+")


@dataclass(slots=True)
class CanonicalWord:
    surah: int
    ayah: int
    word_index_global: int
    word_index_in_ayah: int
    text_uthmani: str
    text_norm: str


class QuranTextStore:
    """Loads canonical Quran text snapshot and provides deterministic word indexing."""

    def __init__(self, data_path: str | Path | None = None) -> None:
        self.data_path = Path(data_path) if data_path else default_text_path()
        if not self.data_path.exists():
            raise FileNotFoundError(
                "Canonical Quran text file not found at "
                f"{self.data_path}. Provide --text-data or add data/quran_text_uthmani_v1.json"
            )
        self._raw = orjson.loads(self.data_path.read_bytes())
        self._surahs: dict[str, Any] = self._raw.get("surahs", {})
        self._variants: dict[str, dict[str, Any]] = self._raw.get("variants", {})

    @property
    def metadata(self) -> dict[str, Any]:
        return self._raw.get("metadata", {})

    def _resolve_surah_map(
        self,
        *,
        text_variant: str | None = None,
        riwaya: str | None = None,
    ) -> dict[str, Any]:
        if text_variant and text_variant in self._variants:
            variant = self._variants[text_variant]
            if isinstance(variant, dict):
                surahs = variant.get("surahs")
                if isinstance(surahs, dict):
                    return surahs
        if riwaya and riwaya in self._variants:
            variant = self._variants[riwaya]
            if isinstance(variant, dict):
                surahs = variant.get("surahs")
                if isinstance(surahs, dict):
                    return surahs
        return self._surahs

    def get_ayah_text(
        self,
        surah: int,
        ayah: int,
        *,
        text_variant: str | None = None,
        riwaya: str | None = None,
    ) -> str:
        surah_map = self._resolve_surah_map(text_variant=text_variant, riwaya=riwaya)
        surah_obj = surah_map.get(str(surah))
        if not isinstance(surah_obj, dict):
            raise KeyError(f"Surah {surah} not found in canonical text snapshot")
        value = surah_obj.get(str(ayah))
        if not isinstance(value, str):
            raise KeyError(f"Ayah {surah}:{ayah} not found in canonical text snapshot")
        return value

    def get_surah_ayahs(
        self,
        surah: int,
        *,
        text_variant: str | None = None,
        riwaya: str | None = None,
    ) -> list[tuple[int, str]]:
        surah_map = self._resolve_surah_map(text_variant=text_variant, riwaya=riwaya)
        surah_obj = surah_map.get(str(surah))
        if not isinstance(surah_obj, dict):
            raise KeyError(f"Surah {surah} not found in canonical text snapshot")
        ayah_pairs: list[tuple[int, str]] = []
        for ayah_str, text in surah_obj.items():
            if not isinstance(text, str):
                continue
            ayah_pairs.append((int(ayah_str), text))
        return sorted(ayah_pairs, key=lambda x: x[0])

    def build_words(
        self,
        *,
        surah: int,
        ayah: int | None = None,
        text_variant: str | None = None,
        riwaya: str | None = None,
    ) -> list[CanonicalWord]:
        words: list[CanonicalWord] = []
        global_idx = 0

        ayah_rows = (
            [
                (
                    ayah,
                    self.get_ayah_text(
                        surah,
                        ayah,
                        text_variant=text_variant,
                        riwaya=riwaya,
                    ),
                )
            ]
            if ayah is not None
            else self.get_surah_ayahs(
                surah,
                text_variant=text_variant,
                riwaya=riwaya,
            )
        )

        for ayah_number, ayah_text in ayah_rows:
            tokens = tokenize_words(ayah_text)
            for idx_in_ayah, token in enumerate(tokens, start=1):
                global_idx += 1
                words.append(
                    CanonicalWord(
                        surah=surah,
                        ayah=ayah_number,
                        word_index_global=global_idx,
                        word_index_in_ayah=idx_in_ayah,
                        text_uthmani=token,
                        text_norm=normalize_arabic(token),
                    )
                )
        return words


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for alignment scoring while preserving token boundaries."""

    value = unicodedata.normalize("NFKC", text)
    value = _AR_DIACRITICS.sub("", value)
    value = value.replace("ـ", "")
    value = _AR_PUNCT.sub(" ", value)

    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
        "ی": "ي",
        "ې": "ي",
        "ك": "ك",
        "ک": "ك",
    }
    for src, dst in replacements.items():
        value = value.replace(src, dst)

    # Normalize Quranic combining marks that are outside the explicit regex ranges.
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = _MULTISPACE.sub(" ", value).strip()
    return value


def tokenize_words(text: str) -> list[str]:
    parts = [part for part in _MULTISPACE.split(text.strip()) if part]
    return parts


def default_text_path() -> Path:
    package_dir = Path(__file__).resolve().parent
    repo_root = package_dir.parent.parent.parent
    return repo_root / "data" / "quran_text_uthmani_v1.json"
