from __future__ import annotations

from pathlib import Path
import csv

from quran_audio_data.manifest import ManifestRowModel, ManifestValidationError
from .types import ManifestRow


def read_manifest(manifest_path: str | Path) -> list[ManifestRow]:
    from .types import PipelineError

    path = Path(manifest_path)
    if not path.exists():
        raise PipelineError(f"Manifest file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    required = {"audio_path", "reciter_id", "surah", "ayah"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise PipelineError("Manifest missing required columns: " + ", ".join(sorted(missing)))

    parsed: list[ManifestRow] = []
    for idx, row in enumerate(rows, start=2):
        try:
            normalized = ManifestRowModel.model_validate(row)
        except Exception as exc:
            if hasattr(exc, "errors"):
                raise PipelineError(str(ManifestValidationError(row_index=idx, error=exc))) from exc
            raise PipelineError(f"manifest row {idx}: {exc}") from exc

        parsed.append(
            ManifestRow(
                audio_path=normalized.audio_path_obj,
                reciter_id=normalized.reciter_id,
                surah=normalized.surah,
                ayah=normalized.ayah,
                source_url=normalized.source_url,
                sha256=normalized.sha256,
                language=normalized.language,
                riwaya=normalized.riwaya,
                text_variant=normalized.text_variant,
                reference_split=normalized.reference_split,
            )
        )

    return parsed


__all__ = ["ManifestRow", "read_manifest"]
