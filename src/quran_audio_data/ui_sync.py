from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import filecmp
import json
import re
import shutil
from urllib.parse import urlparse

import orjson


FULL_JSON_RE = re.compile(r"^(?P<reciter_id>.+)_s(?P<surah>\d{3})_full\.json$")
EVERYAYAH_AUDIO_URL_TEMPLATE = "https://everyayah.com/data/{subfolder}/{surah:03d}{ayah:03d}.mp3"
QCOM_CHAPTER_URL_TEMPLATE = (
    "https://api.quran.com/api/v4/chapter_recitations/{recitation_id}/{surah}"
)


@dataclass(frozen=True, slots=True)
class RunCandidate:
    reciter_id: str
    surah: int
    path: Path
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class AudioContract:
    granularity: str
    primary_asset: str | None
    fallback_order: list[str]
    assets: dict[str, dict[str, Any]]
    ayah_audio_urls: dict[int, str]


@dataclass(frozen=True, slots=True)
class SurahArtifact:
    reciter_id: str
    surah: int
    title: str
    audio_src: str
    metadata_endpoint: str
    timings_endpoint: str
    ayah_count: int
    word_count: int
    duration_s: float | None
    qc_coverage: float | None
    segment_source_type: str
    updated_at: str


def _safe_surah_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = orjson.loads(path.read_bytes())
    except orjson.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _render_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _write_json_if_changed(*, path: Path, payload: dict[str, Any], dry_run: bool) -> bool:
    rendered = _render_json(payload)
    previous = path.read_text(encoding="utf-8") if path.exists() else ""
    changed = rendered != previous
    if changed and not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    return changed


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


def _parse_key_value_fields(raw: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in raw.split(":"):
        idx = part.find("=")
        if idx <= 0:
            continue
        key = part[:idx].strip()
        value = part[idx + 1 :].strip()
        if key:
            fields[key] = value
    return fields


def _looks_like_audio_url(url: str | None) -> bool:
    if not isinstance(url, str) or not url.strip():
        return False
    parsed = urlparse(url.strip())
    suffix = Path(parsed.path).suffix.lower()
    return suffix in {".mp3", ".wav", ".m4a", ".ogg", ".opus", ".aac", ".flac"}


def _collect_ayah_numbers(full_payload: dict[str, Any]) -> list[int]:
    ayahs = full_payload.get("ayahs")
    if not isinstance(ayahs, list):
        return []
    values: set[int] = set()
    for row in ayahs:
        if not isinstance(row, dict):
            continue
        ayah_number = _safe_surah_int(row.get("ayah"))
        if ayah_number is None or ayah_number <= 0:
            continue
        values.add(ayah_number)
    return sorted(values)


def _provider_from_url(url: str | None) -> str:
    if not isinstance(url, str):
        return "unknown"
    lowered = url.lower()
    if "everyayah.com" in lowered:
        return "everyayah"
    if "quranicaudio.com" in lowered or "download.quranicaudio.com" in lowered:
        return "quranicaudio"
    if "quran.com" in lowered:
        return "quran_com"
    return "unknown"


def _read_run_summary(path: Path) -> dict[str, Any] | None:
    run_summary_path = path.parent.parent / "run_summary.json"
    return _read_json(run_summary_path)


def _parse_everyayah_source_url(source_url: str | None) -> tuple[str | None, int | None]:
    if not isinstance(source_url, str) or not source_url.startswith("everyayah:"):
        return None, None
    body = source_url[len("everyayah:") :]
    fields = _parse_key_value_fields(body)
    subfolder = fields.get("subfolder")
    surah = _safe_surah_int(fields.get("surah"))
    return subfolder, surah


def _derive_audio_contract(
    *,
    reciter_id: str,
    surah: int,
    full_payload: dict[str, Any],
    reciter_row: dict[str, Any] | None,
    run_summary: dict[str, Any] | None,
) -> AudioContract:
    supervision_sources = (
        full_payload.get("supervision_sources")
        if isinstance(full_payload.get("supervision_sources"), list)
        else []
    )
    source_obj = reciter_row.get("source") if isinstance(reciter_row, dict) else {}
    source_everyayah = source_obj.get("everyayah") if isinstance(source_obj, dict) else {}
    source_qcom = source_obj.get("quran_com") if isinstance(source_obj, dict) else {}
    run_paths = run_summary.get("paths") if isinstance(run_summary, dict) else {}
    run_mapping = run_summary.get("mapping") if isinstance(run_summary, dict) else {}
    run_source_url = run_paths.get("full_audio_source_url") if isinstance(run_paths, dict) else None

    everyayah_subfolder: str | None = None
    everyayah_surah = surah
    direct_surah_link: str | None = (
        run_source_url if _looks_like_audio_url(run_source_url) else None
    )
    qcom_recitation_id: int | None = None

    for value in supervision_sources:
        raw = str(value or "").strip()
        if raw.startswith("everyayah:"):
            body = raw[len("everyayah:") :].strip()
            if body.startswith("http://") or body.startswith("https://"):
                direct_surah_link = body
                continue
            fields = _parse_key_value_fields(body)
            subfolder = fields.get("subfolder")
            if subfolder:
                everyayah_subfolder = subfolder
            parsed_surah = _safe_surah_int(fields.get("surah"))
            if parsed_surah is not None and 1 <= parsed_surah <= 114:
                everyayah_surah = parsed_surah
            continue

        if raw.startswith("qcom:"):
            parts = raw.split(":")
            if len(parts) >= 3:
                recitation_id = _safe_surah_int(parts[2])
                if recitation_id is not None and recitation_id > 0:
                    qcom_recitation_id = recitation_id

    if not everyayah_subfolder and isinstance(source_everyayah, dict):
        raw = source_everyayah.get("subfolder")
        if isinstance(raw, str) and raw.strip():
            everyayah_subfolder = raw.strip()

    if not everyayah_subfolder and isinstance(run_mapping, dict):
        raw = run_mapping.get("everyayah_subfolder")
        if isinstance(raw, str) and raw.strip():
            everyayah_subfolder = raw.strip()

    if not everyayah_subfolder:
        parsed_subfolder, parsed_surah = _parse_everyayah_source_url(
            run_source_url if isinstance(run_source_url, str) else None
        )
        if parsed_subfolder:
            everyayah_subfolder = parsed_subfolder
        if parsed_surah is not None and 1 <= parsed_surah <= 114:
            everyayah_surah = parsed_surah

    if qcom_recitation_id is None and isinstance(source_qcom, dict):
        qcom_recitation_id = _safe_surah_int(source_qcom.get("recitation_id"))
    if qcom_recitation_id is None and isinstance(run_mapping, dict):
        qcom_recitation_id = _safe_surah_int(run_mapping.get("qcom_recitation_id"))

    ayah_audio_urls: dict[int, str] = {}
    if everyayah_subfolder:
        for ayah in _collect_ayah_numbers(full_payload):
            ayah_audio_urls[ayah] = EVERYAYAH_AUDIO_URL_TEMPLATE.format(
                subfolder=everyayah_subfolder,
                surah=everyayah_surah,
                ayah=ayah,
            )

    assets: dict[str, dict[str, Any]] = {}
    if direct_surah_link and _looks_like_audio_url(direct_surah_link):
        assets["surah_direct"] = {
            "provider": _provider_from_url(direct_surah_link),
            "level": "surah",
            "kind": "direct",
            "url": direct_surah_link,
        }
    if qcom_recitation_id is not None and qcom_recitation_id > 0:
        assets["qcom_surah"] = {
            "provider": "quran_com",
            "level": "surah",
            "kind": "resolver",
            "url": QCOM_CHAPTER_URL_TEMPLATE.format(recitation_id=qcom_recitation_id, surah=surah),
            "resolve_json_path": "$.audio_file.audio_url",
        }
    if everyayah_subfolder:
        assets["everyayah_ayah"] = {
            "provider": "everyayah",
            "level": "ayah",
            "kind": "template",
            "template": (
                f"https://everyayah.com/data/{everyayah_subfolder}/{everyayah_surah:03d}"
                "{ayah:03d}.mp3"
            ),
        }

    primary_asset: str | None = None
    if "surah_direct" in assets:
        primary_asset = "surah_direct"
    elif "qcom_surah" in assets:
        primary_asset = "qcom_surah"
    elif "everyayah_ayah" in assets:
        primary_asset = "everyayah_ayah"

    fallback_order: list[str] = []
    for key in ["surah_direct", "qcom_surah", "everyayah_ayah"]:
        if key in assets and key != primary_asset:
            fallback_order.append(key)

    levels = {str(item.get("level") or "") for item in assets.values()}
    if "surah" in levels and "ayah" in levels:
        granularity = "mixed"
    elif "surah" in levels:
        granularity = "surah"
    elif "ayah" in levels:
        granularity = "ayah"
    else:
        granularity = "none"

    return AudioContract(
        granularity=granularity,
        primary_asset=primary_asset,
        fallback_order=fallback_order,
        assets=assets,
        ayah_audio_urls=ayah_audio_urls,
    )


def _surah_title(surah: int) -> str:
    return f"Surah {surah}"


def _iso_from_mtime_ns(mtime_ns: int) -> str:
    return datetime.fromtimestamp(mtime_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _build_surah_metadata_payload(
    *,
    reciter_id: str,
    reciter_name: str,
    surah: int,
    full_payload: dict[str, Any],
    audio_contract: AudioContract,
    timings_endpoint: str,
    updated_at: str,
) -> dict[str, Any]:
    ayahs = full_payload.get("ayahs") if isinstance(full_payload.get("ayahs"), list) else []
    words = full_payload.get("words") if isinstance(full_payload.get("words"), list) else []
    audio = full_payload.get("audio") if isinstance(full_payload.get("audio"), dict) else {}
    qc = full_payload.get("qc") if isinstance(full_payload.get("qc"), dict) else {}

    return {
        "schema_version": "v2",
        "reciter": {
            "slug": reciter_id,
            "name": reciter_name,
        },
        "surah": {
            "number": surah,
            "title": _surah_title(surah),
            "ayah_count": len(ayahs),
            "word_count": len(words),
        },
        "audio": {
            "granularity": audio_contract.granularity,
            "primary_asset": audio_contract.primary_asset,
            "fallback_order": audio_contract.fallback_order,
            "assets": audio_contract.assets,
            "duration_s": audio.get("duration_s"),
            "sample_rate": audio.get("sample_rate"),
            "channels": audio.get("channels"),
        },
        "quality": {
            "coverage": qc.get("coverage"),
            "monotonic": qc.get("monotonic"),
            "duration_match": qc.get("duration_match"),
            "warnings": qc.get("warnings") if isinstance(qc.get("warnings"), list) else [],
            "reason_codes": qc.get("reason_codes")
            if isinstance(qc.get("reason_codes"), list)
            else [],
            "zero_or_negative_ratio": qc.get("zero_or_negative_ratio"),
            "median_confidence": qc.get("median_confidence"),
            "interpolated_ratio": qc.get("interpolated_ratio"),
            "lexical_match_ratio": qc.get("lexical_match_ratio"),
            "speech_end_delta_ratio": qc.get("speech_end_delta_ratio"),
            "quantization_step_ms": qc.get("quantization_step_ms"),
            "everyayah_stitch_eval": full_payload.get("everyayah_stitch_eval"),
        },
        "provenance": {
            "engine": full_payload.get("engine"),
            "selected_candidate_engine": full_payload.get("selected_candidate_engine"),
            "candidate_scores": full_payload.get("candidate_scores"),
            "pass_trace": full_payload.get("pass_trace"),
            "supervision_sources": full_payload.get("supervision_sources"),
            "segment_source_type": full_payload.get("segment_source_type") or "none",
        },
        "timings_endpoint": timings_endpoint,
        "updated_at": updated_at,
    }


def _build_surah_timings_payload(
    *,
    reciter_id: str,
    surah: int,
    full_payload: dict[str, Any],
    audio_contract: AudioContract,
) -> dict[str, Any]:
    ayahs_payload: list[Any] = []
    ayahs = full_payload.get("ayahs") if isinstance(full_payload.get("ayahs"), list) else []
    for row in ayahs:
        if not isinstance(row, dict):
            ayahs_payload.append(row)
            continue
        ayah_number = _safe_surah_int(row.get("ayah"))
        next_row = dict(row)
        if ayah_number is not None and ayah_number in audio_contract.ayah_audio_urls:
            next_row["audio_asset"] = "everyayah_ayah"
            next_row["audio_key"] = f"{ayah_number:03d}"
            next_row["audio_url"] = audio_contract.ayah_audio_urls[ayah_number]
        ayahs_payload.append(next_row)

    return {
        "schema_version": "v2",
        "reciter_slug": reciter_id,
        "surah": surah,
        "ayahs": ayahs_payload,
        "words": full_payload.get("words") if isinstance(full_payload.get("words"), list) else [],
    }


def export_api_surah_files(
    *,
    latest_candidates: dict[tuple[str, int], RunCandidate],
    api_root: Path,
    reciter_name_by_id: dict[str, str],
    reciter_row_by_id: dict[str, dict[str, Any]],
    dry_run: bool,
    prune: bool,
) -> tuple[list[SurahArtifact], int, int, int]:
    artifacts: list[SurahArtifact] = []
    changed = 0
    unchanged = 0
    pruned = 0

    for candidate in sorted(
        latest_candidates.values(), key=lambda item: (item.reciter_id, item.surah)
    ):
        payload = _read_json(candidate.path)
        if payload is None:
            continue

        reciter_name = reciter_name_by_id.get(candidate.reciter_id, candidate.reciter_id)
        reciter_surah_dir = (
            api_root / "reciters" / candidate.reciter_id / "surahs" / str(candidate.surah)
        )
        metadata_path = reciter_surah_dir / "metadata.json"
        timings_path = reciter_surah_dir / "timings.json"
        metadata_endpoint = (
            f"/data/reciters/{candidate.reciter_id}/surahs/{candidate.surah}/metadata.json"
        )
        timings_endpoint = (
            f"/data/reciters/{candidate.reciter_id}/surahs/{candidate.surah}/timings.json"
        )
        updated_at = _iso_from_mtime_ns(candidate.mtime_ns)
        run_summary = _read_run_summary(candidate.path)
        reciter_row = reciter_row_by_id.get(candidate.reciter_id)
        audio_contract = _derive_audio_contract(
            reciter_id=candidate.reciter_id,
            surah=candidate.surah,
            full_payload=payload,
            reciter_row=reciter_row,
            run_summary=run_summary,
        )

        metadata_payload = _build_surah_metadata_payload(
            reciter_id=candidate.reciter_id,
            reciter_name=reciter_name,
            surah=candidate.surah,
            full_payload=payload,
            audio_contract=audio_contract,
            timings_endpoint=timings_endpoint,
            updated_at=updated_at,
        )
        timings_payload = _build_surah_timings_payload(
            reciter_id=candidate.reciter_id,
            surah=candidate.surah,
            full_payload=payload,
            audio_contract=audio_contract,
        )

        metadata_changed = _write_json_if_changed(
            path=metadata_path, payload=metadata_payload, dry_run=dry_run
        )
        timings_changed = _write_json_if_changed(
            path=timings_path, payload=timings_payload, dry_run=dry_run
        )
        if metadata_changed or timings_changed:
            changed += 1
        else:
            unchanged += 1

        artifacts.append(
            SurahArtifact(
                reciter_id=candidate.reciter_id,
                surah=candidate.surah,
                title=_surah_title(candidate.surah),
                audio_src=(
                    str(audio_contract.assets["surah_direct"].get("url") or "")
                    if "surah_direct" in audio_contract.assets
                    else ""
                ),
                metadata_endpoint=metadata_endpoint,
                timings_endpoint=timings_endpoint,
                ayah_count=len(timings_payload["ayahs"]),
                word_count=len(timings_payload["words"]),
                duration_s=(
                    float(metadata_payload["audio"]["duration_s"])
                    if isinstance(metadata_payload["audio"].get("duration_s"), (int, float))
                    else None
                ),
                qc_coverage=(
                    float(metadata_payload["quality"]["coverage"])
                    if isinstance(metadata_payload["quality"].get("coverage"), (int, float))
                    else None
                ),
                segment_source_type=str(
                    metadata_payload["provenance"].get("segment_source_type") or "none"
                ),
                updated_at=updated_at,
            )
        )

    if not prune:
        return artifacts, changed, unchanged, pruned

    selected_reciters = {item.reciter_id for item in artifacts}
    selected_surahs_by_reciter: dict[str, set[str]] = {}
    for item in artifacts:
        selected_surahs_by_reciter.setdefault(item.reciter_id, set()).add(str(item.surah))

    reciters_root = api_root / "reciters"
    if reciters_root.exists():
        for reciter_dir in reciters_root.iterdir():
            if not reciter_dir.is_dir():
                continue
            reciter_id = reciter_dir.name
            if reciter_id not in selected_reciters:
                pruned += sum(1 for file in reciter_dir.rglob("*.json"))
                if not dry_run:
                    shutil.rmtree(reciter_dir, ignore_errors=True)
                continue

            surahs_dir = reciter_dir / "surahs"
            allowed_surahs = selected_surahs_by_reciter.get(reciter_id, set())
            if not surahs_dir.exists():
                continue
            for surah_dir in surahs_dir.iterdir():
                if not surah_dir.is_dir():
                    continue
                if surah_dir.name in allowed_surahs:
                    continue
                pruned += sum(1 for file in surah_dir.rglob("*.json"))
                if not dry_run:
                    shutil.rmtree(surah_dir, ignore_errors=True)

    return artifacts, changed, unchanged, pruned


def _ensure_reciters_payload(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload is None:
        return {
            "schema_version": "v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "counts": {
                "everyayah_source_reciters": 0,
                "quran_com_source_reciters": 0,
                "configured_reciters": 0,
                "enabled_reciters": 0,
            },
            "sources": {},
            "reciters": [],
        }
    if not isinstance(payload.get("reciters"), list):
        payload["reciters"] = []
    return payload


def update_reciters_index(
    *,
    reciters_index_path: Path,
    artifacts: list[SurahArtifact],
    dry_run: bool,
) -> tuple[dict[str, Any], bool, int]:
    payload = _ensure_reciters_payload(reciters_index_path)
    reciters = payload.get("reciters") if isinstance(payload.get("reciters"), list) else []

    by_slug: dict[str, dict[str, Any]] = {}
    for item in reciters:
        if not isinstance(item, dict):
            continue
        slug = str(item.get("slug") or "").strip().lower()
        if slug:
            by_slug[slug] = item

    surahs_by_slug: dict[str, set[int]] = {}
    for artifact in artifacts:
        surahs_by_slug.setdefault(artifact.reciter_id, set()).add(artifact.surah)

    added = 0
    for slug, surahs in surahs_by_slug.items():
        row = by_slug.get(slug)
        if row is None:
            row = {
                "slug": slug,
                "name": slug,
                "enabled": True,
                "check_type": "model_only",
                "capabilities": {"ayah_by_ayah": False, "word_by_word": False},
                "source": {
                    "everyayah": {"subfolder": None, "reciter_key": None, "name": None},
                    "quran_com": {"recitation_id": None, "name": None},
                },
            }
            reciters.append(row)
            by_slug[slug] = row
            added += 1

        row["surahs_available"] = sorted(surahs)
        row["surah_count"] = len(surahs)
        row["endpoints"] = {
            "metadata": f"/data/reciters/{slug}/metadata.json",
        }

    reciters.sort(key=lambda item: str(item.get("slug") or ""))
    payload["reciters"] = reciters
    counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
    counts["configured_reciters"] = len(reciters)
    counts["enabled_reciters"] = sum(
        1 for item in reciters if isinstance(item, dict) and bool(item.get("enabled"))
    )
    payload["counts"] = counts
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()

    changed = _write_json_if_changed(path=reciters_index_path, payload=payload, dry_run=dry_run)
    return payload, changed, added


def write_reciter_metadata_files(
    *,
    api_root: Path,
    reciters_payload: dict[str, Any],
    artifacts: list[SurahArtifact],
    dry_run: bool,
) -> tuple[int, int]:
    by_slug: dict[str, dict[str, Any]] = {}
    reciters = (
        reciters_payload.get("reciters")
        if isinstance(reciters_payload.get("reciters"), list)
        else []
    )
    for item in reciters:
        if not isinstance(item, dict):
            continue
        slug = str(item.get("slug") or "").strip().lower()
        if slug:
            by_slug[slug] = item

    artifacts_by_slug: dict[str, list[SurahArtifact]] = {}
    for artifact in artifacts:
        artifacts_by_slug.setdefault(artifact.reciter_id, []).append(artifact)

    changed = 0
    unchanged = 0
    for slug, surah_rows in artifacts_by_slug.items():
        reciter_row = by_slug.get(slug, {"slug": slug, "name": slug, "enabled": True})
        source = reciter_row.get("source") if isinstance(reciter_row.get("source"), dict) else {}
        capabilities = (
            reciter_row.get("capabilities")
            if isinstance(reciter_row.get("capabilities"), dict)
            else {}
        )

        surahs_payload = [
            {
                "number": item.surah,
                "title": item.title,
                "audio_src": item.audio_src,
                "metadata_endpoint": item.metadata_endpoint,
                "timings_endpoint": item.timings_endpoint,
                "ayah_count": item.ayah_count,
                "word_count": item.word_count,
                "duration_s": item.duration_s,
                "qc_coverage": item.qc_coverage,
                "segment_source_type": item.segment_source_type,
                "updated_at": item.updated_at,
            }
            for item in sorted(surah_rows, key=lambda row: row.surah)
        ]

        payload = {
            "schema_version": "v2",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "reciter": {
                "slug": slug,
                "name": str(reciter_row.get("name") or slug),
                "enabled": bool(reciter_row.get("enabled")),
                "check_type": reciter_row.get("check_type") or "model_only",
                "capabilities": {
                    "ayah_by_ayah": bool(capabilities.get("ayah_by_ayah")),
                    "word_by_word": bool(capabilities.get("word_by_word")),
                },
                "source": source,
                "endpoints": {
                    "metadata": f"/data/reciters/{slug}/metadata.json",
                },
            },
            "surahs": surahs_payload,
        }

        reciter_metadata_path = api_root / "reciters" / slug / "metadata.json"
        if _write_json_if_changed(path=reciter_metadata_path, payload=payload, dry_run=dry_run):
            changed += 1
        else:
            unchanged += 1

    return changed, unchanged


def copy_api_to_target(
    *,
    api_root: Path,
    reciters_index_path: Path,
    target_dir: Path,
    dry_run: bool,
    prune: bool,
) -> tuple[int, int]:
    copied = 0
    skipped = 0

    if prune and target_dir.exists() and not dry_run:
        shutil.rmtree(target_dir)

    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)

    # Keep only new API structure in served UI data.
    for legacy in target_dir.glob("*_full.json"):
        if dry_run:
            copied += 1
        else:
            legacy.unlink(missing_ok=True)
            copied += 1
    legacy_catalog = target_dir / "catalog.json"
    if legacy_catalog.exists():
        if dry_run:
            copied += 1
        else:
            legacy_catalog.unlink(missing_ok=True)
            copied += 1

    stale_audio_dir = target_dir / "audio"
    if prune and stale_audio_dir.exists():
        if dry_run:
            copied += 1
        else:
            shutil.rmtree(stale_audio_dir)
            copied += 1

    for item in ["reciters"]:
        source = api_root / item
        destination = target_dir / item
        if not source.exists():
            continue

        if prune and destination.exists() and not dry_run:
            shutil.rmtree(destination)

        for source_file in source.rglob("*"):
            if source_file.is_dir():
                continue
            relative = source_file.relative_to(source)
            target_file = destination / relative
            if needs_copy(source_file, target_file):
                copied += 1
                if not dry_run:
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, target_file)
            else:
                skipped += 1

    reciters_target = target_dir / "reciters.json"
    if needs_copy(reciters_index_path, reciters_target):
        copied += 1
        if not dry_run:
            reciters_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(reciters_index_path, reciters_target)
    else:
        skipped += 1

    return copied, skipped


def export_api_from_latest_runs(
    *,
    runs_root: Path,
    api_root: Path,
    reciters_index_path: Path,
    ui_data_dir: Path,
    dist_data_dir: Path,
    sync_dist: bool,
    prune_ui: bool,
    dry_run: bool,
    include_reciters: set[str] | None = None,
    include_surahs: set[int] | None = None,
) -> dict[str, Any]:
    latest_candidates = discover_latest_candidates(runs_root=runs_root)
    if include_reciters:
        allow = {
            str(item or "").strip().lower() for item in include_reciters if str(item or "").strip()
        }
        latest_candidates = {
            key: value
            for key, value in latest_candidates.items()
            if value.reciter_id.strip().lower() in allow
        }
    if include_surahs:
        allow_surahs = {int(item) for item in include_surahs if 1 <= int(item) <= 114}
        latest_candidates = {
            key: value
            for key, value in latest_candidates.items()
            if int(value.surah) in allow_surahs
        }
    if not latest_candidates:
        raise RuntimeError(f"No *_full.json files found under {runs_root}")

    reciters_payload = _ensure_reciters_payload(reciters_index_path)
    reciters = (
        reciters_payload.get("reciters")
        if isinstance(reciters_payload.get("reciters"), list)
        else []
    )
    reciter_name_by_id = {
        str(item.get("slug") or "").strip().lower(): str(
            item.get("name") or item.get("slug") or ""
        ).strip()
        for item in reciters
        if isinstance(item, dict)
    }
    reciter_row_by_id = {
        str(item.get("slug") or "").strip().lower(): item
        for item in reciters
        if isinstance(item, dict)
    }

    audio_copied = 0
    audio_unchanged = 0
    audio_pruned = 0
    legacy_audio_dir = api_root / "audio"
    if prune_ui and legacy_audio_dir.exists():
        audio_pruned = sum(1 for item in legacy_audio_dir.rglob("*") if item.is_file())
        if not dry_run:
            shutil.rmtree(legacy_audio_dir)

    artifacts, surah_changed, surah_unchanged, surah_pruned = export_api_surah_files(
        latest_candidates=latest_candidates,
        api_root=api_root,
        reciter_name_by_id=reciter_name_by_id,
        reciter_row_by_id=reciter_row_by_id,
        dry_run=dry_run,
        prune=prune_ui,
    )

    reciters_payload, reciters_changed, reciters_added = update_reciters_index(
        reciters_index_path=reciters_index_path,
        artifacts=artifacts,
        dry_run=dry_run,
    )

    reciter_meta_changed, reciter_meta_unchanged = write_reciter_metadata_files(
        api_root=api_root,
        reciters_payload=reciters_payload,
        artifacts=artifacts,
        dry_run=dry_run,
    )

    ui_copied, ui_skipped = copy_api_to_target(
        api_root=api_root,
        reciters_index_path=reciters_index_path,
        target_dir=ui_data_dir,
        dry_run=dry_run,
        prune=prune_ui,
    )

    dist_copied = 0
    dist_skipped = 0
    if sync_dist:
        dist_copied, dist_skipped = copy_api_to_target(
            api_root=api_root,
            reciters_index_path=reciters_index_path,
            target_dir=dist_data_dir,
            dry_run=dry_run,
            prune=prune_ui,
        )

    return {
        "dry_run": dry_run,
        "keys_selected": len(latest_candidates),
        "api": {
            "surah_changed": surah_changed,
            "surah_unchanged": surah_unchanged,
            "surah_pruned": surah_pruned,
            "audio_copied": audio_copied,
            "audio_unchanged": audio_unchanged,
            "audio_pruned": audio_pruned,
            "reciter_meta_changed": reciter_meta_changed,
            "reciter_meta_unchanged": reciter_meta_unchanged,
            "api_root": str(api_root),
        },
        "reciters_index": {
            "path": str(reciters_index_path),
            "changed": reciters_changed,
            "added_reciters": reciters_added,
        },
        "ui": {
            "copied": ui_copied,
            "unchanged": ui_skipped,
            "target_dir": str(ui_data_dir),
        },
        "dist": {
            "enabled": sync_dist,
            "copied": dist_copied,
            "unchanged": dist_skipped,
            "target_dir": str(dist_data_dir),
        },
    }


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
    """Backward-compatible wrapper around API export for old call sites."""

    _ = prune_catalog_surahs
    reciters_index_path = reciter_catalog_path or Path("data/reciters.json")
    api_root = Path("data/api")
    summary = export_api_from_latest_runs(
        runs_root=runs_root,
        api_root=api_root,
        reciters_index_path=reciters_index_path,
        ui_data_dir=ui_data_dir,
        dist_data_dir=dist_data_dir,
        sync_dist=sync_dist,
        prune_ui=prune_ui,
        dry_run=dry_run,
    )

    # Preserve a subset of old keys for existing scripts.
    return {
        "dry_run": summary["dry_run"],
        "keys_selected": summary["keys_selected"],
        "ui": {
            "copied": summary["ui"]["copied"],
            "unchanged": summary["ui"]["unchanged"],
            "pruned": summary["api"]["surah_pruned"],
            "audio_copied": summary["api"]["audio_copied"],
            "audio_unchanged": summary["api"]["audio_unchanged"],
            "audio_pruned": summary["api"]["audio_pruned"],
            "target_dir": summary["ui"]["target_dir"],
        },
        "catalog": {
            "changed": summary["reciters_index"]["changed"],
            "bootstrap_added_reciters": 0,
            "added_surahs": summary["api"]["surah_changed"],
            "updated_surahs": 0,
            "skipped_missing_reciter": 0,
            "path": summary["reciters_index"]["path"],
        },
        "dist": summary["dist"],
        "api": summary["api"],
        "reciters_index": summary["reciters_index"],
    }
