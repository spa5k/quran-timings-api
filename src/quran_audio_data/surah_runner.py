from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import subprocess
import statistics
from typing import Any

import orjson

from quran_audio_data.core.http import get_bytes_with_retry
from quran_audio_data.pipeline import run_alignment_pipeline
from quran_audio_data.supervision import (
    DEFAULT_RECITER_CATALOG_PATH,
    build_audio_url,
    fetch_chapter_recitation_by_chapter,
    fetch_chapter_recitations,
    get_configured_reciter_entry,
    is_reciter_enabled,
    resolve_reciter_mapping,
)
from quran_audio_data.text.quran_text import QuranTextStore


class PreparedSurahAudio:
    def __init__(
        self,
        *,
        source_url: str,
        source_type: str,
        everyayah_timeline_path: Path | None = None,
        everyayah_bounds: dict[int, tuple[float, float]] | None = None,
    ) -> None:
        self.source_url = source_url
        self.source_type = source_type
        self.everyayah_timeline_path = everyayah_timeline_path
        self.everyayah_bounds = everyayah_bounds or {}


def run_surah_for_reciter(
    *,
    reciter_id: str,
    surah: int,
    out_root: str | Path,
    text_data: str | Path | None = None,
    cache_dir: str | Path | None = None,
    catalog_path: str | Path = DEFAULT_RECITER_CATALOG_PATH,
) -> dict[str, Any]:
    if surah < 1 or surah > 114:
        raise ValueError(f"surah out of range: {surah}")

    normalized_reciter = reciter_id.strip().lower()
    if not normalized_reciter:
        raise ValueError("reciter_id is required")

    if not is_reciter_enabled(normalized_reciter, catalog_path=catalog_path):
        raise ValueError(
            f"reciter is not enabled in catalog: {normalized_reciter} "
            f"(sync with enabled list first)"
        )

    mapping = resolve_reciter_mapping(normalized_reciter, catalog_path=catalog_path)
    if mapping.everyayah_subfolder is None and mapping.qcom_recitation_id is None:
        raise ValueError(
            f"reciter has no EveryAyah or Quran.com audio source mapping: {normalized_reciter}"
        )

    reciter_entry = get_configured_reciter_entry(
        normalized_reciter,
        catalog_path=catalog_path,
    )

    run_dir = Path(out_root) / f"{normalized_reciter}_s{surah:03d}"
    input_dir = run_dir / "input"
    audio_dir = input_dir / "audio"
    output_dir = run_dir / "outputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = audio_dir / f"{normalized_reciter}_{surah:03d}.mp3"
    prepared_audio = _prepare_surah_audio(
        reciter_id=normalized_reciter,
        surah=surah,
        mapping_everyayah_subfolder=mapping.everyayah_subfolder,
        mapping_qcom_recitation_id=mapping.qcom_recitation_id,
        audio_path=audio_path,
        text_data=text_data,
    )
    source_url = prepared_audio.source_url

    sha = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    manifest_path = input_dir / "surah_manifest.csv"
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
        writer.writerow(
            {
                "audio_path": str(audio_path),
                "reciter_id": normalized_reciter,
                "surah": str(surah),
                "ayah": "",
                "source_url": source_url,
                "sha256": sha,
                "language": "ar",
                "riwaya": "",
                "text_variant": "",
                "reference_split": "surah_run",
            }
        )

    pipeline_summary = run_alignment_pipeline(
        manifest_path=manifest_path,
        out_dir=output_dir,
        text_data=text_data,
        cache_dir=cache_dir,
        enable_remote=True,
    )

    segment_source_counts: dict[str, int] = {}
    files_with_qc_warnings = 0
    reason_code_counts: dict[str, int] = {}
    coverage_values: list[float] = []
    everyayah_eval: dict[str, Any] | None = None
    output_payloads: list[tuple[Path, dict[str, Any]]] = []
    for processed in pipeline_summary.outputs:
        payload = orjson.loads(processed.output_json.read_bytes())
        if isinstance(payload, dict):
            output_payloads.append((processed.output_json, payload))
        segment_source = str(payload.get("segment_source_type") or "none")
        segment_source_counts[segment_source] = segment_source_counts.get(segment_source, 0) + 1

        qc = payload.get("qc") if isinstance(payload.get("qc"), dict) else {}
        warnings = qc.get("warnings") if isinstance(qc.get("warnings"), list) else []
        if warnings:
            files_with_qc_warnings += 1
        coverage = qc.get("coverage")
        if isinstance(coverage, (int, float)):
            coverage_values.append(float(coverage))

        reason_codes = qc.get("reason_codes") if isinstance(qc.get("reason_codes"), list) else []
        for code in reason_codes:
            key = str(code)
            reason_code_counts[key] = reason_code_counts.get(key, 0) + 1

        if prepared_audio.everyayah_bounds and everyayah_eval is None:
            ayahs_payload = payload.get("ayahs")
            if isinstance(ayahs_payload, list):
                candidate = _evaluate_ayah_timing_against_reference(
                    predicted_ayahs=ayahs_payload,
                    reference_bounds=prepared_audio.everyayah_bounds,
                )
                if candidate is not None:
                    everyayah_eval = candidate

    avg_coverage = sum(coverage_values) / len(coverage_values) if coverage_values else 0.0
    min_coverage = min(coverage_values) if coverage_values else 0.0

    if output_payloads:
        timeline_path_str = (
            str(prepared_audio.everyayah_timeline_path)
            if prepared_audio.everyayah_timeline_path is not None
            else None
        )
        for output_path, payload in output_payloads:
            changed = False
            if payload.get("everyayah_stitch_eval") != everyayah_eval:
                payload["everyayah_stitch_eval"] = everyayah_eval
                changed = True
            if payload.get("everyayah_stitch_timeline_path") != timeline_path_str:
                payload["everyayah_stitch_timeline_path"] = timeline_path_str
                changed = True
            if changed:
                output_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))

    summary: dict[str, Any] = {
        "reciter_id": normalized_reciter,
        "surah": surah,
        "ayah_count": None,
        "catalog_path": str(catalog_path),
        "reciter_entry": reciter_entry,
        "mapping": {
            "everyayah_subfolder": mapping.everyayah_subfolder,
            "qcom_recitation_id": mapping.qcom_recitation_id,
            "qcom_word_supervision_supported": mapping.qcom_word_supervision_supported,
        },
        "paths": {
            "run_dir": str(run_dir),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "manifest_path": str(manifest_path),
            "reference_dir": None,
            "full_audio_path": str(audio_path),
            "full_audio_source_url": source_url,
            "everyayah_stitch_timeline_path": (
                str(prepared_audio.everyayah_timeline_path)
                if prepared_audio.everyayah_timeline_path is not None
                else None
            ),
        },
        "pipeline": {
            "schema_version": pipeline_summary.schema_version,
            "total": pipeline_summary.total,
            "succeeded": pipeline_summary.succeeded,
            "failed": pipeline_summary.failed,
            "fallback_used": pipeline_summary.fallback_used,
            "priors_used": pipeline_summary.priors_used,
            "elapsed_s": pipeline_summary.elapsed_s,
            "attempted_engines": pipeline_summary.attempted_engines,
            "errors": pipeline_summary.errors,
        },
        "quality": {
            "files_with_qc_warnings": files_with_qc_warnings,
            "avg_coverage": avg_coverage,
            "min_coverage": min_coverage,
            "segment_source_counts": segment_source_counts,
            "reason_code_counts": reason_code_counts,
            "everyayah_stitch_eval": everyayah_eval,
        },
    }

    summary_path = run_dir / "run_summary.json"
    summary_path.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
    summary["paths"]["summary_path"] = str(summary_path)
    return summary


def _prepare_surah_audio(
    *,
    reciter_id: str,
    surah: int,
    mapping_everyayah_subfolder: str | None,
    mapping_qcom_recitation_id: int | None,
    audio_path: Path,
    text_data: str | Path | None,
) -> PreparedSurahAudio:
    if mapping_everyayah_subfolder:
        return _build_everyayah_surah_audio(
            subfolder=mapping_everyayah_subfolder,
            surah=surah,
            audio_path=audio_path,
            text_data=text_data,
        )

    if mapping_qcom_recitation_id is None:
        raise ValueError(f"reciter has no usable source mapping: {reciter_id}")

    return _download_qcom_chapter_audio(
        qcom_recitation_id=mapping_qcom_recitation_id,
        surah=surah,
        audio_path=audio_path,
    )


def _build_everyayah_surah_audio(
    *,
    subfolder: str,
    surah: int,
    audio_path: Path,
    text_data: str | Path | None,
) -> PreparedSurahAudio:
    store = QuranTextStore(data_path=text_data) if text_data else QuranTextStore()
    ayah_count = len(store.get_surah_ayahs(surah))
    if ayah_count <= 0:
        raise ValueError(f"failed to resolve ayah count for surah {surah}")

    segments_dir = audio_path.parent / f"{audio_path.stem}_everyayah_segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    ayah_paths: list[Path] = []
    ayah_meta: list[dict[str, Any]] = []
    cursor_s = 0.0
    for ayah in range(1, ayah_count + 1):
        url = build_audio_url(subfolder=subfolder, surah=surah, ayah=ayah)
        ayah_path = segments_dir / f"{surah:03d}{ayah:03d}.mp3"
        if not ayah_path.exists():
            content = get_bytes_with_retry(url=url, timeout_s=30.0, retries=8, retry_backoff_s=1.0)
            ayah_path.write_bytes(content)
        sha = hashlib.sha256(ayah_path.read_bytes()).hexdigest()
        duration_s = _probe_duration_s(ayah_path)
        start_s = cursor_s
        end_s = start_s + duration_s
        cursor_s = end_s
        ayah_meta.append(
            {
                "ayah": ayah,
                "start_s": round(start_s, 6),
                "end_s": round(end_s, 6),
                "duration_s": round(duration_s, 6),
                "audio_url": url,
                "audio_path": str(ayah_path),
                "sha256": sha,
            }
        )
        ayah_paths.append(ayah_path)

    _concat_mp3_files(ayah_paths=ayah_paths, output_path=audio_path)
    full_duration_s = _probe_duration_s(audio_path)
    timeline_path = audio_path.parent.parent / "everyayah_stitch_timeline.json"
    timeline_payload = {
        "source_type": "everyayah_stitch",
        "subfolder": subfolder,
        "surah": surah,
        "full_audio_path": str(audio_path),
        "full_audio_duration_s": round(full_duration_s, 6),
        "ayah_boundaries": ayah_meta,
    }
    timeline_path.write_bytes(orjson.dumps(timeline_payload, option=orjson.OPT_INDENT_2))
    bounds = {
        int(item["ayah"]): (float(item["start_s"]), float(item["end_s"]))
        for item in ayah_meta
    }
    return PreparedSurahAudio(
        source_url=f"everyayah:subfolder={subfolder}:surah={surah}:scope=full_surah",
        source_type="everyayah_stitch",
        everyayah_timeline_path=timeline_path,
        everyayah_bounds=bounds,
    )


def _concat_mp3_files(*, ayah_paths: list[Path], output_path: Path) -> None:
    concat_manifest = output_path.parent / f"{output_path.stem}_concat.txt"
    lines = [
        "file '" + str(path.resolve()).replace("'", "'\\''") + "'"
        for path in ayah_paths
    ]
    concat_manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        cmd_copy = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_manifest),
            "-c",
            "copy",
            str(output_path),
        ]
        result_copy = subprocess.run(cmd_copy, check=False, capture_output=True, text=True)
        if result_copy.returncode == 0:
            return

        cmd_encode = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_manifest),
            "-c:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(output_path),
        ]
        result_encode = subprocess.run(cmd_encode, check=False, capture_output=True, text=True)
        if result_encode.returncode != 0:
            raise RuntimeError(result_encode.stderr.strip() or "ffmpeg concat failed")
    finally:
        concat_manifest.unlink(missing_ok=True)


def _download_qcom_chapter_audio(
    *,
    qcom_recitation_id: int,
    surah: int,
    audio_path: Path,
) -> PreparedSurahAudio:
    source_url = _resolve_qcom_chapter_audio_url(
        qcom_recitation_id=qcom_recitation_id,
        surah=surah,
    )
    content = get_bytes_with_retry(url=source_url, timeout_s=30.0, retries=8, retry_backoff_s=1.0)
    audio_path.write_bytes(content)
    return PreparedSurahAudio(
        source_url=source_url,
        source_type="quran_com_chapter",
    )


def _resolve_qcom_chapter_audio_url(*, qcom_recitation_id: int, surah: int) -> str:
    payload = fetch_chapter_recitation_by_chapter(
        qcom_recitation_id,
        surah,
        include_segments=False,
    )
    audio_file = payload.get("audio_file") if isinstance(payload, dict) else None
    if isinstance(audio_file, dict):
        direct = audio_file.get("audio_url")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()

    listing = fetch_chapter_recitations(qcom_recitation_id)
    audio_files = listing.get("audio_files") if isinstance(listing, dict) else None
    if isinstance(audio_files, list):
        for item in audio_files:
            if not isinstance(item, dict):
                continue
            chapter_id = item.get("chapter_id")
            try:
                chapter_val = int(chapter_id)
            except (TypeError, ValueError):
                continue
            if chapter_val != surah:
                continue
            audio_url = item.get("audio_url")
            if isinstance(audio_url, str) and audio_url.strip():
                return audio_url.strip()

    raise ValueError(
        f"unable to resolve Quran.com chapter audio URL for recitation {qcom_recitation_id} surah {surah}"
    )


def _probe_duration_s(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffprobe failed for {path}")
    value = result.stdout.strip()
    try:
        duration_s = float(value)
    except ValueError as exc:
        raise RuntimeError(f"invalid ffprobe duration for {path}: {value}") from exc
    if duration_s <= 0:
        raise RuntimeError(f"non-positive duration for {path}: {duration_s}")
    return duration_s


def _evaluate_ayah_timing_against_reference(
    *,
    predicted_ayahs: list[dict[str, Any]],
    reference_bounds: dict[int, tuple[float, float]],
) -> dict[str, Any] | None:
    if not predicted_ayahs or not reference_bounds:
        return None

    rows: list[dict[str, float | int]] = []
    matched = 0
    valid_predictions = 0
    for item in predicted_ayahs:
        if not isinstance(item, dict):
            continue
        ayah_val = item.get("ayah")
        start_val = item.get("start_s")
        end_val = item.get("end_s")
        try:
            ayah = int(ayah_val)
            pred_start = float(start_val)
            pred_end = float(end_val)
        except (TypeError, ValueError):
            continue
        valid_predictions += 1
        ref = reference_bounds.get(ayah)
        if ref is None:
            continue
        ref_start, ref_end = ref
        rows.append(
            {
                "ayah": ayah,
                "pred_start_s": pred_start,
                "pred_end_s": pred_end,
                "ref_start_s": ref_start,
                "ref_end_s": ref_end,
                "delta_start_s": pred_start - ref_start,
                "delta_end_s": pred_end - ref_end,
            }
        )
        matched += 1

    if not rows:
        return None

    start_errors_ms = [abs(float(row["delta_start_s"])) * 1000.0 for row in rows]
    end_errors_ms = [abs(float(row["delta_end_s"])) * 1000.0 for row in rows]
    boundary_errors_ms = [value for pair in zip(start_errors_ms, end_errors_ms) for value in pair]
    if not boundary_errors_ms:
        return None

    start_deltas_s = [float(row["delta_start_s"]) for row in rows]
    offset_s = float(statistics.median(start_deltas_s)) if start_deltas_s else 0.0
    normalized_errors_ms: list[float] = []
    ayah_differences: list[dict[str, float | int]] = []
    for row in rows:
        ayah = int(row["ayah"])
        pred_start = float(row["pred_start_s"])
        pred_end = float(row["pred_end_s"])
        ref_start = float(row["ref_start_s"])
        ref_end = float(row["ref_end_s"])
        delta_start_ms = (pred_start - ref_start) * 1000.0
        delta_end_ms = (pred_end - ref_end) * 1000.0
        norm_delta_start_ms = (pred_start - (ref_start + offset_s)) * 1000.0
        norm_delta_end_ms = (pred_end - (ref_end + offset_s)) * 1000.0
        abs_start_ms = abs(delta_start_ms)
        abs_end_ms = abs(delta_end_ms)
        norm_abs_start_ms = abs(norm_delta_start_ms)
        norm_abs_end_ms = abs(norm_delta_end_ms)
        normalized_errors_ms.extend([norm_abs_start_ms, norm_abs_end_ms])
        ayah_differences.append(
            {
                "ayah": ayah,
                "pred_start_s": round(pred_start, 3),
                "pred_end_s": round(pred_end, 3),
                "ref_start_s": round(ref_start, 3),
                "ref_end_s": round(ref_end, 3),
                "delta_start_ms": round(delta_start_ms, 3),
                "delta_end_ms": round(delta_end_ms, 3),
                "abs_start_ms": round(abs_start_ms, 3),
                "abs_end_ms": round(abs_end_ms, 3),
                "norm_delta_start_ms": round(norm_delta_start_ms, 3),
                "norm_delta_end_ms": round(norm_delta_end_ms, 3),
                "norm_abs_start_ms": round(norm_abs_start_ms, 3),
                "norm_abs_end_ms": round(norm_abs_end_ms, 3),
            }
        )
    ayah_differences.sort(key=lambda item: int(item["ayah"]))

    return {
        "expected_ayahs": len(reference_bounds),
        "predicted_ayahs": valid_predictions,
        "matched_ayahs": matched,
        "coverage_vs_reference": round(matched / max(1, len(reference_bounds)), 6),
        "start_error_median_ms": round(float(statistics.median(start_errors_ms)), 3),
        "end_error_median_ms": round(float(statistics.median(end_errors_ms)), 3),
        "boundary_error_median_ms": round(float(statistics.median(boundary_errors_ms)), 3),
        "boundary_error_p95_ms": round(_percentile(boundary_errors_ms, 95), 3),
        "boundary_hit_rate_20ms": round(_hit_rate(boundary_errors_ms, 20.0), 6),
        "boundary_hit_rate_50ms": round(_hit_rate(boundary_errors_ms, 50.0), 6),
        "boundary_hit_rate_80ms": round(_hit_rate(boundary_errors_ms, 80.0), 6),
        "start_offset_s": round(offset_s, 6),
        "offset_normalized_boundary_error_median_ms": round(float(statistics.median(normalized_errors_ms)), 3),
        "offset_normalized_boundary_error_p95_ms": round(_percentile(normalized_errors_ms, 95), 3),
        "offset_normalized_hit_rate_20ms": round(_hit_rate(normalized_errors_ms, 20.0), 6),
        "offset_normalized_hit_rate_50ms": round(_hit_rate(normalized_errors_ms, 50.0), 6),
        "offset_normalized_hit_rate_80ms": round(_hit_rate(normalized_errors_ms, 80.0), 6),
        "ayah_differences": ayah_differences,
    }


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (p / 100.0)
    low = int(rank)
    high = min(len(ordered) - 1, low + 1)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _hit_rate(values: list[float], threshold_ms: float) -> float:
    if not values:
        return 0.0
    hits = sum(1 for value in values if value <= (threshold_ms + 1e-6))
    return hits / len(values)


__all__ = ["run_surah_for_reciter"]
