from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import orjson
from rich.console import Console
from rich.table import Table
import typer

from quran_audio_data.benchmark_data import prepare_benchmark_data
from quran_audio_data.core.parsing import parse_csv_ints, parse_csv_strings
from quran_audio_data.core.settings import get_settings
from quran_audio_data.corpus_builder import DEFAULT_SOURCE_URL, build_canonical_corpus
from quran_audio_data.evaluation import evaluate_bakeoff, evaluate_predictions
from quran_audio_data.gpu import doctor_gpu
from quran_audio_data.surah_runner import run_surah_for_reciter
from quran_audio_data.pipeline import (
    benchmark_pipeline,
    run_alignment_pipeline,
    validate_outputs,
)
from quran_audio_data.pipeline.artifacts import output_stem
from quran_audio_data.pipeline.manifest import read_manifest
from quran_audio_data.supervision import (
    DEFAULT_RECITER_CATALOG_PATH,
    build_audio_url,
    get_configured_reciter_entry,
    fetch_best_verse_segments,
    fetch_chapter_recitation_by_chapter,
    fetch_chapter_recitations,
    fetch_recitation_catalog,
    fetch_verse_recitation_by_ayah,
    fetch_verse_recitations_by_chapter,
    is_reciter_enabled,
    read_reciter_catalog,
    resolve_reciter_mapping,
    resolve_verse_audio_url,
    write_reciter_catalog,
)
from quran_audio_data.ui_sync import sync_ui_from_latest_runs


console = Console()
app = typer.Typer(help="Quran audio timing extraction CLI")


def _print_key_values(title: str, values: dict[str, Any]) -> None:
    table = Table(title=title)
    table.add_column("Key")
    table.add_column("Value")
    for key, value in values.items():
        table.add_row(key, str(value))
    console.print(table)


def _print_errors(errors: list[str]) -> None:
    if not errors:
        return
    console.print("[bold red]Errors:[/bold red]")
    for error in errors:
        console.print(f"- {error}")


@app.command("align")
def align_cmd(
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, file_okay=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False, dir_okay=True)],
    text_data: Annotated[Path | None, typer.Option("--text-data")] = None,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(get_settings().cache_dir),
) -> None:
    """Run one-shot max-quality pipeline (always strict multi-pass)."""

    summary = run_alignment_pipeline(
        manifest_path=manifest,
        out_dir=out,
        engine="nemo",
        multi_engine=["nemo", "mfa", "whisperx"],
        accuracy_mode="strict",
        availability_policy="best_effort",
        device="auto",
        text_data=text_data,
        cache_dir=cache_dir,
        enable_remote=True,
    )

    _print_key_values(
        "Alignment Summary",
        {
            "schema_version": summary.schema_version,
            "total": summary.total,
            "succeeded": summary.succeeded,
            "failed": summary.failed,
            "aligned": summary.aligned,
            "fallback": summary.fallback_used,
            "priors_used": summary.priors_used,
            "elapsed_s": f"{summary.elapsed_s:.2f}",
            "attempted_engines": ",".join(summary.attempted_engines),
        },
    )
    _print_errors(summary.errors)
    if summary.errors:
        raise typer.Exit(code=1)


@app.command("validate")
def validate_cmd(
    input_path: Annotated[Path, typer.Option("--input", exists=True)],
) -> None:
    """Validate schema output JSON(s)."""

    valid, invalid, errors = validate_outputs(input_path)
    _print_key_values(
        "Validation Summary",
        {
            "valid": valid,
            "invalid": invalid,
        },
    )
    _print_errors(errors)
    if errors:
        raise typer.Exit(code=1)


@app.command("benchmark")
def benchmark_cmd(
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, file_okay=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False, dir_okay=True)],
    sample_size: Annotated[int, typer.Option("--sample-size", min=1)] = 5,
    text_data: Annotated[Path | None, typer.Option("--text-data")] = None,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(get_settings().cache_dir),
) -> None:
    """Benchmark pipeline over a manifest sample."""

    report = benchmark_pipeline(
        manifest_path=manifest,
        out_dir=out,
        sample_size=sample_size,
        engine="nemo",
        multi_engine=["nemo", "mfa", "whisperx"],
        accuracy_mode="strict",
        availability_policy="best_effort",
        device="auto",
        text_data=text_data,
        cache_dir=cache_dir,
        enable_remote=True,
    )
    _print_key_values(
        "Benchmark Summary",
        {
            "sample_size": report["sample_size"],
            "succeeded": report["succeeded"],
            "failed": report["failed"],
            "aligned": report["aligned"],
            "fallback": report["fallback_used"],
            "priors_used": report["priors_used"],
            "elapsed_s": f"{report['elapsed_s']:.2f}",
            "avg_file_runtime_s": f"{report['avg_file_runtime_s']:.2f}",
            "attempted_engines": ",".join(report.get("attempted_engines", [])),
        },
    )
    _print_errors(report["errors"])
    if report["errors"]:
        raise typer.Exit(code=1)


@app.command("eval")
def eval_cmd(
    pred_dir: Annotated[Path, typer.Option("--pred-dir", exists=True)],
    reference_dir: Annotated[Path, typer.Option("--reference-dir", exists=True)],
    report: Annotated[Path | None, typer.Option("--report")] = None,
) -> None:
    """Evaluate predicted timings against reference annotations."""

    output = evaluate_predictions(pred_dir=pred_dir, reference_dir=reference_dir)
    summary = output["summary"]
    _print_key_values(
        "Evaluation Summary",
        {
            "matched_ayahs": output["matched_ayahs"],
            "missing_predictions": len(output["missing_predictions"]),
            "median_ms": f"{summary['median_abs_error_ms']:.2f}",
            "p95_ms": f"{summary['p95_abs_error_ms']:.2f}",
            "hit50": f"{summary['hit_rate_50ms']:.3f}",
            "passes_targets": output["passes_targets"],
        },
    )
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_bytes(orjson.dumps(output, option=orjson.OPT_INDENT_2))
    if not output["passes_targets"]:
        raise typer.Exit(code=1)


@app.command("eval-bakeoff")
def eval_bakeoff_cmd(
    pred_dir: Annotated[Path, typer.Option("--pred-dir", exists=True)],
    reference_dir: Annotated[Path, typer.Option("--reference-dir", exists=True)],
    report: Annotated[Path | None, typer.Option("--report")] = None,
) -> None:
    """Evaluate with supported/unsupported reciter split and supervision coverage."""

    output = evaluate_bakeoff(pred_dir=pred_dir, reference_dir=reference_dir)
    _print_key_values(
        "Bakeoff Summary",
        {
            "matched_ayahs": output["overall"]["matched_ayahs"],
            "median_ms": f"{output['overall']['summary']['median_abs_error_ms']:.2f}",
            "p95_ms": f"{output['overall']['summary']['p95_abs_error_ms']:.2f}",
            "supported_reciters": output["coverage"]["supported_reciters"],
            "unsupported_reciters": output["coverage"]["unsupported_reciters"],
            "segment_shape_3_field_ratio": f"{output['segment_shape_usage_ratio']['3_field']:.3f}",
            "segment_shape_4_field_ratio": f"{output['segment_shape_usage_ratio']['4_field']:.3f}",
        },
    )
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_bytes(orjson.dumps(output, option=orjson.OPT_INDENT_2))


@app.command("prepare-supervision")
def prepare_supervision_cmd(
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, file_okay=True, dir_okay=False)],
    out_dir: Annotated[Path, typer.Option("--out-dir")] = Path(".cache/supervision"),
) -> None:
    """Fetch and materialize EveryAyah/Quran.com supervision payloads per manifest row."""

    rows = read_manifest(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared = 0
    qcom_supervised = 0
    unsupported = 0
    failures: list[str] = []

    for row in rows:
        mapping = resolve_reciter_mapping(row.reciter_id)
        payload: dict[str, Any] = {
            "reciter_id": row.reciter_id,
            "surah": row.surah,
            "ayah": row.ayah,
            "everyayah_subfolder": mapping.everyayah_subfolder,
            "qcom_recitation_id": mapping.qcom_recitation_id,
            "qcom_word_supervision_supported": mapping.qcom_word_supervision_supported,
            "everyayah_url": (
                build_audio_url(subfolder=mapping.everyayah_subfolder, surah=row.surah, ayah=row.ayah)
                if row.ayah is not None and mapping.everyayah_subfolder
                else None
            ),
            "segment_source_type": "none",
            "segments": [],
        }

        if mapping.qcom_word_supervision_supported and mapping.qcom_recitation_id is not None and row.ayah is not None:
            try:
                segment_payload = fetch_best_verse_segments(
                    recitation_id=mapping.qcom_recitation_id,
                    chapter=row.surah,
                    verse_key=f"{row.surah}:{row.ayah}",
                )
                if segment_payload is not None:
                    payload["segment_source_type"] = segment_payload.source_type
                    payload["segments"] = [
                        {
                            "word_index": segment.word_index,
                            "start_ms": segment.start_ms,
                            "end_ms": segment.end_ms,
                        }
                        for segment in segment_payload.segments
                    ]
                    qcom_supervised += 1
            except Exception as exc:
                failures.append(f"{row.reciter_id}:{row.surah}:{row.ayah}: {exc}")
        elif not mapping.qcom_word_supervision_supported:
            unsupported += 1

        target = out_dir / f"{output_stem(row)}_supervision.json"
        target.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        prepared += 1

    summary = {
        "prepared": prepared,
        "qcom_supervised": qcom_supervised,
        "unsupported_reciter_rows": unsupported,
        "failures": failures,
    }
    console.print_json(orjson.dumps(summary).decode("utf-8"))
    if failures:
        raise typer.Exit(code=1)


@app.command("validate-supervision")
def validate_supervision_cmd(
    reciter_id: Annotated[int, typer.Option("--reciter-id")] = 2,
    chapter: Annotated[int, typer.Option("--chapter")] = 1,
    verse_key: Annotated[str, typer.Option("--verse-key")] = "1:1",
) -> None:
    """Validate live Quran.com audio endpoint contracts used for supervision."""

    checks: list[tuple[str, bool, str]] = []

    try:
        catalog = fetch_recitation_catalog()
        ok = isinstance(catalog, dict) and isinstance(catalog.get("recitations"), list)
        checks.append(("resources/recitations", ok, "" if ok else "missing recitations[]"))
    except Exception as exc:
        checks.append(("resources/recitations", False, str(exc)))

    try:
        chapter_list = fetch_chapter_recitations(reciter_id)
        audio_files = chapter_list.get("audio_files") if isinstance(chapter_list, dict) else None
        ok = isinstance(audio_files, list)
        checks.append(("chapter_recitations/{reciter_id}", ok, "" if ok else "missing audio_files[]"))
    except Exception as exc:
        checks.append(("chapter_recitations/{reciter_id}", False, str(exc)))

    try:
        chapter_payload = fetch_chapter_recitation_by_chapter(reciter_id, chapter, include_segments=True)
        audio_file = chapter_payload.get("audio_file") if isinstance(chapter_payload, dict) else None
        timestamps = audio_file.get("timestamps") if isinstance(audio_file, dict) else None
        ok = isinstance(timestamps, list)
        checks.append(("chapter_recitations/{id}/{chapter}?segments=true", ok, "" if ok else "missing timestamps[]"))
    except Exception as exc:
        checks.append(("chapter_recitations/{id}/{chapter}?segments=true", False, str(exc)))

    try:
        by_chapter = fetch_verse_recitations_by_chapter(reciter_id, chapter)
        audio_files = by_chapter.get("audio_files") if isinstance(by_chapter, dict) else None
        ok = isinstance(audio_files, list)
        if ok and audio_files:
            sample = audio_files[0]
            resolved = resolve_verse_audio_url(sample.get("url") if isinstance(sample, dict) else None)
            ok = isinstance(resolved, str) and resolved.startswith("https://verses.quran.com/")
        checks.append(("recitations/{id}/by_chapter/{chapter}", ok, "" if ok else "missing/invalid verse url"))
    except Exception as exc:
        checks.append(("recitations/{id}/by_chapter/{chapter}", False, str(exc)))

    try:
        by_ayah = fetch_verse_recitation_by_ayah(reciter_id, verse_key)
        audio_files = by_ayah.get("audio_files") if isinstance(by_ayah, dict) else None
        ok = isinstance(audio_files, list)
        checks.append(("recitations/{id}/by_ayah/{verse_key}", ok, "" if ok else "missing audio_files[]"))
    except Exception as exc:
        checks.append(("recitations/{id}/by_ayah/{verse_key}", False, str(exc)))

    passed = all(item[1] for item in checks)
    payload = {
        "passed": passed,
        "checks": [
            {"endpoint": endpoint, "ok": ok, "error": error}
            for endpoint, ok, error in checks
        ],
    }
    console.print_json(orjson.dumps(payload).decode("utf-8"))
    if not passed:
        raise typer.Exit(code=1)


@app.command("doctor-gpu")
def doctor_gpu_cmd() -> None:
    """Report GPU/CUDA readiness for strict alignment mode."""

    report = doctor_gpu()
    console.print_json(orjson.dumps(report).decode("utf-8"))


@app.command("benchmark-data")
def benchmark_data_cmd(
    out_dir: Annotated[Path, typer.Option("--out-dir")] = Path("benchmarks/generated"),
    count: Annotated[int, typer.Option("--count", min=1)] = 200,
    surahs: Annotated[
        str | None,
        typer.Option("--surahs", help="Comma-separated surah list, e.g. 1,2,112"),
    ] = None,
    ayah_keys: Annotated[
        str | None,
        typer.Option("--ayah-keys", help="Comma-separated ayah keys, e.g. 1:1,1:2,112:1"),
    ] = None,
    reciter_key: Annotated[int | None, typer.Option("--reciter-key")] = None,
    reciter_subfolder: Annotated[str | None, typer.Option("--reciter-subfolder")] = None,
    manifest_reciter_id: Annotated[str | None, typer.Option("--manifest-reciter-id")] = None,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    download_audio: Annotated[bool, typer.Option("--download-audio/--no-download-audio")] = True,
    timeout_s: Annotated[float, typer.Option("--timeout-s")] = 20.0,
    request_retries: Annotated[int, typer.Option("--request-retries", min=0)] = 5,
    retry_backoff_s: Annotated[float, typer.Option("--retry-backoff-s", min=0.0)] = 1.0,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    reference_split: Annotated[str, typer.Option("--reference-split")] = "benchmark",
) -> None:
    """Build benchmark manifest + reference templates using Quran.com + EveryAyah."""

    metadata = prepare_benchmark_data(
        out_dir=out_dir,
        count=count,
        surahs=parse_csv_ints(surahs),
        ayah_keys=parse_csv_strings(ayah_keys),
        reciter_key=reciter_key,
        reciter_subfolder=reciter_subfolder,
        manifest_reciter_id=manifest_reciter_id,
        seed=seed,
        download_audio=download_audio,
        timeout_s=timeout_s,
        request_retries=request_retries,
        retry_backoff_s=retry_backoff_s,
        resume=resume,
        reference_split=reference_split,
    )
    console.print_json(orjson.dumps(metadata).decode("utf-8"))


@app.command("sync-reciters")
def sync_reciters_cmd(
    out: Annotated[Path, typer.Option("--out")] = DEFAULT_RECITER_CATALOG_PATH,
    enabled_reciters: Annotated[
        str | None,
        typer.Option(
            "--enabled-reciters",
            help="Comma-separated manifest reciter ids to enable in production runs.",
        ),
    ] = None,
) -> None:
    """Fetch EveryAyah + Quran.com reciters and write unified catalog file."""

    enabled = set(parse_csv_strings(enabled_reciters) or [])
    payload = write_reciter_catalog(path=out, enabled_reciters=enabled or None)
    counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
    _print_key_values(
        "Reciter Catalog",
        {
            "catalog_path": out,
            "everyayah_reciters": counts.get("everyayah_reciters", 0),
            "quran_com_reciters": counts.get("quran_com_reciters", 0),
            "configured_reciters": counts.get("configured_reciters", 0),
            "configured_enabled": counts.get("configured_enabled", 0),
        },
    )


@app.command("list-reciters")
def list_reciters_cmd(
    catalog: Annotated[Path, typer.Option("--catalog")] = DEFAULT_RECITER_CATALOG_PATH,
    enabled_only: Annotated[bool, typer.Option("--enabled-only/--all")] = False,
) -> None:
    """List configured reciters from local catalog with source/check capabilities."""

    payload = read_reciter_catalog(catalog)
    if payload is None:
        raise typer.BadParameter(f"catalog not found or invalid: {catalog}")

    configured = payload.get("configured_reciters")
    if not isinstance(configured, list):
        raise typer.BadParameter(f"catalog has no configured_reciters: {catalog}")

    table = Table(title="Configured Reciters")
    table.add_column("reciter_id")
    table.add_column("enabled")
    table.add_column("everyayah")
    table.add_column("quran_com")
    table.add_column("check_type")
    table.add_column("qcom_word_sup")

    rows = 0
    for item in configured:
        if not isinstance(item, dict):
            continue
        if enabled_only and not bool(item.get("enabled")):
            continue
        everyayah = item.get("everyayah") if isinstance(item.get("everyayah"), dict) else {}
        quran_com = item.get("quran_com") if isinstance(item.get("quran_com"), dict) else {}
        table.add_row(
            str(item.get("manifest_reciter_id") or ""),
            str(bool(item.get("enabled"))),
            str(everyayah.get("subfolder") or "-"),
            str(quran_com.get("recitation_id") or "-"),
            str(item.get("check_type") or "-"),
            str(bool(item.get("qcom_word_supervision_supported"))),
        )
        rows += 1

    console.print(table)
    _print_key_values("List Summary", {"rows": rows, "catalog": str(catalog)})


@app.command("run-surah")
def run_surah_cmd(
    reciter_id: Annotated[str, typer.Option("--reciter-id")],
    surah: Annotated[int, typer.Option("--surah", min=1, max=114)],
    out_root: Annotated[Path, typer.Option("--out-root")] = Path("runs/surah_runs"),
    catalog: Annotated[Path, typer.Option("--catalog")] = DEFAULT_RECITER_CATALOG_PATH,
    ui_data_dir: Annotated[Path, typer.Option("--ui-data-dir")] = Path("ui/public/data"),
    ui_catalog: Annotated[Path, typer.Option("--ui-catalog")] = Path("ui/public/data/catalog.json"),
    dist_data_dir: Annotated[Path, typer.Option("--dist-data-dir")] = Path("ui/dist/data"),
    text_data: Annotated[Path | None, typer.Option("--text-data")] = None,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(get_settings().cache_dir),
) -> None:
    """Run one reciter+surah end-to-end (prepare data, align, print QC summary)."""

    normalized_reciter = reciter_id.strip().lower()
    if not is_reciter_enabled(normalized_reciter, catalog_path=catalog):
        raise typer.BadParameter(
            f"reciter not enabled in catalog: {normalized_reciter}. "
            f"Use `qad sync-reciters --enabled-reciters ...` first."
        )

    mapping = resolve_reciter_mapping(normalized_reciter, catalog_path=catalog)
    reciter_entry = get_configured_reciter_entry(normalized_reciter, catalog_path=catalog)
    _print_key_values(
        "Run Setup",
        {
            "reciter_id": normalized_reciter,
            "surah": surah,
            "enabled": is_reciter_enabled(normalized_reciter, catalog_path=catalog),
            "everyayah_subfolder": mapping.everyayah_subfolder,
            "qcom_recitation_id": mapping.qcom_recitation_id,
            "qcom_word_supervision_supported": mapping.qcom_word_supervision_supported,
            "check_type": (
                reciter_entry.get("check_type")
                if isinstance(reciter_entry, dict)
                else "unknown"
            ),
        },
    )

    summary = run_surah_for_reciter(
        reciter_id=normalized_reciter,
        surah=surah,
        out_root=out_root,
        text_data=text_data,
        cache_dir=cache_dir,
        catalog_path=catalog,
    )

    pipeline = summary.get("pipeline") if isinstance(summary.get("pipeline"), dict) else {}
    quality = summary.get("quality") if isinstance(summary.get("quality"), dict) else {}
    paths = summary.get("paths") if isinstance(summary.get("paths"), dict) else {}
    everyayah_eval = (
        quality.get("everyayah_stitch_eval")
        if isinstance(quality.get("everyayah_stitch_eval"), dict)
        else None
    )
    run_summary_payload: dict[str, object] = {
        "total": pipeline.get("total", 0),
        "succeeded": pipeline.get("succeeded", 0),
        "failed": pipeline.get("failed", 0),
        "fallback_used": pipeline.get("fallback_used", 0),
        "priors_used": pipeline.get("priors_used", 0),
        "files_with_qc_warnings": quality.get("files_with_qc_warnings", 0),
        "avg_coverage": f"{float(quality.get('avg_coverage', 0.0)):.4f}",
        "min_coverage": f"{float(quality.get('min_coverage', 0.0)):.4f}",
        "summary_path": paths.get("summary_path", ""),
        "manifest_path": paths.get("manifest_path", ""),
        "output_dir": paths.get("output_dir", ""),
    }
    if everyayah_eval is not None:
        run_summary_payload["everyayah_matched_ayahs"] = everyayah_eval.get("matched_ayahs")
        run_summary_payload["everyayah_coverage"] = everyayah_eval.get("coverage_vs_reference")
        run_summary_payload["everyayah_boundary_median_ms"] = everyayah_eval.get("boundary_error_median_ms")
        run_summary_payload["everyayah_boundary_p95_ms"] = everyayah_eval.get("boundary_error_p95_ms")
        run_summary_payload["everyayah_start_offset_s"] = everyayah_eval.get("start_offset_s")
        run_summary_payload["everyayah_norm_boundary_median_ms"] = everyayah_eval.get(
            "offset_normalized_boundary_error_median_ms"
        )
        run_summary_payload["everyayah_norm_boundary_p95_ms"] = everyayah_eval.get(
            "offset_normalized_boundary_error_p95_ms"
        )
    if paths.get("everyayah_stitch_timeline_path"):
        run_summary_payload["everyayah_timeline_path"] = paths.get("everyayah_stitch_timeline_path")

    _print_key_values(
        "Run Summary",
        run_summary_payload,
    )

    ui_sync = sync_ui_from_latest_runs(
        runs_root=out_root,
        ui_data_dir=ui_data_dir,
        catalog_path=ui_catalog,
        dist_data_dir=dist_data_dir,
        sync_dist=True,
        prune_ui=True,
        dry_run=False,
        reciter_catalog_path=catalog,
        prune_catalog_surahs=True,
    )
    _print_key_values(
        "UI Sync",
        {
            "keys_selected": ui_sync["keys_selected"],
            "ui_copied": ui_sync["ui"]["copied"],
            "ui_audio_copied": ui_sync["ui"]["audio_copied"],
            "ui_pruned": ui_sync["ui"]["pruned"],
            "ui_audio_pruned": ui_sync["ui"]["audio_pruned"],
            "catalog_changed": ui_sync["catalog"]["changed"],
            "catalog_bootstrap_added_reciters": ui_sync["catalog"]["bootstrap_added_reciters"],
            "catalog_added_surahs": ui_sync["catalog"]["added_surahs"],
            "catalog_updated_surahs": ui_sync["catalog"]["updated_surahs"],
            "catalog_skipped_missing_reciter": ui_sync["catalog"]["skipped_missing_reciter"],
        },
    )


@app.command("build-corpus")
def build_corpus_cmd(
    source_url: Annotated[str, typer.Option("--source-url")] = DEFAULT_SOURCE_URL,
    output: Annotated[Path, typer.Option("--output")] = Path("data/quran_text_uthmani_v1.json"),
    raw_out: Annotated[Path, typer.Option("--raw-out")] = Path("data/ara-quranuthmanienc.json"),
    keep_raw: Annotated[bool, typer.Option("--keep-raw/--no-keep-raw")] = False,
    timeout_s: Annotated[float, typer.Option("--timeout-s")] = 20.0,
    request_retries: Annotated[int, typer.Option("--request-retries", min=0)] = 5,
    retry_backoff_s: Annotated[float, typer.Option("--retry-backoff-s", min=0.0)] = 1.0,
) -> None:
    """Build canonical Quran text snapshot from the configured source."""

    canonical = build_canonical_corpus(
        source_url=source_url,
        output=output,
        raw_out=raw_out,
        keep_raw=keep_raw,
        timeout_s=timeout_s,
        retries=request_retries,
        retry_backoff_s=retry_backoff_s,
    )
    metadata = canonical.get("metadata", {})
    _print_key_values(
        "Corpus Build Summary",
        {
            "output": output,
            "surah_count": metadata.get("surah_count"),
            "ayah_count": metadata.get("ayah_count"),
            "source_url": metadata.get("source_url"),
        },
    )


@app.command("sync-ui-data")
def sync_ui_data_cmd(
    runs_root: Annotated[Path, typer.Option("--runs-root", exists=True, file_okay=False, dir_okay=True)] = Path("runs"),
    ui_data_dir: Annotated[Path, typer.Option("--ui-data-dir", file_okay=False, dir_okay=True)] = Path("ui/public/data"),
    catalog: Annotated[Path, typer.Option("--catalog", file_okay=True, dir_okay=False)] = Path("ui/public/data/catalog.json"),
    reciter_catalog: Annotated[Path, typer.Option("--reciter-catalog", file_okay=True, dir_okay=False)] = Path("data/reciter_catalog.json"),
    dist_data_dir: Annotated[Path, typer.Option("--dist-data-dir", file_okay=False, dir_okay=True)] = Path("ui/dist/data"),
    sync_dist: Annotated[bool, typer.Option("--sync-dist/--no-sync-dist")] = True,
    prune_ui: Annotated[bool, typer.Option("--prune-ui/--no-prune-ui")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run/--no-dry-run")] = False,
) -> None:
    """Sync UI timing JSONs from latest run artifact per reciter/surah key."""

    summary = sync_ui_from_latest_runs(
        runs_root=runs_root,
        ui_data_dir=ui_data_dir,
        catalog_path=catalog,
        dist_data_dir=dist_data_dir,
        sync_dist=sync_dist,
        prune_ui=prune_ui,
        dry_run=dry_run,
        reciter_catalog_path=reciter_catalog,
        prune_catalog_surahs=prune_ui,
    )

    _print_key_values(
        "UI Sync Summary",
        {
            "dry_run": summary["dry_run"],
            "keys_selected": summary["keys_selected"],
            "ui_copied": summary["ui"]["copied"],
            "ui_audio_copied": summary["ui"]["audio_copied"],
            "ui_unchanged": summary["ui"]["unchanged"],
            "ui_audio_unchanged": summary["ui"]["audio_unchanged"],
            "ui_pruned": summary["ui"]["pruned"],
            "ui_audio_pruned": summary["ui"]["audio_pruned"],
            "catalog_changed": summary["catalog"]["changed"],
            "catalog_bootstrap_added_reciters": summary["catalog"]["bootstrap_added_reciters"],
            "catalog_added_surahs": summary["catalog"]["added_surahs"],
            "catalog_updated_surahs": summary["catalog"]["updated_surahs"],
            "catalog_skipped_missing_reciter": summary["catalog"]["skipped_missing_reciter"],
            "dist_enabled": summary["dist"]["enabled"],
            "dist_copied": summary["dist"]["copied"],
            "dist_audio_copied": summary["dist"]["audio_copied"],
            "dist_unchanged": summary["dist"]["unchanged"],
            "dist_audio_unchanged": summary["dist"]["audio_unchanged"],
            "dist_pruned": summary["dist"]["pruned"],
            "dist_audio_pruned": summary["dist"]["audio_pruned"],
        },
    )


if __name__ == "__main__":
    app()
