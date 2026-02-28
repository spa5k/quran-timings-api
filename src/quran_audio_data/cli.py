from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Any, cast

import orjson
from rich.console import Console
from rich.table import Table
import typer

from quran_audio_data.benchmark_data import prepare_benchmark_data
from quran_audio_data.core.parsing import parse_csv_ints, parse_csv_strings
from quran_audio_data.core.settings import get_settings
from quran_audio_data.corpus_builder import DEFAULT_SOURCE_URL, build_canonical_corpus
from quran_audio_data.evaluation import evaluate_predictions, validate_gold_annotations
from quran_audio_data.gold_labeling import auto_label_gold_from_quran_com
from quran_audio_data.gpu import doctor_gpu
from quran_audio_data.pipeline import (
    AccuracyMode,
    DeviceOption,
    EngineAvailabilityPolicy,
    EngineOption,
    benchmark_pipeline,
    run_alignment_pipeline,
    run_resolve_existing_only,
    validate_outputs,
)
from quran_audio_data.ui_sync import sync_ui_from_latest_runs


class EngineChoice(str, Enum):
    nemo = "nemo"
    whisperx = "whisperx"
    mfa = "mfa"


class AccuracyModeChoice(str, Enum):
    standard = "standard"
    strict = "strict"


class DeviceChoice(str, Enum):
    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"


class AvailabilityPolicyChoice(str, Enum):
    best_effort = "best_effort"
    require_requested = "require_requested"
    require_all = "require_all"


console = Console()
app = typer.Typer(help="Quran audio timing extraction CLI")


def _ordered_engine_values(engines: list[EngineChoice]) -> list[str]:
    ordered: list[str] = []
    for item in engines:
        value = item.value
        if value not in ordered:
            ordered.append(value)
    return ordered or [EngineChoice.nemo.value]


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
    engine: Annotated[list[EngineChoice], typer.Option("--engine")] = [EngineChoice.nemo],
    accuracy_mode: Annotated[AccuracyModeChoice, typer.Option("--accuracy-mode")] = AccuracyModeChoice.standard,
    availability_policy: Annotated[
        AvailabilityPolicyChoice,
        typer.Option("--availability-policy"),
    ] = AvailabilityPolicyChoice.best_effort,
    device: Annotated[DeviceChoice, typer.Option("--device")] = DeviceChoice.auto,
    text_data: Annotated[Path | None, typer.Option("--text-data")] = None,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(get_settings().cache_dir),
    no_remote: Annotated[bool, typer.Option("--no-remote")] = False,
) -> None:
    """Run full pipeline: existing resolver first, then alignment engines."""

    selected_engines = _ordered_engine_values(engine)
    requested_engine = cast(EngineOption, selected_engines[0])
    selected_engine_list = cast(list[EngineOption], selected_engines)
    selected_accuracy = cast(AccuracyMode, accuracy_mode.value)
    selected_policy = cast(EngineAvailabilityPolicy, availability_policy.value)
    selected_device = cast(DeviceOption, device.value)
    summary = run_alignment_pipeline(
        manifest_path=manifest,
        out_dir=out,
        engine=requested_engine,
        multi_engine=selected_engine_list,
        accuracy_mode=selected_accuracy,
        availability_policy=selected_policy,
        device=selected_device,
        text_data=text_data,
        cache_dir=cache_dir,
        enable_remote=not no_remote,
    )

    _print_key_values(
        "Alignment Summary",
        {
            "schema_version": summary.schema_version,
            "total": summary.total,
            "succeeded": summary.succeeded,
            "failed": summary.failed,
            "existing": summary.existing_resolved,
            "aligned": summary.aligned,
            "fallback": summary.fallback_used,
            "elapsed_s": f"{summary.elapsed_s:.2f}",
            "attempted_engines": ",".join(summary.attempted_engines),
        },
    )
    _print_errors(summary.errors)
    if summary.errors:
        raise typer.Exit(code=1)


@app.command("resolve-existing")
def resolve_existing_cmd(
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, file_okay=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False, dir_okay=True)],
    text_data: Annotated[Path | None, typer.Option("--text-data")] = None,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(get_settings().cache_dir),
    no_remote: Annotated[bool, typer.Option("--no-remote")] = False,
) -> None:
    """Only resolve and export existing timing sources without model alignment."""

    summary = run_resolve_existing_only(
        manifest_path=manifest,
        out_dir=out,
        text_data=text_data,
        cache_dir=cache_dir,
        enable_remote=not no_remote,
    )
    _print_key_values(
        "Resolve Existing Summary",
        {
            "total": summary.total,
            "succeeded": summary.succeeded,
            "failed": summary.failed,
            "elapsed_s": f"{summary.elapsed_s:.2f}",
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
    engine: Annotated[list[EngineChoice], typer.Option("--engine")] = [EngineChoice.nemo],
    accuracy_mode: Annotated[AccuracyModeChoice, typer.Option("--accuracy-mode")] = AccuracyModeChoice.standard,
    availability_policy: Annotated[
        AvailabilityPolicyChoice,
        typer.Option("--availability-policy"),
    ] = AvailabilityPolicyChoice.best_effort,
    device: Annotated[DeviceChoice, typer.Option("--device")] = DeviceChoice.auto,
    text_data: Annotated[Path | None, typer.Option("--text-data")] = None,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(get_settings().cache_dir),
    no_remote: Annotated[bool, typer.Option("--no-remote")] = False,
) -> None:
    """Benchmark pipeline over a manifest sample."""

    selected_engines = _ordered_engine_values(engine)
    requested_engine = cast(EngineOption, selected_engines[0])
    selected_engine_list = cast(list[EngineOption], selected_engines)
    selected_accuracy = cast(AccuracyMode, accuracy_mode.value)
    selected_policy = cast(EngineAvailabilityPolicy, availability_policy.value)
    selected_device = cast(DeviceOption, device.value)
    report = benchmark_pipeline(
        manifest_path=manifest,
        out_dir=out,
        sample_size=sample_size,
        engine=requested_engine,
        multi_engine=selected_engine_list,
        accuracy_mode=selected_accuracy,
        availability_policy=selected_policy,
        device=selected_device,
        text_data=text_data,
        cache_dir=cache_dir,
        enable_remote=not no_remote,
    )
    _print_key_values(
        "Benchmark Summary",
        {
            "sample_size": report["sample_size"],
            "succeeded": report["succeeded"],
            "failed": report["failed"],
            "existing": report["existing_resolved"],
            "aligned": report["aligned"],
            "fallback": report["fallback_used"],
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
    gold_dir: Annotated[Path, typer.Option("--gold-dir", exists=True)],
    report: Annotated[Path | None, typer.Option("--report")] = None,
) -> None:
    """Evaluate predicted timings against gold annotations."""

    output = evaluate_predictions(pred_dir=pred_dir, gold_dir=gold_dir)
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


@app.command("validate-gold")
def validate_gold_cmd(
    gold_dir: Annotated[Path, typer.Option("--gold-dir", exists=True)],
    report: Annotated[Path | None, typer.Option("--report")] = None,
    max_errors: Annotated[int, typer.Option("--max-errors", min=1)] = 50,
) -> None:
    """Validate gold annotations before running eval."""

    output = validate_gold_annotations(gold_dir=gold_dir, max_errors=max_errors)
    _print_key_values(
        "Gold Validation Summary",
        {
            "gold_files": output["gold_files"],
            "valid": output["valid_files"],
            "invalid": output["invalid_files"],
            "unlabeled_words": output["unlabeled_words"],
            "invalid_durations": output["invalid_duration_words"],
            "non_monotonic": output["non_monotonic_words"],
            "passes": output["passes"],
        },
    )
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_bytes(orjson.dumps(output, option=orjson.OPT_INDENT_2))
    if not output["passes"]:
        raise typer.Exit(code=1)


@app.command("auto-label-gold")
def auto_label_gold_cmd(
    gold_dir: Annotated[Path, typer.Option("--gold-dir", exists=True)],
    chapter_reciter_id: Annotated[int, typer.Option("--chapter-reciter-id", min=1)],
    overwrite_existing: Annotated[
        bool,
        typer.Option("--overwrite-existing/--no-overwrite-existing"),
    ] = False,
    timeout_s: Annotated[float, typer.Option("--timeout-s")] = 20.0,
    request_retries: Annotated[int, typer.Option("--request-retries", min=0)] = 5,
    retry_backoff_s: Annotated[float, typer.Option("--retry-backoff-s", min=0.0)] = 1.0,
    max_errors: Annotated[int, typer.Option("--max-errors", min=1)] = 100,
    report: Annotated[Path | None, typer.Option("--report")] = None,
) -> None:
    """Auto-label gold templates using Quran.com chapter-recitation segments."""

    summary = auto_label_gold_from_quran_com(
        gold_dir=gold_dir,
        chapter_reciter_id=chapter_reciter_id,
        overwrite_existing=overwrite_existing,
        timeout_s=timeout_s,
        request_retries=request_retries,
        retry_backoff_s=retry_backoff_s,
        max_errors=max_errors,
    )
    payload = summary.to_dict()
    _print_key_values(
        "Auto-label Summary",
        {
            "files_total": payload["files_total"],
            "updated": payload["files_updated"],
            "skipped": payload["files_skipped_already_labeled"],
            "files_missing_segments": payload["files_missing_segments"],
            "words_labeled": payload["words_labeled"],
            "words_missing_segments": payload["words_missing_segments"],
            "errors": len(payload["errors"]),
            "passes": payload["passes"],
        },
    )
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    if not payload["passes"]:
        _print_errors(payload["errors"])
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
    seed: Annotated[int, typer.Option("--seed")] = 42,
    download_audio: Annotated[bool, typer.Option("--download-audio/--no-download-audio")] = True,
    timeout_s: Annotated[float, typer.Option("--timeout-s")] = 20.0,
    request_retries: Annotated[int, typer.Option("--request-retries", min=0)] = 5,
    retry_backoff_s: Annotated[float, typer.Option("--retry-backoff-s", min=0.0)] = 1.0,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    gold_split: Annotated[str, typer.Option("--gold-split")] = "benchmark",
) -> None:
    """Build benchmark manifest + gold templates using Quran.com + EveryAyah."""

    metadata = prepare_benchmark_data(
        out_dir=out_dir,
        count=count,
        surahs=parse_csv_ints(surahs),
        ayah_keys=parse_csv_strings(ayah_keys),
        reciter_key=reciter_key,
        reciter_subfolder=reciter_subfolder,
        seed=seed,
        download_audio=download_audio,
        timeout_s=timeout_s,
        request_retries=request_retries,
        retry_backoff_s=retry_backoff_s,
        resume=resume,
        gold_split=gold_split,
    )
    console.print_json(orjson.dumps(metadata).decode("utf-8"))


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
    catalog: Annotated[Path, typer.Option("--catalog", exists=True, file_okay=True, dir_okay=False)] = Path("ui/public/data/catalog.json"),
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
    )

    _print_key_values(
        "UI Sync Summary",
        {
            "dry_run": summary["dry_run"],
            "keys_selected": summary["keys_selected"],
            "ui_copied": summary["ui"]["copied"],
            "ui_unchanged": summary["ui"]["unchanged"],
            "ui_pruned": summary["ui"]["pruned"],
            "catalog_changed": summary["catalog"]["changed"],
            "catalog_added_surahs": summary["catalog"]["added_surahs"],
            "catalog_updated_surahs": summary["catalog"]["updated_surahs"],
            "catalog_skipped_missing_reciter": summary["catalog"]["skipped_missing_reciter"],
            "dist_enabled": summary["dist"]["enabled"],
            "dist_copied": summary["dist"]["copied"],
            "dist_unchanged": summary["dist"]["unchanged"],
            "dist_pruned": summary["dist"]["pruned"],
        },
    )


if __name__ == "__main__":
    app()
