from __future__ import annotations

from pathlib import Path
from typing import Annotated

import orjson
import typer

from quran_audio_data.evaluation import evaluate_predictions, validate_gold_annotations
from quran_audio_data.gold_labeling import auto_label_gold_from_quran_com
from quran_audio_data.gpu import doctor_gpu
from quran_audio_data.benchmark_data import prepare_benchmark_data
from quran_audio_data.pipeline import (
    benchmark_pipeline,
    run_alignment_pipeline,
    run_resolve_existing_only,
    validate_outputs,
)


app = typer.Typer(help="Quran audio timing extraction CLI")


def _parse_multi_engine(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    parsed = [item.strip() for item in raw.split(",") if item.strip()]
    return parsed or None


def _parse_csv_ints(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values: list[int] = []
    for item in raw.split(","):
        cleaned = item.strip()
        if not cleaned:
            continue
        values.append(int(cleaned))
    return values or None


def _parse_csv_strings(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or None


@app.command("align")
def align_cmd(
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, file_okay=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False, dir_okay=True)],
    engine: Annotated[str, typer.Option("--engine")] = "nemo",
    multi_engine: Annotated[
        str | None,
        typer.Option("--multi-engine", help="Comma-separated engines, e.g. nemo,whisperx,mfa"),
    ] = None,
    accuracy_mode: Annotated[str, typer.Option("--accuracy-mode")] = "standard",
    device: Annotated[str, typer.Option("--device")] = "auto",
    text_data: Annotated[Path | None, typer.Option("--text-data")] = None,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(".cache/timings"),
    no_remote: Annotated[bool, typer.Option("--no-remote")] = False,
) -> None:
    """Run full pipeline: existing resolver first, then model alignment."""

    summary = run_alignment_pipeline(
        manifest_path=manifest,
        out_dir=out,
        engine=engine,  # type: ignore[arg-type]
        multi_engine=_parse_multi_engine(multi_engine),  # type: ignore[arg-type]
        accuracy_mode=accuracy_mode,  # type: ignore[arg-type]
        device=device,  # type: ignore[arg-type]
        text_data=text_data,
        cache_dir=cache_dir,
        enable_remote=not no_remote,
    )

    typer.echo(
        " ".join(
            [
                f"total={summary.total}",
                f"succeeded={summary.succeeded}",
                f"failed={summary.failed}",
                f"existing={summary.existing_resolved}",
                f"aligned={summary.aligned}",
                f"fallback={summary.fallback_used}",
                f"elapsed_s={summary.elapsed_s:.2f}",
            ]
        )
    )

    if summary.errors:
        typer.echo("Errors:")
        for error in summary.errors:
            typer.echo(f"- {error}")
        raise typer.Exit(code=1)


@app.command("resolve-existing")
def resolve_existing_cmd(
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, file_okay=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False, dir_okay=True)],
    text_data: Annotated[Path | None, typer.Option("--text-data")] = None,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(".cache/timings"),
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

    typer.echo(
        " ".join(
            [
                f"total={summary.total}",
                f"succeeded={summary.succeeded}",
                f"failed={summary.failed}",
                f"elapsed_s={summary.elapsed_s:.2f}",
            ]
        )
    )

    if summary.errors:
        typer.echo("Errors:")
        for error in summary.errors:
            typer.echo(f"- {error}")
        raise typer.Exit(code=1)


@app.command("validate")
def validate_cmd(
    input_path: Annotated[Path, typer.Option("--input", exists=True)],
) -> None:
    """Validate schema output JSON(s)."""

    valid, invalid, errors = validate_outputs(input_path)
    typer.echo(f"valid={valid} invalid={invalid}")
    if errors:
        typer.echo("Errors:")
        for error in errors:
            typer.echo(f"- {error}")
        raise typer.Exit(code=1)


@app.command("benchmark")
def benchmark_cmd(
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, file_okay=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False, dir_okay=True)],
    sample_size: Annotated[int, typer.Option("--sample-size", min=1)] = 5,
    engine: Annotated[str, typer.Option("--engine")] = "nemo",
    multi_engine: Annotated[
        str | None,
        typer.Option("--multi-engine", help="Comma-separated engines, e.g. nemo,whisperx,mfa"),
    ] = None,
    accuracy_mode: Annotated[str, typer.Option("--accuracy-mode")] = "standard",
    device: Annotated[str, typer.Option("--device")] = "auto",
    text_data: Annotated[Path | None, typer.Option("--text-data")] = None,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(".cache/timings"),
    no_remote: Annotated[bool, typer.Option("--no-remote")] = False,
) -> None:
    """Benchmark pipeline over a manifest sample."""

    report = benchmark_pipeline(
        manifest_path=manifest,
        out_dir=out,
        sample_size=sample_size,
        engine=engine,  # type: ignore[arg-type]
        multi_engine=_parse_multi_engine(multi_engine),  # type: ignore[arg-type]
        accuracy_mode=accuracy_mode,  # type: ignore[arg-type]
        device=device,  # type: ignore[arg-type]
        text_data=text_data,
        cache_dir=cache_dir,
        enable_remote=not no_remote,
    )

    typer.echo(
        " ".join(
            [
                f"sample_size={report['sample_size']}",
                f"succeeded={report['succeeded']}",
                f"failed={report['failed']}",
                f"existing={report['existing_resolved']}",
                f"aligned={report['aligned']}",
                f"fallback={report['fallback_used']}",
                f"elapsed_s={report['elapsed_s']:.2f}",
                f"avg_file_runtime_s={report['avg_file_runtime_s']:.2f}",
            ]
        )
    )

    if report["errors"]:
        typer.echo("Errors:")
        for error in report["errors"]:
            typer.echo(f"- {error}")
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
    typer.echo(
        " ".join(
            [
                f"matched_ayahs={output['matched_ayahs']}",
                f"missing={len(output['missing_predictions'])}",
                f"median_ms={summary['median_abs_error_ms']:.2f}",
                f"p95_ms={summary['p95_abs_error_ms']:.2f}",
                f"hit50={summary['hit_rate_50ms']:.3f}",
                f"passes={output['passes_targets']}",
            ]
        )
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
    typer.echo(
        " ".join(
            [
                f"gold_files={output['gold_files']}",
                f"valid={output['valid_files']}",
                f"invalid={output['invalid_files']}",
                f"unlabeled_words={output['unlabeled_words']}",
                f"invalid_durations={output['invalid_duration_words']}",
                f"non_monotonic={output['non_monotonic_words']}",
                f"passes={output['passes']}",
            ]
        )
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
    """Auto-label gold templates using Quran.com chapter-recitation word segments."""

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
    typer.echo(
        " ".join(
            [
                f"files_total={payload['files_total']}",
                f"updated={payload['files_updated']}",
                f"skipped={payload['files_skipped_already_labeled']}",
                f"files_missing_segments={payload['files_missing_segments']}",
                f"words_labeled={payload['words_labeled']}",
                f"words_missing_segments={payload['words_missing_segments']}",
                f"errors={len(payload['errors'])}",
                f"passes={payload['passes']}",
            ]
        )
    )

    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))

    if not payload["passes"]:
        raise typer.Exit(code=1)


@app.command("doctor-gpu")
def doctor_gpu_cmd() -> None:
    """Report GPU/CUDA readiness for strict alignment mode."""

    report = doctor_gpu()
    typer.echo(orjson.dumps(report, option=orjson.OPT_INDENT_2).decode("utf-8"))


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
        surahs=_parse_csv_ints(surahs),
        ayah_keys=_parse_csv_strings(ayah_keys),
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
    typer.echo(orjson.dumps(metadata, option=orjson.OPT_INDENT_2).decode("utf-8"))


if __name__ == "__main__":
    app()
