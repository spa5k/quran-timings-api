from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from quran_audio_data.pipeline import (
    benchmark_pipeline,
    run_alignment_pipeline,
    run_resolve_existing_only,
    validate_outputs,
)


app = typer.Typer(help="Quran audio timing extraction CLI")


@app.command("align")
def align_cmd(
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, file_okay=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False, dir_okay=True)],
    engine: Annotated[str, typer.Option("--engine")] = "nemo",
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


if __name__ == "__main__":
    app()
