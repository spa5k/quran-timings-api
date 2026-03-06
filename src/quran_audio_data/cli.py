from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

from rich.console import Console
from rich.table import Table
import typer

from quran_audio_data.core.http import get_bytes_with_retry
from quran_audio_data.core.settings import get_settings
from quran_audio_data.pipeline import run_alignment_pipeline
from quran_audio_data.reciters import (
    DEFAULT_RECITERS_PATH,
    list_reciters,
    normalize_reciter_id,
    reciter_exists,
    upsert_reciter,
)


console = Console()
app = typer.Typer(help="Quran audio timing extraction CLI")


@app.callback()
def _main() -> None:
    """Quran audio timing extraction CLI."""


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


def _ensure_mode_guard(
    *,
    setup_reciter: bool,
    list_reciters_mode: bool,
) -> None:
    if setup_reciter and list_reciters_mode:
        raise typer.BadParameter("--setup-reciter and --list-reciters cannot be used together.")


def _setup_example(reciter_id: str) -> str:
    return (
        f"qad detect --setup-reciter --reciter-id {reciter_id} "
        f'--reciter-name "{reciter_id.replace("_", " ").title()}" --source custom'
    )


def _is_interactive_tty() -> bool:
    return sys.stdin.isatty()


def _prompt_setup_values(
    *,
    reciter_id: str,
    reciter_name: str | None,
    source: str | None,
    notes: str | None,
    interactive_only: bool = True,
) -> tuple[str, str, str]:
    name_value = (reciter_name or "").strip()
    source_value = (source or "").strip()
    notes_value = "" if notes is None else notes.strip()
    interactive = _is_interactive_tty()

    if not name_value:
        if interactive and interactive_only:
            name_value = typer.prompt(
                "Reciter name",
                default=reciter_id.replace("_", " ").title(),
            ).strip()
        elif interactive_only:
            raise typer.BadParameter(
                "setup mode requires --reciter-name in non-interactive mode."
            )
        else:
            name_value = reciter_id.replace("_", " ").title()

    if not source_value:
        if interactive and interactive_only:
            source_value = typer.prompt(
                "Source label",
                default="custom",
            ).strip()
        elif interactive_only:
            raise typer.BadParameter("setup mode requires --source in non-interactive mode.")
        else:
            source_value = "custom"

    if notes is None and interactive and interactive_only:
        notes_value = typer.prompt("Notes (optional)", default="", show_default=False).strip()

    return name_value, source_value, notes_value


def _handle_list_mode(*, reciters_path: Path) -> None:
    rows = list_reciters(reciters_path)
    if not rows:
        _print_key_values(
            "Reciters",
            {
                "count": 0,
                "registry_path": reciters_path,
                "hint": "Use `qad detect --setup-reciter --reciter-id ...` to add one.",
            },
        )
        return

    table = Table(title="Configured Reciters")
    table.add_column("id")
    table.add_column("name")
    table.add_column("source")
    table.add_column("notes")
    table.add_column("updated_at")
    for item in rows:
        table.add_row(
            str(item.get("id") or ""),
            str(item.get("name") or ""),
            str(item.get("source") or ""),
            str(item.get("notes") or "-"),
            str(item.get("updated_at") or ""),
        )
    console.print(table)
    _print_key_values(
        "List Summary",
        {
            "count": len(rows),
            "registry_path": reciters_path,
        },
    )


def _handle_setup_mode(
    *,
    reciter_id: str | None,
    reciter_name: str | None,
    source: str | None,
    notes: str | None,
    reciters_path: Path,
) -> None:
    if not reciter_id:
        raise typer.BadParameter("--setup-reciter requires --reciter-id.")
    normalized = normalize_reciter_id(reciter_id)
    name_value, source_value, notes_value = _prompt_setup_values(
        reciter_id=normalized,
        reciter_name=reciter_name,
        source=source,
        notes=notes,
        interactive_only=True,
    )
    row = upsert_reciter(
        reciter_id=normalized,
        name=name_value,
        source=source_value,
        notes=notes_value,
        path=reciters_path,
    )
    _print_key_values(
        "Reciter Setup",
        {
            "reciter_id": row["id"],
            "name": row["name"],
            "source": row["source"],
            "notes": row["notes"] or "-",
            "registry_path": reciters_path,
        },
    )


@app.command("detect")
def detect_cmd(
    audio_url: Annotated[
        str | None,
        typer.Option("--audio-url", help="Public audio URL to align."),
    ] = None,
    reciter_id: Annotated[
        str | None,
        typer.Option("--reciter-id", help="Configured reciter ID."),
    ] = None,
    surah: Annotated[int | None, typer.Option("--surah", min=1, max=114)] = None,
    ayah: Annotated[int | None, typer.Option("--ayah", min=1)] = None,
    setup_reciter: Annotated[
        bool,
        typer.Option("--setup-reciter", help="Create or update reciter metadata."),
    ] = False,
    list_reciters_mode: Annotated[
        bool,
        typer.Option("--list-reciters", help="List configured reciter IDs."),
    ] = False,
    reciter_name: Annotated[str | None, typer.Option("--reciter-name")] = None,
    source: Annotated[str | None, typer.Option("--source")] = None,
    notes: Annotated[str | None, typer.Option("--notes")] = None,
    out_root: Annotated[Path, typer.Option("--out-root")] = Path("runs/detect"),
    text_data: Annotated[Path | None, typer.Option("--text-data")] = None,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(get_settings().cache_dir),
    reciters_path: Annotated[
        Path,
        typer.Option("--reciters-path", hidden=True),
    ] = DEFAULT_RECITERS_PATH,
) -> None:
    """Single-command Quran timing workflow: run, setup reciter, or list reciters."""

    _ensure_mode_guard(setup_reciter=setup_reciter, list_reciters_mode=list_reciters_mode)

    if list_reciters_mode:
        if any(
            value is not None
            for value in (audio_url, reciter_id, surah, ayah, reciter_name, source, notes)
        ):
            raise typer.BadParameter("--list-reciters cannot be combined with run/setup flags.")
        _handle_list_mode(reciters_path=reciters_path)
        return

    if setup_reciter:
        if any(value is not None for value in (audio_url, surah, ayah)):
            raise typer.BadParameter(
                "--setup-reciter cannot be combined with run flags (--audio-url/--surah/--ayah)."
            )
        _handle_setup_mode(
            reciter_id=reciter_id,
            reciter_name=reciter_name,
            source=source,
            notes=notes,
            reciters_path=reciters_path,
        )
        return

    if reciter_name is not None or source is not None or notes is not None:
        raise typer.BadParameter(
            "run mode does not use --reciter-name/--source/--notes. Use --setup-reciter instead."
        )

    if audio_url is None:
        raise typer.BadParameter("run mode requires --audio-url.")
    if reciter_id is None:
        raise typer.BadParameter("run mode requires --reciter-id.")
    if surah is None:
        raise typer.BadParameter("run mode requires --surah.")
    if ayah is not None and surah is None:
        raise typer.BadParameter("--ayah requires --surah.")

    url = audio_url.strip()
    if not url.startswith(("http://", "https://")):
        raise typer.BadParameter("--audio-url must start with http:// or https://")

    resolved_id = normalize_reciter_id(reciter_id)
    reciter_known = reciter_exists(
        resolved_id,
        path=reciters_path,
    )
    if not reciter_known:
        if _is_interactive_tty():
            create_now = typer.confirm(
                f"Reciter '{resolved_id}' is unknown. Create it now?",
                default=True,
            )
            if create_now:
                name_value, source_value, notes_value = _prompt_setup_values(
                    reciter_id=resolved_id,
                    reciter_name=None,
                    source=None,
                    notes=None,
                    interactive_only=True,
                )
                upsert_reciter(
                    reciter_id=resolved_id,
                    name=name_value,
                    source=source_value,
                    notes=notes_value,
                    path=reciters_path,
                )
                _print_key_values(
                    "Reciter Setup",
                    {
                        "reciter_id": resolved_id,
                        "name": name_value,
                        "source": source_value,
                        "notes": notes_value or "-",
                        "registry_path": reciters_path,
                    },
                )
            else:
                raise typer.BadParameter(
                    f"unknown reciter-id '{resolved_id}'. Setup first: {_setup_example(resolved_id)}"
                )
        else:
            raise typer.BadParameter(
                f"unknown reciter-id '{resolved_id}'. Setup first: {_setup_example(resolved_id)}"
            )

    out_root.mkdir(parents=True, exist_ok=True)

    parsed_path = Path(urlparse(url).path)
    suffix = parsed_path.suffix.lower() if parsed_path.suffix else ".mp3"
    if suffix not in {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".aac"}:
        suffix = ".mp3"

    row_surah = int(surah)
    row_ayah = ayah

    run_key_hint = f"s{row_surah:03d}"
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    run_dir = out_root / f"{resolved_id}_{run_key_hint}_{url_hash}"
    input_dir = run_dir / "input"
    audio_dir = input_dir / "audio"
    output_dir = run_dir / "outputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = audio_dir / f"{resolved_id}_input{suffix}"
    audio_bytes = get_bytes_with_retry(url=url, timeout_s=30.0, retries=8, retry_backoff_s=1.0)
    audio_path.write_bytes(audio_bytes)
    audio_sha = hashlib.sha256(audio_bytes).hexdigest()

    manifest_path = input_dir / "manifest.csv"
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
                "reciter_id": resolved_id,
                "surah": str(row_surah),
                "ayah": "" if row_ayah is None else str(row_ayah),
                "source_url": url,
                "sha256": audio_sha,
                "language": "ar",
                "riwaya": "",
                "text_variant": "",
                "reference_split": "detect",
            }
        )

    _print_key_values(
        "Detect Setup",
        {
            "audio_url": url,
            "reciter_id": resolved_id,
            "surah": row_surah,
            "ayah": row_ayah if row_ayah is not None else "full",
            "input_mode": "ayah_file" if row_ayah is not None else "full_surah",
            "downloaded_audio": str(audio_path),
            "manifest": str(manifest_path),
        },
    )

    summary = run_alignment_pipeline(
        manifest_path=manifest_path,
        out_dir=output_dir,
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
        "Detect Alignment Summary",
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
            "output_dir": str(output_dir),
        },
    )
    _print_errors(summary.errors)
    if summary.errors:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
