from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
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
    get_reciter,
    list_reciters,
    normalize_reciter_id,
    reciter_exists,
    upsert_reciter,
)
from quran_audio_data.supervision import (
    DEFAULT_RECITER_CATALOG_PATH,
    read_reciter_catalog,
    write_reciter_catalog,
)
from quran_audio_data.ui_sync import export_api_from_latest_runs


console = Console()
app = typer.Typer(help="Quran audio timing extraction CLI")


@app.callback()
def _main() -> None:
    """Quran audio timing extraction CLI."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _is_interactive_tty() -> bool:
    return sys.stdin.isatty()


def _prompt_setup_values(
    *,
    reciter_id: str,
    reciter_name: str | None,
    source: str | None,
    notes: str | None,
    interactive_only: bool,
) -> tuple[str, str, str]:
    name_value = (reciter_name or "").strip()
    source_value = (source or "").strip()
    notes_value = "" if notes is None else notes.strip()

    if not name_value:
        if interactive_only and _is_interactive_tty():
            name_value = typer.prompt(
                "Reciter name",
                default=reciter_id.replace("_", " ").title(),
            ).strip()
        else:
            name_value = reciter_id.replace("_", " ").title()

    if not source_value:
        if interactive_only and _is_interactive_tty():
            source_value = typer.prompt("Source label", default="custom").strip()
        else:
            source_value = "custom"

    if notes is None and interactive_only and _is_interactive_tty():
        notes_value = typer.prompt("Notes (optional)", default="", show_default=False).strip()

    return name_value, source_value, notes_value


def _normalize_catalog_slug(value: str) -> str:
    return str(value or "").strip().lower()


def _catalog_source_label(item: dict[str, Any]) -> str:
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    everyayah = source.get("everyayah") if isinstance(source.get("everyayah"), dict) else {}
    quran_com = source.get("quran_com") if isinstance(source.get("quran_com"), dict) else {}
    if everyayah.get("subfolder"):
        return "everyayah"
    if quran_com.get("recitation_id"):
        return "quran.com"
    return "catalog"


def _load_public_reciters(catalog_path: Path) -> list[dict[str, Any]]:
    payload = read_reciter_catalog(catalog_path)
    if payload is None:
        return []
    reciters = payload.get("reciters")
    if not isinstance(reciters, list):
        return []
    return [item for item in reciters if isinstance(item, dict)]


def _default_catalog_payload() -> dict[str, Any]:
    return {
        "schema_version": "v1",
        "generated_at": _utc_now(),
        "counts": {
            "everyayah_source_reciters": 0,
            "quran_com_source_reciters": 0,
            "quranicaudio_source_reciters": 0,
            "configured_reciters": 0,
            "enabled_reciters": 0,
        },
        "sources": {},
        "reciters": [],
    }


def _write_catalog_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _upsert_public_catalog_reciter(
    *,
    catalog_path: Path,
    reciter_id: str,
    reciter_name: str,
) -> None:
    payload = read_reciter_catalog(catalog_path) or _default_catalog_payload()
    reciters = payload.get("reciters")
    if not isinstance(reciters, list):
        reciters = []
        payload["reciters"] = reciters

    key = _normalize_catalog_slug(reciter_id)
    row: dict[str, Any] | None = None
    for item in reciters:
        if not isinstance(item, dict):
            continue
        if _normalize_catalog_slug(item.get("slug") or "") == key:
            row = item
            break

    if row is None:
        row = {
            "slug": key,
            "name": reciter_name or key,
            "enabled": True,
            "check_type": "model_only",
            "capabilities": {
                "ayah_by_ayah": False,
                "word_by_word": False,
            },
            "source": {
                "everyayah": {"subfolder": None, "reciter_key": None, "name": None},
                "quran_com": {"recitation_id": None, "name": None},
                "quranicaudio": {"path": None, "name": None},
            },
            "surahs_available": [],
            "surah_count": 0,
            "endpoints": {"metadata": f"/data/reciters/{key}/metadata.json"},
        }
        reciters.append(row)
    else:
        if reciter_name and (not row.get("name") or str(row.get("name")).strip().lower() == key):
            row["name"] = reciter_name
        row["enabled"] = True
        row["endpoints"] = {"metadata": f"/data/reciters/{key}/metadata.json"}
        if not isinstance(row.get("capabilities"), dict):
            row["capabilities"] = {"ayah_by_ayah": False, "word_by_word": False}
        if not isinstance(row.get("source"), dict):
            row["source"] = {
                "everyayah": {"subfolder": None, "reciter_key": None, "name": None},
                "quran_com": {"recitation_id": None, "name": None},
                "quranicaudio": {"path": None, "name": None},
            }
        row["check_type"] = str(row.get("check_type") or "model_only")

    reciters.sort(key=lambda item: str(item.get("slug") or ""))
    counts = payload.get("counts")
    if not isinstance(counts, dict):
        counts = {}
        payload["counts"] = counts
    counts["configured_reciters"] = len(reciters)
    counts["enabled_reciters"] = sum(
        1 for item in reciters if isinstance(item, dict) and bool(item.get("enabled"))
    )
    payload["generated_at"] = _utc_now()
    _write_catalog_payload(catalog_path, payload)


def _build_detect_choices(
    *,
    reciters_path: Path,
    catalog_path: Path,
) -> list[dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}

    for item in _load_public_reciters(catalog_path):
        slug = _normalize_catalog_slug(item.get("slug") or "")
        if not slug:
            continue
        merged[slug] = {
            "id": slug,
            "name": str(item.get("name") or slug).strip() or slug,
            "source": _catalog_source_label(item),
            "notes": "",
        }

    for item in list_reciters(reciters_path):
        if not isinstance(item, dict):
            continue
        slug = _normalize_catalog_slug(item.get("id") or "")
        if not slug:
            continue
        merged[slug] = {
            "id": slug,
            "name": str(item.get("name") or slug).strip() or slug,
            "source": str(item.get("source") or "custom").strip() or "custom",
            "notes": str(item.get("notes") or "").strip(),
        }

    return [merged[key] for key in sorted(merged)]


def _find_catalog_reciter(
    *,
    catalog_path: Path,
    reciter_id: str,
) -> dict[str, Any] | None:
    key = _normalize_catalog_slug(reciter_id)
    if not key:
        return None
    for item in _load_public_reciters(catalog_path):
        if _normalize_catalog_slug(item.get("slug") or "") == key:
            return item
    return None


def _prompt_detect_reciter(
    *,
    reciters_path: Path,
    catalog_path: Path,
) -> tuple[str, str, str, str]:
    choices = _build_detect_choices(reciters_path=reciters_path, catalog_path=catalog_path)
    if choices:
        table = Table(title="Reciter Selection")
        table.add_column("#")
        table.add_column("id")
        table.add_column("name")
        table.add_column("source")
        for index, item in enumerate(choices, start=1):
            table.add_row(str(index), item["id"], item["name"], item["source"])
        console.print(table)

    raw = typer.prompt(
        "Reciter (index, slug, or type 'new')",
        default="new" if not choices else "",
    ).strip()

    if raw.isdigit():
        index = int(raw) - 1
        if 0 <= index < len(choices):
            choice = choices[index]
            return choice["id"], choice["name"], choice["source"], choice["notes"]

    normalized = _normalize_catalog_slug(raw)
    if normalized and normalized not in {"new", "add"}:
        for choice in choices:
            if choice["id"] == normalized:
                return choice["id"], choice["name"], choice["source"], choice["notes"]

    seed = None if normalized in {"", "new", "add"} else raw
    reciter_id = normalize_reciter_id(typer.prompt("Reciter ID", default=seed or "").strip())
    reciter_name, source, notes = _prompt_setup_values(
        reciter_id=reciter_id,
        reciter_name=None,
        source=None,
        notes=None,
        interactive_only=True,
    )
    return reciter_id, reciter_name, source, notes


def _prompt_surah(value: int | None) -> int:
    if value is not None:
        if value < 1 or value > 114:
            raise typer.BadParameter("--surah must be between 1 and 114.")
        return int(value)
    prompted = int(typer.prompt("Surah (1-114)", default=1))
    if prompted < 1 or prompted > 114:
        raise typer.BadParameter("--surah must be between 1 and 114.")
    return prompted


def _prompt_ayah(value: int | None) -> int | None:
    if value is not None:
        if value < 1:
            raise typer.BadParameter("--ayah must be >= 1.")
        return int(value)
    raw = typer.prompt("Ayah (blank = full surah)", default="", show_default=False).strip()
    if not raw:
        return None
    ayah_value = int(raw)
    if ayah_value < 1:
        raise typer.BadParameter("--ayah must be >= 1.")
    return ayah_value


def _prompt_audio_url(value: str | None) -> str:
    url = (value or "").strip()
    if not url:
        url = typer.prompt("Audio URL").strip()
    if not url.startswith(("http://", "https://")):
        raise typer.BadParameter("--audio-url must start with http:// or https://")
    return url


def _collect_detect_inputs(
    *,
    audio_url: str | None,
    reciter_id: str | None,
    surah: int | None,
    ayah: int | None,
    reciter_name: str | None,
    source: str | None,
    notes: str | None,
    reciters_path: Path,
    catalog_path: Path,
) -> tuple[str, str, str, str, str, int, int | None]:
    interactive = _is_interactive_tty()

    chosen_id = (reciter_id or "").strip()
    chosen_name = (reciter_name or "").strip()
    chosen_source = (source or "").strip()
    chosen_notes = notes

    if not chosen_id and not interactive:
        raise typer.BadParameter(
            "detect requires --reciter-id, --surah, and --audio-url unless you run it interactively."
        )

    if not chosen_id:
        chosen_id, chosen_name, chosen_source, chosen_notes_value = _prompt_detect_reciter(
            reciters_path=reciters_path,
            catalog_path=catalog_path,
        )
        chosen_notes = chosen_notes_value

    catalog_entry = _find_catalog_reciter(catalog_path=catalog_path, reciter_id=chosen_id)
    resolved_id = str(catalog_entry.get("slug") or "").strip() if catalog_entry else ""
    if not resolved_id:
        resolved_id = normalize_reciter_id(chosen_id)

    existing = None if catalog_entry is not None else get_reciter(resolved_id, path=reciters_path)

    if catalog_entry is not None:
        if not chosen_name:
            chosen_name = str(catalog_entry.get("name") or resolved_id).strip()
        if not chosen_source:
            chosen_source = _catalog_source_label(catalog_entry)
        if chosen_notes is None:
            chosen_notes = ""
    elif existing is not None:
        if not chosen_name:
            chosen_name = str(existing.get("name") or resolved_id).strip()
        if not chosen_source:
            chosen_source = str(existing.get("source") or "custom").strip() or "custom"
        if chosen_notes is None:
            chosen_notes = str(existing.get("notes") or "").strip()
    else:
        chosen_name, chosen_source, chosen_notes_value = _prompt_setup_values(
            reciter_id=resolved_id,
            reciter_name=chosen_name or None,
            source=chosen_source or None,
            notes=chosen_notes,
            interactive_only=interactive,
        )
        chosen_notes = chosen_notes_value

    run_surah = _prompt_surah(surah if surah is not None else None) if interactive else int(surah)
    if not interactive and (surah is None or audio_url is None):
        raise typer.BadParameter(
            "detect requires --reciter-id, --surah, and --audio-url unless you run it interactively."
        )
    run_ayah = _prompt_ayah(ayah) if interactive else ayah
    run_audio_url = _prompt_audio_url(audio_url)

    return (
        resolved_id,
        chosen_name or resolved_id.replace("_", " ").title(),
        chosen_source or "custom",
        "" if chosen_notes is None else chosen_notes.strip(),
        run_audio_url,
        run_surah,
        run_ayah,
    )


def _publish_detect_run(
    *,
    out_root: Path,
    reciter_id: str,
    surah: int,
    catalog_path: Path,
    api_root: Path,
    ui_data_dir: Path,
    dist_data_dir: Path,
    sync_dist: bool,
) -> dict[str, Any]:
    return export_api_from_latest_runs(
        runs_root=out_root,
        api_root=api_root,
        reciters_index_path=catalog_path,
        ui_data_dir=ui_data_dir,
        dist_data_dir=dist_data_dir,
        sync_dist=sync_dist,
        prune_ui=False,
        dry_run=False,
        include_reciters={reciter_id},
        include_surahs={surah},
    )


@app.command("sync-reciters")
def sync_reciters_cmd(
    out: Annotated[Path, typer.Option("--out")] = DEFAULT_RECITER_CATALOG_PATH,
    enabled_reciters: Annotated[
        str | None,
        typer.Option(
            "--enabled-reciters",
            help="Comma-separated reciter slugs to mark enabled. If omitted, preserves current enablement.",
        ),
    ] = None,
) -> None:
    """Fetch EveryAyah, Quran.com, and Quranicaudio reciters into the public index."""

    enabled_set: set[str] | None = None
    if enabled_reciters is not None:
        enabled_set = {
            _normalize_catalog_slug(item)
            for item in str(enabled_reciters).split(",")
            if _normalize_catalog_slug(item)
        }
    else:
        existing = _load_public_reciters(out)
        if existing:
            enabled_set = {
                _normalize_catalog_slug(item.get("slug") or "")
                for item in existing
                if bool(item.get("enabled")) and item.get("slug")
            }

    payload = write_reciter_catalog(path=out, enabled_reciters=enabled_set or None)
    counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
    _print_key_values(
        "Reciters Synced",
        {
            "path": out,
            "configured_reciters": counts.get("configured_reciters", 0),
            "enabled_reciters": counts.get("enabled_reciters", 0),
            "everyayah_source_reciters": counts.get("everyayah_source_reciters", 0),
            "quran_com_source_reciters": counts.get("quran_com_source_reciters", 0),
            "quranicaudio_source_reciters": counts.get("quranicaudio_source_reciters", 0),
        },
    )


@app.command("list-reciters")
def list_catalog_reciters_cmd(
    catalog: Annotated[Path, typer.Option("--catalog")] = DEFAULT_RECITER_CATALOG_PATH,
    enabled_only: Annotated[bool, typer.Option("--enabled-only/--all")] = False,
) -> None:
    """List configured reciters from public reciters index."""

    reciters = _load_public_reciters(catalog)
    if not reciters:
        raise typer.BadParameter(f"catalog not found/invalid or has no reciters: {catalog}")

    table = Table(title="Configured Reciters")
    table.add_column("slug")
    table.add_column("enabled")
    table.add_column("name")
    table.add_column("check_type")
    table.add_column("ayah_by_ayah")
    table.add_column("word_by_word")
    table.add_column("everyayah")
    table.add_column("quran_com")
    table.add_column("surahs")

    rows = 0
    for item in reciters:
        if enabled_only and not bool(item.get("enabled")):
            continue
        capabilities = (
            item.get("capabilities") if isinstance(item.get("capabilities"), dict) else {}
        )
        source = item.get("source") if isinstance(item.get("source"), dict) else {}
        everyayah = source.get("everyayah") if isinstance(source.get("everyayah"), dict) else {}
        quran_com = source.get("quran_com") if isinstance(source.get("quran_com"), dict) else {}
        table.add_row(
            str(item.get("slug") or ""),
            str(bool(item.get("enabled"))),
            str(item.get("name") or ""),
            str(item.get("check_type") or ""),
            str(bool(capabilities.get("ayah_by_ayah"))),
            str(bool(capabilities.get("word_by_word"))),
            str(everyayah.get("subfolder") or "-"),
            str(quran_com.get("recitation_id") or "-"),
            str(int(item.get("surah_count") or 0)),
        )
        rows += 1

    console.print(table)
    _print_key_values("List Summary", {"rows": rows, "catalog": str(catalog)})


@app.command("detect")
def detect_cmd(
    audio_url: Annotated[
        str | None,
        typer.Option("--audio-url", help="Public audio URL to align."),
    ] = None,
    reciter_id: Annotated[
        str | None,
        typer.Option("--reciter-id", help="Reciter slug."),
    ] = None,
    surah: Annotated[int | None, typer.Option("--surah", min=1, max=114)] = None,
    ayah: Annotated[int | None, typer.Option("--ayah", min=1)] = None,
    reciter_name: Annotated[
        str | None,
        typer.Option("--reciter-name", help="Optional display name for a new reciter."),
    ] = None,
    source: Annotated[
        str | None,
        typer.Option("--source", help="Optional source label for a new reciter."),
    ] = None,
    notes: Annotated[
        str | None,
        typer.Option("--notes", help="Optional notes for a new reciter."),
    ] = None,
    out_root: Annotated[Path, typer.Option("--out-root")] = Path("runs/detect"),
    text_data: Annotated[Path | None, typer.Option("--text-data")] = None,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(get_settings().cache_dir),
    catalog: Annotated[Path, typer.Option("--catalog")] = DEFAULT_RECITER_CATALOG_PATH,
    reciters_path: Annotated[
        Path,
        typer.Option("--reciters-path", hidden=True),
    ] = DEFAULT_RECITERS_PATH,
    api_root: Annotated[Path, typer.Option("--api-root", hidden=True)] = Path("data/api"),
    ui_data_dir: Annotated[Path, typer.Option("--ui-data-dir", hidden=True)] = Path(
        "ui/public/data"
    ),
    dist_data_dir: Annotated[Path, typer.Option("--dist-data-dir", hidden=True)] = Path(
        "ui/dist/data"
    ),
    sync_dist: Annotated[bool, typer.Option("--sync-dist/--no-sync-dist", hidden=True)] = False,
) -> None:
    """Run alignment for one recitation and publish full-surah results into `data/api`."""

    (
        resolved_id,
        resolved_name,
        resolved_source,
        resolved_notes,
        url,
        row_surah,
        row_ayah,
    ) = _collect_detect_inputs(
        audio_url=audio_url,
        reciter_id=reciter_id,
        surah=surah,
        ayah=ayah,
        reciter_name=reciter_name,
        source=source,
        notes=notes,
        reciters_path=reciters_path,
        catalog_path=catalog,
    )

    catalog_entry = _find_catalog_reciter(catalog_path=catalog, reciter_id=resolved_id)
    if catalog_entry is None and not reciter_exists(resolved_id, path=reciters_path):
        upsert_reciter(
            reciter_id=resolved_id,
            name=resolved_name,
            source=resolved_source,
            notes=resolved_notes,
            path=reciters_path,
        )
    _upsert_public_catalog_reciter(
        catalog_path=catalog,
        reciter_id=resolved_id,
        reciter_name=resolved_name,
    )

    out_root.mkdir(parents=True, exist_ok=True)

    parsed_path = Path(urlparse(url).path)
    suffix = parsed_path.suffix.lower() if parsed_path.suffix else ".mp3"
    if suffix not in {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".aac"}:
        suffix = ".mp3"

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
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
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
            "reciter_name": resolved_name,
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

    if row_ayah is not None:
        console.print(
            "[yellow]Ayah-only runs are not auto-published because `/data/api` is surah-level.[/yellow]"
        )
        return

    export_summary = _publish_detect_run(
        out_root=out_root,
        reciter_id=resolved_id,
        surah=row_surah,
        catalog_path=catalog,
        api_root=api_root,
        ui_data_dir=ui_data_dir,
        dist_data_dir=dist_data_dir,
        sync_dist=sync_dist,
    )
    _print_key_values(
        "Detect Publish Summary",
        {
            "keys_selected": export_summary["keys_selected"],
            "surah_changed": export_summary["api"]["surah_changed"],
            "reciters_index_changed": export_summary["reciters_index"]["changed"],
            "ui_copied": export_summary["ui"]["copied"],
            "dist_copied": export_summary["dist"]["copied"]
            if export_summary["dist"]["enabled"]
            else 0,
            "api_root": api_root,
            "catalog": catalog,
        },
    )


if __name__ == "__main__":
    app()
