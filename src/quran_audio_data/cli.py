from __future__ import annotations

import csv
import hashlib
import re
import sys
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

from rich.console import Console
from rich.table import Table
import typer

from quran_audio_data.core.http import get_bytes_with_retry
from quran_audio_data.core.parsing import parse_csv_strings
from quran_audio_data.core.settings import get_settings
from quran_audio_data.pipeline import run_alignment_pipeline
from quran_audio_data.reciters import (
    DEFAULT_RECITERS_PATH,
    list_reciters,
    normalize_reciter_id,
    reciter_exists,
    upsert_reciter,
)
from quran_audio_data.surah_runner import run_surah_for_reciter
from quran_audio_data.supervision import (
    DEFAULT_RECITER_CATALOG_PATH,
    get_configured_reciter_entry,
    is_reciter_enabled,
    read_reciter_catalog,
    write_reciter_catalog,
)
from quran_audio_data.ui_sync import export_api_from_latest_runs


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
            raise typer.BadParameter("setup mode requires --reciter-name in non-interactive mode.")
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


def _normalize_catalog_slug(value: str) -> str:
    return str(value or "").strip().lower()


def _load_public_reciters(catalog_path: Path) -> list[dict[str, Any]]:
    payload = read_reciter_catalog(catalog_path)
    if payload is None:
        return []
    reciters = payload.get("reciters")
    if not isinstance(reciters, list):
        return []
    return [item for item in reciters if isinstance(item, dict)]


def _prompt_selected_reciters(reciters: list[dict[str, Any]]) -> list[str]:
    show_all = typer.confirm("Show all configured reciters?", default=False)
    visible = reciters if show_all else [item for item in reciters if bool(item.get("enabled"))]
    if not visible:
        visible = reciters

    table = Table(title="Reciter Selection")
    table.add_column("#")
    table.add_column("slug")
    table.add_column("enabled")
    table.add_column("name")
    table.add_column("check_type")
    for idx, item in enumerate(visible, start=1):
        table.add_row(
            str(idx),
            str(item.get("slug") or ""),
            str(bool(item.get("enabled"))),
            str(item.get("name") or ""),
            str(item.get("check_type") or ""),
        )
    console.print(table)

    raw = typer.prompt(
        "Select reciters by comma-separated slug or index (empty = all shown)",
        default="",
    ).strip()
    if not raw:
        return sorted(
            {_normalize_catalog_slug(item.get("slug")) for item in visible if item.get("slug")}
        )

    parts = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    by_index = {idx: item for idx, item in enumerate(visible, start=1)}
    selected: set[str] = set()
    for part in parts:
        if part.isdigit():
            item = by_index.get(int(part))
            if item and item.get("slug"):
                selected.add(_normalize_catalog_slug(item["slug"]))
            continue
        selected.add(_normalize_catalog_slug(part))
    return sorted(selected)


_SURAH_RANGE_RE = re.compile(r"^(?P<start>\d{1,3})-(?P<end>\d{1,3})$")


def _parse_surah_selection(raw: str) -> list[int]:
    value = raw.strip().lower()
    if value == "all":
        return list(range(1, 115))

    range_match = _SURAH_RANGE_RE.fullmatch(value)
    if range_match:
        start = int(range_match.group("start"))
        end = int(range_match.group("end"))
        if start < 1 or end > 114 or start > end:
            raise typer.BadParameter(f"invalid surah range: {raw}")
        return list(range(start, end + 1))

    parts = parse_csv_strings(raw)
    if not parts:
        raise typer.BadParameter("surah selection cannot be empty")
    values = sorted({int(part) for part in parts})
    for surah in values:
        if surah < 1 or surah > 114:
            raise typer.BadParameter(f"surah out of range: {surah}")
    return values


def _prompt_surah_selection() -> list[int]:
    raw = typer.prompt(
        "Surahs to process: all, range (e.g. 55-84), or csv list (e.g. 1,2,112)",
        default="all",
    )
    return _parse_surah_selection(raw)


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
    """Fetch EveryAyah + Quran.com reciters and write public reciters index."""

    enabled_set: set[str] | None = None
    if enabled_reciters is not None:
        enabled_set = {
            _normalize_catalog_slug(item) for item in (parse_csv_strings(enabled_reciters) or [])
        }
    else:
        existing = _load_public_reciters(out)
        if existing:
            enabled_set = {
                _normalize_catalog_slug(item.get("slug"))
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


@app.command("run-surah")
def run_surah_cmd(
    reciter_id: Annotated[str, typer.Option("--reciter-id")],
    surah: Annotated[int, typer.Option("--surah", min=1, max=114)],
    out_root: Annotated[Path, typer.Option("--out-root")] = Path("runs/surah_runs"),
    catalog: Annotated[Path, typer.Option("--catalog")] = DEFAULT_RECITER_CATALOG_PATH,
    text_data: Annotated[Path | None, typer.Option("--text-data")] = None,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(get_settings().cache_dir),
) -> None:
    """Run one reciter+surah pipeline job (debugging/verification path)."""

    normalized_reciter = _normalize_catalog_slug(reciter_id)
    if not is_reciter_enabled(normalized_reciter, catalog_path=catalog):
        raise typer.BadParameter(
            f"reciter is not enabled: {normalized_reciter}. "
            f"Enable it first in {catalog} (or use `qad sync-reciters --enabled-reciters ...`)."
        )

    entry = get_configured_reciter_entry(normalized_reciter, catalog_path=catalog) or {}
    source = entry.get("source") if isinstance(entry.get("source"), dict) else {}
    everyayah = source.get("everyayah") if isinstance(source.get("everyayah"), dict) else {}
    quran_com = source.get("quran_com") if isinstance(source.get("quran_com"), dict) else {}
    capabilities = entry.get("capabilities") if isinstance(entry.get("capabilities"), dict) else {}
    _print_key_values(
        "Run Setup",
        {
            "reciter_id": normalized_reciter,
            "surah": surah,
            "enabled": is_reciter_enabled(normalized_reciter, catalog_path=catalog),
            "check_type": entry.get("check_type") or "unknown",
            "ayah_by_ayah": bool(capabilities.get("ayah_by_ayah")),
            "word_by_word": bool(capabilities.get("word_by_word")),
            "everyayah_subfolder": everyayah.get("subfolder") or "-",
            "quran_com_recitation_id": quran_com.get("recitation_id") or "-",
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
    paths = summary.get("paths") if isinstance(summary.get("paths"), dict) else {}
    pipeline = summary.get("pipeline") if isinstance(summary.get("pipeline"), dict) else {}
    quality = summary.get("quality") if isinstance(summary.get("quality"), dict) else {}
    _print_key_values(
        "Run Summary",
        {
            "summary_path": paths.get("summary_path", ""),
            "output_dir": paths.get("output_dir", ""),
            "succeeded": pipeline.get("succeeded", 0),
            "failed": pipeline.get("failed", 0),
            "fallback_used": pipeline.get("fallback_used", 0),
            "avg_coverage": f"{float(quality.get('avg_coverage', 0.0)):.4f}",
            "min_coverage": f"{float(quality.get('min_coverage', 0.0)):.4f}",
        },
    )


@app.command("build-api")
def build_api_cmd(
    reciters: Annotated[
        str | None,
        typer.Option(
            "--reciters", help="Comma-separated reciter slugs. Interactive prompt if omitted."
        ),
    ] = None,
    surahs: Annotated[
        str | None,
        typer.Option("--surahs", help="all, range (e.g. 55-84), or csv list (e.g. 1,2,112)."),
    ] = None,
    force: Annotated[bool, typer.Option("--force/--no-force")] = False,
    refresh_reciters: Annotated[
        bool | None,
        typer.Option(
            "--refresh-reciters/--no-refresh-reciters",
            help="Refresh reciter catalog from sources before selecting reciters.",
        ),
    ] = None,
    interactive: Annotated[bool, typer.Option("--interactive/--no-interactive")] = True,
    export_only: Annotated[
        bool,
        typer.Option(
            "--export-only/--run-and-export",
            help="Skip running jobs and only export/sync API files.",
        ),
    ] = False,
    catalog: Annotated[Path, typer.Option("--catalog")] = DEFAULT_RECITER_CATALOG_PATH,
    runs_root: Annotated[Path, typer.Option("--runs-root")] = Path("runs"),
    out_root: Annotated[Path, typer.Option("--out-root")] = Path("runs/api_build"),
    api_root: Annotated[Path, typer.Option("--api-root")] = Path("data/api"),
    ui_data_dir: Annotated[Path, typer.Option("--ui-data-dir")] = Path("ui/public/data"),
    dist_data_dir: Annotated[Path, typer.Option("--dist-data-dir")] = Path("ui/dist/data"),
    sync_dist: Annotated[bool, typer.Option("--sync-dist/--no-sync-dist")] = False,
    prune_ui: Annotated[bool, typer.Option("--prune-ui/--no-prune-ui")] = True,
    text_data: Annotated[Path | None, typer.Option("--text-data")] = None,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(get_settings().cache_dir),
) -> None:
    """Interactive API build command: run selected reciter/surah jobs, then export `/data/reciters/*` endpoints."""

    use_interactive = interactive and _is_interactive_tty()

    if not catalog.exists():
        payload = write_reciter_catalog(path=catalog, enabled_reciters=None)
        _print_key_values(
            "Reciter Catalog Bootstrap",
            {
                "catalog": catalog,
                "configured_reciters": payload.get("counts", {}).get("configured_reciters", 0),
                "enabled_reciters": payload.get("counts", {}).get("enabled_reciters", 0),
            },
        )

    should_refresh = refresh_reciters
    if should_refresh is None and use_interactive:
        should_refresh = typer.confirm(
            "Refresh reciters from EveryAyah + Quran.com first?", default=False
        )
    should_refresh = bool(should_refresh)
    if should_refresh:
        existing = _load_public_reciters(catalog)
        enabled_set = {
            _normalize_catalog_slug(item.get("slug"))
            for item in existing
            if bool(item.get("enabled")) and item.get("slug")
        }
        write_reciter_catalog(path=catalog, enabled_reciters=enabled_set or None)

    reciter_rows = _load_public_reciters(catalog)
    if not reciter_rows:
        raise typer.BadParameter(f"catalog has no reciters: {catalog}")

    if reciters is not None:
        selected_reciters = sorted(
            {_normalize_catalog_slug(item) for item in (parse_csv_strings(reciters) or [])}
        )
    elif use_interactive:
        selected_reciters = _prompt_selected_reciters(reciter_rows)
    else:
        selected_reciters = sorted(
            {
                _normalize_catalog_slug(item.get("slug"))
                for item in reciter_rows
                if bool(item.get("enabled")) and item.get("slug")
            }
        )

    if not selected_reciters:
        raise typer.BadParameter("no reciters selected")

    known_slugs = {
        _normalize_catalog_slug(item.get("slug")) for item in reciter_rows if item.get("slug")
    }
    unknown = [item for item in selected_reciters if item not in known_slugs]
    if unknown:
        raise typer.BadParameter(f"unknown reciters: {', '.join(unknown)}")

    if surahs is not None:
        selected_surahs = _parse_surah_selection(surahs)
    elif use_interactive:
        selected_surahs = _prompt_surah_selection()
    else:
        selected_surahs = list(range(1, 115))

    tasks = [(reciter_id, surah) for reciter_id in selected_reciters for surah in selected_surahs]
    to_run: list[tuple[str, int]] = []
    skipped = 0
    for reciter_id, surah in tasks:
        metadata_path = api_root / "reciters" / reciter_id / "surahs" / str(surah) / "metadata.json"
        timings_path = api_root / "reciters" / reciter_id / "surahs" / str(surah) / "timings.json"
        if not force and metadata_path.exists() and timings_path.exists():
            skipped += 1
            continue
        to_run.append((reciter_id, surah))

    _print_key_values(
        "Build Preview",
        {
            "selected_reciters": len(selected_reciters),
            "selected_surahs": len(selected_surahs),
            "selected_pairs": len(tasks),
            "to_run": 0 if export_only else len(to_run),
            "skipped_existing": skipped if not force else 0,
            "force": force,
            "runs_root": runs_root,
            "out_root": out_root,
            "api_root": api_root,
        },
    )

    if use_interactive and not typer.confirm("Continue?", default=True):
        raise typer.Exit(code=0)

    failures: list[str] = []
    if not export_only:
        out_root.mkdir(parents=True, exist_ok=True)
        total = len(to_run)
        for idx, (reciter_id, surah) in enumerate(to_run, start=1):
            console.print(f"[{idx}/{total}] run {reciter_id} surah={surah}")
            try:
                run_surah_for_reciter(
                    reciter_id=reciter_id,
                    surah=surah,
                    out_root=out_root,
                    text_data=text_data,
                    cache_dir=cache_dir,
                    catalog_path=catalog,
                )
            except Exception as exc:
                failures.append(f"{reciter_id}:{surah}: {exc}")

    export_summary = export_api_from_latest_runs(
        runs_root=runs_root,
        api_root=api_root,
        reciters_index_path=catalog,
        ui_data_dir=ui_data_dir,
        dist_data_dir=dist_data_dir,
        sync_dist=sync_dist,
        prune_ui=prune_ui,
        dry_run=False,
        include_reciters=set(selected_reciters),
        include_surahs=set(selected_surahs),
    )

    _print_key_values(
        "API Export Summary",
        {
            "keys_selected": export_summary["keys_selected"],
            "surah_changed": export_summary["api"]["surah_changed"],
            "audio_copied": export_summary["api"]["audio_copied"],
            "reciters_index_changed": export_summary["reciters_index"]["changed"],
            "ui_copied": export_summary["ui"]["copied"],
            "dist_copied": export_summary["dist"]["copied"]
            if export_summary["dist"]["enabled"]
            else 0,
        },
    )
    _print_errors(failures)
    if failures:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
