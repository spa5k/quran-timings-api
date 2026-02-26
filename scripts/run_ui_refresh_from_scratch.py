from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import json
import shutil
import subprocess
import time


@dataclass(slots=True)
class JobRow:
    audio_path: str
    reciter_id: str
    surah: int
    ayah: str
    source_url: str
    sha256: str
    language: str


def _build_rows(catalog_path: Path, repo_root: Path) -> list[JobRow]:
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    recitations = payload.get("recitations")
    if not isinstance(recitations, list):
        raise ValueError("catalog.json missing recitations list")

    rows: list[JobRow] = []
    for recitation in recitations:
        if not isinstance(recitation, dict):
            continue
        reciter_id = str(recitation.get("id", "")).strip()
        if not reciter_id:
            continue
        surahs = recitation.get("surahs")
        if not isinstance(surahs, list):
            continue

        for entry in surahs:
            if not isinstance(entry, dict):
                continue
            surah = int(entry["surah"])
            audio_path = repo_root / "samples" / "audio" / f"{reciter_id}_{surah:03d}.mp3"
            rows.append(
                JobRow(
                    audio_path=str(audio_path),
                    reciter_id=reciter_id,
                    surah=surah,
                    ayah="",
                    source_url=str(entry.get("audioSrc", "")),
                    sha256="",
                    language="ar",
                )
            )

    rows.sort(key=lambda row: (row.reciter_id, row.surah))
    return rows


def _delete_existing_ui_timings(ui_public_dir: Path, ui_dist_dir: Path) -> None:
    prefixes = (
        "abdulbaset_warsh_s",
        "abdurrahmaan_as-sudays_s",
        "sa3ood_al-shuraym_s",
        "yasser_ad-dussary_s",
    )
    for target_dir in (ui_public_dir, ui_dist_dir):
        if not target_dir.exists():
            continue
        for file_path in target_dir.glob("*_full.json"):
            if file_path.name.startswith(prefixes):
                file_path.unlink(missing_ok=True)


def _write_manifest(path: Path, row: JobRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
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
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "audio_path": row.audio_path,
                "reciter_id": row.reciter_id,
                "surah": str(row.surah),
                "ayah": row.ayah,
                "source_url": row.source_url,
                "sha256": row.sha256,
                "language": row.language,
            }
        )


def _expected_output_name(row: JobRow) -> str:
    return f"{row.reciter_id}_s{row.surah:03d}_full.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a full from-scratch UI timing refresh one row at a time.")
    parser.add_argument("--repo-root", default=".", help="Repository root path")
    parser.add_argument(
        "--catalog",
        default="ui/public/data/catalog.json",
        help="UI catalog path used to enumerate reciter/surah rows",
    )
    parser.add_argument("--out-dir", default="runs/ui_refresh_from_scratch", help="Output run directory")
    parser.add_argument("--cache-dir", default=".cache/timings_from_scratch", help="Fresh cache directory")
    parser.add_argument("--accuracy-mode", default="strict", choices=["standard", "strict"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-remote", action="store_true", default=True)
    parser.add_argument("--keep-remote", action="store_true", help="Allow remote existing timing resolver")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    catalog_path = (repo_root / args.catalog).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    ui_public_dir = (repo_root / "ui/public/data").resolve()
    ui_dist_dir = (repo_root / "ui/dist/data").resolve()
    cache_dir = (repo_root / args.cache_dir).resolve()

    rows = _build_rows(catalog_path=catalog_path, repo_root=repo_root)
    if not rows:
        raise RuntimeError("No rows were found in catalog.")

    missing_audio = [row.audio_path for row in rows if not Path(row.audio_path).exists()]
    if missing_audio:
        raise FileNotFoundError(f"Missing audio files: {missing_audio[:10]}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    _delete_existing_ui_timings(ui_public_dir=ui_public_dir, ui_dist_dir=ui_dist_dir)

    manifest_path = out_dir / "_single_row_manifest.csv"
    log_path = out_dir / "run.log"
    failed_path = out_dir / "failed_rows.csv"

    with failed_path.open("w", newline="", encoding="utf-8") as failed_fh:
        failed_writer = csv.DictWriter(
            failed_fh,
            fieldnames=["index", "reciter_id", "surah", "return_code"],
        )
        failed_writer.writeheader()

        started = time.time()
        succeeded = 0
        failed = 0

        with log_path.open("a", encoding="utf-8") as log_fh:
            for idx, row in enumerate(rows, start=1):
                _write_manifest(manifest_path, row)
                expected_name = _expected_output_name(row)

                cmd = [
                    "uv",
                    "run",
                    "qad",
                    "align",
                    "--manifest",
                    str(manifest_path),
                    "--out",
                    str(out_dir),
                    "--accuracy-mode",
                    args.accuracy_mode,
                    "--device",
                    args.device,
                    "--cache-dir",
                    str(cache_dir),
                ]
                if not args.keep_remote:
                    cmd.append("--no-remote")

                row_started = time.time()
                log_fh.write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START {idx}/{len(rows)} "
                    f"{row.reciter_id} surah={row.surah}\n"
                )
                log_fh.flush()

                proc = subprocess.run(
                    cmd,
                    cwd=str(repo_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                elapsed = time.time() - row_started
                log_fh.write(proc.stdout)
                if proc.returncode != 0:
                    failed += 1
                    failed_writer.writerow(
                        {
                            "index": idx,
                            "reciter_id": row.reciter_id,
                            "surah": row.surah,
                            "return_code": proc.returncode,
                        }
                    )
                    log_fh.write(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FAIL {idx}/{len(rows)} "
                        f"{row.reciter_id} surah={row.surah} elapsed_s={elapsed:.1f}\n"
                    )
                    log_fh.flush()
                    continue

                output_json = out_dir / expected_name
                if not output_json.exists():
                    failed += 1
                    failed_writer.writerow(
                        {
                            "index": idx,
                            "reciter_id": row.reciter_id,
                            "surah": row.surah,
                            "return_code": 99,
                        }
                    )
                    log_fh.write(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FAIL-MISSING {idx}/{len(rows)} "
                        f"{row.reciter_id} surah={row.surah} elapsed_s={elapsed:.1f}\n"
                    )
                    log_fh.flush()
                    continue

                shutil.copy2(output_json, ui_public_dir / output_json.name)
                shutil.copy2(output_json, ui_dist_dir / output_json.name)
                succeeded += 1
                log_fh.write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] OK {idx}/{len(rows)} "
                    f"{row.reciter_id} surah={row.surah} elapsed_s={elapsed:.1f}\n"
                )
                log_fh.flush()

        total_elapsed = time.time() - started
        summary = {
            "total": len(rows),
            "succeeded": succeeded,
            "failed": failed,
            "elapsed_s": round(total_elapsed, 2),
            "out_dir": str(out_dir),
            "log_path": str(log_path),
            "failed_rows_path": str(failed_path),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
