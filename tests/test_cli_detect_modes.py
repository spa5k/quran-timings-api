from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import quran_audio_data.cli as cli_module
from quran_audio_data.cli import app
from quran_audio_data.reciters import load_registry


runner = CliRunner()


class _Summary:
    schema_version = "1.0.0"
    total = 1
    succeeded = 1
    failed = 0
    aligned = 1
    fallback_used = 0
    priors_used = 0
    elapsed_s = 0.12
    attempted_engines = ["nemo"]
    errors: list[str] = []


def test_help_exposes_public_api_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "detect" in result.output
    assert "run-surah" in result.output
    assert "sync-reciters" in result.output
    assert "build-api" in result.output


def test_list_mode_reads_registry(tmp_path) -> None:
    reciters_path = tmp_path / "reciters.json"
    reciters_path.write_text(
        (
            "{\n"
            '  "version": 1,\n'
            '  "updated_at": "2026-03-06T00:00:00+00:00",\n'
            '  "reciters": [\n'
            "    {\n"
            '      "id": "test_reciter",\n'
            '      "name": "Test Reciter",\n'
            '      "source": "custom",\n'
            '      "notes": "",\n'
            '      "created_at": "2026-03-06T00:00:00+00:00",\n'
            '      "updated_at": "2026-03-06T00:00:00+00:00"\n'
            "    }\n"
            "  ]\n"
            "}\n"
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app, ["detect", "--list-reciters", "--reciters-path", str(reciters_path)]
    )
    assert result.exit_code == 0
    assert "test_reciter" in result.output


def test_mode_conflict_fails() -> None:
    result = runner.invoke(app, ["detect", "--setup-reciter", "--list-reciters"])
    assert result.exit_code != 0
    assert "cannot be used together" in result.output


def test_setup_mode_writes_registry(tmp_path) -> None:
    reciters_path = tmp_path / "reciters.json"
    result = runner.invoke(
        app,
        [
            "detect",
            "--setup-reciter",
            "--reciter-id",
            "new_reciter",
            "--reciter-name",
            "New Reciter",
            "--source",
            "quranicaudio",
            "--notes",
            "note",
            "--reciters-path",
            str(reciters_path),
        ],
    )
    assert result.exit_code == 0
    payload = load_registry(reciters_path)
    ids = [str(item.get("id")) for item in payload.get("reciters", [])]
    assert "new_reciter" in ids


def test_setup_mode_missing_optional_in_non_interactive_fails(tmp_path) -> None:
    reciters_path = tmp_path / "reciters.json"
    result = runner.invoke(
        app,
        [
            "detect",
            "--setup-reciter",
            "--reciter-id",
            "new_reciter",
            "--reciters-path",
            str(reciters_path),
        ],
    )
    assert result.exit_code != 0
    assert "--reciter-name" in result.output


def test_run_mode_missing_required_flags_fails() -> None:
    result = runner.invoke(
        app,
        [
            "detect",
            "--audio-url",
            "https://example.com/001.mp3",
            "--surah",
            "1",
        ],
    )
    assert result.exit_code != 0
    assert "--reciter-id" in result.output


def test_unknown_reciter_non_interactive_fails_with_setup_example(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli_module, "reciter_exists", lambda *args, **kwargs: False)
    result = runner.invoke(
        app,
        [
            "detect",
            "--audio-url",
            "https://example.com/001.mp3",
            "--reciter-id",
            "unknown_reciter",
            "--surah",
            "1",
            "--out-root",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code != 0
    assert "unknown reciter-id 'unknown_reciter'" in result.output
    assert "--setup-reciter --reciter-id unknown_reciter" in result.output


def test_run_mode_known_reciter_executes_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli_module, "reciter_exists", lambda *args, **kwargs: True)
    monkeypatch.setattr(cli_module, "get_bytes_with_retry", lambda **kwargs: b"fake-mp3")
    monkeypatch.setattr(cli_module, "run_alignment_pipeline", lambda **kwargs: _Summary())

    out_root = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "detect",
            "--audio-url",
            "https://example.com/001.mp3",
            "--reciter-id",
            "known_reciter",
            "--surah",
            "1",
            "--out-root",
            str(out_root),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )
    assert result.exit_code == 0
    manifests = list(out_root.glob("**/manifest.csv"))
    assert len(manifests) == 1
    manifest_text = manifests[0].read_text(encoding="utf-8")
    assert "known_reciter" in manifest_text
    assert ",1," in manifest_text


def _build_api_summary() -> dict[str, object]:
    return {
        "keys_selected": 0,
        "api": {"surah_changed": 0, "audio_copied": 0},
        "reciters_index": {"changed": False},
        "ui": {"copied": 0},
        "dist": {"enabled": False, "copied": 0},
    }


def _write_public_catalog(path: Path) -> None:
    path.write_text(
        (
            "{\n"
            '  "schema_version": "v1",\n'
            '  "generated_at": "2026-03-06T00:00:00+00:00",\n'
            '  "counts": {"configured_reciters": 1, "enabled_reciters": 1},\n'
            '  "sources": {},\n'
            '  "reciters": [\n'
            "    {\n"
            '      "slug": "reciter-1",\n'
            '      "name": "Reciter One",\n'
            '      "enabled": true,\n'
            '      "check_type": "word_by_word",\n'
            '      "capabilities": {"ayah_by_ayah": false, "word_by_word": true},\n'
            '      "source": {\n'
            '        "everyayah": {"subfolder": null, "reciter_key": null, "name": null},\n'
            '        "quran_com": {"recitation_id": 3, "name": "Reciter One"}\n'
            "      },\n"
            '      "endpoints": {"metadata": "/data/reciters/reciter-1/metadata.json"},\n'
            '      "surahs_available": [],\n'
            '      "surah_count": 0\n'
            "    }\n"
            "  ]\n"
            "}\n"
        ),
        encoding="utf-8",
    )


def test_build_api_skips_existing_surah_files_without_force(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "reciters.json"
    _write_public_catalog(catalog)

    api_root = tmp_path / "api"
    target = api_root / "reciters" / "reciter-1" / "surahs" / "1"
    target.mkdir(parents=True, exist_ok=True)
    (target / "metadata.json").write_text("{}", encoding="utf-8")
    (target / "timings.json").write_text("{}", encoding="utf-8")

    calls: list[tuple[str, int]] = []

    def _fake_run_surah_for_reciter(*, reciter_id, surah, **kwargs):
        calls.append((reciter_id, surah))
        return {}

    monkeypatch.setattr(cli_module, "run_surah_for_reciter", _fake_run_surah_for_reciter)
    monkeypatch.setattr(
        cli_module, "export_api_from_latest_runs", lambda **kwargs: _build_api_summary()
    )

    result = runner.invoke(
        app,
        [
            "build-api",
            "--no-interactive",
            "--catalog",
            str(catalog),
            "--reciters",
            "reciter-1",
            "--surahs",
            "1",
            "--api-root",
            str(api_root),
            "--runs-root",
            str(tmp_path / "runs"),
            "--out-root",
            str(tmp_path / "out"),
            "--no-sync-dist",
            "--no-prune-ui",
        ],
    )
    assert result.exit_code == 0
    assert calls == []


def test_build_api_force_rebuilds_selected_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "reciters.json"
    _write_public_catalog(catalog)

    api_root = tmp_path / "api"
    target = api_root / "reciters" / "reciter-1" / "surahs" / "1"
    target.mkdir(parents=True, exist_ok=True)
    (target / "metadata.json").write_text("{}", encoding="utf-8")
    (target / "timings.json").write_text("{}", encoding="utf-8")

    calls: list[tuple[str, int]] = []

    def _fake_run_surah_for_reciter(*, reciter_id, surah, **kwargs):
        calls.append((reciter_id, surah))
        return {}

    monkeypatch.setattr(cli_module, "run_surah_for_reciter", _fake_run_surah_for_reciter)
    monkeypatch.setattr(
        cli_module, "export_api_from_latest_runs", lambda **kwargs: _build_api_summary()
    )

    result = runner.invoke(
        app,
        [
            "build-api",
            "--no-interactive",
            "--catalog",
            str(catalog),
            "--reciters",
            "reciter-1",
            "--surahs",
            "1",
            "--api-root",
            str(api_root),
            "--runs-root",
            str(tmp_path / "runs"),
            "--out-root",
            str(tmp_path / "out"),
            "--force",
            "--no-sync-dist",
            "--no-prune-ui",
        ],
    )
    assert result.exit_code == 0
    assert calls == [("reciter-1", 1)]
