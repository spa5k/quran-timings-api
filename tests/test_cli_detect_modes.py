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


def test_help_exposes_only_detect_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "detect" in result.output
    assert "run-surah" not in result.output
    assert "sync-reciters" not in result.output
    assert "build-corpus" not in result.output
    assert "sync-ui-data" not in result.output


def test_list_mode_reads_registry(tmp_path) -> None:
    reciters_path = tmp_path / "reciters.json"
    reciters_path.write_text(
        (
            '{\n'
            '  "version": 1,\n'
            '  "updated_at": "2026-03-06T00:00:00+00:00",\n'
            '  "reciters": [\n'
            '    {\n'
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

    result = runner.invoke(app, ["detect", "--list-reciters", "--reciters-path", str(reciters_path)])
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
