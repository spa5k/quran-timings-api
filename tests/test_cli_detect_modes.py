from __future__ import annotations

import json
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


def _build_export_summary() -> dict[str, object]:
    return {
        "keys_selected": 1,
        "api": {"surah_changed": 1, "audio_copied": 0},
        "reciters_index": {"changed": True},
        "ui": {"copied": 1},
        "dist": {"enabled": False, "copied": 0},
    }


def test_help_exposes_detect_only() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "detect" in result.output
    assert "sync-reciters" in result.output
    assert "list-reciters" in result.output
    assert "run-surah" not in result.output
    assert "build-api" not in result.output


def test_detect_missing_required_flags_fails_non_interactive() -> None:
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
    assert "requires --reciter-id, --surah, and --audio-url" in result.output


def test_detect_unknown_reciter_non_interactive_registers_and_publishes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli_module, "get_bytes_with_retry", lambda **kwargs: b"fake-mp3")
    monkeypatch.setattr(cli_module, "run_alignment_pipeline", lambda **kwargs: _Summary())

    export_calls: list[dict[str, object]] = []

    def _fake_export(**kwargs):
        export_calls.append(kwargs)
        return _build_export_summary()

    monkeypatch.setattr(cli_module, "export_api_from_latest_runs", _fake_export)

    out_root = tmp_path / "runs"
    reciters_path = tmp_path / "detect_reciters.json"
    catalog = tmp_path / "reciters.json"

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
            str(out_root),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--reciters-path",
            str(reciters_path),
            "--catalog",
            str(catalog),
            "--api-root",
            str(tmp_path / "api"),
            "--ui-data-dir",
            str(tmp_path / "ui"),
        ],
    )
    assert result.exit_code == 0

    payload = load_registry(reciters_path)
    ids = [str(item.get("id")) for item in payload.get("reciters", [])]
    assert "unknown_reciter" in ids

    catalog_payload = json.loads(catalog.read_text(encoding="utf-8"))
    assert catalog_payload["reciters"][0]["slug"] == "unknown_reciter"
    assert catalog_payload["reciters"][0]["enabled"] is True

    manifests = list(out_root.glob("**/manifest.csv"))
    assert len(manifests) == 1
    assert "unknown_reciter" in manifests[0].read_text(encoding="utf-8")

    assert len(export_calls) == 1
    assert export_calls[0]["runs_root"] == out_root
    assert export_calls[0]["include_reciters"] == {"unknown_reciter"}
    assert export_calls[0]["include_surahs"] == {1}
    assert export_calls[0]["prune_ui"] is False


def test_detect_without_params_runs_interactive_flow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli_module, "_is_interactive_tty", lambda: True)
    monkeypatch.setattr(cli_module, "get_bytes_with_retry", lambda **kwargs: b"fake-mp3")
    monkeypatch.setattr(cli_module, "run_alignment_pipeline", lambda **kwargs: _Summary())

    export_calls: list[dict[str, object]] = []

    def _fake_export(**kwargs):
        export_calls.append(kwargs)
        return _build_export_summary()

    monkeypatch.setattr(cli_module, "export_api_from_latest_runs", _fake_export)

    reciters_path = tmp_path / "detect_reciters.json"
    catalog = tmp_path / "reciters.json"

    result = runner.invoke(
        app,
        [
            "detect",
            "--out-root",
            str(tmp_path / "runs"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--reciters-path",
            str(reciters_path),
            "--catalog",
            str(catalog),
            "--api-root",
            str(tmp_path / "api"),
            "--ui-data-dir",
            str(tmp_path / "ui"),
        ],
        input="\ninteractive_reciter\nInteractive Reciter\ncustom\n\n114\n\nhttps://example.com/114.mp3\n",
    )

    assert result.exit_code == 0
    payload = load_registry(reciters_path)
    rows = payload.get("reciters", [])
    ids = [str(item.get("id")) for item in rows]
    assert "interactive_reciter" in ids
    interactive_row = next(item for item in rows if str(item.get("id")) == "interactive_reciter")
    assert interactive_row["name"] == "Interactive Reciter (custom)"
    assert len(export_calls) == 1
    assert export_calls[0]["include_reciters"] == {"interactive_reciter"}
    assert export_calls[0]["include_surahs"] == {114}


def test_detect_ayah_run_skips_publish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli_module, "get_bytes_with_retry", lambda **kwargs: b"fake-mp3")
    monkeypatch.setattr(cli_module, "run_alignment_pipeline", lambda **kwargs: _Summary())

    called = {"count": 0}

    def _fake_export(**kwargs):
        called["count"] += 1
        return _build_export_summary()

    monkeypatch.setattr(cli_module, "export_api_from_latest_runs", _fake_export)

    result = runner.invoke(
        app,
        [
            "detect",
            "--audio-url",
            "https://example.com/001.mp3",
            "--reciter-id",
            "ayah_clip_reciter",
            "--surah",
            "1",
            "--ayah",
            "1",
            "--out-root",
            str(tmp_path / "runs"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--reciters-path",
            str(tmp_path / "detect_reciters.json"),
            "--catalog",
            str(tmp_path / "reciters.json"),
            "--api-root",
            str(tmp_path / "api"),
            "--ui-data-dir",
            str(tmp_path / "ui"),
        ],
    )

    assert result.exit_code == 0
    assert called["count"] == 0
    assert "Ayah-only runs are not auto-published" in result.output


def test_detect_known_catalog_reciter_preserves_catalog_slug_and_skips_detect_registry_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli_module, "get_bytes_with_retry", lambda **kwargs: b"fake-mp3")
    monkeypatch.setattr(cli_module, "run_alignment_pipeline", lambda **kwargs: _Summary())
    monkeypatch.setattr(
        cli_module, "export_api_from_latest_runs", lambda **kwargs: _build_export_summary()
    )

    reciters_path = tmp_path / "detect_reciters.json"
    catalog = tmp_path / "reciters.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "generated_at": "2026-03-10T00:00:00+00:00",
                "counts": {
                    "everyayah_source_reciters": 0,
                    "quran_com_source_reciters": 1,
                    "quranicaudio_source_reciters": 0,
                    "configured_reciters": 1,
                    "enabled_reciters": 1,
                },
                "sources": {},
                "reciters": [
                    {
                        "slug": "yasser_ad-dussary",
                        "name": "Yasser Ad-Dussary",
                        "enabled": True,
                        "check_type": "word_by_word",
                        "capabilities": {"ayah_by_ayah": False, "word_by_word": True},
                        "source": {
                            "everyayah": {"subfolder": None, "reciter_key": None, "name": None},
                            "quran_com": {"recitation_id": 10, "name": "Yasser Ad-Dussary"},
                            "quranicaudio": {"path": None, "name": None},
                        },
                        "surahs_available": [],
                        "surah_count": 0,
                        "endpoints": {"metadata": "/data/reciters/yasser_ad-dussary/metadata.json"},
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "detect",
            "--audio-url",
            "https://example.com/114.mp3",
            "--reciter-id",
            "yasser_ad-dussary",
            "--surah",
            "114",
            "--out-root",
            str(tmp_path / "runs"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--reciters-path",
            str(reciters_path),
            "--catalog",
            str(catalog),
            "--api-root",
            str(tmp_path / "api"),
            "--ui-data-dir",
            str(tmp_path / "ui"),
        ],
    )

    assert result.exit_code == 0
    assert not reciters_path.exists()
    assert "yasser_ad-dussary" in result.output
    assert "yasser_ad_dussary" not in result.output


def test_detect_legacy_everyayah_id_resolves_to_canonical_catalog_slug(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli_module, "get_bytes_with_retry", lambda **kwargs: b"fake-mp3")
    monkeypatch.setattr(cli_module, "run_alignment_pipeline", lambda **kwargs: _Summary())
    monkeypatch.setattr(
        cli_module, "export_api_from_latest_runs", lambda **kwargs: _build_export_summary()
    )

    reciters_path = tmp_path / "detect_reciters.json"
    catalog = tmp_path / "reciters.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "generated_at": "2026-03-10T00:00:00+00:00",
                "counts": {
                    "everyayah_source_reciters": 1,
                    "quran_com_source_reciters": 0,
                    "quranicaudio_source_reciters": 0,
                    "configured_reciters": 1,
                    "enabled_reciters": 1,
                },
                "sources": {},
                "reciters": [
                    {
                        "slug": "eya_yasser_ad_dussary_128kbps",
                        "name": "Yasser_Ad-Dussary",
                        "enabled": True,
                        "check_type": "ayah_by_ayah",
                        "capabilities": {"ayah_by_ayah": True, "word_by_word": False},
                        "source": {
                            "everyayah": {
                                "subfolder": "Yasser_Ad-Dussary_128kbps",
                                "reciter_key": 67,
                                "name": "Yasser_Ad-Dussary",
                            },
                            "quran_com": {"recitation_id": None, "name": None},
                            "quranicaudio": {"path": None, "name": None},
                        },
                        "surahs_available": [],
                        "surah_count": 0,
                        "endpoints": {
                            "metadata": "/data/reciters/eya_yasser_ad_dussary_128kbps/metadata.json"
                        },
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "detect",
            "--audio-url",
            "https://example.com/114.mp3",
            "--reciter-id",
            "yasser_ad-dussary",
            "--surah",
            "114",
            "--out-root",
            str(tmp_path / "runs"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--reciters-path",
            str(reciters_path),
            "--catalog",
            str(catalog),
            "--api-root",
            str(tmp_path / "api"),
            "--ui-data-dir",
            str(tmp_path / "ui"),
        ],
    )

    assert result.exit_code == 0
    assert not reciters_path.exists()
    assert "eya_yasser_ad_dussary_128kbps" in result.output


def test_list_reciters_reads_catalog(tmp_path: Path) -> None:
    catalog = tmp_path / "reciters.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "generated_at": "2026-03-10T00:00:00+00:00",
                "counts": {
                    "everyayah_source_reciters": 0,
                    "quran_com_source_reciters": 0,
                    "quranicaudio_source_reciters": 0,
                    "configured_reciters": 1,
                    "enabled_reciters": 1,
                },
                "sources": {},
                "reciters": [
                    {
                        "slug": "test_reciter",
                        "name": "Test Reciter",
                        "enabled": True,
                        "check_type": "model_only",
                        "capabilities": {"ayah_by_ayah": False, "word_by_word": False},
                        "source": {
                            "everyayah": {"subfolder": None, "reciter_key": None, "name": None},
                            "quran_com": {"recitation_id": None, "name": None},
                            "quranicaudio": {"path": None, "name": None},
                        },
                        "surahs_available": [1],
                        "surah_count": 1,
                        "endpoints": {"metadata": "/data/reciters/test_reciter/metadata.json"},
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["list-reciters", "--catalog", str(catalog)])
    assert result.exit_code == 0
    assert "Configured Reciters" in result.output
    assert "List Summary" in result.output
