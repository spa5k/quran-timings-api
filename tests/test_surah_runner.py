from __future__ import annotations

from quran_audio_data.surah_runner import _evaluate_ayah_timing_against_reference
from pathlib import Path

import orjson

import quran_audio_data.surah_runner as surah_runner
from quran_audio_data.pipeline.types import PipelineReportV3, ProcessedFile


def test_evaluate_ayah_timing_against_reference_metrics() -> None:
    reference = {
        1: (0.0, 1.0),
        2: (1.0, 2.0),
    }
    predicted = [
        {"ayah": 1, "start_s": 0.01, "end_s": 1.02},
        {"ayah": 2, "start_s": 1.03, "end_s": 2.04},
    ]

    result = _evaluate_ayah_timing_against_reference(
        predicted_ayahs=predicted,
        reference_bounds=reference,
    )

    assert result is not None
    assert result["expected_ayahs"] == 2
    assert result["predicted_ayahs"] == 2
    assert result["matched_ayahs"] == 2
    assert result["coverage_vs_reference"] == 1.0
    assert result["boundary_error_median_ms"] == 25.0
    assert result["boundary_error_p95_ms"] == 38.5
    assert result["boundary_hit_rate_20ms"] == 0.5
    assert result["boundary_hit_rate_50ms"] == 1.0
    assert result["boundary_hit_rate_80ms"] == 1.0
    assert result["start_offset_s"] == 0.02
    assert result["offset_normalized_boundary_error_median_ms"] == 10.0
    assert result["offset_normalized_boundary_error_p95_ms"] == 18.5
    assert result["offset_normalized_hit_rate_20ms"] == 1.0


def test_evaluate_ayah_timing_against_reference_empty_or_unmatched() -> None:
    assert (
        _evaluate_ayah_timing_against_reference(
            predicted_ayahs=[],
            reference_bounds={1: (0.0, 1.0)},
        )
        is None
    )

    result = _evaluate_ayah_timing_against_reference(
        predicted_ayahs=[{"ayah": 5, "start_s": 1.0, "end_s": 2.0}],
        reference_bounds={1: (0.0, 1.0)},
    )
    assert result is None


def test_run_surah_for_reciter_writes_consistent_fallback_summary(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "runs" / "test_reciter_s001" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "test_reciter_s001_full.json"
    output_json.write_bytes(
        orjson.dumps(
            {
                "schema_version": "v3",
                "audio": {
                    "path": str(tmp_path / "audio.mp3"),
                    "duration_s": 1.0,
                    "sample_rate": 16000,
                    "channels": 1,
                },
                "meta": {
                    "reciter_id": "test_reciter",
                    "surah": 1,
                    "input_mode": "full_surah",
                },
                "engine": {
                    "name": "ensemble",
                    "model": "ayah-wise",
                    "device": "cpu",
                    "fallback_used": True,
                },
                "ayahs": [{"ayah": 1, "start_s": 0.0, "end_s": 1.0}],
                "words": [],
                "qc": {"coverage": 1.0, "monotonic": True, "duration_match": True, "warnings": []},
                "segment_source_type": "none",
            }
        )
    )

    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"fake-mp3")

    monkeypatch.setattr(surah_runner, "is_reciter_enabled", lambda reciter_id, catalog_path: True)
    monkeypatch.setattr(
        surah_runner,
        "resolve_reciter_mapping",
        lambda reciter_id, catalog_path: type(
            "Mapping",
            (),
            {
                "everyayah_subfolder": "Test_128kbps",
                "qcom_recitation_id": None,
                "qcom_word_supervision_supported": False,
            },
        )(),
    )
    monkeypatch.setattr(
        surah_runner,
        "get_configured_reciter_entry",
        lambda reciter_id, catalog_path: {"slug": reciter_id, "name": "Test Reciter"},
    )
    monkeypatch.setattr(
        surah_runner,
        "_prepare_surah_audio",
        lambda **kwargs: (
            kwargs["audio_path"].write_bytes(b"fake-mp3"),
            surah_runner.PreparedSurahAudio(
                source_url="https://example.com/audio.mp3",
                source_type="everyayah",
            ),
        )[1],
    )

    def fake_run_alignment_pipeline(**kwargs):
        return PipelineReportV3(
            total=1,
            succeeded=1,
            failed=0,
            aligned=1,
            fallback_used=1,
            elapsed_s=1.0,
            outputs=[
                ProcessedFile(
                    row=None,  # type: ignore[arg-type]
                    output_json=output_json,
                    output_ayah_csv=output_dir / "dummy_ayah.csv",
                    output_words_csv=output_dir / "dummy_words.csv",
                    qc_report_json=output_dir / "dummy_qc.json",
                    text_audit_json=None,
                    source="fallback",
                    fallback_used=True,
                    elapsed_s=1.0,
                )
            ],
            errors=[],
            error_details=[],
            attempted_engines=["nemo", "whisperx", "mfa"],
            priors_used=0,
        )

    monkeypatch.setattr(surah_runner, "run_alignment_pipeline", fake_run_alignment_pipeline)

    summary = surah_runner.run_surah_for_reciter(
        reciter_id="test_reciter",
        surah=1,
        out_root=tmp_path / "runs",
        text_data=None,
        cache_dir=tmp_path / ".cache",
        catalog_path=tmp_path / "catalog.json",
    )

    assert summary["pipeline"]["fallback_used"] == 1
    written = orjson.loads(
        (tmp_path / "runs" / "test_reciter_s001" / "run_summary.json").read_bytes()
    )
    assert written["pipeline"]["fallback_used"] == 1
