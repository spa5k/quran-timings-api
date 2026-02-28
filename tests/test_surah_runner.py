from __future__ import annotations

from quran_audio_data.surah_runner import _evaluate_ayah_timing_against_reference


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
