from quran_audio_data.schema import QCThresholds, WordTiming, compute_qc, qc_requires_fallback


def _word(
    *,
    idx: int,
    start_s: float,
    end_s: float,
    alignment_origin: str = "native",
    match_score: float | None = 100.0,
) -> WordTiming:
    return WordTiming(
        surah=1,
        ayah=1,
        word_index_global=idx,
        word_index_in_ayah=idx,
        text_uthmani="x",
        text_norm="x",
        start_s=start_s,
        end_s=end_s,
        confidence=0.9,
        alignment_origin=alignment_origin,  # type: ignore[arg-type]
        match_score=match_score,
        engine_candidate="nemo",
    )


def test_qc_fallback_includes_duration_mismatch() -> None:
    words = [
        _word(idx=1, start_s=0.0, end_s=0.2),
        _word(idx=2, start_s=0.2, end_s=0.5),
    ]
    thresholds = QCThresholds(max_duration_delta_ratio=0.05)
    qc = compute_qc(
        words=words,
        expected_word_count=2,
        audio_duration_s=1.0,
        thresholds=thresholds,
    )

    assert not qc.duration_match
    assert qc_requires_fallback(qc, thresholds)


def test_qc_uses_speech_end_reference_when_provided() -> None:
    words = [
        _word(idx=1, start_s=0.0, end_s=0.2),
        _word(idx=2, start_s=0.2, end_s=0.5),
    ]
    thresholds = QCThresholds(max_duration_delta_ratio=0.05)
    qc = compute_qc(
        words=words,
        expected_word_count=2,
        audio_duration_s=1.0,
        speech_end_s=0.5,
        thresholds=thresholds,
    )

    assert qc.duration_match
    assert not qc_requires_fallback(qc, thresholds)


def test_qc_fallback_includes_interpolated_ratio_in_strict_mode() -> None:
    words = [
        _word(idx=1, start_s=0.0, end_s=0.2, alignment_origin="interpolated", match_score=None),
        _word(idx=2, start_s=0.2, end_s=0.4, alignment_origin="interpolated", match_score=None),
        _word(idx=3, start_s=0.4, end_s=0.6, alignment_origin="native", match_score=95.0),
    ]
    thresholds = QCThresholds.strict()
    qc = compute_qc(
        words=words,
        expected_word_count=3,
        audio_duration_s=0.6,
        thresholds=thresholds,
    )

    assert qc.interpolated_ratio > thresholds.max_interpolated_ratio
    assert qc_requires_fallback(qc, thresholds)
