from quran_audio_data.alignment.whisperx_fallback import WhisperWord, _interpolate_slot


def test_interpolate_slot_uses_local_unmatched_window_for_leading_gap() -> None:
    predicted = [
        WhisperWord(text_norm="a", start_s=1.0, end_s=1.2, confidence=0.9),
    ]
    matched_idx = {2: 0}

    start_s, end_s = _interpolate_slot(
        index=0,
        total=10,
        matched_idx=matched_idx,
        predicted_words=predicted,
        audio_duration_s=5.0,
    )

    assert start_s == 0.0
    assert end_s == 0.5
