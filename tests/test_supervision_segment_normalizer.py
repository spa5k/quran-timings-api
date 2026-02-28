from quran_audio_data.supervision.segment_normalizer import normalize_segments


def test_normalize_segments_handles_three_field_shape() -> None:
    segments = normalize_segments([
        [1, 120.0, 250.0],
        [2, 260.0, 410.0],
    ])

    assert [segment.word_index for segment in segments] == [1, 2]
    assert segments[0].start_ms == 120.0
    assert segments[0].end_ms == 250.0


def test_normalize_segments_handles_four_field_shape() -> None:
    segments = normalize_segments([
        [0, 1, 50.0, 100.0],
        [1, 2, 105.0, 210.0],
    ])

    assert [segment.word_index for segment in segments] == [1, 2]
    assert segments[1].start_ms == 105.0
    assert segments[1].end_ms == 210.0


def test_normalize_segments_prefers_longer_duplicate_window() -> None:
    segments = normalize_segments([
        [1, 10.0, 20.0],
        [1, 10.0, 30.0],
    ])

    assert len(segments) == 1
    assert segments[0].word_index == 1
    assert segments[0].end_ms == 30.0
