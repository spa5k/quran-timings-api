from quran_audio_data.schema import AyahTiming, WordTiming
from quran_audio_data.sources.existing_timings import validate_external_timing


def test_validate_external_timing_rejects_malformed_durations() -> None:
    ayahs = [
        AyahTiming(surah=1, ayah=1, start_s=0.0, end_s=1.0, source="existing"),
    ]
    words = [
        WordTiming.model_construct(
            surah=1,
            ayah=1,
            word_index_global=1,
            word_index_in_ayah=1,
            text_uthmani="بِسْمِ",
            text_norm="بسم",
            start_s=0.4,
            end_s=0.2,
            confidence=0.9,
        )
    ]

    result = validate_external_timing(
        ayahs=ayahs,
        words=words,
        expected_word_count=1,
        audio_duration_s=1.0,
    )

    assert not result.ok
    assert "negative_word_duration" in result.warnings


def test_validate_external_timing_requires_lexical_scores() -> None:
    ayahs = [
        AyahTiming(surah=1, ayah=1, start_s=0.0, end_s=1.0, source="existing"),
    ]
    words = [
        WordTiming(
            surah=1,
            ayah=1,
            word_index_global=1,
            word_index_in_ayah=1,
            text_uthmani="بِسْمِ",
            text_norm="بسم",
            start_s=0.0,
            end_s=0.5,
            confidence=0.9,
            alignment_origin="native",
            match_score=None,
        )
    ]

    result = validate_external_timing(
        ayahs=ayahs,
        words=words,
        expected_word_count=1,
        audio_duration_s=1.0,
    )

    assert not result.ok
    assert "missing_lexical_scores" in result.warnings


def test_validate_external_timing_can_skip_lexical_score_requirement() -> None:
    ayahs = [
        AyahTiming(surah=1, ayah=1, start_s=0.0, end_s=1.0, source="existing"),
    ]
    words = [
        WordTiming(
            surah=1,
            ayah=1,
            word_index_global=1,
            word_index_in_ayah=1,
            text_uthmani="بِسْمِ",
            text_norm="بسم",
            start_s=0.0,
            end_s=1.0,
            confidence=0.9,
            alignment_origin="native",
            match_score=None,
        )
    ]

    result = validate_external_timing(
        ayahs=ayahs,
        words=words,
        expected_word_count=1,
        audio_duration_s=1.0,
        require_lexical_scores=False,
    )

    assert result.ok
    assert "missing_lexical_scores" not in result.warnings
