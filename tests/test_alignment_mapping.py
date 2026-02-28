from __future__ import annotations

from quran_audio_data.alignment.mapping import MappingConfig, map_canonical_words, to_prediction_spans
from quran_audio_data.text.quran_text import CanonicalWord, normalize_arabic


def _canonical_words() -> list[CanonicalWord]:
    words = ["بسم", "الله", "الرحمن", "الرحيم"]
    out: list[CanonicalWord] = []
    for idx, text in enumerate(words, start=1):
        out.append(
            CanonicalWord(
                surah=1,
                ayah=1,
                word_index_global=idx,
                word_index_in_ayah=idx,
                text_uthmani=text,
                text_norm=normalize_arabic(text),
            )
        )
    return out


def test_mapping_preserves_count_and_monotonicity_with_interpolation() -> None:
    predicted = [
        {"text": "بسم", "start": 0.0, "end": 0.2},
        {"text": "الرحيم", "start": 0.8, "end": 1.0},
    ]
    spans = to_prediction_spans(
        predicted_words=predicted,
        text_getter=lambda item: item["text"],
        start_getter=lambda item: item["start"],
        end_getter=lambda item: item["end"],
    )
    mapped = map_canonical_words(
        canonical_words=_canonical_words(),
        predicted_words=spans,
        audio_duration_s=1.0,
        config=MappingConfig(engine_candidate="test"),
    )

    assert len(mapped) == 4
    starts = [word.start_s for word in mapped]
    assert starts == sorted(starts)
    assert all(0.0 <= word.start_s <= word.end_s <= 1.0 for word in mapped)
    interpolated_or_distributed = [
        word for word in mapped if word.alignment_origin in {"interpolated", "distributed"}
    ]
    assert len(interpolated_or_distributed) >= 1
