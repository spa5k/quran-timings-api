from __future__ import annotations

from quran_audio_data.sources.existing_timings import ExistingTimingResolver
from quran_audio_data.text.quran_text import CanonicalWord, normalize_arabic


def test_resolver_accepts_qf_payload_without_lexical_scores(monkeypatch) -> None:
    resolver = ExistingTimingResolver(enable_remote=False)

    payload = {
        "audio_file": {
            "timestamps": [
                {
                    "verse_key": "1:1",
                    "timestamp_from": 1000,
                    "timestamp_to": 2000,
                    "segments": [
                        [1, 1000.0, 1400.0],
                        [2, 1400.0, 2000.0],
                    ],
                }
            ]
        }
    }
    monkeypatch.setattr(
        resolver,
        "_build_candidates",
        lambda **_kwargs: [("qf:https://api.quran.foundation/example", payload)],
    )

    canonical_words = [
        CanonicalWord(
            surah=1,
            ayah=1,
            word_index_global=1,
            word_index_in_ayah=1,
            text_uthmani="بِسْمِ",
            text_norm=normalize_arabic("بِسْمِ"),
        ),
        CanonicalWord(
            surah=1,
            ayah=1,
            word_index_global=2,
            word_index_in_ayah=2,
            text_uthmani="ٱللَّهِ",
            text_norm=normalize_arabic("ٱللَّهِ"),
        ),
    ]

    resolved = resolver.resolve(
        reciter_id="reciter-1",
        surah=1,
        ayah=1,
        canonical_words=canonical_words,
        audio_duration_s=1.0,
        source_url=None,
    )

    assert resolved is not None
    assert resolved.source_name.startswith("qf:")
    assert len(resolved.words) == 2
