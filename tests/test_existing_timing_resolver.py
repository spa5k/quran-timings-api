from __future__ import annotations

import orjson

from quran_audio_data.sources.existing_timings import ExistingTimingResolver
from quran_audio_data.text.quran_text import CanonicalWord, normalize_arabic


def test_resolver_accepts_local_schema_like_payload(tmp_path) -> None:
    reciter_dir = tmp_path / "r1"
    reciter_dir.mkdir(parents=True)
    payload = {
        "ayahs": [
            {
                "surah": 1,
                "ayah": 1,
                "start_s": 0.0,
                "end_s": 1.0,
                "source": "existing",
            }
        ],
        "words": [
            {
                "surah": 1,
                "ayah": 1,
                "word_index_in_ayah": 1,
                "start_s": 0.0,
                "end_s": 0.4,
                "match_score": 90.0,
            },
            {
                "surah": 1,
                "ayah": 1,
                "word_index_in_ayah": 2,
                "start_s": 0.4,
                "end_s": 1.0,
                "match_score": 90.0,
            },
        ],
    }
    (reciter_dir / "001_001.json").write_bytes(orjson.dumps(payload))

    resolver = ExistingTimingResolver(cache_dir=tmp_path, enable_remote=False)

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
        reciter_id="r1",
        surah=1,
        ayah=1,
        canonical_words=canonical_words,
        audio_duration_s=1.0,
        source_url=None,
    )

    assert resolved is not None
    assert resolved.source_name.startswith("cache:")
    assert len(resolved.words) == 2
